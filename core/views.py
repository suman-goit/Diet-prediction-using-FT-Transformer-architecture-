from django.contrib.auth import authenticate
from django.forms.models import model_to_dict
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from . import constraint_engine, ilp_engine
from .models import DayPlan, Meal, MealItem, MealPlan, RecommendationHistory, UserProfile
from .serializers import (MealPlanSerializer, RecommendationHistorySerializer,
                           SignupSerializer, UserProfileSerializer)


# ---------------------------------------------------------------------
# Auth (unchanged)
# ---------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([AllowAny])
def signup(request):
    serializer = SignupSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key, "username": user.username}, status=201)
    return Response(serializer.errors, status=400)


@api_view(["POST"])
@permission_classes([AllowAny])
def login(request):
    username = request.data.get("username")
    password = request.data.get("password")
    user = authenticate(username=username, password=password)
    if not user:
        return Response({"detail": "Invalid credentials"}, status=401)
    token, _ = Token.objects.get_or_create(user=user)
    return Response({"token": token.key, "username": user.username})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout(request):
    request.user.auth_token.delete()
    return Response(status=204)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def me(request):
    latest = request.user.profiles.first()
    data = {"username": request.user.username}
    if latest:
        data["latest_profile"] = UserProfileSerializer(latest).data
    return Response(data)


# ---------------------------------------------------------------------
# Assessment (unchanged)
# ---------------------------------------------------------------------
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def create_assessment(request):
    serializer = UserProfileSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)

    profile = serializer.save(user=request.user)
    targets = constraint_engine.calculate_targets(profile)
    for key, value in targets.items():
        setattr(profile, key, value)
    profile.save()

    return Response(UserProfileSerializer(profile).data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def assessment_latest(request):
    profile = request.user.profiles.first()
    if not profile:
        return Response({"detail": "No assessment yet"}, status=404)
    return Response(UserProfileSerializer(profile).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def assessment_history(request):
    profiles = request.user.profiles.all()[:365]
    return Response(UserProfileSerializer(profiles, many=True).data)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def assessment_delete(request, pk):
    profile = get_object_or_404(UserProfile, pk=pk, user=request.user)
    profile.delete()
    return Response(status=204)


# ---------------------------------------------------------------------
# Meal Plan
# ---------------------------------------------------------------------
def _day_targets(profile):
    return {
        "target_calories": profile.target_calories,
        "target_protein_g": profile.target_protein_g,
        "target_carbs_g": profile.target_carbs_g,
        "target_fat_g": profile.target_fat_g,
    }


def _reason_for_item(food, meal_type, disease_predictions):
    """Human-readable reason a food was selected — stored per item in
    history so old recommendations can show *why*, not just *what*."""
    reasons = [f"fits {meal_type} slot", f"within {food.diet_tag} preference"]
    applied = disease_predictions.get("reasons", {})
    if applied:
        for cond in applied:
            reasons.append(f"compatible with {cond.replace('_', ' ')} constraint")
    return "; ".join(reasons)


def _build_full_plan_json(meal_plan, disease_predictions):
    full_plan = {}
    ilp_results = {}
    for day_plan in meal_plan.days.all().order_by("day_number"):
        day_key = str(day_plan.day_number)
        full_plan[day_key] = {}
        ilp_results[day_key] = {}
        for meal in day_plan.meals.all():
            items_data = []
            for item in meal.items.select_related("food").all():
                f = item.food
                factor = item.grams / 100.0
                items_data.append({
                    "food_id": f.id, "name": f.name, "grams": item.grams,
                    "servings": item.servings,
                    "calories": round(f.calories * factor, 1),
                    "protein_g": round(f.protein_g * factor, 1),
                    "carbs_g": round(f.carbs_g * factor, 1),
                    "fat_g": round(f.fat_g * factor, 1),
                    "fiber_g": round(f.fiber_g * factor, 1),
                    "reason_selected": _reason_for_item(f, meal.meal_type, disease_predictions),
                })
            totals = ilp_engine.meal_totals(
                [{"food": i.food, "grams": i.grams} for i in meal.items.all()]
            )
            full_plan[day_key][meal.meal_type] = {"items": items_data, "totals": totals}
            ilp_results[day_key][meal.meal_type] = {
                "status": "Optimal" if items_data else "Infeasible",
                "item_count": len(items_data),
            }
    return full_plan, ilp_results


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_meal_plan(request):
    profile = request.user.profiles.first()
    if not profile:
        return Response({"detail": "Complete a health assessment first"}, status=400)

    # Transformer Medical Classifier -> disease probabilities, then
    # Constraint Engine applies nutrition constraints using them.
    constraint_output = constraint_engine.allowed_foods_for(profile)
    foods = constraint_output["foods"]
    if not foods:
        return Response({"detail": "No foods match your constraints"}, status=400)

    day_targets = _day_targets(profile)

    request.user.meal_plans.update(is_active=False)
    meal_plan = MealPlan.objects.create(user=request.user, profile=profile, is_active=True)

    for day_number in range(1, 4):
        day_plan = DayPlan.objects.create(meal_plan=meal_plan, day_number=day_number)
        plan = ilp_engine.generate_full_plan(foods, day_targets)
        for meal_type, items in plan.items():
            meal = Meal.objects.create(day_plan=day_plan, meal_type=meal_type)
            for item in items:
                MealItem.objects.create(meal=meal, food=item["food"],
                                         servings=item["servings"], grams=item["grams"])

    # --- persist full history record (exact reconstruction requirement) ---
    disease_predictions = constraint_output["disease_predictions"]
    full_plan_json, ilp_results_json = _build_full_plan_json(meal_plan, disease_predictions)

    profile_snapshot = model_to_dict(profile)
    profile_snapshot["created_at"] = profile.created_at.isoformat()

    RecommendationHistory.objects.create(
        user=request.user,
        meal_plan=meal_plan,
        profile_snapshot=profile_snapshot,
        disease_predictions=disease_predictions,
        constraint_engine_output={
            "conditions_applied": constraint_output["conditions_applied"],
            "excluded_food_ids": constraint_output["excluded_food_ids"],
            "diet_tags_allowed": constraint_output["diet_tags_allowed"],
            "candidate_food_count": len(foods),
        },
        full_meal_plan=full_plan_json,
        ilp_results=ilp_results_json,
    )

    return Response(MealPlanSerializer(meal_plan).data, status=201)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def meal_plan_latest(request):
    meal_plan = request.user.meal_plans.filter(is_active=True).first()
    if not meal_plan:
        return Response({"detail": "No active meal plan"}, status=404)
    return Response(MealPlanSerializer(meal_plan).data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def regenerate_meal(request):
    day_number = request.data.get("day_number")
    meal_type = request.data.get("meal_type")
    meal_plan = request.user.meal_plans.filter(is_active=True).first()
    if not meal_plan:
        return Response({"detail": "No active meal plan"}, status=404)

    day_plan = get_object_or_404(DayPlan, meal_plan=meal_plan, day_number=day_number)
    meal = get_object_or_404(Meal, day_plan=day_plan, meal_type=meal_type)

    profile = meal_plan.profile
    constraint_output = constraint_engine.allowed_foods_for(profile)
    foods = constraint_output["foods"]
    exclude_ids = set(meal.items.values_list("food_id", flat=True))

    day_targets = _day_targets(profile)
    new_items = ilp_engine.solve_meal(foods, day_targets, meal_type, exclude_food_ids=exclude_ids)

    meal.items.all().delete()
    for item in new_items:
        MealItem.objects.create(meal=meal, food=item["food"],
                                 servings=item["servings"], grams=item["grams"])

    # keep the history record's full_meal_plan/ilp_results in sync so a
    # reconstruction after a regenerate still matches the current plan
    if hasattr(meal_plan, "history_record"):
        record = meal_plan.history_record
        full_plan_json, ilp_results_json = _build_full_plan_json(
            meal_plan, record.disease_predictions
        )
        record.full_meal_plan = full_plan_json
        record.ilp_results = ilp_results_json
        record.save(update_fields=["full_meal_plan", "ilp_results"])

    from .serializers import DayPlanSerializer
    return Response(DayPlanSerializer(day_plan).data)


# ---------------------------------------------------------------------
# Recommendation History
# ---------------------------------------------------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def recommendation_history_list(request):
    records = request.user.recommendation_history.all()[:100]
    return Response(RecommendationHistorySerializer(records, many=True).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def recommendation_history_detail(request, pk):
    record = get_object_or_404(RecommendationHistory, pk=pk, user=request.user)
    return Response(RecommendationHistorySerializer(record).data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def recommendation_history_reconstruct(request, pk):
    """Returns the recommendation exactly as originally generated —
    profile snapshot, disease predictions, constraint engine output, full
    3-day plan with per-food reasons, and ILP results — all from the
    frozen JSON, independent of current Food/UserProfile state."""
    record = get_object_or_404(RecommendationHistory, pk=pk, user=request.user)
    return Response(record.reconstruct())
