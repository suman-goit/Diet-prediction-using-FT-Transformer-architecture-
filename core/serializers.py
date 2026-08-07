from django.contrib.auth.models import User
from rest_framework import serializers

from .models import (DayPlan, Food, Meal, MealItem, MealPlan,
                      RecommendationHistory, UserProfile)


class SignupSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=6)

    class Meta:
        model = User
        fields = ["username", "password"]

    def create(self, validated_data):
        user = User(username=validated_data["username"])
        user.set_password(validated_data["password"])
        user.save()
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = "__all__"
        read_only_fields = ["user", "created_at", "bmi", "bmr", "tdee",
                             "target_calories", "target_protein_g", "target_carbs_g",
                             "target_fat_g", "target_fiber_g", "target_omega3_g",
                             "target_sugar_g", "target_sodium_mg", "target_cholesterol_mg",
                             "target_iron_mg", "target_calcium_mg", "target_vitamin_c_mg"]


class FoodSerializer(serializers.ModelSerializer):
    class Meta:
        model = Food
        fields = ["id", "food_id", "name", "nepali_name", "category", "diet_tag",
                  "calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "omega3_g",
                  "sodium_mg", "sugar_g", "portion_grams", "household_measure"]


class MealItemSerializer(serializers.ModelSerializer):
    food = FoodSerializer(read_only=True)

    class Meta:
        model = MealItem
        fields = ["id", "food", "servings", "grams"]


class MealSerializer(serializers.ModelSerializer):
    items = MealItemSerializer(many=True, read_only=True)
    totals = serializers.SerializerMethodField()

    class Meta:
        model = Meal
        fields = ["id", "meal_type", "items", "totals"]

    def get_totals(self, obj):
        from . import ilp_engine
        items = [{"food": mi.food, "grams": mi.grams} for mi in obj.items.all()]
        return ilp_engine.meal_totals(items)


class DayPlanSerializer(serializers.ModelSerializer):
    meals = MealSerializer(many=True, read_only=True)

    class Meta:
        model = DayPlan
        fields = ["id", "day_number", "meals"]


class MealPlanSerializer(serializers.ModelSerializer):
    days = DayPlanSerializer(many=True, read_only=True)

    class Meta:
        model = MealPlan
        fields = ["id", "created_at", "is_active", "days"]


class RecommendationHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = RecommendationHistory
        fields = [
            "id", "created_at", "profile_snapshot", "disease_predictions",
            "constraint_engine_output", "full_meal_plan", "ilp_results",
        ]
