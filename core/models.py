from django.conf import settings
from django.db import models


class Food(models.Model):
    DIET_TAG_CHOICES = [
        ("Vegetarian", "Vegetarian"),
        ("Vegan", "Vegan"),
        ("Non-Vegetarian", "Non-Vegetarian"),
        ("Eggetarian", "Eggetarian"),
    ]

    food_id = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=200)
    nepali_name = models.CharField(max_length=200, blank=True, default="")
    description = models.CharField(max_length=255, blank=True, default="")
    category = models.CharField(max_length=50)
    diet_tag = models.CharField(max_length=20, choices=DIET_TAG_CHOICES, default="Vegetarian")
    suitable_meals = models.CharField(max_length=100, default="Breakfast,Lunch,Dinner")

    # the 6 tracked "vitals" (architecture doc section 7)
    calories = models.FloatField(default=0)
    protein_g = models.FloatField(default=0)
    carbs_g = models.FloatField(default=0)
    fat_g = models.FloatField(default=0)
    fiber_g = models.FloatField(default=0)
    omega3_g = models.FloatField(default=0)

    # medical-condition-only fields (architecture doc section 3)
    sodium_mg = models.FloatField(default=0)
    sugar_g = models.FloatField(default=0)

    portion_grams = models.FloatField(default=100)
    household_measure = models.CharField(max_length=100, default="100g serving")
    price_tier = models.IntegerField(default=2)
    available_year_round = models.BooleanField(default=True)
    region_mountain = models.BooleanField(default=True)
    region_hills = models.BooleanField(default=True)
    region_terai = models.BooleanField(default=True)
    nutrient_estimate_source = models.CharField(max_length=30, blank=True, default="")

    def __str__(self):
        return self.name


class UserProfile(models.Model):
    ACTIVITY_CHOICES = [
        ("Sedentary", "Sedentary"),
        ("Lightly Active", "Lightly Active"),
        ("Moderate", "Moderate"),
        ("Very Active", "Very Active"),
        ("Extra Active", "Extra Active"),
    ]
    GOAL_CHOICES = [
        ("Weight Loss", "Weight Loss"),
        ("Maintenance", "Maintenance"),
        ("Muscle Gain", "Muscle Gain"),
        ("Healthy Living", "Healthy Living"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                              related_name="profiles")
    created_at = models.DateTimeField(auto_now_add=True)

    gender = models.CharField(max_length=20)
    age = models.PositiveIntegerField()
    height_cm = models.FloatField()
    weight_kg = models.FloatField()
    activity_level = models.CharField(max_length=20, choices=ACTIVITY_CHOICES)
    goal = models.CharField(max_length=20, choices=GOAL_CHOICES)
    food_preference = models.CharField(max_length=20, default="Vegetarian")
    medical_conditions = models.CharField(max_length=200, blank=True, default="")
    allergies = models.CharField(max_length=200, blank=True, default="")
    region = models.CharField(max_length=20, default="Hills")

    bmi = models.FloatField(null=True, blank=True)
    bmr = models.FloatField(null=True, blank=True)
    tdee = models.FloatField(null=True, blank=True)
    target_calories = models.FloatField(null=True, blank=True)
    target_protein_g = models.FloatField(null=True, blank=True)
    target_carbs_g = models.FloatField(null=True, blank=True)
    target_fat_g = models.FloatField(null=True, blank=True)
    target_fiber_g = models.FloatField(null=True, blank=True)
    target_omega3_g = models.FloatField(null=True, blank=True)

    # micronutrient targets — feed the Transformer Medical Classifier and
    # power future nutrition-summary UI
    target_sugar_g = models.FloatField(null=True, blank=True)
    target_sodium_mg = models.FloatField(null=True, blank=True)
    target_cholesterol_mg = models.FloatField(null=True, blank=True)
    target_iron_mg = models.FloatField(null=True, blank=True)
    target_calcium_mg = models.FloatField(null=True, blank=True)
    target_vitamin_c_mg = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.username} @ {self.created_at:%Y-%m-%d %H:%M}"


class MealPlan(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                              related_name="meal_plans")
    profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE,
                                 related_name="meal_plans")
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"MealPlan#{self.pk} for {self.user.username}"


class DayPlan(models.Model):
    meal_plan = models.ForeignKey(MealPlan, on_delete=models.CASCADE, related_name="days")
    day_number = models.PositiveSmallIntegerField()

    class Meta:
        ordering = ["day_number"]
        unique_together = ("meal_plan", "day_number")

    def __str__(self):
        return f"Day {self.day_number}"


class Meal(models.Model):
    MEAL_TYPE_CHOICES = [
        ("Breakfast", "Breakfast"),
        ("Lunch", "Lunch"),
        ("Dinner", "Dinner"),
    ]

    day_plan = models.ForeignKey(DayPlan, on_delete=models.CASCADE, related_name="meals")
    meal_type = models.CharField(max_length=20, choices=MEAL_TYPE_CHOICES)

    class Meta:
        unique_together = ("day_plan", "meal_type")

    def __str__(self):
        return self.meal_type


class MealItem(models.Model):
    meal = models.ForeignKey(Meal, on_delete=models.CASCADE, related_name="items")
    food = models.ForeignKey(Food, on_delete=models.CASCADE)
    servings = models.FloatField(default=1)
    grams = models.FloatField(default=100)

    def __str__(self):
        return f"{self.grams}g {self.food.name}"


class RecommendationHistory(models.Model):
    """One row per generate_meal_plan() call. Stores everything needed to
    reconstruct that recommendation exactly, independent of later edits to
    Food/UserProfile: a frozen profile snapshot, the Transformer's disease
    predictions, the Constraint Engine's filtering output, the complete
    3-day meal plan (with per-food selection reasons), and ILP results."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                              related_name="recommendation_history")
    meal_plan = models.OneToOneField(MealPlan, on_delete=models.CASCADE,
                                      related_name="history_record")
    created_at = models.DateTimeField(auto_now_add=True)

    profile_snapshot = models.JSONField()
    disease_predictions = models.JSONField()
    constraint_engine_output = models.JSONField()
    full_meal_plan = models.JSONField()
    ilp_results = models.JSONField()

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"RecommendationHistory#{self.pk} for {self.user.username} @ {self.created_at:%Y-%m-%d}"

    def reconstruct(self) -> dict:
        return {
            "recommendation_id": self.pk,
            "timestamp": self.created_at.isoformat(),
            "user_profile_snapshot": self.profile_snapshot,
            "disease_predictions": self.disease_predictions,
            "constraint_engine_output": self.constraint_engine_output,
            "meal_plan": self.full_meal_plan,
            "ilp_results": self.ilp_results,
        }
