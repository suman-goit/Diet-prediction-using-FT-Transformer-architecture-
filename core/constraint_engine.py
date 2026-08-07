"""
Constraint Engine — "what should this person eat, how much, and what's allowed?"

Pipeline: BMI -> BMR (Mifflin-St Jeor) -> TDEE -> target calories (goal-adjusted)
-> target macros/micros (ratio + RDA-based) -> filtered food list (diet/allergy/
region/availability + Transformer Medical Classifier disease-probability filter).

Decision Tree is gone. No new Rule-Based Engine was added — the threshold
logic that used to live inside decision_tree.py's is_suitable() now lives
here, driven by the Transformer's per-condition probabilities instead of
per-food sklearn predictions. This is still "the Constraint Engine doing
constraint filtering," per your requirement — just fed richer input.
"""
from .ml import transformer_classifier
from .models import Food

ACTIVITY_MULTIPLIER = {
    "Sedentary": 1.2,
    "Lightly Active": 1.375,
    "Moderate": 1.55,
    "Very Active": 1.725,
    "Extra Active": 1.9,
}

GOAL_CALORIE_ADJUSTMENT = {
    "Weight Loss": -500,
    "Maintenance": 0,
    "Healthy Living": 0,
    "Muscle Gain": 300,
}

MACRO_RATIO = {
    "Weight Loss": (0.30, 0.40, 0.30),
    "Maintenance": (0.20, 0.55, 0.25),
    "Healthy Living": (0.20, 0.55, 0.25),
    "Muscle Gain": (0.30, 0.45, 0.25),
}

FIBER_G_PER_1000KCAL = 14
OMEGA3_G_PER_DAY = 1.6

# RDA-ish flat daily targets for the newly tracked micronutrients (used both
# as UserProfile targets and as transformer input features)
SUGAR_G_PER_DAY = 25
SODIUM_MG_PER_DAY = 2000
CHOLESTEROL_MG_PER_DAY = 300
IRON_MG_PER_DAY = 8
CALCIUM_MG_PER_DAY = 1000
VITAMIN_C_MG_PER_DAY = 90

# Disease probability above which the corresponding food-side nutrient
# constraint is activated. Tunable independently per condition.
CONDITION_PROBABILITY_THRESHOLD = {
    "Diabetes": 0.5,
    "Hypertension": 0.5,
    "Obesity": 0.5,
    "Underweight": 0.5,
    "High_Cholesterol": 0.5,
    "Anemia": 0.5,
    "Metabolic_Syndrome_Risk": 0.6,
    "Vitamin_C_Deficiency_Risk": 0.6,
    "Sarcopenia_Risk": 0.6,
    "Low_Fiber_Risk": 0.6,
    "Sodium_Sensitivity_Risk": 0.6,
}

# Per-100g food-side nutrient thresholds applied when a condition fires
# (same WHO/ADA-style reference points the old decision_tree.py used,
# extended to the new conditions)
CONDITION_FOOD_FILTERS = {
    "Hypertension": lambda f: f.sodium_mg <= 400,
    "Sodium_Sensitivity_Risk": lambda f: f.sodium_mg <= 400,
    "Diabetes": lambda f: f.sugar_g <= 15,
    "Metabolic_Syndrome_Risk": lambda f: f.sugar_g <= 15,
    "Low_Fiber_Risk": lambda f: f.fiber_g >= 2,
}


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)


def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    if gender.strip().lower() == "male":
        return round(base + 5, 1)
    return round(base - 161, 1)


def calculate_tdee(bmr: float, activity_level: str) -> float:
    mult = ACTIVITY_MULTIPLIER.get(activity_level, 1.2)
    return round(bmr * mult, 1)


def calculate_targets(profile) -> dict:
    """profile: UserProfile instance (unsaved is fine, just needs the fields)."""
    bmi = calculate_bmi(profile.weight_kg, profile.height_cm)
    bmr = calculate_bmr(profile.weight_kg, profile.height_cm, profile.age, profile.gender)
    tdee = calculate_tdee(bmr, profile.activity_level)

    target_calories = tdee + GOAL_CALORIE_ADJUSTMENT.get(profile.goal, 0)
    target_calories = max(target_calories, 1200)

    p_ratio, c_ratio, f_ratio = MACRO_RATIO.get(profile.goal, MACRO_RATIO["Maintenance"])

    return {
        "bmi": bmi, "bmr": bmr, "tdee": tdee,
        "target_calories": round(target_calories, 1),
        "target_protein_g": round((target_calories * p_ratio) / 4, 1),
        "target_carbs_g": round((target_calories * c_ratio) / 4, 1),
        "target_fat_g": round((target_calories * f_ratio) / 9, 1),
        "target_fiber_g": round((target_calories / 1000) * FIBER_G_PER_1000KCAL, 1),
        "target_omega3_g": OMEGA3_G_PER_DAY,
        # newly tracked micronutrient targets — feed the transformer AND
        # future nutrition-summary UI
        "target_sugar_g": SUGAR_G_PER_DAY,
        "target_sodium_mg": SODIUM_MG_PER_DAY,
        "target_cholesterol_mg": CHOLESTEROL_MG_PER_DAY,
        "target_iron_mg": IRON_MG_PER_DAY,
        "target_calcium_mg": CALCIUM_MG_PER_DAY,
        "target_vitamin_c_mg": VITAMIN_C_MG_PER_DAY,
    }


DIET_ALLOWED = {
    "Vegetarian": {"Vegetarian", "Vegan"},
    "Vegan": {"Vegan"},
    "Eggetarian": {"Vegetarian", "Vegan", "Eggetarian"},
    "Non-Vegetarian": {"Vegetarian", "Vegan", "Eggetarian", "Non-Vegetarian"},
}

ALLERGY_CATEGORY_KEYWORDS = {
    "Milk": ["milk"],
    "Nuts": ["nuts & oilseeds"],
    "Fish": ["fish"],
    "Gluten": ["wheat", "barley"],
}

REGION_FIELD = {
    "Mountain": "region_mountain",
    "Hills": "region_hills",
    "Terai": "region_terai",
}


def get_disease_predictions(profile) -> dict:
    """Runs the Transformer Medical Classifier. Returns
    {"probabilities": {...}, "reasons": {...}}."""
    nutrition = {
        "sugar_g": profile.target_sugar_g or SUGAR_G_PER_DAY,
        "sodium_mg": profile.target_sodium_mg or SODIUM_MG_PER_DAY,
        "cholesterol_mg": profile.target_cholesterol_mg or CHOLESTEROL_MG_PER_DAY,
        "iron_mg": profile.target_iron_mg or IRON_MG_PER_DAY,
        "calcium_mg": profile.target_calcium_mg or CALCIUM_MG_PER_DAY,
        "vitamin_c_mg": profile.target_vitamin_c_mg or VITAMIN_C_MG_PER_DAY,
    }
    return transformer_classifier.predict_with_reasons(profile, nutrition)


def allowed_foods_for(profile, disease_predictions: dict | None = None) -> dict:
    """Applies diet/allergy/region/availability filters, then the
    Transformer-driven medical filter. Returns
    {"foods": [...], "conditions_applied": [...], "excluded_food_ids": [...]}
    so the caller can persist the full constraint-engine output to history.
    """
    qs = Food.objects.filter(available_year_round=True)

    region_field = REGION_FIELD.get(profile.region)
    if region_field:
        qs = qs.filter(**{region_field: True})

    diet_tags = DIET_ALLOWED.get(profile.food_preference, DIET_ALLOWED["Non-Vegetarian"])
    qs = qs.filter(diet_tag__in=diet_tags)

    foods = list(qs)
    pre_medical_ids = {f.id for f in foods}

    allergies = [a.strip() for a in (profile.allergies or "").split(",") if a.strip()]
    for allergy in allergies:
        keywords = ALLERGY_CATEGORY_KEYWORDS.get(allergy, [allergy.lower()])
        foods = [
            f for f in foods
            if not any(kw in f.category.lower() or kw in f.name.lower() for kw in keywords)
        ]

    if disease_predictions is None:
        disease_predictions = get_disease_predictions(profile)
    probs = disease_predictions["probabilities"]

    conditions_applied = [
        cond for cond, prob in probs.items()
        if prob >= CONDITION_PROBABILITY_THRESHOLD.get(cond, 0.5) and cond in CONDITION_FOOD_FILTERS
    ]
    for cond in conditions_applied:
        filter_fn = CONDITION_FOOD_FILTERS[cond]
        foods = [f for f in foods if filter_fn(f)]

    post_medical_ids = {f.id for f in foods}
    excluded_food_ids = list(pre_medical_ids - post_medical_ids)

    return {
        "foods": foods,
        "conditions_applied": conditions_applied,
        "excluded_food_ids": excluded_food_ids,
        "diet_tags_allowed": sorted(diet_tags),
        "disease_predictions": disease_predictions,
    }
