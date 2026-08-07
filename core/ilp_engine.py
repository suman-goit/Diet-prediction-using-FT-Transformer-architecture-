"""
ILP Engine — "given allowed foods and targets, what exact combo/quantities fit best?"

PuLP-based per-meal solver. Splits the day's targets 25/40/35 across
Breakfast/Lunch/Dinner, then for each meal picks a small set of foods and
serving counts that minimize weighted absolute deviation from that meal's
calorie/protein/carb/fat sub-targets (goal programming, not a hard feasibility
solve — real food data is rarely exactly satisfiable, so we minimize distance
instead of requiring an exact match).

`Regenerate` reuses this same function against the same day_targets, just
with the excluded food id(s) added — see architecture doc section 4.
"""
import pulp

MEAL_SPLIT = {"Breakfast": 0.25, "Lunch": 0.40, "Dinner": 0.35}

# how many distinct foods a single meal is allowed to draw from
MAX_ITEMS_PER_MEAL = 5
MIN_ITEMS_PER_MEAL = 2

# each chosen food can be scaled 0.5x-3x its standard portion
MIN_SERVINGS = 0.5
MAX_SERVINGS = 3.0

# relative weights in the objective — calories matter most, then protein
DEVIATION_WEIGHTS = {"calories": 1.0, "protein_g": 2.0, "carbs_g": 0.5, "fat_g": 0.5}


def _meal_targets(day_targets: dict, meal_type: str) -> dict:
    split = MEAL_SPLIT.get(meal_type, 1 / 3)
    return {
        "calories": day_targets["target_calories"] * split,
        "protein_g": day_targets["target_protein_g"] * split,
        "carbs_g": day_targets["target_carbs_g"] * split,
        "fat_g": day_targets["target_fat_g"] * split,
    }


def _food_suits_meal(food, meal_type: str) -> bool:
    allowed = [m.strip() for m in (food.suitable_meals or "").split(",") if m.strip()]
    return not allowed or meal_type in allowed


def solve_meal(foods, day_targets: dict, meal_type: str, exclude_food_ids=None):
    """foods: iterable of Food instances (already filtered by Constraint
    Engine + Decision Tree). Returns a list of {"food": Food, "servings": float,
    "grams": float} for the foods chosen for this meal."""
    exclude_food_ids = exclude_food_ids or set()
    candidates = [f for f in foods if f.id not in exclude_food_ids and _food_suits_meal(f, meal_type)]

    if not candidates:
        return []

    targets = _meal_targets(day_targets, meal_type)
    n_available = len(candidates)
    max_items = min(MAX_ITEMS_PER_MEAL, n_available)
    min_items = min(MIN_ITEMS_PER_MEAL, n_available)

    prob = pulp.LpProblem("meal_plan", pulp.LpMinimize)

    chosen = {f.id: pulp.LpVariable(f"chosen_{f.id}", cat="Binary") for f in candidates}
    servings = {
        f.id: pulp.LpVariable(f"servings_{f.id}", lowBound=0, upBound=MAX_SERVINGS)
        for f in candidates
    }

    # link servings to the chosen flag: 0 if not chosen, MIN..MAX if chosen
    for f in candidates:
        prob += servings[f.id] <= MAX_SERVINGS * chosen[f.id]
        prob += servings[f.id] >= MIN_SERVINGS * chosen[f.id]

    prob += pulp.lpSum(chosen.values()) <= max_items
    prob += pulp.lpSum(chosen.values()) >= min_items

    # nutrient totals as linear expressions (per-100g values scaled by grams)
    def total_for(attr):
        return pulp.lpSum(
            servings[f.id] * f.portion_grams * getattr(f, attr) / 100.0 for f in candidates
        )

    totals = {attr: total_for(attr) for attr in DEVIATION_WEIGHTS}

    # goal-programming deviation variables (over + under = |actual - target|)
    over = {}
    under = {}
    for attr in DEVIATION_WEIGHTS:
        over[attr] = pulp.LpVariable(f"over_{attr}", lowBound=0)
        under[attr] = pulp.LpVariable(f"under_{attr}", lowBound=0)
        prob += totals[attr] - targets[attr] == over[attr] - under[attr]

    prob += pulp.lpSum(
        DEVIATION_WEIGHTS[attr] * (over[attr] + under[attr]) for attr in DEVIATION_WEIGHTS
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    items = []
    for f in candidates:
        if pulp.value(chosen[f.id]) and pulp.value(chosen[f.id]) > 0.5:
            serv = pulp.value(servings[f.id]) or 0
            if serv <= 0:
                continue
            grams = round(serv * f.portion_grams, 1)
            items.append({"food": f, "servings": round(serv, 2), "grams": grams})

    return items


def generate_full_plan(foods, day_targets: dict) -> dict:
    """Solves each meal independently against its own slice of the day's
    targets. Returns {meal_type: [items...]}."""
    plan = {}
    used_ids = set()
    for meal_type in ("Breakfast", "Lunch", "Dinner"):
        items = solve_meal(foods, day_targets, meal_type, exclude_food_ids=used_ids)
        plan[meal_type] = items
        # keep some variety across meals within the same day, without being
        # so strict that a small food list makes later meals infeasible
        if len(foods) > MAX_ITEMS_PER_MEAL * 3:
            used_ids |= {item["food"].id for item in items}
    return plan


def meal_totals(items) -> dict:
    """items: iterable of {"food": Food, "grams": float}. Returns summed
    nutrient totals for display (Meal.totals in the serializer)."""
    totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0,
              "fiber_g": 0.0, "omega3_g": 0.0}
    for item in items:
        food = item["food"]
        grams = item["grams"]
        factor = grams / 100.0
        totals["calories"] += food.calories * factor
        totals["protein_g"] += food.protein_g * factor
        totals["carbs_g"] += food.carbs_g * factor
        totals["fat_g"] += food.fat_g * factor
        totals["fiber_g"] += food.fiber_g * factor
        totals["omega3_g"] += food.omega3_g * factor
    return {k: round(v, 1) for k, v in totals.items()}
