# Nepali Diet Planner — Backend (Django + DRF)

Implements the architecture doc: Constraint Engine, Decision Tree (medical
suitability), and ILP Engine (meal generation).

## What was already yours vs. what I filled in

You'd already built: `config/` (settings/urls/wsgi/asgi), `core/admin.py`,
`core/apps.py`, `core/constraint_engine.py`, `core/decision_tree.py`,
`core/serializers.py`, `core/views.py`, `core/urls.py`, the migration
`0001_initial.py`, and the trained `decision_tree.joblib`.

Missing pieces I reconstructed to make the app actually run, inferred from
the migration's field list and from how `views.py`/`serializers.py` call them:

- **`core/models.py`** — written to match `0001_initial.py` field-for-field.
- **`core/ilp_engine.py`** — PuLP goal-programming solver (`solve_meal`,
  `generate_full_plan`, `meal_totals`), matching the signatures your `views.py`
  and `serializers.py` already call.
- **`core/management/commands/seed_data.py`** — loads `data/food_table_clean.csv`.
  `diet_tag` isn't a column in that CSV, so it's derived from `category` (see
  `CATEGORY_*` constants in the file — plant-only categories tagged `Vegan`,
  Meat/Fish `Non-Vegetarian`, Eggs `Eggetarian`, everything else
  `Vegetarian`). Documented heuristic, not ground truth.
- **`core/management/commands/train_decision_tree.py`** — rebuilds training
  data from whatever's in the `Food` table and re-saves the joblib.
- **One fix in `constraint_engine.py`**: `DIET_ALLOWED` previously treated
  `Vegan` as disjoint from `Vegetarian`, which meant Vegan users would match
  zero foods (nothing in the source data was tagged `Vegan`) and Vegetarian
  users would never see Vegan-tagged food either. Changed it to a proper
  subset relationship (`Vegetarian ⊇ Vegan`, etc.) now that `seed_data` tags
  plant-only foods as `Vegan`.

## ⚠️ Not run in my sandbox

I don't have outbound network access in this environment, so I could not
`pip install` Django/PuLP/scikit-learn here to actually execute
`migrate`/`seed_data`/`runserver`. Everything above was written against your
existing files' exact function signatures and the migration's exact field
names, and every `.py` file passes `python -m py_compile` (syntax-checked),
but **please run the smoke test below on your machine before relying on it**
— that's the first real execution.

## Setup

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

python manage.py migrate
python manage.py seed_data              # loads data/food_table_clean.csv -> Food table
python manage.py train_decision_tree    # trains + saves the medical-suitability tree
python manage.py createsuperuser        # optional, for /admin/

python manage.py runserver
```

Server runs at http://localhost:8000 — API base path `/api/`.

## Smoke test (curl)

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/auth/signup/ \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demopass123"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")

curl -s -X POST localhost:8000/api/assessment/ -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","age":30,"height_cm":160,"weight_kg":60,"activity_level":"Moderate","goal":"Maintenance","food_preference":"Vegetarian","medical_conditions":"Hypertension","allergies":"","region":"Hills"}'

curl -s -X POST localhost:8000/api/meal-plan/generate/ -H "Authorization: Token $TOKEN"
```

If `meal-plan/generate/` returns `{"detail":"No foods match your constraints"}`,
check that `seed_data` actually ran and `region`/`food_preference` match a
real subset of the seeded foods.

## What's implemented

- **Models**: `Food`, `UserProfile` (append-only history), `MealPlan` ->
  `DayPlan` -> `Meal` -> `MealItem`
- **Constraint Engine** (`core/constraint_engine.py`): BMI/BMR(Mifflin-St
  Jeor)/TDEE -> goal-adjusted target calories -> macro targets, plus
  diet/allergy/region/availability filtering.
- **Decision Tree** (`core/decision_tree.py`): shallow
  `DecisionTreeClassifier` trained on sodium/sugar threshold labels
  (Hypertension/Diabetes scope only). Cached at
  `core/ml_artifacts/decision_tree.joblib`.
- **ILP Engine** (`core/ilp_engine.py`): PuLP-based per-meal solver,
  minimizes weighted deviation from calorie/protein/carb/fat sub-targets,
  splits a day 25/40/35 across Breakfast/Lunch/Dinner.
- **API** (`core/views.py`, `core/urls.py`): signup/login/logout/me,
  assessment CRUD + history, meal-plan generate/latest/regenerate-meal.
- **Data**: `data/food_table_clean.csv` (225 foods) is the single source
  loaded by `seed_data`.

## Known gaps / next steps

1. RDA/BDP lookup tables (`data/rda_benchmarks_clean.csv`,
   `data/balanced_diet_prescriptions_clean.csv`) aren't wired into
   `constraint_engine.calculate_targets` yet — targets are pure ratio-based
   off TDEE. Architecture doc section 4 flags this too.
2. No pagination/rate limiting on the history endpoint.
3. `MealItem` grams are per-meal only — no shopping-list aggregation
   (Groceries page is explicitly out of scope).
4. `seed_data`'s `diet_tag` heuristic (see above) may mistag a handful of
   "Cooked Foods" that actually contain meat (e.g. some momo/pakauda
   variants) — category alone can't distinguish these.
5. No automated tests yet (`core/tests.py` is still the Django stub).
6. Swap SQLite (dev default) for Postgres before any real deployment.
7. No `/api/assessment/history/pdf/` endpoint — the frontend's "Download as
   PDF" currently falls back to the browser print dialog.
