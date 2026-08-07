"""
python manage.py seed_data

Loads data/food_table_clean.csv into the Food table. diet_tag isn't a column
in the source CSV, so it's derived from `category` with a documented
heuristic — see CATEGORY_DIET_TAG below. This is a simplification for the
prototype: a handful of "Cooked Foods" (e.g. momo) may actually contain meat
but get tagged Vegetarian by default since category alone can't tell.
"""
import os

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from core.models import Food

DATA_PATH = os.path.join(settings.BASE_DIR, "data", "food_table_clean.csv")

# Plant-only categories -> Vegan (a subset of Vegetarian, handled in
# constraint_engine's DIET_ALLOWED superset mapping)
VEGAN_CATEGORIES = {
    "Cereals", "Pulses", "Green Leafy", "Vegetables", "Roots & Tubers",
    "Fruits", "Condiments", "Nuts & Oilseeds", "Veg Products",
}
NON_VEG_CATEGORIES = {"Meat", "Fish"}
EGG_CATEGORIES = {"Eggs"}
# everything else (Milk, Fats & Oils, Misc, Cooked Foods, Supplementary)
# defaults to Vegetarian


def diet_tag_for(category: str) -> str:
    if category in NON_VEG_CATEGORIES:
        return "Non-Vegetarian"
    if category in EGG_CATEGORIES:
        return "Eggetarian"
    if category in VEGAN_CATEGORIES:
        return "Vegan"
    return "Vegetarian"


class Command(BaseCommand):
    help = "Seed the Food table from data/food_table_clean.csv"

    def handle(self, *args, **options):
        if not os.path.exists(DATA_PATH):
            self.stderr.write(self.style.ERROR(f"File not found: {DATA_PATH}"))
            return

        df = pd.read_csv(DATA_PATH)
        created, updated = 0, 0

        for _, row in df.iterrows():
            defaults = dict(
                name=row["food_name"],
                nepali_name=row.get("nepali_name") or "",
                category=row["category"],
                diet_tag=diet_tag_for(row["category"]),
                calories=float(row["calories"]),
                protein_g=float(row["protein_g"]),
                carbs_g=float(row["carbs_g"]),
                fat_g=float(row["fat_g"]),
                fiber_g=float(row["fiber_g"]),
                omega3_g=float(row.get("omega3_g", 0) or 0),
                sodium_mg=float(row.get("sodium_mg", 0) or 0),
                sugar_g=float(row.get("sugar_g", 0) or 0),
                portion_grams=float(row.get("portion_grams", 100) or 100),
                household_measure=row.get("household_measure") or "100g serving",
                price_tier=int(row.get("price_tier", 2) or 2),
                available_year_round=bool(row.get("available_year_round", True)),
                region_mountain=bool(row.get("region_mountain", True)),
                region_hills=bool(row.get("region_hills", True)),
                region_terai=bool(row.get("region_terai", True)),
                nutrient_estimate_source=row.get("nutrient_estimate_source") or "",
            )
            _, was_created = Food.objects.update_or_create(
                food_id=row["food_id"], defaults=defaults
            )
            created += was_created
            updated += not was_created

        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded {len(df)} foods ({created} created, {updated} updated)."
            )
        )
