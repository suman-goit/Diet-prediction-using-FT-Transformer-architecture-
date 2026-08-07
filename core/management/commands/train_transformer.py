"""
python manage.py train_transformer

Trains the FT-Transformer multi-label classifier on synthetic labels
generated from documented clinical/dietary reference thresholds (same
philosophy as the old decision_tree.py: no user behavior data, transparent
thresholds, but now applied to *user profiles* rather than individual foods,
and across 11 conditions instead of 2).

Data source: backend/data/synthetic_user_profiles_clean.csv. That file's
raw columns (age, sex, weight_kg, height_cm, bmi, activity_level, bmr_kcal,
tdee_kcal, diet_type, region, food_allergy, ...) don't line up 1:1 with the
NUMERIC_FEATURES/CATEGORICAL_FEATURES schema the transformer expects, so
`_prepare_dataframe()` below does the mapping + derives the target_* fields
(calories/protein/carbs/fat/fiber via constraint_engine's own formulas, and
sugar/sodium/cholesterol/iron/calcium/vitaminC as flat RDA targets with
small deterministic jitter so the model sees realistic variance). If you
later add real per-user micronutrient columns to the CSV, drop the jitter
step and read them directly instead.
"""
import os
import hashlib
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from django.conf import settings
from django.core.management.base import BaseCommand
from sklearn.model_selection import train_test_split

from core import constraint_engine
from core.ml.transformer_classifier import (
    FTTransformer, NUMERIC_FEATURES, CATEGORICAL_FEATURES, CONDITIONS,
    MODEL_DIR, MODEL_PATH, PREPROC_PATH,
)

DATA_PATH = os.path.join(settings.BASE_DIR, "data", "synthetic_user_profiles_clean.csv")


def _jitter(user_id: str, base: float, spread: float) -> float:
    """Deterministic pseudo-random jitter in [base-spread, base+spread],
    seeded from the row's user_id so training data is reproducible."""
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    frac = (h % 1000) / 1000.0  # 0..1
    return round(base + (frac - 0.5) * 2 * spread, 1)


def _prepare_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["age"] = raw["age"].astype(float)
    df["height_cm"] = raw["height_cm"].astype(float)
    df["weight_kg"] = raw["weight_kg"].astype(float)
    df["bmi"] = raw["bmi"].astype(float)
    df["bmr"] = raw["bmr_kcal"].astype(float)
    df["tdee"] = raw["tdee_kcal"].astype(float)

    # goal-agnostic default (Maintenance) target macros, same formulas as
    # constraint_engine.calculate_targets, applied row-wise for training data
    p_ratio, c_ratio, f_ratio = constraint_engine.MACRO_RATIO["Maintenance"]
    df["target_calories"] = df["tdee"]
    df["target_protein_g"] = (df["target_calories"] * p_ratio) / 4
    df["target_carbs_g"] = (df["target_calories"] * c_ratio) / 4
    df["target_fat_g"] = (df["target_calories"] * f_ratio) / 9
    df["target_fiber_g"] = (df["target_calories"] / 1000) * constraint_engine.FIBER_G_PER_1000KCAL

    df["target_sugar_g"] = [
        _jitter(uid, constraint_engine.SUGAR_G_PER_DAY, 10) for uid in raw["user_id"]
    ]
    df["target_sodium_mg"] = [
        _jitter(uid, constraint_engine.SODIUM_MG_PER_DAY, 600) for uid in raw["user_id"]
    ]
    df["target_cholesterol_mg"] = [
        _jitter(uid, constraint_engine.CHOLESTEROL_MG_PER_DAY, 100) for uid in raw["user_id"]
    ]
    df["target_iron_mg"] = [
        _jitter(uid, constraint_engine.IRON_MG_PER_DAY, 4) for uid in raw["user_id"]
    ]
    df["target_calcium_mg"] = [
        _jitter(uid, constraint_engine.CALCIUM_MG_PER_DAY, 300) for uid in raw["user_id"]
    ]
    df["target_vitamin_c_mg"] = [
        _jitter(uid, constraint_engine.VITAMIN_C_MG_PER_DAY, 30) for uid in raw["user_id"]
    ]

    df["gender"] = raw["sex"].astype(str)
    df["activity_level"] = raw["activity_level"].astype(str)
    df["food_preference"] = raw["diet_type"].astype(str)
    df["region"] = raw["region"].astype(str)
    df["allergies_bucket"] = raw["food_allergy"].astype(str).apply(
        lambda v: "None" if v.strip().lower() in ("", "none", "nan") else "HasAllergy"
    )
    return df

# Reference thresholds used to derive weak-supervision labels where the
# dataset doesn't already carry a ground-truth condition column.
THRESHOLDS = {
    "Diabetes": lambda r: r["target_sugar_g"] > 25 or r["bmi"] > 30,
    "Hypertension": lambda r: r["target_sodium_mg"] > 2000,
    "Obesity": lambda r: r["bmi"] >= 30,
    "Underweight": lambda r: r["bmi"] < 18.5,
    "High_Cholesterol": lambda r: r["target_cholesterol_mg"] > 300,
    "Anemia": lambda r: r["target_iron_mg"] < 8,
    "Metabolic_Syndrome_Risk": lambda r: r["bmi"] >= 27 and r["target_sugar_g"] > 20,
    "Vitamin_C_Deficiency_Risk": lambda r: r["target_vitamin_c_mg"] < 60,
    "Sarcopenia_Risk": lambda r: (r["target_protein_g"] / max(r["weight_kg"], 1)) < 0.8,
    "Low_Fiber_Risk": lambda r: r["target_fiber_g"] < 21,
    "Sodium_Sensitivity_Risk": lambda r: r["target_sodium_mg"] > 1500,
}


class Command(BaseCommand):
    help = "Train the FT-Transformer multi-label medical condition classifier"

    def add_arguments(self, parser):
        parser.add_argument("--epochs", type=int, default=40)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--lr", type=float, default=1e-3)

    def handle(self, *args, **opts):
        if not os.path.exists(DATA_PATH):
            self.stderr.write(f"Missing dataset: {DATA_PATH}")
            return
        raw = pd.read_csv(DATA_PATH)
        df = _prepare_dataframe(raw)

        missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
        if missing:
            self.stderr.write(f"Internal error: prepared dataframe still missing {missing}")
            return

        # Labels derived via documented thresholds (weak supervision)
        for cond in CONDITIONS:
            df[cond] = df.apply(THRESHOLDS.get(cond, lambda r: False), axis=1).astype(int)

        # --- preprocessing ---
        num = df[NUMERIC_FEATURES].astype(float).values
        num_mean, num_std = num.mean(axis=0), num.std(axis=0) + 1e-6
        num_norm = (num - num_mean) / num_std

        cat_encoders = {}
        cat_cardinalities = []
        cat_cols = []
        for col in CATEGORICAL_FEATURES:
            values = df[col].astype(str).unique().tolist()
            mapping = {v: i for i, v in enumerate(values)}
            cat_encoders[col] = mapping
            cat_cardinalities.append(len(mapping))
            cat_cols.append(df[col].astype(str).map(mapping).values)
        cat_arr = np.stack(cat_cols, axis=1)

        y = df[CONDITIONS].astype(float).values

        X_num_tr, X_num_te, X_cat_tr, X_cat_te, y_tr, y_te = train_test_split(
            num_norm, cat_arr, y, test_size=0.15, random_state=42
        )

        model = FTTransformer(
            n_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cardinalities,
            n_labels=len(CONDITIONS),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=opts["lr"], weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()

        X_num_tr_t = torch.tensor(X_num_tr, dtype=torch.float32)
        X_cat_tr_t = torch.tensor(X_cat_tr, dtype=torch.long)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

        n = X_num_tr_t.size(0)
        batch_size = opts["batch_size"]

        model.train()
        for epoch in range(opts["epochs"]):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                logits = model(X_num_tr_t[idx], X_cat_tr_t[idx])
                loss = loss_fn(logits, y_tr_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.stdout.write(f"epoch {epoch+1}/{opts['epochs']}  loss={epoch_loss/n:.4f}")

        # --- eval ---
        model.eval()
        with torch.no_grad():
            test_logits = model(torch.tensor(X_num_te, dtype=torch.float32),
                                 torch.tensor(X_cat_te, dtype=torch.long))
            test_probs = torch.sigmoid(test_logits).numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        per_cond_acc = (test_preds == y_te).mean(axis=0)
        for cond, acc in zip(CONDITIONS, per_cond_acc):
            self.stdout.write(f"  {cond}: test accuracy {acc:.3f}")

        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        joblib.dump({
            "num_mean": num_mean, "num_std": num_std,
            "cat_encoders": cat_encoders, "cat_cardinalities": cat_cardinalities,
        }, PREPROC_PATH)
        self.stdout.write(self.style.SUCCESS(f"Saved model to {MODEL_PATH}"))
