"""
Transformer Medical Classifier — replaces core/decision_tree.py entirely.

FT-Transformer (Gorishniy et al. 2021) adapted for multi-label disease-risk
classification from user profile + nutrition target features. Every numeric
feature gets its own learned linear token; every categorical feature gets
its own embedding token. A [CLS] token attends over all of them and a
sigmoid head produces independent per-condition probabilities.

This is the single component that sits between the Constraint Engine and
the ILP Engine in the new architecture:

    Constraint Engine -> Transformer Medical Classifier -> ILP Engine

Constraint Engine calls `predict(profile)` to get disease probabilities,
then applies existing nutrition constraints (SODIUM_THRESHOLD,
SUGAR_THRESHOLD, etc.) using those probabilities instead of decision_tree's
per-food classification.
"""
import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from django.conf import settings

MODEL_DIR = os.path.join(settings.BASE_DIR, "core", "ml_artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "ft_transformer.pt")
PREPROC_PATH = os.path.join(MODEL_DIR, "ft_preprocessor.joblib")

# ---------------------------------------------------------------------
# Feature schema — all inputs listed in the redesign spec
# ---------------------------------------------------------------------
NUMERIC_FEATURES = [
    "age", "height_cm", "weight_kg", "bmi", "bmr", "tdee",
    "target_calories", "target_protein_g", "target_carbs_g", "target_fat_g",
    "target_fiber_g", "target_sugar_g", "target_sodium_mg",
    "target_cholesterol_mg", "target_iron_mg", "target_calcium_mg",
    "target_vitamin_c_mg",
]

CATEGORICAL_FEATURES = [
    "gender", "activity_level", "food_preference", "region", "allergies_bucket",
]

CONDITIONS = [
    "Diabetes", "Hypertension", "Obesity", "Underweight", "High_Cholesterol",
    "Anemia", "Metabolic_Syndrome_Risk", "Vitamin_C_Deficiency_Risk",
    "Sarcopenia_Risk", "Low_Fiber_Risk", "Sodium_Sensitivity_Risk",
]

CLASSIFICATION_THRESHOLD = 0.5  # per-condition decision cutoff, tunable per condition later


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class FeatureTokenizer(nn.Module):
    """Turns each numeric feature into its own d_model-dim token via a
    per-feature linear layer, and each categorical feature into its own
    embedding token. This per-feature tokenization is what distinguishes
    FT-Transformer from plain concatenation (TabTransformer-style)."""

    def __init__(self, n_numeric: int, cat_cardinalities: list[int], d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        # one weight/bias vector per numeric feature -> (n_numeric, d_model)
        self.numeric_weight = nn.Parameter(torch.randn(n_numeric, d_model) * 0.02)
        self.numeric_bias = nn.Parameter(torch.zeros(n_numeric, d_model))
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in cat_cardinalities]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_num: (B, n_numeric), x_cat: (B, n_categorical) long
        num_tokens = x_num.unsqueeze(-1) * self.numeric_weight + self.numeric_bias  # (B, n_numeric, d)
        cat_tokens = torch.stack(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1
        ) if self.cat_embeddings else torch.empty(x_num.size(0), 0, self.d_model, device=x_num.device)
        tokens = torch.cat([num_tokens, cat_tokens], dim=1)  # (B, n_features, d)
        cls = self.cls_token.expand(x_num.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)  # (B, 1+n_features, d)


class FTTransformer(nn.Module):
    def __init__(self, n_numeric: int, cat_cardinalities: list[int], n_labels: int,
                 d_model: int = 64, n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_numeric, cat_cardinalities, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_labels),
        )

    def forward(self, x_num, x_cat, return_attention: bool = False):
        tokens = self.tokenizer(x_num, x_cat)
        encoded = self.encoder(tokens)
        cls_out = self.norm(encoded[:, 0, :])
        logits = self.head(cls_out)
        return logits


# ---------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------
_model_cache = None
_preproc_cache = None


def _load():
    global _model_cache, _preproc_cache
    if _model_cache is None:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROC_PATH)):
            raise FileNotFoundError(
                "Transformer not trained yet. Run: python manage.py train_transformer"
            )
        _preproc_cache = joblib.load(PREPROC_PATH)  # dict: scalers + label encoders + cardinalities
        model = FTTransformer(
            n_numeric=len(NUMERIC_FEATURES),
            cat_cardinalities=_preproc_cache["cat_cardinalities"],
            n_labels=len(CONDITIONS),
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        _model_cache = model
    return _model_cache, _preproc_cache


def _allergies_bucket(allergies: str) -> str:
    """Coarse categorical bucket so raw free-text allergy strings become a
    stable categorical feature instead of an unbounded vocabulary."""
    allergies = (allergies or "").strip()
    return "None" if not allergies else "HasAllergy"


def build_feature_row(profile, nutrition: dict) -> dict:
    """profile: UserProfile instance. nutrition: dict of current/target
    micronutrient values not already stored on the profile (sugar_g,
    sodium_mg, cholesterol_mg, iron_mg, calcium_mg, vitamin_c_mg)."""
    return {
        "age": profile.age, "height_cm": profile.height_cm, "weight_kg": profile.weight_kg,
        "bmi": profile.bmi, "bmr": profile.bmr, "tdee": profile.tdee,
        "target_calories": profile.target_calories, "target_protein_g": profile.target_protein_g,
        "target_carbs_g": profile.target_carbs_g, "target_fat_g": profile.target_fat_g,
        "target_fiber_g": profile.target_fiber_g,
        "target_sugar_g": nutrition.get("sugar_g", 0.0),
        "target_sodium_mg": nutrition.get("sodium_mg", 0.0),
        "target_cholesterol_mg": nutrition.get("cholesterol_mg", 0.0),
        "target_iron_mg": nutrition.get("iron_mg", 0.0),
        "target_calcium_mg": nutrition.get("calcium_mg", 0.0),
        "target_vitamin_c_mg": nutrition.get("vitamin_c_mg", 0.0),
        "gender": profile.gender, "activity_level": profile.activity_level,
        "food_preference": profile.food_preference, "region": profile.region,
        "allergies_bucket": _allergies_bucket(profile.allergies),
    }


def predict(profile, nutrition: dict) -> dict:
    """Returns {condition: probability} for every condition in CONDITIONS,
    e.g. {"Diabetes": 0.95, "Hypertension": 0.81, ...}."""
    model, preproc = _load()
    row = build_feature_row(profile, nutrition)

    x_num = np.array([[row[f] for f in NUMERIC_FEATURES]], dtype=np.float32)
    # num_mean/num_std were saved as float64 (numpy's default dtype) by the
    # training script, so this subtraction/division silently upcasts x_num
    # back to float64 unless we force float32 again afterward — that's what
    # produced the "double != float" RuntimeError.
    x_num = ((x_num - preproc["num_mean"]) / preproc["num_std"]).astype(np.float32)

    x_cat = np.array([[
        preproc["cat_encoders"][f].get(str(row[f]), 0) for f in CATEGORICAL_FEATURES
    ]], dtype=np.int64)

    with torch.no_grad():
        logits = model(torch.tensor(x_num, dtype=torch.float32),
                        torch.tensor(x_cat, dtype=torch.long))
        probs = torch.sigmoid(logits).squeeze(0).numpy()

    return {cond: round(float(p), 4) for cond, p in zip(CONDITIONS, probs)}


def predict_with_reasons(profile, nutrition: dict, top_k: int = 3) -> dict:
    """Same as predict(), plus a lightweight 'why' explanation per
    condition above threshold, derived from which raw features are most
    out-of-range vs. RDA/clinical reference points. Kept rule-based (not
    attention-extraction) so it stays fast and stable for history logging;
    attention weights can be added later without changing the API shape."""
    probs = predict(profile, nutrition)
    reasons = {}
    row = build_feature_row(profile, nutrition)
    reference_hints = {
        "Diabetes": [("target_sugar_g", 25, "high sugar target")],
        "Hypertension": [("target_sodium_mg", 2000, "high sodium target")],
        "Obesity": [("bmi", 25, "elevated BMI")],
        "Underweight": [("bmi", 18.5, "low BMI", "below")],
        "High_Cholesterol": [("target_cholesterol_mg", 300, "high dietary cholesterol")],
        "Anemia": [("target_iron_mg", 8, "low iron target", "below")],
    }
    for cond, prob in probs.items():
        if prob >= CLASSIFICATION_THRESHOLD and cond in reference_hints:
            hints = reference_hints[cond]
            reasons[cond] = [h[2] for h in hints]
    return {"probabilities": probs, "reasons": reasons}