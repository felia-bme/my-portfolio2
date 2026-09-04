"""
train_model.py
----------------------------------------------------------------------
Melatih model klasifikasi Breast Cancer Wisconsin (Diagnostic) dan
menyimpannya sebagai `model.pkl` agar bisa dipakai langsung oleh app.py
(Streamlit).

Cara pakai (di VS Code / terminal):
    python train_model.py

Output:
    model.pkl   -> pipeline sklearn siap pakai (StandardScaler + model terbaik)

PENTING - konsistensi dengan app.py:
    app.py melakukan train_test_split(test_size=0.20, random_state=42,
    stratify=y) pada `breast_cancer_data.csv` untuk MENGHITUNG metrik
    evaluasi (seolah-olah X_test adalah data yang belum pernah dilihat
    model). Supaya metrik itu valid, script ini memakai split yang PERSIS
    SAMA, dan model HANYA dilatih dengan X_train (bukan seluruh dataset).
    Jangan ubah test_size/random_state di sini tanpa mengubahnya juga di
    app.py.
----------------------------------------------------------------------
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ----------------------------------------------------------------
# 1. KONSTANTA (harus identik dengan FEATURE_NAMES di app.py)
# ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "breast_cancer_data.csv"
MODEL_FILE = BASE_DIR / "model.pkl"

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]

RANDOM_STATE = 42


def clean_target(y: pd.Series) -> pd.Series:
    """Ubah kolom diagnosis (M/B, Malignant/Benign, atau 0/1) menjadi 0/1."""
    numeric = pd.to_numeric(y, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        return numeric.astype(int)

    mapping = {"B": 0, "BENIGN": 0, "0": 0, "M": 1, "MALIGNANT": 1, "1": 1}
    cleaned = y.astype(str).str.strip().str.upper().map(mapping)
    if cleaned.isna().any():
        raise ValueError("Kolom target berisi nilai yang tidak dikenali.")
    return cleaned.astype(int)


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Tidak menemukan {DATA_FILE}. Taruh CSV di folder yang sama.")
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom fitur berikut tidak ditemukan di CSV: {missing}")

    X = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Ada nilai kosong/non-numerik pada kolom fitur.")

    target_col = "target" if "target" in df.columns else "diagnosis"
    y = clean_target(df[target_col])
    return X, y


def build_candidates():
    """Beberapa kandidat model, semua dibungkus StandardScaler agar konsisten
    dengan cara app.py memanggil model.predict(X) langsung pada data mentah."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def main():
    print("Memuat data...")
    X, y = load_data()
    print(f"  -> {X.shape[0]} sampel, {X.shape[1]} fitur, "
          f"{int(y.sum())} malignant / {int((y == 0).sum())} benign")

    # Split IDENTIK dengan yang dipakai app.py untuk evaluasi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print("\nMembandingkan beberapa model dengan 5-fold cross-validation (ROC-AUC) "
          "di data training...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates()
    scores = {}
    for name, pipe in candidates.items():
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        scores[name] = cv_scores.mean()
        print(f"  {name:<18s} ROC-AUC = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    best_name = max(scores, key=scores.get)
    best_model = candidates[best_name]
    print(f"\nModel terbaik: {best_name} (CV ROC-AUC = {scores[best_name]:.4f})")

    print("\nMelatih ulang model terbaik pada seluruh data training...")
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("\n=== Evaluasi di data test (20% - belum pernah dilihat model) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    joblib.dump(best_model, MODEL_FILE)
    print(f"\nModel tersimpan di: {MODEL_FILE}")


if __name__ == "__main__":
    main()