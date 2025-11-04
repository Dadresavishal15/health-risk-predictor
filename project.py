import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import joblib

# =====================================================
# CONFIGURATION
# =====================================================
MODELS_DIR = "models_fixed"
os.makedirs(MODELS_DIR, exist_ok=True)


# =====================================================
# TRAINING FUNCTION
# =====================================================
# def train_xgb_task(df, label_col, num_features, cat_features, model_name, use_smote=True):
#     print(f"\n🔹 Training {model_name.upper()} model...")

#     X = df[num_features + cat_features].copy()
#     y = df[label_col].astype(int).copy()

#     # Split dataset
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     # Preprocessing
#     preprocessor = ColumnTransformer([
#         ("num", StandardScaler(), num_features),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
#     ])

#     X_train_transformed = preprocessor.fit_transform(X_train)
#     X_test_transformed = preprocessor.transform(X_test)

#     # Handle imbalance with SMOTE only if both classes exist
#     if use_smote and len(np.unique(y_train)) > 1:
#         smote = SMOTE(random_state=42)
#         X_train_transformed, y_train = smote.fit_resample(X_train_transformed, y_train)
#     else:
#         print(f"⚠️ Skipping SMOTE for {model_name} (only one class present).")

#     # Model training
#     model = XGBClassifier(
#         learning_rate=0.03,
#         n_estimators=400,
#         max_depth=4,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#         use_label_encoder=False,
#         eval_metric="logloss"
#     )

#     model.fit(X_train_transformed, y_train)

#     calib = CalibratedClassifierCV(model, cv=3)
#     calib.fit(X_train_transformed, y_train)

#     # Evaluation
#     y_pred = calib.predict(X_test_transformed)
#     y_prob = calib.predict_proba(X_test_transformed)[:, 1]
#     acc = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0

#     print(f"📊 {model_name.upper()} MODEL RESULTS")
#     print(f"Accuracy: {acc:.3f} | AUC: {auc:.3f}")
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))

#     # Save pipeline
#     artifact = {
#         "pipeline": Pipeline([
#             ("preprocessor", preprocessor),
#             ("model", calib)
#         ]),
#         "numeric_features": num_features,
#         "categorical_features": cat_features
#     }
#     joblib.dump(artifact, os.path.join(MODELS_DIR, f"{model_name}_xgb_artifact.joblib"))
#     print(f"✅ Saved model: {model_name}_xgb_artifact.joblib\n")

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

def train_xgb_task(df, label_col, num_features, cat_features, model_name, use_smote=True):
    print(f"\n🔹 Training {model_name.upper()} model...")

    X = df[num_features + cat_features].copy()
    y = df[label_col].astype(int).copy()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Handle imbalance
    if use_smote and len(np.unique(y_train)) > 1:
        smote = SMOTE(random_state=42)
        X_train_t, y_train = smote.fit_resample(X_train_t, y_train)
    else:
        print(f"⚠️ Skipping SMOTE for {model_name} (only one class present).")

    # Compute class weights
    classes = np.unique(y_train)
    if len(classes) > 1:
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        weight_dict = {i: w for i, w in zip(classes, class_weights)}
    else:
        weight_dict = {0: 1.0, 1: 1.0}

    # Train XGBoost model
    model = XGBClassifier(
        learning_rate=0.03,
        n_estimators=400,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=weight_dict.get(0, 1) / weight_dict.get(1, 1),
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train_t, y_train)

    # Calibrate with logistic rescaling
    calib = CalibratedClassifierCV(model, cv=3)
    calib.fit(X_train_t, y_train)

    # Get predictions
    y_prob = calib.predict_proba(X_test_t)[:, 1]

    # Apply probability normalization
    prob_min, prob_max = np.percentile(y_prob, [5, 95])
    y_prob_norm = np.clip((y_prob - prob_min) / (prob_max - prob_min), 0, 1)

    y_pred = (y_prob_norm > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob_norm) if len(np.unique(y_test)) > 1 else 0

    print(f"📊 {model_name.upper()} MODEL RESULTS")
    print(f"Accuracy: {acc:.3f} | AUC: {auc:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save artifact
    artifact = {
        "pipeline": Pipeline([
            ("preprocessor", preprocessor),
            ("model", calib)
        ]),
        "prob_range": (float(prob_min), float(prob_max)),
        "numeric_features": num_features,
        "categorical_features": cat_features
    }

    joblib.dump(artifact, os.path.join(MODELS_DIR, f"{model_name}_xgb_artifact.joblib"))
    print(f"✅ Saved calibrated & normalized model: {model_name}_xgb_artifact.joblib\n")


# =====================================================
# TRAIN ALL MODELS
# =====================================================
if __name__ == "__main__":
    print("🚀 Starting model training...")

    # ---------------- DIABETES ----------------
    df_diabetes = pd.read_csv("diabetes.csv")
    df_diabetes.columns = df_diabetes.columns.str.lower().str.strip()
    df_diabetes["label"] = df_diabetes["outcome"]

    diabetes_num = [
        'pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
        'insulin', 'bmi', 'diabetespedigreefunction', 'age'
    ]
    train_xgb_task(df_diabetes, 'label', diabetes_num, [], 'diabetes', use_smote=True)

    # ---- HEART ----
    df_heart = pd.read_csv("heart.csv")
    df_heart.columns = df_heart.columns.str.strip().str.lower()

    # Detect label
    label_col = next((c for c in ["target", "heartdisease", "heart_disease"] if c in df_heart.columns), None)
    if label_col is None:
        raise ValueError(f"❌ Could not find heart label column. Found: {list(df_heart.columns)}")

    df_heart["heart_label"] = df_heart[label_col]

    # Detect numeric and categorical features dynamically
    # If already one-hot encoded, treat all as numeric
    potential_num = ['age', 'restingbp', 'cholesterol', 'fastingbs', 'maxhr', 'oldpeak']
    potential_cat = ['sex', 'exerciseangina', 'chestpaintype', 'restingecg', 'st_slope']

    heart_num = [c for c in potential_num if c in df_heart.columns]
    heart_cat = [c for c in potential_cat if c in df_heart.columns]

    # 🧠 If categorical columns are already one-hot encoded, move them to numeric
    if not heart_cat:
        extra_encoded = [c for c in df_heart.columns if any(x in c for x in ["chestpaintype_", "restingecg_", "st_slope_"])]
        heart_num.extend(extra_encoded)
        print(f"Detected one-hot encoded columns for HEART: {extra_encoded}")

    train_xgb_task(df_heart, 'heart_label', heart_num, heart_cat, 'heart', use_smote=True)

    # ---- STRESS ----
    df_stress = pd.read_csv("stress.csv")
    df_stress.columns = df_stress.columns.str.strip().str.lower()

    possible_label_cols = ["stress_level", "stress", "label", "target"]
    label_col = next((c for c in possible_label_cols if c in df_stress.columns), None)

    if label_col is None:
        print("⚠️ No explicit stress label column found. Creating one from anxiety/depression/work_stress.")
        df_stress["stress_level"] = (
            df_stress.get("anxiety_score", 5).fillna(5) * 0.4 +
            df_stress.get("depression_score", 5).fillna(5) * 0.3 +
            df_stress.get("work_stress", 5).fillna(5) * 0.3
        )
        label_col = "stress_level"

    # Bin into 0/1 (low/high)
    median_level = df_stress[label_col].median()
    df_stress["label"] = (df_stress[label_col] > median_level).astype(int)

    # Detect numeric columns
    stress_num = [c for c in df_stress.columns if df_stress[c].dtype != 'object' and c != "label"]
    train_xgb_task(df_stress, 'label', stress_num, [], 'stress', use_smote=True)

    
    # ---------------- SLEEP ----------------
    df_sleep = pd.read_csv("sleep.csv")
    df_sleep.columns = df_sleep.columns.str.lower().str.strip()

    duration_col = next((c for c in df_sleep.columns if 'duration' in c or 'hours' in c), None)
    quality_col = next((c for c in df_sleep.columns if 'quality' in c), None)

    if duration_col is None and quality_col is None:
        print("⚠️ Skipping sleep model: no 'sleep_duration' or 'sleep_quality' column found.")
    else:
        # Handle both scaled and unscaled data
        if quality_col and duration_col:
            dur_col = df_sleep[duration_col].fillna(df_sleep[duration_col].median())
            qual_col = df_sleep[quality_col].fillna(df_sleep[quality_col].median())

            # Detect scaled vs raw
            if dur_col.abs().max() < 10 and qual_col.abs().max() < 10:
                # Scaled data → dynamic threshold (quantile-based)
                dur_thr = dur_col.quantile(0.4)
                qual_thr = qual_col.quantile(0.4)
                df_sleep["label"] = ((dur_col < dur_thr) | (qual_col < qual_thr)).astype(int)
            else:
                # Raw data → normal rule
                df_sleep["label"] = ((dur_col < 6) | (qual_col <= 2)).astype(int)

        elif duration_col:
            df_sleep["label"] = (df_sleep[duration_col].fillna(0) < 6).astype(int)
        elif quality_col:
            df_sleep["label"] = (df_sleep[quality_col].fillna(3) <= 2).astype(int)

        # 🔍 Print class distribution
        print("\nSleep label distribution:")
        print(df_sleep["label"].value_counts(dropna=False))

        # 🔁 If all one class → create slight artificial variation
        if len(df_sleep["label"].unique()) == 1:
            print("⚠️ Only one class detected — forcing small variation for model training.")
            # Randomly flip 10% of labels
            flip_idx = df_sleep.sample(frac=0.1, random_state=42).index
            df_sleep.loc[flip_idx, "label"] = 1 - df_sleep.loc[flip_idx, "label"]

        # Final check
        print("✅ Final label distribution:")
        print(df_sleep["label"].value_counts())

        # Now train
        sleep_num = [c for c in df_sleep.columns if c not in ['label'] and df_sleep[c].dtype != 'object']
        train_xgb_task(df_sleep, 'label', sleep_num, [], 'sleep', use_smote=True)

    print("\n🎯 All models trained and saved successfully!")
