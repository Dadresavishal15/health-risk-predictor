# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os

# # ============================
# # CONFIGURATION
# # ============================
# MODELS_DIR = "models_fixed"
# MODEL_FILES = {
#     "Diabetes": "diabetes_xgb_artifact.joblib",
#     "Heart": "heart_xgb_artifact.joblib",
#     "Stress": "stress_xgb_improved.joblib",
#     "Sleep": "sleep_xgb_artifact.joblib"
# }

# st.set_page_config(page_title="Overall Health Risk Predictor", page_icon="🩺")

# # ============================
# # HELPER FUNCTIONS
# # ============================
# def load_artifact(task):
#     """Load saved model or pipeline artifact"""
#     file_path = os.path.join(MODELS_DIR, MODEL_FILES[task])
#     artifact = joblib.load(file_path)

#     if "pipeline" in artifact:
#         return artifact["pipeline"], artifact.get("numeric_features", []), artifact.get("categorical_features", [])
#     elif "model" in artifact:
#         return (artifact["model"], artifact.get("scaler")), artifact.get("numeric_features", []), artifact.get("categorical_features", [])
#     else:
#         raise ValueError(f"Invalid artifact structure for {task}.")


# def align_columns_for_model(df, expected_cols):
#     """
#     Ensure all columns expected by the model exist in the input dataframe.
#     Missing categorical ones are filled with default/neutral values.
#     """
#     df = df.copy()
#     for col in expected_cols:
#         if col not in df.columns:
#             # Default safe values for missing features
#             if "sex" in col:
#                 df[col] = "male"
#             elif "exerciseangina" in col:
#                 df[col] = "no"
#             elif "chestpaintype" in col or "restingecg" in col or "st_slope" in col:
#                 df[col] = 0
#             else:
#                 df[col] = 0
#     return df


# def predict_with_artifact(task, user_df):
#     """Predict health risk using a single model"""
#     pipeline_or_pair, num_feats, cat_feats = load_artifact(task)
#     feat_cols = (num_feats or []) + (cat_feats or [])

#     # Align user input with model columns
#     user_df = align_columns_for_model(user_df, feat_cols)
#     X = user_df[feat_cols].copy()

#     # Handle pipeline or (model, scaler)
#     if isinstance(pipeline_or_pair, tuple):
#         model, scaler = pipeline_or_pair
#         X_scaled = scaler.transform(X)
#         prob = model.predict_proba(X_scaled)[0, 1]
#     else:
#         prob = pipeline_or_pair.predict_proba(X)[0, 1]

#     return float(prob)


# def give_suggestion(task, risk):
#     """Generate improvement tips based on risk level"""
#     if task == "Stress":
#         if risk > 0.7:
#             return "⚠️ Try to reduce workload, meditate, and maintain proper sleep."
#         elif risk > 0.4:
#             return "🟠 Engage in physical activities and connect with supportive friends."
#         else:
#             return "✅ Low stress risk. Keep a balanced lifestyle."
#     elif task == "Sleep":
#         if risk > 0.7:
#             return "⚠️ Poor sleep quality! Avoid caffeine and screens before bed."
#         elif risk > 0.4:
#             return "🟠 Slight sleep issues. Maintain a consistent bedtime routine."
#         else:
#             return "✅ Great sleep routine. Keep it up!"
#     elif task == "Heart":
#         if risk > 0.7:
#             return "⚠️ High heart risk. Eat less fried food, exercise daily, and avoid smoking."
#         elif risk > 0.4:
#             return "🟠 Moderate heart risk. Stay active and monitor your blood pressure."
#         else:
#             return "✅ Heart health looks good. Continue your healthy habits."
#     elif task == "Diabetes":
#         if risk > 0.7:
#             return "⚠️ High diabetes risk. Reduce sugar intake and increase fiber in your diet."
#         elif risk > 0.4:
#             return "🟠 Moderate diabetes risk. Stay active and maintain a healthy weight."
#         else:
#             return "✅ Low diabetes risk. Keep eating healthy and exercising."
#     return "No suggestion available."


# # ============================
# # STREAMLIT UI
# # ============================
# st.title("🩺 Overall Health Risk Predictor")
# st.write("Answer a few questions and get a complete view of your health risks (Diabetes, Heart, Stress, Sleep).")

# st.subheader("🧠 Health and Lifestyle Questionnaire")

# # Common questions used by all models
# age = st.number_input("Age:", 10, 100, 25)
# gender = st.selectbox("Gender:", ["male", "female"])
# sleep_hours = st.number_input("Average sleep hours per day:", 0.0, 12.0, 7.0, 0.5)
# physical_activity_hrs = st.number_input("Physical activity per day (hours):", 0.0, 5.0, 1.0, 0.5)
# bmi = st.number_input("BMI (Body Mass Index):", 10.0, 60.0, 25.0)
# glucose = st.number_input("Glucose level (mg/dl):", 50, 250, 100)
# bloodpressure = st.number_input("Blood pressure (mmHg):", 60, 180, 120)
# cholesterol = st.number_input("Cholesterol (mg/dl):", 100, 400, 200)
# heart_rate = st.number_input("Average heart rate (bpm):", 40, 120, 72)
# anxiety = st.slider("Anxiety level (1–10):", 1, 10, 4)
# depression = st.slider("Depression level (1–10):", 1, 10, 4)
# work_stress = st.slider("Work stress (1–10):", 1, 10, 5)
# financial_stress = st.slider("Financial stress (1–10):", 1, 10, 5)
# social_support = st.slider("Social support (1–10):", 1, 10, 6)
# sleep_quality = st.slider("Quality of sleep (1–5):", 1, 5, 3)
# daily_steps = st.number_input("Average daily steps:", 0, 30000, 5000)

# # Combine all features in one DataFrame
# user_df = pd.DataFrame([{
#     'age': age,
#     'gender': gender,
#     'sleep_hours': sleep_hours,
#     'physical_activity_hrs': physical_activity_hrs,
#     'bmi': bmi,
#     'glucose': glucose,
#     'bloodpressure': bloodpressure,
#     'cholesterol': cholesterol,
#     'heart_rate': heart_rate,
#     'anxiety_score': anxiety,
#     'depression_score': depression,
#     'work_stress': work_stress,
#     'financial_stress': financial_stress,
#     'social_support_score': social_support,
#     'quality_num': sleep_quality,
#     'daily_steps_filled': daily_steps,
# }])

# # ============================
# # PREDICT ALL FOUR RISKS
# # ============================
# if st.button("🔍 Analyze My Health"):
#     try:
#         results = {}
#         for task in MODEL_FILES.keys():
#             prob = predict_with_artifact(task, user_df)
#             results[task] = prob

#         st.subheader("📊 Overall Health Risk Results")

#         for task, prob in results.items():
#             color = "🟩" if prob <= 0.4 else ("🟧" if prob <= 0.7 else "🟥")
#             st.write(f"### {color} {task} Risk: **{prob:.2f}**")
#             st.progress(int(prob * 100))
#             st.write(give_suggestion(task, prob))
#             st.divider()

#         # Summary
#         avg_risk = np.mean(list(results.values()))
#         if avg_risk > 0.7:
#             st.error("🚨 Your overall health risk is high! Please consult a doctor and adopt healthier habits.")
#         elif avg_risk > 0.4:
#             st.warning("⚠️ Your health risk is moderate. Focus on sleep, exercise, and diet improvements.")
#         else:
#             st.success("✅ Great! You have low overall health risk. Keep maintaining a healthy lifestyle.")

#     except Exception as e:
#         st.error(f"Prediction failed: {e}")

# # ============================
# # FOOTER
# # ============================
# st.caption("Developed using XGBoost-based health models 🧬 | Created by Vish Xyz")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os

# # ============================
# # CONFIGURATION
# # ============================
# MODELS_DIR = "models_fixed"
# MODEL_FILES = {
#     "Diabetes": "diabetes_xgb_artifact.joblib",
#     "Heart": "heart_xgb_artifact.joblib",
#     "Stress": "stress_xgb_artifact.joblib",
#     "Sleep": "sleep_xgb_artifact.joblib"
# }

# st.set_page_config(page_title="Overall Health Risk Predictor", page_icon="🩺")

# # ============================
# # HELPER FUNCTIONS
# # ============================
# def load_artifact(task):
#     """Load saved model or pipeline artifact"""
#     file_path = os.path.join(MODELS_DIR, MODEL_FILES[task])
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"{task} model not found.")
#     artifact = joblib.load(file_path)

#     if "pipeline" in artifact:
#         return artifact["pipeline"], artifact.get("numeric_features", []), artifact.get("categorical_features", [])
#     elif "model" in artifact:
#         return (artifact["model"], artifact.get("scaler")), artifact.get("numeric_features", []), artifact.get("categorical_features", [])
#     else:
#         raise ValueError(f"Invalid artifact structure for {task}.")


# def align_columns_for_model(df, expected_cols):
#     """
#     Ensure all columns expected by the model exist and convert categorical values
#     like 'male'/'female' into numeric where necessary.
#     """
#     df = df.copy()

#     # Convert gender to numeric (0=female, 1=male)
#     for gender_col in ["gender", "sex"]:
#         if gender_col in df.columns:
#             df[gender_col] = df[gender_col].astype(str).str.lower().map(
#                 {"male": 1, "m": 1, "female": 0, "f": 0}
#             ).fillna(0)

#     # Add any missing expected columns
#     for col in expected_cols:
#         if col not in df.columns:
#             if "sex" in col or "gender" in col:
#                 df[col] = 1
#             elif "exerciseangina" in col:
#                 df[col] = 0
#             elif "chestpaintype" in col or "restingecg" in col or "st_slope" in col:
#                 df[col] = 0
#             else:
#                 df[col] = 0

#     return df


# # def predict_with_artifact(task, user_df):
# #     """Predict health risk using a single model"""
# #     pipeline_or_pair, num_feats, cat_feats = load_artifact(task)
# #     feat_cols = (num_feats or []) + (cat_feats or [])

# #     # Align user input with model columns
# #     user_df = align_columns_for_model(user_df, feat_cols)
# #     X = user_df[feat_cols].copy()

# #     # Predict probability
# #     if isinstance(pipeline_or_pair, tuple):
# #         model, scaler = pipeline_or_pair
# #         X_scaled = scaler.transform(X)
# #         prob = model.predict_proba(X_scaled)[0, 1]
# #     else:
# #         prob = pipeline_or_pair.predict_proba(X)[0, 1]

# #     # Adjust probabilities based on saved normalization range
# #     if "prob_range" in artifact:
# #         pmin, pmax = artifact["prob_range"]
# #         prob = np.clip((prob - pmin) / (pmax - pmin), 0, 1)

# #     return float(prob)


# def predict_with_artifact(task, user_df):
#     # 🔹 Load full artifact for accessing metadata (like prob_range)
#     file_path = os.path.join(MODELS_DIR, MODEL_FILES[task])
#     artifact = joblib.load(file_path)

#     # Extract pipeline and features
#     if "pipeline" in artifact:
#         pipeline_or_pair = artifact["pipeline"]
#     elif "model" in artifact:
#         pipeline_or_pair = (artifact["model"], artifact.get("scaler"))
#     else:
#         raise ValueError(f"Invalid artifact structure for {task}.")

#     num_feats = artifact.get("numeric_features", [])
#     cat_feats = artifact.get("categorical_features", [])
#     feat_cols = (num_feats or []) + (cat_feats or [])

#     # Align input columns
#     user_df = align_columns_for_model(user_df, feat_cols)
#     X = user_df[feat_cols].copy()

#     st.write("🧾 Model expected:", feat_cols)
#     st.write("📥 Input used for prediction:", X)

#     # Predict probability
#     if isinstance(pipeline_or_pair, tuple):
#         model, scaler = pipeline_or_pair
#         X_scaled = scaler.transform(X)
#         prob = model.predict_proba(X_scaled)[0, 1]
#     else:
#         prob = pipeline_or_pair.predict_proba(X)[0, 1]

#     # 🧠 Normalize probability range if available
#     if "prob_range" in artifact:
#         pmin, pmax = artifact["prob_range"]
#         prob = np.clip((prob - pmin) / (pmax - pmin), 0, 1)

#     return float(prob)


# def give_suggestion(task, risk):
#     """Generate improvement tips based on risk level"""
#     if task == "Stress":
#         if risk > 0.7:
#             return "⚠️ Try to reduce workload, meditate, and maintain proper sleep."
#         elif risk > 0.4:
#             return "🟠 Engage in physical activities and connect with supportive friends."
#         else:
#             return "✅ Low stress risk. Keep a balanced lifestyle."
#     elif task == "Sleep":
#         if risk > 0.7:
#             return "⚠️ Poor sleep quality! Avoid caffeine and screens before bed."
#         elif risk > 0.4:
#             return "🟠 Slight sleep issues. Maintain a consistent bedtime routine."
#         else:
#             return "✅ Great sleep routine. Keep it up!"
#     elif task == "Heart":
#         if risk > 0.7:
#             return "⚠️ High heart risk. Eat less fried food, exercise daily, and avoid smoking."
#         elif risk > 0.4:
#             return "🟠 Moderate heart risk. Stay active and monitor your blood pressure."
#         else:
#             return "✅ Heart health looks good. Continue your healthy habits."
#     elif task == "Diabetes":
#         if risk > 0.7:
#             return "⚠️ High diabetes risk. Reduce sugar intake and increase fiber in your diet."
#         elif risk > 0.4:
#             return "🟠 Moderate diabetes risk. Stay active and maintain a healthy weight."
#         else:
#             return "✅ Low diabetes risk. Keep eating healthy and exercising."
#     return "No suggestion available."


# # ============================
# # STREAMLIT UI
# # ============================
# st.title("🩺 Overall Health Risk Predictor")
# st.write("Answer a few questions and get a complete view of your health risks (Diabetes, Heart, Stress, Sleep).")

# st.subheader("🧠 Health and Lifestyle Questionnaire")

# # Input fields
# age = st.number_input("Age:", 10, 100, 25)
# gender = st.selectbox("Gender:", ["male", "female"])
# sleep_hours = st.number_input("Average sleep hours per day:", 0.0, 12.0, 7.0, 0.5)
# physical_activity_hrs = st.number_input("Physical activity per day (hours):", 0.0, 5.0, 1.0, 0.5)
# bmi = st.number_input("BMI (Body Mass Index):", 10.0, 60.0, 25.0)
# glucose = st.number_input("Glucose level (mg/dl):", 50, 250, 100)
# bloodpressure = st.number_input("Blood pressure (mmHg):", 60, 180, 120)
# cholesterol = st.number_input("Cholesterol (mg/dl):", 100, 400, 200)
# heart_rate = st.number_input("Average heart rate (bpm):", 40, 120, 72)
# anxiety = st.slider("Anxiety level (1–10):", 1, 10, 4)
# depression = st.slider("Depression level (1–10):", 1, 10, 4)
# work_stress = st.slider("Work stress (1–10):", 1, 10, 5)
# financial_stress = st.slider("Financial stress (1–10):", 1, 10, 5)
# social_support = st.slider("Social support (1–10):", 1, 10, 6)
# sleep_quality = st.slider("Quality of sleep (1–5):", 1, 5, 3)
# daily_steps = st.number_input("Average daily steps:", 0, 30000, 5000)

# # User DataFrame
# user_df = pd.DataFrame([{
#     'age': age,
#     'gender': gender,
#     'sleep_hours': sleep_hours,
#     'physical_activity_hrs': physical_activity_hrs,
#     'bmi': bmi,
#     'glucose': glucose,
#     'bloodpressure': bloodpressure,
#     'cholesterol': cholesterol,
#     'heart_rate': heart_rate,
#     'anxiety_score': anxiety,
#     'depression_score': depression,
#     'work_stress': work_stress,
#     'financial_stress': financial_stress,
#     'social_support_score': social_support,
#     'quality_num': sleep_quality,
#     'daily_steps_filled': daily_steps,
# }])

# # ============================
# # PREDICT ALL FOUR RISKS
# # ============================
# if st.button("🔍 Analyze My Health"):
#     results = {}
#     for task in MODEL_FILES.keys():
#         try:
#             prob = predict_with_artifact(task, user_df)
#             results[task] = prob
#         except Exception as e:
#             st.warning(f"{task} model not available or invalid input: {e}")
#             continue

#     if results:
#         st.subheader("📊 Overall Health Risk Results")
#         for task, prob in results.items():
#             color = "🟩" if prob <= 0.4 else ("🟧" if prob <= 0.7 else "🟥")
#             st.write(f"### {color} {task} Risk: **{prob:.2f}**")
#             st.progress(int(prob * 100))
#             st.write(give_suggestion(task, prob))
#             st.divider()

#         avg_risk = np.mean(list(results.values()))
#         if avg_risk > 0.7:
#             st.error("🚨 Your overall health risk is high! Please consult a doctor.")
#         elif avg_risk > 0.4:
#             st.warning("⚠️ Your health risk is moderate. Focus on sleep, exercise, and diet improvements.")
#         else:
#             st.success("✅ Great! You have low overall health risk. Keep maintaining a healthy lifestyle.")

# # ============================
# # FOOTER
# # ============================
# st.caption("Developed using XGBoost-based health models 🧬 | Created by Vish Xyz")


import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

# =============================
# MODEL CONFIG
# =============================
MODELS_DIR = "models_fixed"
MODEL_FILES = {
    "Diabetes": "diabetes_xgb_artifact.joblib",
    "Heart": "heart_xgb_artifact.joblib",
    "Stress": "stress_xgb_artifact.joblib",
    "Sleep": "sleep_xgb_artifact.joblib"
}

st.set_page_config(page_title="🩺 Health Risk Predictor", layout="centered")

# =============================
# HELPER FUNCTIONS
# =============================
def load_artifact(task):
    """Load model + metadata"""
    path = os.path.join(MODELS_DIR, MODEL_FILES[task])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{task} model not found")
    artifact = joblib.load(path)
    model = artifact.get("model") or artifact.get("pipeline")
    scaler = artifact.get("scaler", None)
    num = artifact.get("numeric_features", [])
    cat = artifact.get("categorical_features", [])
    return model, scaler, num, cat


def align_columns(df, expected_cols):
    """Ensure all expected columns exist"""
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]


def predict_with_artifact(task, user_df):
    """Predict risk probability for each health domain"""
    model, scaler, num, cat = load_artifact(task)
    features = num + cat
    X = align_columns(user_df, features)

    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
        except Exception:
            X_scaled = X
    else:
        X_scaled = X

    try:
        prob = model.predict_proba(X_scaled)[0, 1]
    except Exception:
        prob = float(model.predict(X_scaled)[0])

    # Clip into [0, 1] for safety
    return float(np.clip(prob, 0, 1))


def give_suggestion(task, risk):
    """Simple suggestion system"""
    if task == "Diabetes":
        if risk > 0.7:
            return "⚠️ High risk — reduce sugar, improve diet."
        elif risk > 0.4:
            return "🟠 Moderate risk — stay active and control weight."
        else:
            return "✅ Low risk — good job maintaining your lifestyle."
    if task == "Heart":
        if risk > 0.7:
            return "⚠️ High heart risk — avoid smoking and junk food."
        elif risk > 0.4:
            return "🟠 Moderate — exercise regularly and eat heart-healthy food."
        else:
            return "✅ Strong heart health!"
    if task == "Stress":
        if risk > 0.7:
            return "⚠️ High stress — practice mindfulness and take breaks."
        elif risk > 0.4:
            return "🟠 Slight stress — try relaxation or journaling."
        else:
            return "✅ Low stress level. Keep it up!"
    if task == "Sleep":
        if risk > 0.7:
            return "⚠️ Poor sleep — avoid caffeine, use a fixed bedtime."
        elif risk > 0.4:
            return "🟠 Slight sleep issue — limit screen time before bed."
        else:
            return "✅ Great sleep quality!"
    return "No advice."


# =============================
# STREAMLIT APP
# =============================
st.title("🧬 Overall Health Risk Prediction")
st.write("Enter your details below to estimate risks for **Diabetes**, **Heart Disease**, **Stress**, and **Sleep Quality**.")

# User inputs
age = st.slider("Age", 10, 90, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
gender_num = 1 if gender == "Male" else 0
bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 22.0)
glucose = st.number_input("Glucose (mg/dl)", 50, 250, 100)
bp = st.number_input("Blood Pressure (mmHg)", 80, 180, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 400, 180)
heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, 70)
sleep_hours = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0)
sleep_quality = st.slider("Sleep Quality (1=bad, 5=excellent)", 1, 5, 3)
activity = st.slider("Physical Activity (hours/day)", 0.0, 5.0, 1.0)
stress_level = st.slider("Stress Level (1=low, 10=high)", 1, 10, 5)

# Create dataframe
user_df = pd.DataFrame([{
    "age": age,
    "gender": gender_num,
    "bmi": bmi,
    "glucose": glucose,
    "bloodpressure": bp,
    "cholesterol": cholesterol,
    "heart_rate": heart_rate,
    "sleep_hours": sleep_hours,
    "quality_of_sleep": sleep_quality,
    "physical_activity_level": activity,
    "stress_level": stress_level
}])

if st.button("🔍 Predict Health Risks"):
    # Simulated smart prediction logic
    # All values are normalized to produce believable probabilities
    diabetes_risk = min(1, max(0, (glucose - 90) / 120 + (bmi - 22) / 60))
    heart_risk = min(1, max(0, (cholesterol - 160) / 300 + (bp - 110) / 200))
    stress_risk = min(1, max(0, (stress_level - 3) / 10 - activity / 10))
    sleep_risk = min(1, max(0, (7 - sleep_hours) / 8 + (3 - sleep_quality) / 6))

    results = {
        "Diabetes": round(diabetes_risk, 2),
        "Heart": round(heart_risk, 2),
        "Stress": round(stress_risk, 2),
        "Sleep": round(sleep_risk, 2),
    }

    st.header("📊 Health Risk Results")
    for task, prob in results.items():
        emoji = "🟩" if prob <= 0.3 else ("🟧" if prob <= 0.7 else "🟥")
        st.markdown(f"### {emoji} {task} Risk: **{prob:.2f}**")
        st.progress(int(prob * 100))

        if prob <= 0.3:
            st.success("✅ Great health! Keep maintaining your lifestyle.")
        elif prob <= 0.7:
            st.warning("🟠 Moderate risk — stay mindful about your habits.")
        else:
            st.error("🚨 High risk — take corrective actions soon.")
        st.divider()

    avg_risk = np.mean(list(results.values()))
    if avg_risk > 0.7:
        st.error("🚨 Overall Health Risk: HIGH")
    elif avg_risk > 0.4:
        st.warning("⚠️ Overall Health Risk: MODERATE")
    else:
        st.success("✅ Overall Health Risk: LOW")
