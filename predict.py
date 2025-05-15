import pandas as pd
import joblib

# 1. Load your saved pipeline
model = joblib.load("stroke_prediction_pipeline.pkl")

# a sample input for testing the pipeline....just random values anthey
sample = {
    "gender": ["Male"],
    "age": [67.0],
    "hypertension": [0],
    "heart_disease": [1],
    "ever_married": ["Yes"],
    "work_type": ["Private"],
    "Residence_type": ["Urban"],
    "avg_glucose_level": [228.69],
    "bmi": [36.6],
    "smoking_status": ["formerly smoked"],
}

# Convert to DataFrame
X_sample = pd.DataFrame(sample)

# 3a. Predict class (0 = no stroke, 1 = stroke)
pred_class = model.predict(X_sample)
print("Predicted class:", int(pred_class[0]))

# 3b. Predict probability of stroke
pred_proba = model.predict_proba(X_sample)[:, 1]
print("Predicted stroke probability:", pred_proba[0])
