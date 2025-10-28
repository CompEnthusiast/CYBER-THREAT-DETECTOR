# scripts/phishing_model.py

import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from feature_extractor import extract_features_from_url  # ✅ Ensure correct import path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


# ======== Paths ========
DATA_PATH = "dataset/phishing1.csv"   # adjust if needed
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ======== Load Dataset ========
print(f"📂 Loading dataset from: {DATA_PATH}")
data = pd.read_csv(DATA_PATH)

# Check and clean column names
data.columns = [col.strip().lower() for col in data.columns]

if "url" not in data.columns or "label" not in data.columns:
    raise KeyError("Dataset must contain 'url' and 'label' columns.")

# ======== Convert Labels to Numeric ========
# Converts: legit → 0, phish → 1
data["label"] = data["label"].map({"legit": 0, "phish": 1})
if data["label"].isnull().any():
    # fallback if dataset already has numeric labels or other text
    le = LabelEncoder()
    data["label"] = le.fit_transform(data["label"])

# ======== Feature Extraction ========
print("⚙️ Extracting features from URLs...")
features = [extract_features_from_url(url) for url in data["url"]]
X = pd.DataFrame(features)
y = data["label"]

# ======== Handle Missing/Infinite Values ========
pd.set_option("future.no_silent_downcasting", True)
X = X.fillna(0).replace([float("inf"), float("-inf")], 0).infer_objects(copy=False)

# ======== Feature Scaling ========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======== Train/Test Split ========
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ======== Train Models ========
print("🚀 Training models...")
log_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(eval_metric="logloss", n_estimators=200, random_state=42)

for model in [log_model, rf_model, xgb_model]:
    model.fit(X_train, y_train)

# ======== Evaluate ========
acc_log = accuracy_score(y_test, log_model.predict(X_test))
acc_rf = accuracy_score(y_test, rf_model.predict(X_test))
acc_xgb = accuracy_score(y_test, xgb_model.predict(X_test))

print("\n🎯 Model Accuracies:")
print(f"✅ Logistic Regression: {acc_log:.4f}")
print(f"✅ Random Forest:       {acc_rf:.4f}")
print(f"✅ XGBoost:             {acc_xgb:.4f}")

# ======== Save Models ========
pickle.dump(log_model, open(f"{MODEL_DIR}/logistic_regression.pkl", "wb"))
pickle.dump(rf_model, open(f"{MODEL_DIR}/random_forest.pkl", "wb"))
pickle.dump(xgb_model, open(f"{MODEL_DIR}/xgboost.pkl", "wb"))
pickle.dump(scaler, open(f"{MODEL_DIR}/scaler.pkl", "wb"))

# Save accuracies for reference
model_accuracies = {
    "Logistic Regression": acc_log,
    "Random Forest": acc_rf,
    "XGBoost": acc_xgb
}
pickle.dump(model_accuracies, open(f"{MODEL_DIR}/model_accuracy.pkl", "wb"))

print("\n✅ All models trained and saved successfully in /models/")
