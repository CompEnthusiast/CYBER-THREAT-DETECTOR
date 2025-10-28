# scripts/spam_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# ---------- Load Dataset ----------
with open('dataset/spam1.csv', 'r', encoding='latin-1') as f:
    data = pd.read_csv(f)

# Rename columns if needed
if 'text' in data.columns:
    data = data.rename(columns={'text': 'message'})
if 'label' not in data.columns:
    raise ValueError("Dataset must have a 'label' column with 'spam' and 'ham' values.")

# ---------- Encode Labels ----------
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# ---------- TF-IDF Vectorization ----------
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(data['message'])
y = data['label_num']

# ---------- Split Data ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- Logistic Regression ----------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("\n📊 Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log, target_names=['ham', 'spam']))

# ---------- Random Forest ----------
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\n🌲 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=['ham', 'spam']))

# ---------- XGBoost ----------
xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("\n🚀 XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, target_names=['ham', 'spam']))

# ---------- Save Models ----------
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
pickle.dump(log_model, open('models/spam_logistic.pkl', 'wb'))
pickle.dump(rf_model, open('models/spam_rf.pkl', 'wb'))
pickle.dump(xgb_model, open('models/spam_xgb.pkl', 'wb'))

print("\n✅ Models and vectorizer saved in 'models/' folder successfully.")
