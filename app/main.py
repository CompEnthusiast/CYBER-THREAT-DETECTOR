# app/main.py

import sys
import os
import streamlit as st
import pickle
import pandas as pd

# ---------- Path Setup ----------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import feature extractor
from scripts.feature_extractor import extract_features_from_url

# ---------- Page Setup ----------
st.set_page_config(
    page_title="AI Cyber Threat Detector",
    page_icon="🧠",
    layout="wide",
)

# ---------- Custom CSS (Header + Footer + Theme) ----------
st.markdown("""
    <style>
        /* Main background and font */
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
            font-family: 'Poppins', sans-serif;
        }

        /* --- HEADER --- */
        .main-header {
            text-align: center;
            margin-top: -40px;
            padding: 30px 10px 10px 10px;
            border-bottom: 1px solid rgba(0, 255, 204, 0.3);
            background: linear-gradient(90deg, #0e1117, #112233);
            box-shadow: 0px 0px 20px rgba(0, 255, 204, 0.2);
        }
        .main-title {
            font-size: 44px;
            font-weight: 700;
            background: linear-gradient(90deg, #00ffcc, #00b3ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: glow 2s infinite alternate;
        }
        .sub-title {
            color: #cccccc;
            font-size: 18px;
            margin-top: -10px;
            margin-bottom: 30px;
        }

        @keyframes glow {
            from { text-shadow: 0 0 10px #00ffcc; }
            to { text-shadow: 0 0 25px #00b3ff, 0 0 35px #00ffcc; }
        }

        /* --- RESULTS BOX --- */
        .result-box {
            border-radius: 12px;
            padding: 20px;
            font-size: 22px;
            text-align: center;
        }
        .phish {
            background-color: #ff4d4d;
            color: white;
        }
        .safe {
            background-color: #00cc66;
            color: white;
        }

        /* --- FOOTER --- */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: linear-gradient(90deg, rgba(0,255,204,0.08), rgba(0,179,255,0.1));
            color: #00ffcc;
            text-align: center;
            font-size: 15px;
            padding: 12px 0;
            border-top: 1px solid rgba(0,255,204,0.2);
            box-shadow: 0px -2px 10px rgba(0,255,204,0.15);
            z-index: 999;
        }
        .footer a {
            color: #00ffcc;
            text-decoration: none;
            margin: 0 10px;
            transition: 0.3s;
            font-weight: 500;
        }
        .footer a:hover {
            color: #00ffaa;
            text-shadow: 0px 0px 8px #00ffaa;
        }
        .footer img {
            vertical-align: middle;
            margin-right: 6px;
            filter: drop-shadow(0px 0px 4px #00ffcc);
            transition: 0.3s;
        }
        .footer img:hover {
            transform: scale(1.15);
        }

        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🧑‍💻 AI Cyber Threat Detector🛡️</h1>
        <p class="sub-title">Advanced Machine Learning Tool for <b>Phishing URL</b> and <b>Spam Email Detection</b></p>
    </div>
""", unsafe_allow_html=True)

# ---------- Load Models ----------
try:
    log_model = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'logistic_regression.pkl'), 'rb'))
    rf_model = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'random_forest.pkl'), 'rb'))
    xgb_model = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'xgboost.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'scaler.pkl'), 'rb'))
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🌐 Phishing URL Detection", "📧 Spam Email Detection"])

# --------------------------------------------------------------------
# 🌐 TAB 1: PHISHING URL DETECTION
# --------------------------------------------------------------------
with tab1:
    st.markdown("### 🔍 Enter a Website URL to Analyze")
    url_input = st.text_input("Website URL:", placeholder="https://example.com/login")
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])

    if st.button("🚨 Detect Phishing"):
        if url_input:
            try:
                # Extract features
                features = extract_features_from_url(url_input)
                input_df = pd.DataFrame([features])

                # Clean data
                input_df = input_df.fillna(0)
                input_df = input_df.replace([float('inf'), float('-inf')], 0)

                # Scale safely
                if scaler is not None:
                    try:
                        input_df = scaler.transform(input_df)
                    except Exception as e:
                        st.warning(f"⚠️ Scaling issue: {e}")

                # Predict
                if model_choice == "Logistic Regression":
                    prediction = log_model.predict(input_df)[0]
                elif model_choice == "Random Forest":
                    prediction = rf_model.predict(input_df)[0]
                else:
                    prediction = xgb_model.predict(input_df)[0]

                # Display result
                if prediction == 1:
                    st.markdown("<div class='result-box phish'>🚫 Phishing / Malicious Website Detected!</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-box safe'>✅ This Website Appears Safe.</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                st.write("Debug Info:", input_df)
        else:
            st.warning("Please enter a valid website URL.")

# --------------------------------------------------------------------
# 📧 TAB 2: SPAM EMAIL DETECTION
# --------------------------------------------------------------------
with tab2:
    st.markdown("### 📨 Paste Email Text Below to Check for Spam")
    email_input = st.text_area("Email Message:", height=200, placeholder="Enter email content...")
    model_choice_email = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"], key="email_model")

    if st.button("📬 Detect Spam"):
        if email_input:
            try:
                # Load spam models
                spam_log = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'spam_logistic.pkl'), 'rb'))
                spam_rf = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'spam_rf.pkl'), 'rb'))
                spam_xgb = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'spam_xgb.pkl'), 'rb'))
                vectorizer = pickle.load(open(os.path.join(ROOT_DIR, 'models', 'vectorizer.pkl'), 'rb'))

                # Transform text
                X_input = vectorizer.transform([email_input])

                # Predict
                if model_choice_email == "Logistic Regression":
                    pred = spam_log.predict(X_input)[0]
                elif model_choice_email == "Random Forest":
                    pred = spam_rf.predict(X_input)[0]
                else:
                    pred = spam_xgb.predict(X_input)[0]

                # Display result
                if pred == 1:
                    st.markdown("<div class='result-box phish'>🚨 This Email is Likely SPAM!</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-box safe'>📩 This Email Appears Legitimate.</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
        else:
            st.warning("Please paste an email message to analyze.")

# ---------- FOOTER ----------
st.markdown("""
    <div class="footer">
        Developed with ❤️ by <b>Vansh S. Bhandari</b> |
        <a href="https://github.com/CompEnthusiast" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="18">
            GitHub
        </a> |
        <a href="https://www.linkedin.com/in/vansh-s-bhandari-08911630a/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="18">
            LinkedIn
        </a>
        <br>© 2025 Cyber Threat Detector | All Rights Reserved
    </div>
""", unsafe_allow_html=True)
