# 🧠 Cyber Threat Detector

A **Machine Learning–based Cyber Threat Detection System** that identifies **phishing URLs** and **spam messages** using trained models.  
Built with **Python**, **Streamlit**, and **scikit-learn**, it provides an interactive web interface for real-time threat classification.

---

## 🚀 Features

- 🔗 **Phishing URL Detection** — Extracts features from URLs and classifies them as safe or malicious.  
- 📩 **Spam Message Detection** — Uses NLP models to identify spam or ham text messages.  
- 🧮 **Multiple ML Models** — Logistic Regression, Random Forest, and XGBoost for both tasks.  
- ⚙️ **Scalable Design** — Modular scripts for easy retraining and extension.  
- 💻 **Streamlit Web App** — User-friendly interface for testing URLs or text inputs.

---

## 🗂️ Project Structure

```
cyber_threat_detector/
│
├── app/
│   └── main.py                # Streamlit frontend
├── scripts/
│   ├── feature_extractor.py   # URL feature engineering
│   ├── phishing_model.py      # Phishing URL model training
│   ├── spam_model.py          # Spam message model training
├── dataset/
│   ├── phishing1.csv          # Phishing dataset
│   └── spam1.csv              # SMS spam dataset
├── models/
│   ├── *.pkl                  # Trained models and scalers
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

---

## 🧩 Tech Stack

- **Python 3.10+**
- **Streamlit** — Web UI  
- **pandas, numpy** — Data processing  
- **scikit-learn, xgboost** — ML algorithms  
- **pickle** — Model serialization  

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/CompEnthusiast/CYBER-THREAT-DETECTOR.git
   cd cyber_threat_detector
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate     # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   cd app
   streamlit run main.py
   ```

---

## 📊 Model Overview

| Task | Algorithm | Model File | Description |
|------|------------|-------------|--------------|
| Phishing Detection | Logistic Regression / Random Forest / XGBoost | `models/*.pkl` | Classifies URLs |
| Spam Detection | Logistic Regression / Random Forest / XGBoost | `models/spam_*.pkl` | Detects spam messages |

Feature extraction is handled by `scripts/feature_extractor.py`, which converts URL attributes into numeric vectors suitable for ML models.

---

## 🧠 Example Usage

### 🔗 URL Threat Check
Enter a URL in the Streamlit input box to get a prediction:
```
https://example.com/login-update
```
Output:
```
⚠️ Potentially Phishing
```

### 📩 Spam Message Detection
```
Congratulations! You’ve won a $500 gift card!
```
Output:
```
🚫 Spam Message
```

---

## 📚 Future Enhancements

- 🧾 Integration with real-time web scrapers  
- 🕵️‍♂️ Browser plugin for phishing detection  
- 🧩 Deep learning model integration  
- ☁️ Cloud-based deployment (AWS / Streamlit Cloud)

---

## 👨‍💻 Author

**Vansh S. Bhandari**  
Cybersecurity Enthusiast  | Focused on Ethical Hacking & Threat Detection  
📍 Chandigarh, India  


---

> ⚠️ *Disclaimer:* This tool is for educational and research purposes only. It should not be used for malicious activities.
