

# 🎵 Music Popularity Prediction App

A **Machine Learning–powered Streamlit web application** that predicts the **popularity class of a song** (Low, Medium, High) using Spotify audio features.
The app also provides **global model evaluation** and **local explainability using LIME**, making predictions transparent and interpretable.

---

## 🚀 Live Demo

🔗 **App Link:** *(add your Streamlit Cloud URL here once deployed)*

---

## 📌 Features

### 🔍 Data Overview

* Displays a sample of the Spotify dataset
* Uses real audio features such as:

  * Danceability
  * Energy
  * Loudness
  * Tempo
  * Acousticness
  * Valence
  * Speechiness
  * Liveness

---

### 📊 Global Model Performance

* **Confusion Matrix** (normalized)
* **Feature Importance** from Random Forest
* **Classification Report**

  * Precision
  * Recall
  * F1-score
  * Accuracy

---

### 🎯 Local Prediction & Explainability

* Interactive sliders for audio features
* Predicts:

  * **Popularity Class**
  * **Prediction Confidence**
* **LIME (Local Interpretable Model-agnostic Explanations)** to explain individual predictions visually

---

## 🧠 Machine Learning Model

* **Algorithm:** Random Forest Classifier
* **Target Variable:** Popularity Class

  * Low Popularity (0–33)
  * Medium Popularity (34–66)
  * High Popularity (67–100)
* **Train/Test Split:** 80% / 20%

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend / ML:** Scikit-learn
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Explainability:** LIME
* **Model Persistence:** Joblib

---



---

## ⚙️ Installation & Local Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/music-popularity-streamlit.git
cd music-popularity-streamlit
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 📦 Requirements

```txt
streamlit==1.28.1
scikit-learn==1.3.0
pandas==2.1.1
numpy==1.24.3
joblib==1.3.2
lime==0.2.0.1
matplotlib==3.8.0
seaborn==0.12.2
```

---

## 🌐 Deployment

The app is deployed using **Streamlit Community Cloud**:

1. Push project to GitHub
2. Connect repository on Streamlit Cloud
3. Set `app.py` as the main file
4. Deploy 🚀

---

## 📈 Future Enhancements

* Regression-based popularity prediction
* Spotify API integration for real-time song input
* User authentication
* Downloadable prediction reports
* Model comparison dashboard

---

## 👩‍💻 Author

**Geetanjali Saini**
🎓 Engineering Student | Machine Learning & Web Development
📌 Built as a **portfolio-ready applied ML project**

---


