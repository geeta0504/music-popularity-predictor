

---

# 🎵 Music Popularity Predictor

🚀 **Live App:**
👉 [https://music-popularity-predictor-pawgmryk8tepucedvnpdjk.streamlit.app/](https://music-popularity-predictor-pawgmryk8tepucedvnpdjk.streamlit.app/)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Live Demo](#live-demo)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Machine Learning](#machine-learning)
8. [Model Interpretation](#model-interpretation)


## Project Overview

**Music Popularity Predictor** is a Python-based machine learning project designed to **predict the popularity of songs** based on audio and metadata features such as acoustic properties, tempo, duration, and more.

The project covers the **full ML pipeline**:

* Data preprocessing
* Model training
* Evaluation
* Interpretation
* Deployment using **Streamlit**

---

## Live Demo

🎧 Try the deployed Streamlit app here:

🔗 **[https://music-popularity-predictor-pawgmryk8tepucedvnpdjk.streamlit.app/](https://music-popularity-predictor-pawgmryk8tepucedvnpdjk.streamlit.app/)**

The app allows users to:

* Input song features
* Predict popularity score
* View model behavior interactively

---

## Features

* Load and preprocess song dataset
* Train a **Random Forest** machine learning model
* Predict song popularity score
* Evaluate performance using:

  * RMSE
  * R²
  * MAE
* Explain predictions using **LIME**
* Deploy an interactive UI using **Streamlit**

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-link>
cd music-popularity-predictor
```

2. Create a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. (Optional) Launch Jupyter Notebook:

```powershell
jupyter notebook
```

---

## Usage

### Jupyter Notebook

1. Open `notebook.ipynb`
2. Run cells step-by-step to:

   * Load dataset
   * Train model
   * Make predictions
   * Evaluate results
   * Interpret predictions with LIME

### Streamlit App

Run locally:

```powershell
streamlit run app.py
```

---

## Project Structure

```
music-popularity-predictor/
│
├─ .venv/                  # Python virtual environment
├─ data/                   # Dataset (e.g., songs.csv)
├─ notebook.ipynb          # Model development notebook
├─ app.py                  # Streamlit application
├─ requirements.txt        # Project dependencies
└─ README.md               # Documentation
```

---

## Machine Learning

* **Algorithm:** Random Forest Regressor
* **Input Features:**
  Acousticness, danceability, energy, tempo, duration, etc.
* **Target Variable:**
  Song popularity score (numerical)

### Evaluation Metrics

* RMSE (Root Mean Squared Error)
* R² (Coefficient of Determination)
* MAE (Mean Absolute Error)

---

## Model Interpretation

* **LIME (Local Interpretable Model-agnostic Explanations)** is used
* Helps explain:

  * Why a song received a certain popularity score
  * Feature importance for individual predictions


