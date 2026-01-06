
# Music Popularity Predictor 🎵

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Machine Learning](#machine-learning)
7. [Model Interpretation](#model-interpretation)
8. [Future Improvements](#future-improvements)
9. [License](#license)

---

## Project Overview

**Music Popularity Predictor** is a Python-based machine learning project designed to **predict the popularity of songs** based on various features like acoustic properties, tempo, duration, and more.
The project includes data processing, model training, evaluation, and interpretation.

---

## Features

* Load and preprocess song dataset.
* Train a **Random Forest** (or other ML) model to predict song popularity.
* Evaluate model performance using metrics like **RMSE, R², and MAE**.
* Interpret model predictions using **LIME**.
* Optionally deploy prediction interface using **Streamlit**.

---

## Installation

1. Clone or download the project:

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

1. Open `notebook.ipynb` in Jupyter.

2. Follow the **step-by-step cells** to:

   * Load dataset
   * Train model
   * Make predictions
   * Evaluate model
   * Interpret results

3. (Optional) Run Streamlit app:

```powershell
streamlit run app.py
```

---

## Project Structure

```
music-popularity-predictor/
│
├─ .venv/                  # Python virtual environment
├─ data/                   # Place your dataset here (e.g., songs.csv)
├─ notebook.ipynb          # Main Jupyter Notebook
├─ app.py                  # Optional Streamlit app
├─ requirements.txt        # Dependencies
└─ README.md               # Project documentation
```

---

## Machine Learning

* **Algorithm:** Random Forest Regressor (default)
* **Input Features:** Acoustic features, tempo, duration, etc.
* **Target:** Song popularity score (numerical)

**Evaluation Metrics:**

* RMSE (Root Mean Squared Error)
* R² (Coefficient of Determination)
* MAE (Mean Absolute Error)

---


## Future Improvements

* Add more **advanced models** (XGBoost, LightGBM).
* Integrate **hyperparameter tuning**.
* Deploy **web-based prediction app** using Streamlit with file upload.
* Include **real-time data scraping** for new songs.

---

