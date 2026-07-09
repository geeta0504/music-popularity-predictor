---
# 🎵 Track Intelligence Dashboard (Music Popularity Predictor)

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
8. [Audio Feature Extraction](#audio-feature-extraction)
9. [Recommendation Engine](#recommendation-engine)
10. [Model Interpretation](#model-interpretation)
11. [Known Limitations](#known-limitations)

---

## Project Overview

**Track Intelligence Dashboard** is a Python-based machine learning app that classifies a song's **popularity tier** (Low / Medium / High) from its audio characteristics, and goes further than a typical classroom project by:

* Extracting real audio features directly from an **uploaded MP3/WAV file** (not just manual sliders), using signal processing
* Explaining individual predictions with **LIME**, so results aren't a black box
* Recommending **similar tracks** and **trending tracks** from a 114,000-song dataset, based on audio-feature similarity

The project covers the full pipeline: data preparation → model training → interactive prediction → explainability → recommendation → deployment.

---

## Live Demo

🎧 Try the deployed Streamlit app here:
🔗 **[https://music-popularity-predictor-pawgmryk8tepucedvnpdjk.streamlit.app/](https://music-popularity-predictor-pawgmryk8tepucedvnpdjk.streamlit.app/)**

The app lets you:
* Adjust song features manually via sliders, **or** upload a real audio clip and have features extracted automatically
* Get a predicted popularity class (Low / Medium / High) with confidence score
* See a LIME explanation of *why* that prediction was made
* Browse similar and trending tracks in a genre of your choice

---

## Features

* Load and cache a 114,000-song dataset (audio features + genre + popularity)
* Classify popularity into 3 tiers using a **Random Forest Classifier**
* **Two input modes:**
  * Manual sliders for each audio feature
  * Direct audio file upload, with features extracted via **Librosa** (tempo, loudness, energy, acousticness, danceability, speechiness, and rough valence/liveness proxies)
* Local, per-prediction explainability via **LIME**
* Content-based **recommendation engine**: finds the 5 most similar tracks (by normalized feature distance) and the top trending tracks, filtered by genre
* Global model diagnostics: confusion matrix, feature importances, classification report, accuracy
* Cached data loading and LIME explainer (`@st.cache_data` / `@st.cache_resource`) for responsive performance

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/geeta0504/music-popularity-predictor.git
cd music-popularity-predictor
```

2. Create a virtual environment (Windows PowerShell):
```powershell
python -m venv venv
venv\Scripts\activate
```

3. Upgrade pip:
```powershell
python -m pip install --upgrade pip
```

4. Install dependencies:
```powershell
pip install -r requirements.txt
```

---

## Usage

Run locally:
```powershell
streamlit run app.py
```

In the sidebar:
1. Choose **Manual sliders** or **Upload audio file** as your input method
2. If uploading, select a short MP3/WAV clip — extracted features are shown as JSON before prediction
3. Pick a genre for the recommendation sections
4. View the predicted class, confidence, LIME explanation, similar tracks, and trending tracks in the main panel

---

## Project Structure
```
music-popularity-predictor/
│
├─ app.py                          # Streamlit application (main entry point)
├─ audio_features.py               # Librosa-based feature extraction from uploaded audio
├─ recommender.py                  # Similarity-based and trending track recommendation logic
├─ dataset.csv.zip                 # 114k-song dataset (audio features, genre, popularity)
├─ music_popularity_classifier.model  # Pre-trained Random Forest classifier (joblib)
├─ requirements.txt                # Project dependencies
└─ README.md                       # Documentation
```

---

## Machine Learning

* **Algorithm:** Random Forest Classifier
* **Input Features:** danceability, energy, loudness, tempo, acousticness, valence, speechiness, liveness
* **Target Variable:** Popularity class — Low (0–33) / Medium (34–66) / High (67–100), binned from Spotify's 0–100 popularity score

### Evaluation Metrics
* Accuracy
* Per-class Precision / Recall / F1 (classification report)
* Confusion matrix (normalized)
* Global feature importances

---

## Audio Feature Extraction

Uploading a real audio file computes approximate versions of the model's input features directly from the waveform using **Librosa**:

| Feature | Method |
|---|---|
| Tempo | Beat tracking |
| Loudness | RMS energy converted to dB |
| Energy | Normalized RMS |
| Acousticness | Harmonic-to-percussive energy ratio |
| Speechiness | Zero-crossing rate |
| Danceability | Beat/onset strength consistency |
| Valence | Spectral centroid (rough proxy) |
| Liveness | Spectral flatness (rough proxy) |

These are engineered approximations, not Spotify's proprietary algorithm — see [Known Limitations](#known-limitations).

---

## Recommendation Engine

Given the current input song's feature vector:
* **Similar Tracks** — finds the 5 closest songs in the dataset (filtered by selected genre) using normalized Euclidean distance across all 8 audio features
* **Trending Tracks** — surfaces the top 5 highest-popularity songs in the selected genre

Feature values are standardized (zero mean, unit variance) before distance is computed, so no single feature (e.g. loudness) dominates the similarity score purely due to its numeric scale.

---

## Model Interpretation

**LIME (Local Interpretable Model-agnostic Explanations)** is used to explain individual predictions — showing which specific feature ranges pushed a given song toward its predicted class. This is distinct from global feature importance (which reflects the model's behavior on average across all songs): LIME explains *this one prediction*, not the model as a whole.

---

## Known Limitations

* **Valence and liveness proxies are weak.** Spectral centroid and spectral flatness are rough stand-ins for Spotify's proprietary "musical positivity" and "live performance" scores — they correlate loosely at best and shouldn't be read as precise.
* **Genre is user-selected, not auto-detected.** The app does not classify genre from audio; the user picks it manually to filter recommendations.
* **Audio-derived features are approximations**, not exact matches to Spotify's own feature computation, since that algorithm isn't public.
