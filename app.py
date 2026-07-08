import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular

# ===============================
# Load Dataset
# ===============================
zip_path = "dataset.csv.zip"

@st.cache_data
def load_data(path):
    try:
        data = pd.read_csv(path, compression='zip')
        return data, True
    except Exception:
        np.random.seed(42)
        data = pd.DataFrame({
            'danceability': np.random.rand(1000),
            'energy': np.random.rand(1000),
            'loudness': np.random.rand(1000) * -60,
            'tempo': np.random.rand(1000) * 200,
            'acousticness': np.random.rand(1000),
            'valence': np.random.rand(1000),
            'speechiness': np.random.rand(1000),
            'liveness': np.random.rand(1000),
            'popularity': np.random.randint(0, 101, 1000)
        })
        return data, False

df, loaded_ok = load_data(zip_path)
if loaded_ok:
    st.success("Dataset loaded successfully!")
else:
    st.warning("Failed to load dataset. Using synthetic fallback data.")
# ===============================
# Prepare Data
# ===============================
features = ['danceability', 'energy', 'loudness', 'tempo', 
            'acousticness', 'valence', 'speechiness', 'liveness']
features = [f for f in features if f in df.columns]

X = df[features]
bins = [0, 33, 66, 100]
labels = ['Low Popularity', 'Medium Popularity', 'High Popularity']
df['popularity_class'] = pd.cut(df['popularity'], bins=bins, labels=labels, include_lowest=True)
y = df['popularity_class'].cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

# ===============================
# Load Model
# ===============================
try:
    rf_classif = load("music_popularity_classifier.model")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model not found: {e}. Train and save the model first.")
    st.stop()

# ===============================
# Predictions for Metrics
# ===============================
y_test_preds = rf_classif.predict(X_test)
accuracy = accuracy_score(y_test, y_test_preds)

# ===============================
# Streamlit App Layout
# ===============================
st.set_page_config(page_title="Music Popularity Predictor", layout="wide")
st.title("🎵 Music Popularity Predictor")
st.markdown("Predict the **popularity class** of a song based on its audio features.")
@st.cache_resource
def get_explainer(training_values, feature_names, class_names):
    return lime_tabular.LimeTabularExplainer(
        training_data=training_values,
        mode="classification",
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        random_state=42
    )

explainer = get_explainer(X_train.values, features, labels)
tab1, tab2, tab3 = st.tabs(["📋 Data", "📊 Global Performance", "🧩 Local Performance"])

# -------------------------------
# Tab 1: Dataset
# -------------------------------
with tab1:
    st.header("Dataset Overview")
    st.dataframe(df.head(100))
    st.markdown(f"**Number of samples:** {df.shape[0]} | **Number of features:** {len(features)}")
    st.bar_chart(df['popularity_class'].value_counts())

# -------------------------------
# Tab 2: Global Performance
# -------------------------------
with tab2:
    st.header("Model Performance Metrics")
    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        conf_fig = plt.figure(figsize=(6,6))
        ax = conf_fig.add_subplot(111)
        ConfusionMatrixDisplay.from_predictions(
            y_test, 
            y_test_preds, 
            normalize='true', 
            display_labels=labels, 
            ax=ax, 
            cmap='Blues'
        )
        ax.set_title("Normalized Confusion Matrix")
        st.pyplot(conf_fig, use_container_width=True)

    # Feature Importances
    with col2:
        feat_imp_fig = plt.figure(figsize=(6,6))
        ax = feat_imp_fig.add_subplot(111)
        importances = rf_classif.feature_importances_
        indices = np.argsort(importances)
        ax.barh(range(len(indices)), importances[indices], color='skyblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([features[i] for i in indices])
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importances")
        st.pyplot(feat_imp_fig, use_container_width=True)

    st.divider()
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_test_preds, target_names=labels))
    st.metric(label="Overall Accuracy", value=f"{accuracy*100:.2f} %")

# -------------------------------
# Tab 3: Local Prediction & LIME
# -------------------------------
with tab3:
    
    st.header("Predict a Song's Popularity")
    sliders = []
    col1, col2 = st.columns(2)

    # Feature sliders
    with col1:
        st.markdown("### Adjust Song Features")
        for feature in features:
            sliders.append(
                st.slider(
                    label=feature,
                    min_value=float(df[feature].min()),
                    max_value=float(df[feature].max()),
                    value=float(df[feature].mean())
                )
            )

    # Prediction & LIME
    with col2:
        st.markdown("### Prediction & Confidence")
        prediction = rf_classif.predict([sliders])[0]
        probs = rf_classif.predict_proba([sliders])[0]
        probability = probs[prediction]

        st.markdown(f"**Predicted Popularity Class:** <span style='color:tomato'>{labels[prediction]}</span>", unsafe_allow_html=True)
        st.metric(label="Model Confidence", value=f"{probability*100:.2f} %")

        # LIME Explanation
        st.markdown("### LIME Feature Importance")
        try:
            explanation = explainer.explain_instance(
                data_row=np.array(sliders),
                predict_fn=rf_classif.predict_proba,
                num_features=len(features)
            )

            # Ensure the predicted label exists in explanation
            if prediction not in explanation.local_exp:
                prediction_for_lime = list(explanation.local_exp.keys())[0]
            else:
                prediction_for_lime = prediction

            fig = explanation.as_pyplot_figure(label=prediction_for_lime)
            st.pyplot(fig, use_container_width=True)

        except Exception as e:
            st.error(f"LIME explanation failed: {e}")
