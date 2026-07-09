import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular
import tempfile
from recommender import get_similar_songs, get_top_tracks
from audio_features import extract_features

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
st.set_page_config(page_title="Track Intelligence Dashboard", page_icon="🎵", layout="wide")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎵 Track Intelligence Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict popularity potential, understand what drives it, and discover similar tracks — powered by Random Forest + LIME explainability.</p>', unsafe_allow_html=True)

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
        conf_fig = plt.figure(figsize=(6, 6))
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
        feat_imp_fig = plt.figure(figsize=(6, 6))
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
    st.header("Predict a Track's Popularity")

    with st.sidebar:
        st.markdown("## 🎛️ Track Input")
        input_mode = st.radio("Choose input method:", ["Manual sliders", "Upload audio file"])

        sliders = []

        if input_mode == "Manual sliders":
            st.caption("Adjust these to match your song")
            for feature in features:
                sliders.append(
                    st.slider(
                        label=feature,
                        min_value=float(df[feature].min()),
                        max_value=float(df[feature].max()),
                        value=float(df[feature].mean())
                    )
                )
        else:
            st.caption("Upload a short MP3 or WAV clip")
            uploaded_file = st.file_uploader("Audio file", type=["mp3", "wav"])
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                with st.spinner("Analyzing audio..."):
                    extracted = extract_features(tmp_path)

                st.markdown("**Extracted features:**")
                st.json(extracted)
                sliders = [extracted[feature] for feature in features]
            else:
                st.info("Upload a file to get a prediction.")

        st.divider()
        selected_genre = st.selectbox("Genre for recommendations:", sorted(df['track_genre'].unique()))

    col2 = st.container()
    with col2:
        if sliders:
            prediction = rf_classif.predict([sliders])[0]
            probs = rf_classif.predict_proba([sliders])[0]
            probability = probs[prediction]

            color_map = {0: "#EF4444", 1: "#F59E0B", 2: "#10B981"}  # red, amber, green
            with st.container(border=True):
                pcol1, pcol2 = st.columns([2, 1])
                with pcol1:
                    st.markdown("#### Predicted Class")
                    st.markdown(
                        f"<h2 style='color:{color_map[prediction]}; margin-top:-10px;'>{labels[prediction]}</h2>",
                        unsafe_allow_html=True
                    )
                with pcol2:
                    st.metric(label="Confidence", value=f"{probability*100:.1f}%")

            # LIME Explanation
            st.divider()
            st.markdown("#### 🔍 Why This Prediction")
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

            st.divider()
            st.markdown("#### 🎧 Similar Tracks")
            input_features_dict = dict(zip(features, sliders))
            similar = get_similar_songs(input_features_dict, df, features, genre=selected_genre, top_n=5)
            if not similar.empty:
                st.dataframe(similar[['track_name', 'artists', 'popularity']], use_container_width=True)
            else:
                st.info(f"No songs found in genre '{selected_genre}'.")

            st.divider()
            st.markdown(f"####  Trending in {selected_genre}")
            top_tracks = get_top_tracks(df, genre=selected_genre, top_n=5)
            st.dataframe(top_tracks[['track_name', 'artists', 'popularity']], use_container_width=True)
        else:
            st.info("👈 Choose an input method in the sidebar to get a prediction.")