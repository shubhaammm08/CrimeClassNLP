import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import plotly.express as px

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Crime Prediction",
    page_icon="🚔",
    layout="wide"
)

# -------------------- Model Folder Selection --------------------
BASE_DIR = r"C:\Users\ssk08\OneDrive\Desktop\NLP Project\Model 2\saved_models"

if not os.path.exists(BASE_DIR):
    st.error(f"Base directory does not exist: {BASE_DIR}")
    st.stop()

# Get all subfolders inside saved_models
subdirs = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
]

if not subdirs:
    st.error(f"No model folders found in {BASE_DIR}")
    st.stop()

# Sidebar dropdown to select model folder
selected_folder = st.sidebar.selectbox("Select Model Folder", subdirs)

MODEL_DIR = os.path.join(BASE_DIR, selected_folder)
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Verify files exist
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()
if not os.path.exists(VECTORIZER_PATH):
    st.error(f"Vectorizer file not found at {VECTORIZER_PATH}")
    st.stop()

# -------------------- Load Pre-trained Model --------------------
@st.cache_resource
def load_model_and_vectorizer():
    # Load model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception:
        model = joblib.load(MODEL_PATH)

    # Load vectorizer
    try:
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
    except Exception:
        vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# -------------------- Helper Functions --------------------
def predict_state(crime_text):
    text_tfidf = vectorizer.transform([crime_text])
    prediction = model.predict(text_tfidf)[0]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(text_tfidf)[0]
        conf = np.max(probs)
        prob_dict = dict(zip(model.classes_, probs))
    else:
        conf = 1.0
        prob_dict = {prediction: 1.0}
    return prediction, conf, prob_dict


def display_prediction_results(prediction, confidence, probabilities):
    st.markdown(f"""
    <div style="background-color:#e8f4fd;padding:1.5rem;border-radius:0.5rem;border:2px solid #1f77b4;margin:1rem 0;">
        <h2 style="color: #1f77b4;">Prediction Results</h2>
        <p><strong>Predicted State:</strong> <span style="font-size:1.5em; color:#d62728;">{prediction}</span></p>
        <p><strong>Confidence Score:</strong> {confidence:.3f} ({confidence*100:.1f}%)</p>
    </div>
    """, unsafe_allow_html=True)

    # Top 5 probabilities
    top_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
    df = pd.DataFrame([
        {"Rank": i+1, "State": s, "Probability": f"{p:.3f}", "Percentage": f"{p*100:.1f}%"}
        for i, (s, p) in enumerate(top_probs)
    ])
    st.subheader("Top 5 State Predictions")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Plot chart
    if len(top_probs) > 1:
        fig = px.bar(
            x=[p for _, p in top_probs],
            y=[s for s, _ in top_probs],
            orientation="h",
            title="State Prediction Probabilities",
            labels={"x": "Probability", "y": "State"},
            color=[p for _, p in top_probs],
            color_continuous_scale="viridis"
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

# -------------------- Pages --------------------
def prediction_page():
    st.header("🔮 Crime Pattern Prediction")
    user_input = st.text_area("Enter Crime Pattern (example: district_mumbai crime_theft period_recent):")
    if st.button("Predict State", type="primary"):
        if user_input.strip():
            prediction, confidence, probs = predict_state(user_input)
            display_prediction_results(prediction, confidence, probs)
        else:
            st.error("Please enter a crime pattern.")


def analysis_page():
    st.header("📈 Data Analysis")
    uploaded_file = st.file_uploader("Upload your CSV file for analytics", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("States/UTs", df["STATE/UT"].nunique())
        col3.metric("Districts", df["DISTRICT"].nunique())
        st.dataframe(df.head(), use_container_width=True)

# -------------------- Main --------------------
def main():
    st.title("🚔 Crime Classification Predictor")
    page = st.sidebar.selectbox("Navigation", ["Model Prediction", "Data Analysis"])
    if page == "Model Prediction":
        prediction_page()
    else:
        analysis_page()

if __name__ == "__main__":
    main()
