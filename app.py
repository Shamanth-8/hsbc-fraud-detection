import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection", layout="wide")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("fraud_autoencoder.keras")
    with open("fraud_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, meta["scaler"], meta["threshold"]

model, scaler, saved_threshold = load_artifacts()

st.title("AI-Powered Credit Card Fraud Detection")

uploaded_file = st.file_uploader("Upload transaction CSV (no Class column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Separate features from potential target/extra columns
    X = df.copy()
    if "Class" in X.columns:
        X = X.drop(columns=["Class"])
    
    # Basic validation to prevent crashing with an obscure error
    if X.shape[1] != 30:
        st.error(f"Shape Error: Expected 30 features (Time, V1-V28, Amount), but got {X.shape[1]}. Please check your CSV columns.")
        st.stop()

    X_scaled = scaler.transform(X.values)
    recon = model.predict(X_scaled, verbose=0)
    scores = np.mean(np.square(X_scaled - recon), axis=1)

    st.subheader("Threshold Control")
    threshold = st.slider(
        "Fraud Threshold",
        min_value=float(scores.min()),
        max_value=float(scores.max()),
        value=float(saved_threshold),
        step=float((scores.max() - scores.min()) / 1000)
    )

    preds = (scores >= threshold).astype(int)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Detected Frauds", int(preds.sum()))
    col3.metric("Fraud Rate (%)", round(100 * preds.mean(), 3))

    df["AnomalyScore"] = scores
    df["FraudPrediction"] = preds

    st.subheader("Fraud Predictions")
    st.dataframe(df.head(20))

    # ================= CHART =================
    st.subheader("Anomaly Score Distribution")

    fig, ax = plt.subplots()
    ax.hist(scores, bins=100, color="skyblue", alpha=0.8)
    ax.axvline(threshold, color="red", linestyle="--", label="Threshold")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.legend()

    st.pyplot(fig)

    st.download_button(
        "Download Results",
        df.to_csv(index=False),
        "fraud_predictions.csv",
        "text/csv"
    )