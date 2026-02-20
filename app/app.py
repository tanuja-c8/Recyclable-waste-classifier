import random
import numpy as np
import tensorflow as tf
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

from model.inference import load_model, predict_image
from utils import log_prediction

# --------------------------- Streamlit Page Config --------------------------- #
st.set_page_config(
    page_title="EcoVision ♻️ Waste Classifier",
    layout="wide",
    page_icon="♻️",
    initial_sidebar_state="expanded"
)

# --------------------------- Custom CSS --------------------------- #
st.markdown("""
<style>
.main {
    background: radial-gradient(circle at top left, #e8fff1 0%, #f6fff9 100%);
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
    color: #065f46;
    font-weight: 700;
}
.header {
    text-align: center;
    font-size: 48px;
    background: linear-gradient(90deg, #16a085, #27ae60);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.upload-card {
    background-color: white;
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 8px 18px rgba(0,0,0,0.08);
}
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #27ae60, #16a085);
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg, #16a085, #27ae60);
    color: white;
    border-radius: 12px;
    padding: 0.7rem 1.4rem;
    font-weight: 600;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# --------------------------- Sidebar --------------------------- #
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=90)
    st.markdown("<h2 style='color:white;'>EcoVision</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:white;'>Smart Waste Classifier 🌍</p>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Navigate", ["🏠 Home", "📸 Classify Waste", "📊 Dashboard"])
    st.markdown("---")
    st.caption("AI-powered recycling for a cleaner planet")

# --------------------------- Load Model --------------------------- #
@st.cache_resource
def get_model():
    return load_model()

model = get_model()
CLASS_NAMES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

# --------------------------- Home Page --------------------------- #
if menu == "🏠 Home":
    st.markdown("""
<div style="
    background: linear-gradient(135deg, #16a085, #27ae60);
    padding: 60px;
    border-radius: 20px;
    text-align: center;
    color: white;
">
    <h1 style="font-size:52px;">♻️ EcoVision</h1>
    <p style="font-size:22px;">
        AI-powered waste classification system for smart recycling
    </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        st.button("📸 Classify Waste", use_container_width=True)
    with b2:
        st.button("📊 View Dashboard", use_container_width=True)
    with b3:
        st.button("🌍 Recycling Tips", use_container_width=True)

    st.markdown("### ✨ Why EcoVision?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### ⚡ Instant Detection")
        st.write("Upload an image and get predictions in seconds.")
    with c2:
        st.markdown("### ♻️ Multi-Class Support")
        st.write("Detects plastic, metal, paper, glass, cardboard and trash.")
    with c3:
        st.markdown("### 📊 Analytics Dashboard")
        st.write("Visualize predictions and confidence trends.")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Waste Classes", "6")
    m2.metric("Model Type", "MobileNetV2")
    m3.metric("Avg Accuracy", "87%")
    m4.metric("Prediction Time", "0.3s")

# --------------------------- Classify Waste Page --------------------------- #
elif menu == "📸 Classify Waste":
    st.markdown("<h1 class='header'>Smart Waste Classification</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<div class='upload-card'>", unsafe_allow_html=True)

        st.subheader("📷 Capture or Upload Image")

        captured_image = st.camera_input("Use camera")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        image = None
        image_name = None

        if captured_image is not None:
            image = Image.open(captured_image)
            image_name = "camera_capture"

        elif uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_name = uploaded_file.name

        if image is not None:
            st.image(image, use_container_width=True)

            if st.button("Analyze Waste"):
                with st.spinner("Analyzing image..."):
                    label, confidence, predictions = predict_image(
                        model, image, CLASS_NAMES
                    )

                    log_prediction(
                        image_name,
                        label,
                        confidence,
                        predictions
                    )

                    st.session_state["result"] = (label, confidence, predictions)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if "result" in st.session_state:
            label, confidence, predictions = st.session_state["result"]

            st.markdown(f"### Predicted Category: **{label}**")
            st.metric("Confidence", f"{confidence:.2f}%")

            df_pred = pd.DataFrame(
                list(predictions.items()),
                columns=["Category", "Confidence (%)"]
            )

            fig = px.bar(
                df_pred,
                x="Category",
                y="Confidence (%)",
                text_auto=".2f",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)


# --------------------------- Dashboard Page --------------------------- #
elif menu == "📊 Dashboard":
    st.markdown("<h1 class='header'>Analytics Dashboard</h1>", unsafe_allow_html=True)

    if not os.path.exists("logs/predictions.csv"):
        st.warning("No predictions logged yet.")
        st.stop()

    df = pd.read_csv("logs/predictions.csv")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", len(df))
    col2.metric("Most Common Waste", df["predicted_label"].mode()[0])
    col3.metric("Average Confidence", f"{df['confidence'].mean():.2f}%")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Visual Analytics", "📋 Prediction Logs", "🧠 Insights"])

    with tab1:
        chart = st.radio("Chart Type", ["Pie Chart", "Bar Chart", "Line Chart"], horizontal=True)

        if chart == "Pie Chart":
            fig = px.pie(df, names="predicted_label", title="Waste Category Distribution")
        elif chart == "Bar Chart":
            count_df = df["predicted_label"].value_counts().reset_index()
            count_df.columns = ["Category", "Count"]
            fig = px.bar(count_df, x="Category", y="Count", text="Count")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            fig = px.line(df, x="timestamp", y="confidence", markers=True, title="Confidence Over Time")

        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(px.histogram(df, x="confidence", nbins=10), use_container_width=True)

    with tab2:
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

    with tab3:
        top_class = df["predicted_label"].value_counts().idxmax()
        high_conf = df[df["confidence"] > 80].shape[0]
        st.success(f"Most detected waste: **{top_class}**")
        st.info(f"High-confidence predictions (>80%): {high_conf}")
