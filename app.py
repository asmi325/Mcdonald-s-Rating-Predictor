# filename: app.py
import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np


st.set_page_config(
    page_title="✨ McDonald's Rate Predictor!",
    page_icon="🍟",
    layout="centered",
    initial_sidebar_state="collapsed"
)


## --- CSS ---
st.markdown(
    """
    <style>
    /* Base */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        color: #E0E0E0;
        background-color: #1E1E1E; /* Dark Gray BG */
    }

    /* --- New Header Box --- */
    .header-box {
        background-color: #ff4d4d; /* Lighter Dark Gray */
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 30px;
        border: 1px solid #383838;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Titles (inside header box) */
    .title {
        text-align: center;
        color: #FFBC0D; /* Soft Gold */
        font-size: 38px;
        font-weight: 900;
        letter-spacing: 1px;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-family: 'Georgia', serif;
    }

    .subtitle {
        text-align: center;
        color: #AEC6CF; /* Pastel Blue */
        font-size: 17px;
        margin-bottom: 0; /* No margin needed inside the box */
        font-style: italic;
    }

    /* Main Button */
    .stButton>button {
        background-color: #E57373; /* Soft Red */
        color: #1E1E1E; /* Dark Text */
        border: none;
        padding: 12px 28px;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 4px #BF5F5F; /* Darker Red */
        transition: all 0.2s;
        font-size: 16px;
    }

    .stButton>button:hover {
        background-color: #BF5F5F;
        box-shadow: 0 2px #A95353;
        transform: translateY(2px);
    }

    /* File Uploader */
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #2C2C2C; /* Lighter Dark Gray */
        border: 2px dashed #AEC6CF; /* Pastel Blue */
        border-radius: 10px;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        background-color: #383838;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #2C2C2C; /* Lighter Dark Gray */
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #383838; /* Subtle border */
        margin: 10px 0;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5em;
        font-weight: bold;
        color: #FADF7E; /* Soft Gold */
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9em;
        color: #AEC6CF; /* Pastel Blue */
    }
    
    /* Info/Success/Error Boxes */
    div[data-testid="stInfo"],
    div[data-testid="stSuccess"],
    div[data-testid="stError"] {
        background-color: #2C2C2C;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid;
    }
    div[data-testid="stInfo"] {
        border-color: #AEC6CF; /* Pastel Blue */
        color: #E0E0E0;
    }
    div[data-testid="stSuccess"] {
        border-color: #C1E1C1; /* Pastel Green */
        color: #E0E0E0;
    }
    div[data-testid="stError"] {
        border-color: #E57373; /* Soft Red */
        color: #E0E0E0;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Header ---
# Wrapped title and subtitle in the new "header-box" div
st.markdown(
    """
    <div class="header-box">
        <div class="title">🍔🍟 McDonald's Rating Predictor! 🥤</div>
        <div class="subtitle">Upload your <b>preprocessed</b> test CSV and let our 🤖 predict ratings for fast-food places.</div>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Model Loading ---
MODEL_FILE = "final_lightgbm_model.txt"

try:
    model = lgb.Booster(model_file=MODEL_FILE)
except FileNotFoundError:
    st.error(f"❌ Model file not found! Please ensure **'{MODEL_FILE}'** is in the same folder as 'app.py'.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()


# --- File Upload ---
st.markdown("### **Step 1:** 📂 Upload Your Data")
# Removed the st.info box as requested
# st.info("Remember: The CSV must be **fully preprocessed** (columns and order must match the training data)!") 

uploaded_file = st.file_uploader(
    "Upload preprocessed CSV here",
    type=["csv"],
    accept_multiple_files=False,
    # Moved the warning to the help tooltip
    help="Remember: The CSV must be fully preprocessed (columns and order must match the training data)!"
)


# --- Prediction Logic ---
if uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        st.write("#### 👀 Data Preview (First 5 Rows)")
        st.dataframe(df_test.head(), use_container_width=True)

        if st.button("🚀 Predict Ratings"):
            with st.spinner('Calculating ratings... almost there! ⏳'):
                preds = model.predict(df_test)

                # Handle probability outputs
                if preds.ndim > 1:
                    preds = np.argmax(preds, axis=1)

                preds = np.asarray(preds).reshape(-1)
                preds = preds + 1  # Convert 0–4 → 1–5 scale

                # Combine with input data
                pred_df = pd.DataFrame(preds, columns=["Predicted_Rating"])
                results = pd.concat([df_test.reset_index(drop=True), pred_df], axis=1)

            st.balloons()
            st.success("🎉 Woohoo! Ratings predicted successfully!")

            # Summary metrics
            st.markdown("### 📊 Rating Insights")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Places", len(results))
            col2.metric("Average Predicted Rating", f"{round(pred_df['Predicted_Rating'].mean(), 2)} ⭐")
            col3.metric("Unique Ratings Predicted", int(pred_df["Predicted_Rating"].nunique()))

            # Show top predictions
            st.markdown("### 📋 Top 10 Predictions")
            st.dataframe(results.head(10), use_container_width=True)

            # Download predictions
            st.markdown("---")
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predicted Results (CSV)",
                data=csv,
                file_name="predicted_ratings_yummy.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"🐛 Oops! An error occurred during prediction: {e}")
        st.info("Check your CSV file again! It needs to be perfect (columns and order) for the model to work its magic. ✨")

else:
    st.info("Waiting for your preprocessed CSV upload... 👆")
