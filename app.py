# filename: app.py
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ==============================
# 🎨 PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="✨ McDonald's Rating Predictor!",
    page_icon="🍟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================
# 🎨 CUSTOM CSS
# ==============================
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        color: #E0E0E0;
        background-color: #1a1a2e;
    }
    .title { text-align: center; color: #BB86FC; font-size: 40px; font-weight: 900; letter-spacing: 2px; margin-bottom: 5px; text-shadow: 3px 3px 6px rgba(0,0,0,0.3); font-family: 'Georgia', serif; }
    .subtitle { text-align: center; color: #A0A0B0; font-size: 17px; margin-bottom: 35px; font-style: italic; }
    .stButton>button { background-color: #FFD700; color: #333333; border: none; padding: 12px 28px; border-radius: 15px; font-weight: bold; box-shadow: 0 5px #E4C100; transition: all 0.2s; font-size: 16px; }
    .stButton>button:hover { background-color: #E4C100; box-shadow: 0 3px #C5A800; transform: translateY(2px); }
    .stContainer { border: 2px solid #BB86FC; border-radius: 20px; padding: 30px; margin-bottom: 30px; background-color: #2e1a3e; box-shadow: 0 6px 12px rgba(0,0,0,0.2); }
    div[data-testid="stFileUploaderDropzone"] { background-color: #4a1c6a; border: 2px dashed #BB86FC; border-radius: 10px; padding: 20px; min-height: 150px; display: flex; align-items: center; justify-content: center; }
    div[data-testid="stFileUploaderDropzone"]:hover { background-color: #5d2583; }
    div[data-testid="stMetric"] { background-color: #4a1c6a; border-radius: 12px; padding: 15px; border: 1px solid #BB86FC; margin: 10px 0; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); text-align: center; color: #E0E0E0; }
    div[data-testid="stMetricValue"] { font-size: 2.5em; font-weight: bold; color: #FFD700; }
    div[data-testid="stMetricLabel"] { font-size: 0.9em; color: #BB86FC; margin-bottom: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# 🧾 HEADER
# ==============================
st.markdown('<div class="title">🍔🍟✨ McDonald\'s Rating Predictor! 🥤🍕</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict fast-food ratings either for one restaurant or upload a full dataset! 🤖✨</div>', unsafe_allow_html=True)

# ==============================
# 🧩 MODEL LOADING
# ==============================
MODEL_FILE = "final_lightgbm_model.txt"
try:
    model = lgb.Booster(model_file=MODEL_FILE)
    st.success("✅ Prediction Model Loaded & Ready to Serve!")
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}")
    st.stop()

# ==============================
# 🧭 MODE SELECTION
# ==============================
mode = st.radio("Choose Prediction Mode:", ["🔹 Single Restaurant", "📂 Batch CSV Upload"])

# Common preprocessing helper
def preprocess_input(df):
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    imputer = SimpleImputer(strategy="mean")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df

# ==============================
# 🌟 SINGLE RESTAURANT MODE
# ==============================
if "Single" in mode:
    st.markdown("### 🍔 Enter Restaurant Details Manually")

    with st.form("single_form"):
        col1, col2 = st.columns(2)
        with col1:
            store_name = st.text_input("Store Name", "McDonald's Downtown")
            category = st.selectbox("Category", ["Fast Food", "Cafe", "Burger", "Dessert", "Other"])
            store_address = st.text_input("Store Address", "123 Food Street")
            city = st.text_input("City", "New York")
            state = st.text_input("State", "NY")
        with col2:
            latitude = st.number_input("Latitude", value=40.7128)
            longitude = st.number_input("Longitude", value=-74.0060)
            rating_count = st.number_input("Rating Count", value=200)
            review_time = st.text_input("Review Time", "2023-05-10")
            review = st.text_area("Customer Review", "Tasty food and quick service!")
        submitted = st.form_submit_button("🍟 Predict Rating")

    if submitted:
        single_df = pd.DataFrame([{
            "id": 0,
            "store_name": store_name,
            "category": category,
            "store_address": store_address,
            "latitude": latitude,
            "longitude": longitude,
            "rating_count": rating_count,
            "review_time": review_time,
            "review": review,
            "city": city,
            "state": state,
            "zip": "00000",
        }])

        try:
            X_proc = preprocess_input(single_df)
            pred = model.predict(X_proc)[0]
            st.balloons()
            st.success(f"🎯 Predicted Rating: ⭐ **{round(pred, 2)} / 5**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ==============================
# 📂 BATCH MODE
# ==============================
else:
    st.markdown("### 📂 Upload Preprocessed CSV File")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df_test = pd.read_csv(uploaded_file)
            st.write("#### 👀 Data Preview")
            st.dataframe(df_test.head(), use_container_width=True)
            if st.button("🚀 Predict Ratings"):
                with st.spinner("Predicting delicious ratings... 🍔"):
                    X_proc = preprocess_input(df_test)
                    preds = model.predict(X_proc)
                    if preds.ndim > 1 and preds.shape[1] > 1:
                        preds = np.argmax(preds, axis=1)
                    df_test["Predicted_Rating"] = preds
                    st.success("🎉 Ratings Predicted Successfully!")
                    st.dataframe(df_test.head(10), use_container_width=True)
                    csv = df_test.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Predicted Results", csv, "predicted_ratings.csv", "text/csv")
        except Exception as e:
            st.error(f"🐛 Error during batch prediction: {e}")
