import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load Model & Encoders
# -------------------------------
model = joblib.load("../backend/ctr_model.pkl")
encoders = joblib.load("../backend/encoders.pkl")
feature_order = joblib.load("../backend/feature_order.pkl")

st.set_page_config(
    page_title="Ad Click Prediction",
    layout="wide"
)

st.title("📊 Ad Click Prediction App")
st.write("Predict whether an ad will be clicked based on campaign, app, site, and device info.")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Input Ad & User Details")

def user_input():
    input_data = {}
    
    # Campaign info
    input_data['hour'] = st.sidebar.number_input("Hour of the Day (0-23)", 0, 23, 12)
    input_data['banner_pos'] = st.sidebar.selectbox("Banner Position", [0,1,2,3,4])
    
    # Site info
    input_data['site_id'] = st.sidebar.text_input("Site ID", "1fbe01fe")
    input_data['site_domain'] = st.sidebar.text_input("Site Domain", "f3845767")
    input_data['site_category'] = st.sidebar.text_input("Site Category", "28905ebd")
    
    # App info
    input_data['app_id'] = st.sidebar.text_input("App ID", "ecad2386")
    input_data['app_domain'] = st.sidebar.text_input("App Domain", "7801e8d9")
    input_data['app_category'] = st.sidebar.text_input("App Category", "0")
    
    # Device info
    input_data['device_id'] = st.sidebar.text_input("Device ID", "55298")
    input_data['device_ip'] = st.sidebar.text_input("Device IP", "271188")
    input_data['device_model'] = st.sidebar.text_input("Device Model", "1266")
    input_data['device_type'] = st.sidebar.selectbox("Device Type", [0,1,2])
    input_data['device_conn_type'] = st.sidebar.selectbox("Connection Type", [0,1,2,3])
    
    return pd.DataFrame([input_data])

df_input = user_input()

# -------------------------------
# Encode Categorical Columns
# -------------------------------
for col in encoders:
    if col in df_input.columns:
        le = encoders[col]
        # safe transform: unseen labels -> -1
        df_input[col] = df_input[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Ensure columns are in the same order as during training
df_input = df_input.reindex(columns=feature_order, fill_value=0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Click"):
    pred_prob = model.predict_proba(df_input)[:,1][0]
    pred_label = model.predict(df_input)[0]
    
    if pred_label == 1:
        st.success(f"✅ This ad is likely to be clicked! (Probability: {pred_prob:.2f})")
    else:
        st.warning(f"❌ This ad is unlikely to be clicked. (Probability: {pred_prob:.2f})")
    
    # -------------------------------
    # Visual Feedback
    # -------------------------------
    st.subheader("Prediction Probability")
    st.bar_chart({"Click Probability": [pred_prob], "No Click Probability": [1-pred_prob]})
