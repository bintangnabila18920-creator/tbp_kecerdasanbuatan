import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Prediksi Payung - Perceptron", page_icon="☔")

# 1. Load Model dan Scaler yang sudah disimpan (.pkl)
@st.cache_resource
def load_ml_models():
    with open('model_payung.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_ml_models()

# 2. Tampilan Website Streamlit
st.title("☔ Aplikasi Rekomendasi Payung (Model Perceptron)")
st.markdown("Masukkan data cuaca hari ini untuk memprediksi apakah Anda perlu membawa payung!")

st.divider()

# Form Input untuk Pengguna
col1, col2 = st.columns(2)

with col1:
    rainfall = st.number_input("Curah Hujan (mm)", min_value=0.0, value=0.0, step=0.1)
    wind_9am = st.number_input("Kecepatan Angin 9 Pagi (km/h)", min_value=0, value=15)
    wind_3pm = st.number_input("Kecepatan Angin 3 Sore (km/h)", min_value=0, value=20)
    hum_9am = st.number_input("Kelembapan 9 Pagi (%)", min_value=0, max_value=100, value=60)
    hum_3pm = st.number_input("Kelembapan 3 Sore (%)", min_value=0, max_value=100, value=50)

with col2:
    press_9am = st.number_input("Tekanan Udara 9 Pagi (hPa)", value=1012.0)
    press_3pm = st.number_input("Tekanan Udara 3 Sore (hPa)", value=1009.0)
    temp_9am = st.number_input("Suhu 9 Pagi (°C)", value=25.0)
    temp_3pm = st.number_input("Suhu 3 Sore (°C)", value=32.0)
    rain_today = st.selectbox("Apakah Hari Ini Hujan?", options=["No", "Yes"])

# Konversi RainToday ke Angka (1 / 0)
rain_today_val = 1 if rain_today == "Yes" else 0

# Tombol Prediksi
if st.button("Cek Apakah Perlu Bawa Payung", type="primary"):
    # Susun data input sesuai urutan fitur saat training
    input_data = np.array([[rainfall, wind_9am, wind_3pm, hum_9am, hum_3pm, 
                            press_9am, press_3pm, temp_9am, temp_3pm, rain_today_val]])
    
    # Normalisasi data menggunakan scaler yang sama
    input_scaled = scaler.transform(input_data)
    
    # Prediksi dengan model Perceptron
    prediksi = model.predict(input_scaled)
    
    st.markdown("---")
    st.subheader("Hasil Rekomendasi:")
    
    if prediksi[0] == 1:
        st.error("### ☔ **WAJIB MEMBAWA PAYUNG!**\nModel memprediksi kemungkinan besar akan turun hujan.")
    else:
        st.success("### ☀️ **TIDAK PERLU MEMBAWA PAYUNG**\nCuaca diprediksi aman dan cerah.")
