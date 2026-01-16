import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# ==========================================
# 1. SETUP & LOAD RESOURCES
# ==========================================
st.set_page_config(page_title="Heart Attack Prediction", page_icon="💓", layout="centered")

# CSS untuk mempercantik tombol
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('model_ann_final.h5')
        preprocessor = joblib.load('preprocessor_final.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Gagal memuat file resource: {e}")
        return None, None

model, preprocessor = load_resources()

if not model or not preprocessor:
    st.stop()

# ==========================================
# 2. INISIALISASI STATE (ANTI-ERROR)
# ==========================================
# Bagian ini PENTING agar error "AttributeError" tidak muncul
# Kita set nilai default untuk semua variabel input

default_values = {
    'step': 1,
    # Slide 1
    'age': 45, 'gender': 'Male', 'income_level': 'Low',
    # Slide 2
    'smoking_status': 'Never', 'alcohol_consumption': 'Unknown', 
    'dietary_habits': 'Healthy', 'physical_activity': 'Moderate',
    # Slide 3
    'region': 'Urban', 'sleep_hours': 7.0, 
    'air_pollution_exposure': 'Low', 'stress_level': 'Low',
    # Slide 4
    'has_diabetes': 'Tidak', 'has_hypertension': 'Tidak', 
    'has_obesity': 'Tidak', 'participated_in_free_screening': 'Tidak',
    # Slide 5
    'family_history': 'Tidak', 'previous_heart_disease': 'Tidak', 
    'medication_usage': 'Tidak', 'EKG_results': 'Normal',
    # Slide 6
    'bp_systolic': 120, 'bp_diastolic': 80,
    'cholesterol_level': 200, 'cholesterol_ldl': 100, 'cholesterol_hdl': 50,
    'fasting_blood_sugar': 90, 'triglycerides': 150,
    'waist_circumference': 80, 'heart_rate': 75
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Fungsi Navigasi
def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1
def restart(): 
    for key, value in default_values.items():
        st.session_state[key] = value
    st.rerun()

# ==========================================
# 3. LOGIKA WIZARD (SLIDE PER SLIDE)
# ==========================================
st.title("💓 Prediksi Risiko Jantung")

# Progress Bar
steps_label = ["Data Diri", "Gaya Hidup", "Lingkungan", "Riwayat Medis", "Jantung & Obat", "Lab & Vital"]
current_step = st.session_state.step
st.progress((current_step - 1) / 5)
st.caption(f"Tahap {current_step} dari 6: {steps_label[current_step-1]}")
st.markdown("---")

# --- SLIDE 1: DATA DIRI ---
if current_step == 1:
    st.header("1. Data Diri")
    st.number_input("Usia", 18, 100, key='age')
    st.selectbox("Jenis Kelamin", ["Male", "Female"], key='gender')
    st.selectbox("Pendapatan", ["Low", "Middle", "High"], key='income_level')
    
    if st.button("Lanjut ➡️"): next_step(); st.rerun()

# --- SLIDE 2: GAYA HIDUP ---
elif current_step == 2:
    st.header("2. Gaya Hidup")
    st.selectbox("Merokok", ["Never", "Past", "Current"], key='smoking_status')
    st.selectbox("Alkohol", ["Unknown", "Moderate", "High"], key='alcohol_consumption')
    st.selectbox("Pola Makan", ["Healthy", "Unhealthy"], key='dietary_habits')
    st.selectbox("Aktivitas Fisik", ["Low", "Moderate", "High"], key='physical_activity')
    
    c1, c2 = st.columns(2)
    if c1.button("⬅️ Kembali"): prev_step(); st.rerun()
    if c2.button("Lanjut ➡️"): next_step(); st.rerun()

# --- SLIDE 3: LINGKUNGAN ---
elif current_step == 3:
    st.header("3. Lingkungan")
    st.selectbox("Wilayah", ["Urban", "Rural"], key='region')
    st.number_input("Jam Tidur/Hari", 0.0, 24.0, key='sleep_hours')
    st.selectbox("Polusi Udara", ["Low", "Moderate", "High"], key='air_pollution_exposure')
    st.selectbox("Tingkat Stres", ["Low", "Moderate", "High"], key='stress_level')

    c1, c2 = st.columns(2)
    if c1.button("⬅️ Kembali"): prev_step(); st.rerun()
    if c2.button("Lanjut ➡️"): next_step(); st.rerun()

# --- SLIDE 4: RIWAYAT MEDIS ---
elif current_step == 4:
    st.header("4. Riwayat Medis")
    st.radio("Diabetes", ["Tidak", "Ya"], horizontal=True, key='has_diabetes')
    st.radio("Hipertensi", ["Tidak", "Ya"], horizontal=True, key='has_hypertension')
    st.radio("Obesitas", ["Tidak", "Ya"], horizontal=True, key='has_obesity')
    st.selectbox("Ikut Skrining Gratis?", ["Tidak", "Ya"], key='participated_in_free_screening')

    c1, c2 = st.columns(2)
    if c1.button("⬅️ Kembali"): prev_step(); st.rerun()
    if c2.button("Lanjut ➡️"): next_step(); st.rerun()

# --- SLIDE 5: JANTUNG & OBAT ---
elif current_step == 5:
    st.header("5. Riwayat Jantung")
    st.radio("Riwayat Keluarga Sakit Jantung", ["Tidak", "Ya"], horizontal=True, key='family_history')
    st.radio("Riwayat Jantung Sebelumnya", ["Tidak", "Ya"], horizontal=True, key='previous_heart_disease')
    st.radio("Minum Obat Rutin?", ["Tidak", "Ya"], horizontal=True, key='medication_usage')
    st.radio("Hasil EKG", ["Normal", "Abnormal"], horizontal=True, key='EKG_results')

    c1, c2 = st.columns(2)
    if c1.button("⬅️ Kembali"): prev_step(); st.rerun()
    if c2.button("Lanjut ➡️"): next_step(); st.rerun()

# --- SLIDE 6: LAB & VITAL ---
elif current_step == 6:
    st.header("6. Hasil Lab & Vital")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.number_input("Tensi Sistolik", 90, 200, key='bp_systolic')
        st.number_input("Total Kolesterol", 100, 400, key='cholesterol_level')
        st.number_input("LDL (Jahat)", 50, 250, key='cholesterol_ldl')
        st.number_input("Gula Darah Puasa", 60, 300, key='fasting_blood_sugar')
    
    with col_b:
        st.number_input("Tensi Diastolik", 60, 130, key='bp_diastolic')
        st.number_input("HDL (Baik)", 20, 100, key='cholesterol_hdl')
        st.number_input("Trigliserida", 50, 500, key='triglycerides')
        st.number_input("Lingkar Pinggang (cm)", 50, 150, key='waist_circumference')
        st.number_input("Detak Jantung", 50, 150, key='heart_rate')

    st.markdown("---")
    c1, c2 = st.columns(2)
    
    if c1.button("⬅️ Kembali"): prev_step(); st.rerun()
    
    # --- PROSES PREDIKSI ---
    if c2.button("🔍 PREDIKSI HASIL", type="primary"):
        ss = st.session_state # Singkatan
        
        # 1. Kumpulkan Data (Aman karena sudah di-init di atas)
        input_data = {
            'age': ss.age, 'gender': ss.gender, 'region': ss.region, 'income_level': ss.income_level,
            'hypertension': 1 if ss.has_hypertension == "Ya" else 0,
            'diabetes': 1 if ss.has_diabetes == "Ya" else 0,
            'cholesterol_level': ss.cholesterol_level,
            'obesity': 1 if ss.has_obesity == "Ya" else 0,
            'waist_circumference': ss.waist_circumference,
            'family_history': 1 if ss.family_history == "Ya" else 0,
            'smoking_status': ss.smoking_status,
            'alcohol_consumption': ss.alcohol_consumption,
            'physical_activity': ss.physical_activity,
            'dietary_habits': ss.dietary_habits,
            'air_pollution_exposure': ss.air_pollution_exposure,
            'stress_level': ss.stress_level,
            'sleep_hours': ss.sleep_hours,
            'blood_pressure_systolic': ss.bp_systolic,
            'blood_pressure_diastolic': ss.bp_diastolic,
            'fasting_blood_sugar': ss.fasting_blood_sugar,
            'cholesterol_hdl': ss.cholesterol_hdl,
            'cholesterol_ldl': ss.cholesterol_ldl,
            'triglycerides': ss.triglycerides,
            'EKG_results': ss.EKG_results,
            'previous_heart_disease': 1 if ss.previous_heart_disease == "Ya" else 0,
            'medication_usage': 1 if ss.medication_usage == "Ya" else 0,
            'participated_in_free_screening': 1 if ss.participated_in_free_screening == "Ya" else 0,
            'heart_attack': 0 # Dummy
        }
        
        del input_data['heart_attack']
        df_input = pd.DataFrame([input_data])

        # 2. Feature Engineering
        if ss.age <= 35: df_input['Age_Group'] = 'Young'
        elif ss.age <= 55: df_input['Age_Group'] = 'Middle'
        else: df_input['Age_Group'] = 'Senior'
        
        df_input['is_smoker'] = 1 if ss.smoking_status in ['Current', 'Past'] else 0
        df_input['Total_Risk_Factors'] = (df_input['hypertension'] + df_input['diabetes'] + 
                                          df_input['obesity'] + df_input['is_smoker'])

        # 3. Prediksi
        try:
            final_input = preprocessor.transform(df_input)
            prob = model.predict(final_input)[0][0]
            
            st.divider()
            col_res1, col_res2 = st.columns([1,3])
            
            with col_res1:
                st.metric("Skor Risiko", f"{prob:.1%}")
            
            with col_res2:
                if prob > 0.5:
                    st.error("⚠️ **BERISIKO TINGGI**")
                    st.write("Model mendeteksi pola yang mengarah pada risiko serangan jantung. Segera konsultasikan ke dokter.")
                else:
                    st.success("✅ **RISIKO RENDAH**")
                    st.write("Profil kesehatan Anda terpantau aman. Pertahankan gaya hidup sehat.")
            
            if st.button("🔄 Mulai Ulang"): restart()

        except Exception as e:
            st.error(f"Terjadi kesalahan teknis: {e}")