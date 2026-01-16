import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# ==========================================
# 0. KONFIGURASI TEMA
# ==========================================
if not os.path.exists(".streamlit"):
    os.makedirs(".streamlit")

with open(".streamlit/config.toml", "w") as f:
    f.write("""
[theme]
base="light"
primaryColor="#0055AA"
backgroundColor="#F0F2F6"
secondaryBackgroundColor="#F5F5F5"
textColor="#333333"
font="sans serif"
    """)

# ==========================================
# 1. SETUP APLIKASI
# ==========================================
st.set_page_config(
    page_title="MediHeart AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS CUSTOM
# ==========================================
st.markdown("""
    <style>
    /* 1. CONFIG GLOBAL */
    [data-testid="stAppViewContainer"], .stApp { background-color: #F0F2F6 !important; }
    
    /* 2. HEADER & JUDUL (NAVY GELAP) */
    h1, h2, h3, h4, h5, h6, strong { 
        color: #003366 !important; 
        font-family: 'Segoe UI', Tahoma, sans-serif;
    }

    /* 3. LABEL & TEKS BIASA (ABU GELAP) */
    p, label, span, div, .stText {
        color: #444444 !important;
        font-family: 'Segoe UI', Tahoma, sans-serif;
    }

    /* 4. KOTAK INPUT */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"] {
        background-color: #FFFFFF !important;
        border: 2px solid #0055AA !important;
        border-radius: 8px !important;
    }
    input[type="number"], div[data-baseweb="select"] div {
        color: #222222 !important;
        -webkit-text-fill-color: #222222 !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] svg { fill: #0055AA !important; }

    /* 5. MENU & RADIO */
    ul[data-baseweb="menu"] { background-color: #FFFFFF !important; border: 2px solid #0055AA !important; }
    li[data-baseweb="option"] { color: #333333 !important; -webkit-text-fill-color: #333333 !important; }
    li[data-baseweb="option"]:hover { background-color: #E6F0FF !important; }
    
    div[data-testid="stWidgetLabel"] label, div[data-testid="stWidgetLabel"] p {
        color: #333333 !important; font-weight: 700 !important; font-size: 15px !important;
    }
    div[role="radiogroup"] label p { color: #444444 !important; font-weight: 500 !important; }

    /* 6. TOMBOL */
    .stButton button {
        background-color: #0055AA !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border-radius: 8px !important; border: none !important; font-weight: bold !important;
    }
    /* Tombol Kembali */
    div[data-testid="column"]:first-child .stButton button {
        background-color: #FFFFFF !important;
        color: #0055AA !important;
        -webkit-text-fill-color: #0055AA !important;
        border: 2px solid #0055AA !important;
    }

    /* 7. CONTAINER & SIDEBAR */
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important; 
        border: 1px solid #0055AA !important;
        border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #CCCCCC; }
    [data-testid="stSidebar"] * { color: #003366 !important; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white; border-radius: 10px; margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('model_ann_final.h5')
        preprocessor = joblib.load('preprocessor_final.pkl')
        return model, preprocessor
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        return None, None

model, preprocessor = load_resources()

if not model or not preprocessor:
    st.stop()

# ==========================================
# 4. STATE MANAGEMENT & RESET FUNCTION
# ==========================================
default_values = {
    'step': 1,
    'age': 45, 'gender': 'Laki-laki', 'income_level': 'Rendah',
    'smoking_status': 'Tidak Pernah', 'alcohol_consumption': 'Tidak Pernah',
    'dietary_habits': 'Sehat', 'physical_activity': 'Sedang',
    'region': 'Perkotaan', 'sleep_hours': 7.0, 
    'air_pollution_exposure': 'Rendah', 'stress_level': 'Rendah',
    'has_diabetes': 'Tidak', 'has_hypertension': 'Tidak', 
    'has_obesity': 'Tidak', 'participated_in_free_screening': 'Tidak',
    'family_history': 'Tidak', 'previous_heart_disease': 'Tidak', 
    'medication_usage': 'Tidak', 'EKG_results': 'Normal',
    'bp_systolic': 120, 'bp_diastolic': 80,
    'cholesterol_level': 200, 'cholesterol_ldl': 100, 'cholesterol_hdl': 50,
    'fasting_blood_sugar': 90, 'triglycerides': 150,
    'waist_circumference': 80, 'heart_rate': 75
}

# --- PERBAIKAN: INISIALISASI STATE ---
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- PERBAIKAN UTAMA: FUNGSI RESTART (CALLBACK) ---
# Fungsi ini akan dipanggil oleh on_click, jadi tidak boleh ada st.rerun() di dalamnya
def restart(): 
    # Reset semua variabel input
    for key, value in default_values.items():
        st.session_state[key] = value
    
    # Hapus hasil prediksi sebelumnya
    if 'result_prob' in st.session_state:
        del st.session_state['result_prob']

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1

# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
    st.markdown("### MediHeart AI")
    st.info("Prediksi Risiko Jantung Cerdas")
    
    steps = ["Biodata", "Gaya Hidup", "Lingkungan", "Riwayat Medis", "Jantung", "Lab & Vital"]
    curr = st.session_state.step
    
    st.markdown("---")
    st.write(f"**Tahap {curr} / 6**")
    st.progress(curr/6)
    
    for i, s in enumerate(steps, 1):
        if i == curr:
            st.markdown(f"🔵 **{s}**")
        else:
            st.markdown(f"⚪ {s}")

# ==========================================
# 6. HALAMAN UTAMA
# ==========================================
st.markdown("<h1 style='text-align: center;'>Formulir Analisis Risiko</h1>", unsafe_allow_html=True)

with st.form("wizard_form"):
    
    # --- SLIDE 1: BIODATA ---
    if curr == 1:
        st.markdown("### 👤 Data Pasien")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.number_input("Usia (Tahun)", 18, 100, key='age')
            with st.container(border=True):
                st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], key='gender')
        with c2:
            with st.container(border=True):
                st.selectbox("Tingkat Pendapatan", ["Rendah", "Menengah", "Tinggi"], key='income_level')
            with st.container(border=True):
                st.info("ℹ️ Masukkan identitas pasien.")

        st.markdown("---")
        c_sub1, c_sub2 = st.columns([5, 1])
        if c_sub2.form_submit_button("Lanjut ➡️"):
            next_step()
            st.rerun()

    # --- SLIDE 2: GAYA HIDUP ---
    elif curr == 2:
        st.markdown("### 🏃 Gaya Hidup")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.selectbox("Status Merokok", ["Tidak Pernah", "Pernah (Dulu)", "Perokok Aktif"], key='smoking_status')
            with st.container(border=True):
                st.selectbox("Konsumsi Alkohol", ["Tidak Pernah", "Sedang", "Tinggi"], key='alcohol_consumption')
        with c2:
            with st.container(border=True):
                st.selectbox("Pola Makan", ["Sehat", "Tidak Sehat"], key='dietary_habits')
            with st.container(border=True):
                st.selectbox("Aktivitas Fisik", ["Rendah", "Sedang", "Tinggi"], key='physical_activity')

        st.markdown("---")
        b1, b2, b3 = st.columns([1, 4, 1])
        if b1.form_submit_button("⬅️ Kembali"): prev_step(); st.rerun()
        if b3.form_submit_button("Lanjut ➡️"): next_step(); st.rerun()

    # --- SLIDE 3: LINGKUNGAN ---
    elif curr == 3:
        st.markdown("### 🏠 Lingkungan")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.selectbox("Wilayah Tinggal", ["Perkotaan", "Pedesaan"], key='region')
            with st.container(border=True):
                st.number_input("Jam Tidur (Jam/Hari)", 0.0, 24.0, key='sleep_hours')
        with c2:
            with st.container(border=True):
                st.selectbox("Paparan Polusi Udara", ["Rendah", "Sedang", "Tinggi"], key='air_pollution_exposure')
            with st.container(border=True):
                st.selectbox("Tingkat Stres", ["Rendah", "Sedang", "Tinggi"], key='stress_level')

        st.markdown("---")
        b1, b2, b3 = st.columns([1, 4, 1])
        if b1.form_submit_button("⬅️ Kembali"): prev_step(); st.rerun()
        if b3.form_submit_button("Lanjut ➡️"): next_step(); st.rerun()

    # --- SLIDE 4: RIWAYAT MEDIS ---
    elif curr == 4:
        st.markdown("### 🩺 Riwayat Medis")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.radio("Apakah Diabetes?", ["Tidak", "Ya"], horizontal=True, key='has_diabetes')
            with st.container(border=True):
                st.radio("Apakah Hipertensi?", ["Tidak", "Ya"], horizontal=True, key='has_hypertension')
        with c2:
            with st.container(border=True):
                st.radio("Apakah Obesitas?", ["Tidak", "Ya"], horizontal=True, key='has_obesity')
            with st.container(border=True):
                st.selectbox("Pernah Skrining Gratis?", ["Tidak", "Ya"], key='participated_in_free_screening')

        st.markdown("---")
        b1, b2, b3 = st.columns([1, 4, 1])
        if b1.form_submit_button("⬅️ Kembali"): prev_step(); st.rerun()
        if b3.form_submit_button("Lanjut ➡️"): next_step(); st.rerun()

    # --- SLIDE 5: JANTUNG ---
    elif curr == 5:
        st.markdown("### ❤️ Riwayat Jantung")
        
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            with st.container(border=True):
                st.warning("Riwayat Keluarga")
                st.radio("Ada keluarga sakit jantung?", ["Tidak", "Ya"], horizontal=True, key='family_history')
        with r1_c2:
            with st.container(border=True):
                st.success("Pengobatan")
                st.radio("Minum obat rutin?", ["Tidak", "Ya"], horizontal=True, key='medication_usage')

        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            with st.container(border=True):
                st.error("Riwayat Pasien")
                st.radio("Pernah serangan jantung?", ["Tidak", "Ya"], horizontal=True, key='previous_heart_disease')
        with r2_c2:
            with st.container(border=True):
                st.info("Hasil EKG")
                st.radio("Hasil EKG Terakhir", ["Normal", "Abnormal"], horizontal=True, key='EKG_results')

        st.markdown("---")
        b1, b2, b3 = st.columns([1, 4, 1])
        if b1.form_submit_button("⬅️ Kembali"): prev_step(); st.rerun()
        if b3.form_submit_button("Lanjut ➡️"): next_step(); st.rerun()

    # --- SLIDE 6: LAB & VITAL (FINAL) ---
    elif curr == 6:
        st.markdown("### 🩸 Hasil Lab & Tanda Vital")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            with st.container(border=True):
                st.markdown("##### 💓 Sistolik")
                st.number_input("mmHg", 90, 200, key='bp_systolic', label_visibility="collapsed")
            with st.container(border=True):
                st.markdown("##### 🍬 Gula Darah")
                st.number_input("mg/dL", 60, 300, key='fasting_blood_sugar', label_visibility="collapsed")
            with st.container(border=True):
                st.markdown("##### 🍟 Trigliserida")
                st.number_input("mg/dL", 50, 500, key='triglycerides', label_visibility="collapsed")
        
        with c2:
            with st.container(border=True):
                st.markdown("##### 🩸 Diastolik")
                st.number_input("mmHg", 60, 130, key='bp_diastolic', label_visibility="collapsed")
            with st.container(border=True):
                st.markdown("##### 🍔 Total Kolesterol")
                st.number_input("mg/dL", 100, 400, key='cholesterol_level', label_visibility="collapsed")
            with st.container(border=True):
                st.markdown("##### ✅ HDL (Baik)")
                st.number_input("mg/dL", 20, 100, key='cholesterol_hdl', label_visibility="collapsed")
        
        with c3:
            with st.container(border=True):
                st.markdown("##### 🩺 Detak Jantung")
                st.number_input("BPM", 50, 150, key='heart_rate', label_visibility="collapsed")
            with st.container(border=True):
                st.markdown("##### ❌ LDL (Jahat)")
                st.number_input("mg/dL", 50, 250, key='cholesterol_ldl', label_visibility="collapsed")
            with st.container(border=True):
                st.markdown("##### 👖 Lingkar Pinggang")
                st.number_input("cm", 50, 150, key='waist_circumference', label_visibility="collapsed")

        st.markdown("---")
        b1, b2, b3 = st.columns([1, 4, 1])
        if b1.form_submit_button("⬅️ Kembali"): prev_step(); st.rerun()
        
        submitted = b3.form_submit_button("🏥 ANALISIS")
        
        if submitted:
            ss = st.session_state
            
            # MAPPING BAHASA
            map_gender = {'Laki-laki': 'Male', 'Perempuan': 'Female'}
            map_income = {'Rendah': 'Low', 'Menengah': 'Middle', 'Tinggi': 'High'}
            map_region = {'Perkotaan': 'Urban', 'Pedesaan': 'Rural'}
            map_smoking = {'Tidak Pernah': 'Never', 'Pernah (Dulu)': 'Past', 'Perokok Aktif': 'Current'}
            map_alcohol = {'Tidak Pernah': 'Unknown', 'Sedang': 'Moderate', 'Tinggi': 'High'}
            map_diet = {'Sehat': 'Healthy', 'Tidak Sehat': 'Unhealthy'}
            map_activity = {'Rendah': 'Low', 'Sedang': 'Moderate', 'Tinggi': 'High'}
            map_poll_stress = {'Rendah': 'Low', 'Sedang': 'Moderate', 'Tinggi': 'High'}

            input_data = {
                'age': ss.age, 
                'gender': map_gender[ss.gender], 
                'region': map_region[ss.region], 
                'income_level': map_income[ss.income_level],
                'hypertension': 1 if ss.has_hypertension == "Ya" else 0,
                'diabetes': 1 if ss.has_diabetes == "Ya" else 0,
                'cholesterol_level': ss.cholesterol_level,
                'obesity': 1 if ss.has_obesity == "Ya" else 0,
                'waist_circumference': ss.waist_circumference,
                'family_history': 1 if ss.family_history == "Ya" else 0,
                'smoking_status': map_smoking[ss.smoking_status],
                'alcohol_consumption': map_alcohol[ss.alcohol_consumption],
                'physical_activity': map_activity[ss.physical_activity],
                'dietary_habits': map_diet[ss.dietary_habits],
                'air_pollution_exposure': map_poll_stress[ss.air_pollution_exposure],
                'stress_level': map_poll_stress[ss.stress_level],
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
                'heart_attack': 0
            }
            del input_data['heart_attack']
            df_input = pd.DataFrame([input_data])

            # Feature Engineering
            if ss.age <= 35: df_input['Age_Group'] = 'Young'
            elif ss.age <= 55: df_input['Age_Group'] = 'Middle'
            else: df_input['Age_Group'] = 'Senior'
            
            df_input['is_smoker'] = 1 if input_data['smoking_status'] in ['Current', 'Past'] else 0
            df_input['Total_Risk_Factors'] = (df_input['hypertension'] + df_input['diabetes'] + 
                                              df_input['obesity'] + df_input['is_smoker'])

            # Prediction
            try:
                final_input = preprocessor.transform(df_input)
                prob = model.predict(final_input)[0][0]
                st.session_state['result_prob'] = prob
                st.rerun()
            except Exception as e:
                st.error(f"System Error: {e}")

# --- POP-UP HASIL ---
if 'result_prob' in st.session_state:
    prob = st.session_state['result_prob']
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
        st.markdown(f"""
        <div style="background-color: #FFFFFF; padding: 20px; border-radius: 15px; text-align: center; border: 2px solid #0055AA; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
            <h4 style="color: #003366; margin:0;">Probabilitas</h4>
            <h1 style="color: {'#D90429' if prob > 0.5 else '#008800'}; font-size: 3.5em; margin:0;">
                {prob:.1%}
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        # --- PERBAIKAN UTAMA DI SINI ---
        # Menggunakan on_click=restart (Tanpa if statement)
        st.button("🔄 Analisis Baru", on_click=restart, use_container_width=True)

    with col_res2:
        if prob > 0.5:
            st.error("⚠️ **HASIL: RISIKO TINGGI**")
            st.write("""
            **Analisis:**
            Terdeteksi pola risiko kardiovaskular tinggi.
            
            **Rekomendasi:**
            1. Segera konsultasikan ke dokter spesialis jantung.
            2. Lakukan pemeriksaan EKG lanjutan.
            """)
        else:
            st.success("✅ **HASIL: RISIKO RENDAH**")
            st.write("""
            **Analisis:**
            Profil kesehatan aman.
            
            **Rekomendasi:**
            1. Pertahankan pola hidup sehat.
            2. Cek rutin tahunan.
            """)
