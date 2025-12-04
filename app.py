import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Prediksi Berat Ikan Lele",
    page_icon="🐟",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_estimasi_lele.pkl')
        return model
    except:
        return None

model = load_model()

# Title and description
st.title("🐟 Sistem Prediksi Berat Ikan Lele")
st.markdown("**Aplikasi Machine Learning untuk Estimasi Berat Ikan Berdasarkan Parameter Lingkungan**")
st.divider()

# Sidebar for navigation
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman:", ["Prediksi", "Visualisasi Data", "Tentang Model"])

if page == "Prediksi":
    st.header("📊 Input Parameter untuk Prediksi Berat Ikan")
    
    if model is None:
        st.error("⚠️ Model belum tersedia. Silakan jalankan training.ipynb terlebih dahulu untuk membuat model.")
    else:
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameter Lingkungan")
            temperature = st.slider("Suhu (°C)", 0.0, 30.0, 24.0, 0.1)
            turbidity = st.slider("Kekeruhan (NTU)", -15.0, 100.0, 20.0, 1.0)
            dissolved_oxygen = st.slider("Oksigen Terlarut (mg/L)", 0.0, 17.0, 6.0, 0.1)
            ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1)
        
        with col2:
            st.subheader("Parameter Kimia & Morfometrik")
            ammonia = st.slider("Amonia (mg/L)", 0.0, 5.0, 0.5, 0.1)
            nitrate = st.slider("Nitrat (mg/L)", 0.0, 181.0, 10.0, 1.0)
            length = st.slider("Panjang Ikan (cm)", 0.0, 50.0, 20.0, 0.5)
            population = st.number_input("Populasi Kolam", 0, 50, 50, 10)
        
        # Prediction button
        if st.button("🎯 Prediksi Berat Ikan", type="primary"):
            # Prepare input data
            input_data = pd.DataFrame({
                'TEMPERATURE': [temperature],
                'TURBIDITY': [turbidity],
                'DISOLVED OXYGEN': [dissolved_oxygen],
                'pH': [ph],
                'AMMONIA': [ammonia],
                'NITRATE': [nitrate],
                'Length': [length],
                'Population': [population]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success(f"### Prediksi Berat Ikan: **{prediction:.2f} gram**")
            
            # Additional info
            st.info(f"""
            **Parameter Input:**
            - Suhu: {temperature}°C
            - Kekeruhan: {turbidity} NTU
            - Oksigen Terlarut: {dissolved_oxygen} mg/L
            - pH: {ph}
            - Amonia: {ammonia} mg/L
            - Nitrat: {nitrate} mg/L
            - Panjang: {length} cm
            - Populasi: {population}
            """)

elif page == "Visualisasi Data":
    st.header("📈 Visualisasi Data Training")
    
    try:
        # Load dataset
        df = pd.read_csv('dataset/IoTPond10.csv')
        
        # Clean data
        df_clean = df.replace([-127, -97, np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()
        df_clean = df[
            (df['TEMPERATURE'] > 0) &      
            (df['pH'] > 0) &               
            (df['pH'] < 14) &              
            (df['AMMONIA'] >= 0) &         
            (df['AMMONIA'] < 5.0)          
        ].copy()
        
        st.write(f"📊 Jumlah data: **{len(df_clean)} baris**")
        
        # Show data preview
        st.subheader("Preview Data")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        # Statistics
        st.subheader("Statistik Deskriptif")
        st.dataframe(df_clean.describe(), use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Matriks Korelasi")
        fig, ax = plt.subplots(figsize=(12, 10))
        cols_corr = ['TEMPERATURE','TURBIDITY','DISOLVED OXYGEN','pH','AMMONIA','NITRATE','Length','Weight']
        sns.heatmap(df_clean[cols_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, square=True, cbar_kws={'shrink': 0.8})
        ax.set_title('Matriks Korelasi: Faktor Penentu Berat Ikan', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Scatter plot
        st.subheader("Pola Morfometrik: Panjang vs Berat")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_clean, x='Length', y='Weight', color='blue', alpha=0.6, ax=ax, s=50)
        ax.set_title('Pola Morfometrik: Panjang vs Berat Ikan', fontsize=14, fontweight='bold')
        ax.set_xlabel('Panjang (cm)', fontsize=12)
        ax.set_ylabel('Berat (gram)', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")

elif page == "Tentang Model":
    st.header("ℹ️ Tentang Model")
    
    st.markdown("""
    ### Model Linear Regression
    
    Model ini menggunakan algoritma **Linear Regression** untuk memprediksi berat ikan lele berdasarkan 8 parameter:
    
    1. **TEMPERATURE** - Suhu air kolam (°C)
    2. **TURBIDITY** - Tingkat kekeruhan air (NTU)
    3. **DISOLVED OXYGEN** - Kadar oksigen terlarut (mg/L)
    4. **pH** - Tingkat keasaman air
    5. **AMMONIA** - Kadar amonia (mg/L)
    6. **NITRATE** - Kadar nitrat (mg/L)
    7. **Length** - Panjang ikan (cm)
    8. **Population** - Jumlah populasi dalam kolam
    
    ### Cara Menggunakan Aplikasi
    
    1. **Halaman Prediksi**: Masukkan parameter lingkungan dan morfometrik ikan, kemudian klik tombol prediksi
    2. **Halaman Visualisasi**: Lihat statistik dan visualisasi dari data training
    3. **Halaman Tentang Model**: Informasi tentang model (halaman ini)
    
    ### Teknologi yang Digunakan
    
    - **Python** - Bahasa pemrograman
    - **Scikit-learn** - Library machine learning
    - **Streamlit** - Framework web app
    - **Pandas & NumPy** - Data processing
    - **Matplotlib & Seaborn** - Visualisasi data
    
    ### Catatan
    
    Model ini dilatih menggunakan dataset **IoTPond10.csv** dan menggunakan split 80:20 untuk training dan testing.
    """)
    
    if model is not None:
        st.success("✅ Model berhasil dimuat dan siap digunakan!")
    else:
        st.warning("⚠️ Model belum tersedia. Jalankan notebook training terlebih dahulu.")

# Footer
st.divider()
st.markdown("---")
st.markdown("**🐟 Fish Weight Prediction System** | Powered by Machine Learning")
