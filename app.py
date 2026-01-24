import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import urllib.request
import pandas as pd
import datetime

# 1. Konfigurasi halaman
st.set_page_config(
    page_title="Asphalt Pavement Distress Detection",
    page_icon="🛣️",
    layout="centered"
)

# 2. Kamus terjemahan dan penjelasan teknis
class_mapping = {
    "D00": "Longitudinal Crack (Retak Memanjang)",
    "D10": "Transverse Crack (Retak Melintang)",
    "D20": "Alligator Crack (Retak Buaya)",
    "D40": "Pothole (Lubang/Ambles)",
    "Repair": "Road Repair (Tambalan Jalan)"
}

# 3. Fungsi untuk load model dengan auto-download dari GitHub Release
@st.cache_resource
def load_model():
    model_path = "model/best.1.pt"
    if not os.path.exists("model"):
        os.makedirs("model")
    
    if not os.path.exists(model_path):
        with st.spinner("Sedang mengunduh model YOLOv9c dari GitHub Release... Mohon tunggu sebentar."):
            url = "https://github.com/refn06/AsphaltDistress/releases/download/v1.0/best.1.pt"
            urllib.request.urlretrieve(url, model_path)
            
    return YOLO(model_path)

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")

# 4. Sidebar: Pengaturan dan Kamus Kerusakan
st.sidebar.header("⚙️ Pengaturan Deteksi")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)
st.sidebar.info("Gunakan threshold ~0.45 untuk hasil yang paling seimbang berdasarkan grafik F1-Curve.")

st.sidebar.divider()
st.sidebar.subheader("ℹ️ Kamus Jenis Kerusakan")
with st.sidebar.expander("Lihat Penjelasan"):
    st.markdown("""
    * **D00 (Longitudinal - Retak memanjang):** Retak memanjang searah jalur jalan. Biasanya akibat beban kendaraan atau sambungan aspal.
    * **D10 (Transverse - Retak melintang):** Retak melintang tegak lurus jalur jalan. Akibat penyusutan aspal karena perubahan suhu.
    * **D20 (Alligator - Retak buaya/berpola):** Retak berpola kulit buaya. Menandakan kerusakan struktural pada fondasi jalan.
    * **D40 (Pothole - Lubang):** Lubang/ambles. Tahap kerusakan lanjut yang berbahaya bagi kendaraan.
    * **Repair:** Area tambalan. Menandakan lokasi yang sudah pernah diperbaiki sebelumnya.
    """)

# 5. Konten Utama (Header)
st.title("🛣️ Asphalt Pavement Distress Detection")
st.write("Sistem berbasis YOLOv9c untuk mendeteksi kerusakan jalan secara otomatis.")
st.divider()

# 6. Upload File
uploaded_file = st.file_uploader(
    "Upload foto kerusakan jalan (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Simpan sementara untuk diproses YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Jalankan Prediksi
    with st.spinner('Menganalisis gambar dengan YOLOv9c...'):
        results = model.predict(temp_path, conf=conf_threshold)

    with col2:
        st.subheader("Detection Result")
        annotated_img = results[0].plot()
        st.image(annotated_img, use_container_width=True)

    st.divider()

    # 7. Detail Analisis dan Ringkasan Statistik
    st.subheader("📋 Laporan Hasil Analisis")
    
    # Menampilkan Waktu Analisis
    waktu_sekarang = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    st.info(f"Analisis diselesaikan pada: **{waktu_sekarang} WIB**")
    
    boxes = results[0].boxes
    names = model.names

    if boxes is not None and len(boxes) > 0:
        counts = {}
        conf_scores = boxes.conf.tolist()
        
        for cls in boxes.cls.tolist():
            label = names[int(cls)]
            counts[label] = counts.get(label, 0) + 1

        # Kolom Ringkasan
        st.write("### 📊 Ringkasan Objek")
        data_for_df = []
        for label, count in counts.items():
            label_display = class_mapping.get(label, label)
            st.write(f"- **{label_display}**: {count} titik")
            data_for_df.append({"Kode": label, "Jenis Kerusakan": label_display, "Jumlah": count})
        
        st.success(f"**Total keseluruhan objek terdeteksi: {len(boxes)}**")
        
        # Grafik Confidence Score
        st.write("### 📈 Confidence Level per Objek")
        df_conf = pd.DataFrame({
            "Objek ke-": range(1, len(conf_scores) + 1),
            "Confidence Score": conf_scores
        })
        st.bar_chart(df_conf.set_index("Objek ke-"))
        st.caption("Grafik menunjukkan tingkat kepastian model (0.0 - 1.0) untuk setiap deteksi.")
        
        # Fitur Download CSV
        df_report = pd.DataFrame(data_for_df)
        csv = df_report.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Laporan Deteksi (CSV)",
            data=csv,
            file_name=f'laporan_jalan_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
        )
    else:
        st.warning("Tidak ada kerusakan yang terdeteksi dengan threshold ini.")
        st.info("""
        💡 **Info Teknis:**
        * Coba turunkan **Confidence Threshold** di sidebar jika kerusakan terlihat namun tidak terdeteksi.
        * Pastikan permukaan aspal tidak tertutup objek lain (seperti kendaraan atau bayangan yang sangat pekat).
        """)

    # Hapus file sementara
    os.remove(temp_path)

# Footer
st.divider()
st.caption("Developed for Pavement Inspection Workshop | MK Praktikum Unggulan - Universitas Gunadarma")
