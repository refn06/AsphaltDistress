import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Asphalt Pavement Distress Detection",
    page_icon="🛣️",
    layout="centered"
)

# Kamus terjemahan untuk Dosen Sipil/Arsitektur
class_mapping = {
    "D00": "Longitudinal Crack (Retak Memanjang)",
    "D10": "Transverse Crack (Retak Melintang)",
    "D20": "Alligator Crack (Retak Buaya)",
    "D40": "Pothole (Lubang/Ambles)",
    "Repair": "Road Repair (Tambalan Jalan)"
}

# Fungsi untuk load model dengan cache agar tidak berat
@st.cache_resource
def load_model():
    # Pastikan file best.pt sudah ada di folder model/
    return YOLO("model/best.pt")

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'model/best.pt' sudah diupload ke GitHub. Error: {e}")

# Header
st.title("🛣️ Asphalt Pavement Distress Detection")
st.write("Sistem berbasis YOLOv9c untuk mendeteksi kerusakan jalan secara otomatis.")
st.divider()

# Sidebar untuk pengaturan
st.sidebar.header("Pengaturan Deteksi")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)
st.sidebar.info("Gunakan threshold ~0.45 untuk hasil yang paling seimbang berdasarkan grafik F1-Curve.")

# Upload File
uploaded_file = st.file_uploader(
    "Upload foto kerusakan jalan (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Buka gambar
    image = Image.open(uploaded_file)
    
    # Layout kolom untuk sebelum/sesudah
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

    # Menampilkan Gambar Hasil Deteksi
    with col2:
        st.subheader("Detection Result")
        annotated_img = results[0].plot()
        # Konversi BGR (OpenCV style) ke RGB untuk Streamlit
        st.image(annotated_img, use_container_width=True)

    st.divider()

    # Ringkasan Statistik
    st.subheader("📊 Ringkasan Kerusakan Terdeteksi")
    
    boxes = results[0].boxes
    names = model.names

    if boxes is not None and len(boxes) > 0:
        # Menghitung jumlah per kelas
        counts = {}
        for cls in boxes.cls.tolist():
            label = names[int(cls)]
            counts[label] = counts.get(label, 0) + 1

        # Tampilkan ringkasan yang ramah pengguna
        for label, count in counts.items():
            label_display = class_mapping.get(label, label)
            st.write(f"- **{label_display}**: {count} titik")
        
        st.success(f"**Total keseluruhan objek terdeteksi: {len(boxes)}**")
    else:
        st.warning("Tidak ada kerusakan yang terdeteksi dengan threshold ini. Coba turunkan 'Confidence Threshold' di sidebar.")

    # Hapus file sementara
    os.remove(temp_path)

# Footer
st.caption("Developed for Pavement Inspection Workshop | YOLOv9c State-of-the-Art Model")
