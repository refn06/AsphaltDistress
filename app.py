import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Asphalt Pavement Distress Detection",
    layout="centered"
)

st.title("🛣️ Asphalt Pavement Distress Detection")
st.write("YOLO-based model for detecting pavement damage")

@st.cache_resource
def load_model():
    return YOLO("model/best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload pavement image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    results = model(temp_path)

    # ===== TAMPILKAN GAMBAR HASIL =====
    annotated_img = results[0].plot()
    st.image(annotated_img, caption="Detection Result", use_column_width=True)

    # ===== 🔽 RINGKASAN DETEKSI =====
    st.subheader("📊 Ringkasan Kerusakan Terdeteksi")

    boxes = results[0].boxes
    names = model.names

    if boxes is not None and len(boxes) > 0:
        detected_classes = {}

        for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            label = names[int(cls)]
            conf = float(conf)

            if label not in detected_classes or conf > detected_classes[label]:
                detected_classes[label] = conf

        for label, conf in detected_classes.items():
            st.write(f"- **{label}** ({conf:.2f})")

        st.write(f"\n**Total objek terdeteksi: {len(boxes)}**")
    else:
        st.write("Tidak ada kerusakan terdeteksi.")

    os.remove(temp_path)
