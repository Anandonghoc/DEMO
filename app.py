import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown
import json

# ============ Cấu hình trang ============ #
st.set_page_config(
    page_title="🍅 Phân loại bệnh cây",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============ CSS giao diện ============ #
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Quicksand', sans-serif;
            background-color: #e6f4ea;
        }

        .main-title {
            font-size: 58px;
            color: #40916c;
            text-align: center;
            margin-top: 20px;
            font-weight: 600;
        }

        .subtitle {
            font-size: 20px;
            color: #6c757d;
            text-align: center;
            margin-bottom: 40px;
        }

        .result-box {
            background-color: #d8f3dc;
            color: #1b4332;
            padding: 25px;
            border-radius: 15px;
            font-size: 22px;
            text-align: center;
            margin-top: 10px;
            border: 1px solid #b7e4c7;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }

        .footer {
            text-align: center;
            font-size: 18px;
            color: #6c757d;
            margin-top: 60px;
        }

        .stButton > button {
            background-color: #74c69d;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            font-weight: bold;
        }

        .stFileUploader {
            background-color: #f0fdf4;
            border: 1px solid #caffbf;
            padding: 15px;
            border-radius: 10px;
        }

        .stSidebar {
            background-color: #f1faee;
            width: 400px !important;
            padding: 10px;
        }

        .stSidebar img {
            max-width: 150px;
            margin: 0 auto;
            display: block;
        }

        .language-box {
            position: absolute;
            top: 15px;
            right: 30px;
            z-index: 100;
            background-color: #ffffffcc;
            padding: 10px 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ============ Tải mô hình ============ #
MODEL_PATH = "plant_disease_model.keras"
CLASS_INDEX_PATH = "class_names.json"
FILE_ID = "1qK6cnyVpIwYfuzhC-qtCdisJPmyfaXgs"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("⏳ Đang tải mô hình phân loại..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

classifier_model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# ============ Tải dữ liệu bệnh ============ #
with open("disease_info.json", "r", encoding="utf-8") as f:
    disease_info = json.load(f)

# ============ Header chính ============ #
language = st.radio("🌐 Chọn ngôn ngữ / Select language", ("vi", "en"), horizontal=True)
st.markdown("<div class='main-title'>🌱 Kiểm tra tình trạng lá</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Tải nhiều ảnh và nhận diện bệnh với độ chính xác cao</div>", unsafe_allow_html=True)

# ============ Sidebar ============ #
with st.sidebar:
    st.image("logo2.png", use_container_width=True)
    uploaded_files = st.file_uploader(" Tải lên ảnh lá cây (có thể chọn nhiều)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    st.markdown("---")
    st.markdown("**Mô hình:** MobileNetV3Large")
    st.markdown("Phát hiện bệnh lá giúp bảo vệ năng suất cây trồng, tiết kiệm chi phí, giảm lạm dụng thuốc hóa học và hiện đại hóa nông nghiệp.")

# ============ Dự đoán từng ảnh ============ #
if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        prediction = classifier_model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        predicted_class = index_to_class[predicted_index]
        confidence = float(np.max(prediction)) * 100

        info = disease_info.get(predicted_class, {})
        if language == "vi":
            name = info.get("vi_name", "Không rõ")
            symptoms = info.get("symptoms_vi", "-")
            treatment = info.get("treatment_vi", "-")
            note = info.get("note_vi", "-")
        else:
            name = info.get("en_name", "Unknown")
            symptoms = info.get("symptoms_en", "-")
            treatment = info.get("treatment_en", "-")
            note = info.get("note_en", "-")

        # Hiển thị ảnh và kết quả dự đoán bên cạnh nhau
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img.resize((300, 300)), caption=f"🖼️ Ảnh: {uploaded_file.name}", use_container_width=False)

        with col2:
            st.markdown(f"""
            <div class="result-box">
                Kết quả dự đoán: <strong>{predicted_class}</strong>
            </div>
            <div class="disease-card">
                <h4 style='color:#dc3545;'>📌 {name}</h4>
                <p><strong>{'Triệu chứng' if language == 'vi' else 'Symptoms'}:</strong> {symptoms}</p>
                <p><strong>{'Biện pháp xử lý' if language == 'vi' else 'Treatment'}:</strong> {treatment}</p>
                <p><strong>{'Ghi chú' if language == 'vi' else 'Note'}:</strong> {note}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("📤 Vui lòng tải lên ít nhất một ảnh để bắt đầu.")

# ============ Footer ============ #
st.markdown("<div class='footer'>🌿 Ứng dụng AI hỗ trợ nông dân chẩn đoán bệnh ở cây nhanh chóng và chính xác thông qua tình trạng của lá.</div>", unsafe_allow_html=True)
