import streamlit as st
import cv2
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# 1. Модельдерді жүктеу (Cache арқылы жылдамдату)
@st.cache_resource
def load_models():
    # Объектілерді табу үшін YOLOv8
    det_model = YOLO('yolov8n.pt') 
    # Сипаттама бойынша іздеу үшін CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return det_model, clip_model, clip_processor

det_model, clip_model, clip_processor = load_models()

st.title("🕵️ FindMe AI - GovTech Solution")
st.markdown("### Decentrathon 5.0: AI for Government (Case 3)")

# UI: Сипаттама енгізу
query = st.text_input("Адамның сипаттамасын енгізіңіз (мысалы: 'person in red jacket'):", "person in red clothes")

# UI: Файл жүктеу
uploaded_file = st.file_uploader("CCTV суретін немесе видео кадрын жүктеңіз", type=['jpg', 'jpeg', 'png'])

if uploaded_file and query:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.image(image, caption="Бастапқы кескін", use_container_width=True)
    
    with st.spinner('Жүйе іздеуде...'):
        # 2. YOLO арқылы адамдарды табу (Detection)
        results = det_model(img_array, classes=[0]) # class 0 - адамдар
        
        crops = []
        bboxes = []
        for res in results:
            for box in res.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                crop = img_array[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(Image.fromarray(crop))
                    bboxes.append([x1, y1, x2, y2])

        if not crops:
            st.warning("Кадрдан ешқандай адам табылмады.")
        else:
            # 3. CLIP арқылы сипаттамамен салыстыру (Explainability)
            inputs = clip_processor(text=[query], images=crops, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            
            # Сәйкестік пайызы (Logits)
            logits_per_image = outputs.logits_per_image 
            probs = logits_per_image.softmax(dim=0).detach().numpy()

            # Нәтижелерді шығару
            st.subheader("Табылған нәтижелер (Score бойынша):")
            
            cols = st.columns(3)
            for i, prob in enumerate(probs):
                score = float(prob[0])
                if score > 0.1: # Табалдырық (Threshold)
                    with cols[i % 3]:
                        st.image(crops[i], caption=f"Сәйкестік: {score:.2%}")
                        # Түсіндірме (Explainability талабы үшін) [cite: 98, 116]
                        st.write(f"Локация: {bboxes[i]}")
                        if score > 0.5:
                            st.success("Жоғары ұқсастық!")

st.sidebar.info("Бұл жүйе мемлекеттік органдарға (ІІМ) қоғамдық орындарда жоғалған адамдарды жедел табуға көмектеседі. [cite: 12, 114]")
