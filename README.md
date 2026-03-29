# FindMe - AI Missing Person Locator (GovTech)

### Жоба сипаттамасы
[cite_start]**FindMe** — Decentrathon 5.0 хакатонының "AI for Government" трегі аясында әзірленген прототип. [cite: 3] [cite_start]Жүйе CCTV камераларынан алынған деректер негізінде табиғи тілдегі сипаттама арқылы (Natural Language Processing + Computer Vision) адамдарды іздеуге арналған. [cite: 93, 97]

### Негізгі функционал:
- **Object Detection:** YOLOv8 арқылы адамдарды нақты уақытта анықтау.
- **Zero-shot Matching:** CLIP моделі арқылы алдын ала үйретусіз кез келген сипаттамамен (түс, киім, аксессуар) іздеу.
- [cite_start]**Explainability:** Әрбір табылған адам үшін сәйкестік пайызы мен локация деректерін ұсыну. [cite: 117]

### Мемлекетке пайдасы:
1. [cite_start]**ІІМ мен ТЖД:** Жоғалған балалар мен қылмыскерлерді іздеу уақытын 10 есеге дейін қысқарту. [cite: 11, 20]
2. [cite_start]**Ауқымдылық:** Қалалық "Сергек" жүйесімен интеграциялану мүмкіндігі. [cite: 128]

### Технологиялық стек:
- Python, Streamlit, PyTorch
- YOLOv8 (Ultralytics)
- CLIP (OpenAI / HuggingFace)

### Іске қосу:
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
