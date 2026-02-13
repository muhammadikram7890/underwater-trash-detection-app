# 🌊 Underwater Plastic Waste Detection using YOLOv8

This project focuses on detecting underwater plastic waste using **YOLOv8** and deploying the model through a **Flask-based web application**.  

It is developed as part of an undergraduate thesis research aimed at improving object detection efficiency and accuracy in underwater.

---

## 📌 Overview

Marine plastic pollution is one of the most critical environmental issues worldwide.  
This system is designed to detect underwater plastic waste using a deep learning-based object detection approach.

### 🔍 Key Components

- YOLOv8 (Object Detection)
- Efficient Multi-Scale Attention (EMA)
- Web Deployment using Flask

The system supports:
- Image upload detection
- Video upload detection
- Real-time inference (depending on implementation)

---

# 🛠️ Tech Stack

- Python 3.11+
- Flask
- Ultralytics (YOLOv8)
- PyTorch
---

# 🚀 Installation Guide

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/nama-repo.git
cd nama-repo
```

---

## 2️⃣ Upgrade pip

```bash
pip install --upgrade pip
```

---

## 3️⃣ Install Dependencies

```bash
pip install flask
pip install ultralytics
pip install torch torchvision torchaudio
```

# ▶️ Running the Flask Application

Make sure the trained model file is placed in:

```
models/best.pt
```

Then run:

```bash
python app.py
```

Open your browser and access:

```
http://127.0.0.1:5000/
```

---

# 📊 Training Notebook

The full training notebook is available at:

👉 [https://www.kaggle.com/code/ikram0703/underwater-trash-detection-c2f-ema-tvsbd](https://www.kaggle.com/code/ikram0703/underwater-trash-detection-c2f-ema-tvsbd)

---


# 📄 License

This project is developed for academic research purposes only.
