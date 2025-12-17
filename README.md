# 🌾 AI‑Powered Farmers Disease Diagnostic Portal

An intelligent web application that empowers farmers with **AI‑driven disease diagnosis** for crops and livestock. This platform combines **computer vision, machine learning, weather intelligence, and community knowledge** to provide instant disease identification, treatment recommendations, and real‑time analytics.

---

## 📌 Project Overview

Farmers often lack immediate access to agricultural experts when crops or livestock show disease symptoms. This project bridges that gap using lightweight, production‑ready AI models optimized for real‑world deployment. Farmers can upload images and receive **fast, accurate, and context‑aware disease insights** directly on web or mobile devices.

---

## 🎯 Key Features

### 🔬 AI Disease Recognition

* Upload crop or livestock images for instant diagnosis
* Deep‑learning powered predictions
* Models used:

  * **MobileNetV2** – Plant disease detection
  * **EfficientNet‑B0** – Livestock disease detection

### 🌤️ Weather Integration

* Real‑time weather data via OpenWeather API
* Weather‑based disease risk correlation
* Improves prediction reliability and alerts

### 💊 Treatment Database

* Scientific treatment protocols
* Preventive care recommendations
* Coverage for **50+ agricultural diseases**

### 👥 Community Knowledge Sharing

* Farmers share traditional and field‑tested remedies
* Rate and validate remedy effectiveness
* Expert verification workflow

### 📊 Real‑time Analytics

* Interactive dashboards
* Disease outbreak tracking
* Trend and severity analysis

### 📱 Mobile Optimized (PWA)

* Fully responsive UI
* Progressive Web App (PWA) support
* Works offline in low‑connectivity areas

### 🗺️ Location‑based Insights

* Geographic disease tracking
* Region‑specific risk alerts

---

## 🛠️ Technology Stack

### Frontend

* **Streamlit** (Python web framework)

### Machine Learning

* **PyTorch 2.7.0**
* **MobileNetV2** (Plant diseases)
* **EfficientNet‑B0** (Livestock diseases)

### Computer Vision

* OpenCV
* Pillow

### Data & Analytics

* Pandas
* NumPy
* Plotly
* Matplotlib
* Seaborn

### APIs

* OpenWeather API

---

## 📂 Project Structure

```text
farmers-disease-portal/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (API keys)
├── .gitignore                      # Git ignore rules
│
├── models/                         # ML models and training scripts
│   ├── pretrained/                # Pre-trained model files (.pth, .pkl)
│   └── model_trainer.py            # Model training pipeline
│
├── data/                           # Data storage
│   ├── datasets/                  # Training datasets (excluded from git)
│   │   ├── plantvillage/           # Plant disease images (50K+ images)
│   │   └── livestock/              # Livestock disease images
│   ├── disease_submissions.json    # User submissions
│   ├── community_remedies.json     # Community remedies
│   └── analytics_data.json         # Analytics metrics
│
├── knowledge_base/                 # Rule-based reasoning system
│   ├── __init__.py
│   └── disease_rules.py            # Diagnosis rules & treatments
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── database.py                 # JSON database handler
│   ├── analytics.py                # Analytics & visualization
│   ├── image_processor.py          # Image preprocessing
│   └── augment_blackleg.py         # Data augmentation
│
├── assets/                         # Static assets
│   ├── logo.png
│   └── sample_images/              # Sample images
│
└── setup_scripts/                  # Automation scripts
    ├── setup_datasets.py           # Dataset setup
    └── complete_project_setup.py   # Full initialization
```

---

## 📊 Datasets

This project uses real agricultural disease datasets:

* **PlantVillage Dataset** – 50,000+ images across 38 plant disease classes
* **Plant Pathology 2020** – Apple disease detection dataset
* **Custom Livestock Dataset** – Curated cattle and poultry disease images

> ⚠️ **Important:** Datasets are **not included** due to size constraints. Use setup scripts or rely on pre‑trained models.

---

## 🧠 AI Models

### 🌱 Plant Disease Model

* Architecture: **MobileNetV2**
* Input Size: `224 × 224 × 3`
* Classes: 38 plant diseases
* Accuracy: ~**89.2%**
* Inference Time: < **2 seconds**

### 🐄 Livestock Disease Model

* Architecture: **EfficientNet‑B0**
* Input Size: `224 × 224 × 3`
* Classes: 5 livestock conditions
* Accuracy: ~**85.7%**
* Inference Time: < **2 seconds**

---

## 🎨 Feature Breakdown

### 1️⃣ Disease Diagnosis

* Image upload for crops & animals
* AI‑powered classification
* Confidence scoring (85–95% typical)
* Weather‑enhanced predictions
* Differential diagnosis support

### 2️⃣ Treatment Recommendations

* Scientific treatment protocols
* Preventive strategies
* Severity assessment
* Follow‑up guidance
* Community‑validated remedies

### 3️⃣ Analytics Dashboard

* Real‑time disease tracking
* Geographic outbreak maps
* Severity distribution charts
* Temporal trend analysis
* Risk alerts

### 4️⃣ Community Features

* Share successful treatments
* Rate remedies
* Location‑based insights
* Expert verification system

---

## ⚙️ Model Configuration

* Pre‑trained models are loaded from `models/pretrained/`
* To use custom models:

  1. Train using `models/model_trainer.py`
  2. Save `.pth` files in `models/pretrained/`
  3. Update model metadata JSON files

---

## 📱 Mobile & Device Support

The application works seamlessly on:

* ✅ Desktop browsers (Chrome, Firefox, Edge, Safari)
* ✅ Mobile browsers (Android & iOS)
* ✅ Progressive Web App (PWA)
* ✅ 4G / 5G and low‑bandwidth networks

---

## ⭐ Support the Project

If this project helped you or inspired your work, please consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is released under the **MIT License**.
