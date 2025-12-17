# 🌾 AI-Powered Farmers Disease Diagnostic Portal

## 📌 Project Overview

An intelligent web application that empowers farmers with **AI-driven disease diagnosis** for crops and livestock. This comprehensive platform combines **computer vision, machine learning, weather intelligence, and community knowledge** to provide instant disease identification, treatment recommendations, and real-time analytics.

---

## 🎯 Key Features

- 🔬 **AI Disease Recognition**  
  Upload images for instant diagnosis using deep learning models  
  - MobileNetV2 for plant diseases  
  - EfficientNet-B0 for livestock diseases  

- 🌤️ **Weather Integration**  
  Real-time weather data correlated with disease risk assessment  

- 💊 **Treatment Database**  
  Scientific treatment protocols and prevention strategies for 50+ diseases  

- 👥 **Community Knowledge Sharing**  
  Farmers can share and discover effective traditional remedies  

- 📊 **Real-time Analytics**  
  Interactive dashboards tracking disease outbreaks and trends  

- 📱 **Mobile Optimized**  
  Fully responsive design with PWA capabilities for offline use  

- 🗺️ **Location-based Insights**  
  Geographic disease tracking and risk alerts  

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit (Python web framework)  
- **ML Framework:** PyTorch 2.7.0  
- **Computer Vision:** OpenCV, Pillow  
- **Data Visualization:** Plotly, Matplotlib, Seaborn  
- **Data Processing:** Pandas, NumPy  
- **API Integration:** OpenWeather API  
- **Models:** MobileNetV2 (Plant), EfficientNet-B0 (Livestock)  

---

## 📂 Project Structure

```text
farmers-disease-portal/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (API keys)
├── .gitignore                      # Git ignore file
│
├── models/                         # ML models and training scripts
│   ├── pretrained/                # Pre-trained model files (.pth, .pkl)
│   └── model_trainer.py            # Model training script
│
├── data/                           # Data storage
│   ├── datasets/                  # Training datasets (not in git due to size)
│   │   ├── plantvillage/           # Plant disease images (50K+ images)
│   │   └── livestock/              # Livestock disease images
│   ├── disease_submissions.json    # User submissions database
│   ├── community_remedies.json     # Community remedies database
│   └── analytics_data.json         # Analytics and metrics
│
├── knowledge_base/                 # Rule-based reasoning system
│   ├── __init__.py
│   └── disease_rules.py            # Disease diagnosis rules and treatments
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── database.py                 # JSON database management
│   ├── analytics.py                # Analytics and visualization
│   ├── image_processor.py          # Image preprocessing utilities
│   └── augment_blackleg.py         # Data augmentation script
│
├── assets/                         # Static assets
│   ├── logo.png                    # Application logo
│   └── sample_images/              # Sample disease images for testing
│
└── setup scripts/                  # Setup and automation scripts
    ├── setup_datasets.py           # Dataset download and setup
    └── complete_project_setup.py   # Full project initialization

📊 Datasets
The project uses real agricultural disease datasets:

PlantVillage Dataset: 50,000+ images covering 38 plant disease classes
Plant Pathology 2020: Competition dataset for apple disease detection
Custom Livestock Dataset: Curated images for cattle, poultry diseases

⚠️ Important: Due to size constraints, datasets are NOT included in this repository. Use the setup scripts to download them, or the application will work with the existing pre-trained models.

🧠 AI Models
Plant Disease Model

Architecture: MobileNetV2 (lightweight, mobile-optimized)
Input Size: 224x224x3
Classes: 38 plant diseases
Accuracy: ~89.2%
Inference Time: <2 seconds

Livestock Disease Model

Architecture: EfficientNet-B0
Input Size: 224x224x3
Classes: 5 livestock conditions
Accuracy: ~85.7%
Inference Time: <2 seconds

🎨 Features Breakdown
1. Disease Diagnosis

Upload crop/animal images
AI-powered disease identification
Confidence scoring (85-95% typical)
Weather-context enhanced predictions
Multi-disease differential diagnosis

2. Treatment Recommendations

Scientific treatment protocols
Prevention strategies
Severity assessment
Follow-up guidance
Community-validated remedies

3. Analytics Dashboard

Real-time disease tracking
Geographic outbreak visualization
Severity distribution charts
Temporal trend analysis
Risk assessment alerts

4. Community Features

Share successful treatments
Rate remedy effectiveness
Location-based insights
Expert verification system

Model Configuration
Models are loaded from models/pretrained/. To use custom models:

Train using models/model_trainer.py
Place .pth files in models/pretrained/
Update model metadata JSON files

📱 Mobile Usage
The application is fully responsive and works on:

✅ Desktop browsers (Chrome, Firefox, Safari, Edge)
✅ Mobile browsers (iOS Safari, Android Chrome)
✅ Progressive Web App (PWA) - installable on mobile
✅ Works on 4G/5G networks

⭐ If this project helped you, please consider giving it a star!
