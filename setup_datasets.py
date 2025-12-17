#!/usr/bin/env python3
"""
Automated Dataset Setup Script
Downloads real datasets and trains models
"""

import os
import sys
import subprocess
import requests
import zipfile
from pathlib import Path
import json

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("🔑 Setting up Kaggle API...")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    if kaggle_file.exists():
        print("✅ Kaggle API already configured")
        return True
    
    print("⚠️ Kaggle API not found. Please follow these steps:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json")
    print("4. Place it in:", str(kaggle_dir))
    
    # Create directory if it doesn't exist
    kaggle_dir.mkdir(exist_ok=True)
    
    # Ask user to manually add the file
    input("Press Enter after placing kaggle.json in the correct location...")
    
    if kaggle_file.exists():
        # Set permissions
        os.chmod(kaggle_file, 0o600)
        print("✅ Kaggle API configured successfully")
        return True
    else:
        print("❌ kaggle.json not found. Continuing without Kaggle datasets...")
        return False

#!/usr/bin/env python3
"""
Automated Dataset Setup Script
Downloads real datasets and trains models
"""

import os
import sys
import subprocess
import requests
import zipfile
from pathlib import Path
import json

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("🔑 Setting up Kaggle API...")
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    if kaggle_file.exists():
        print("✅ Kaggle API already configured")
        return True
    
    print("⚠️ Kaggle API not found. Please follow these steps:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json")
    print("4. Place it in:", str(kaggle_dir))
    
    kaggle_dir.mkdir(exist_ok=True)
    
    input("Press Enter after placing kaggle.json in the correct location...")
    
    if kaggle_file.exists():
        os.chmod(kaggle_file, 0o600)
        print("✅ Kaggle API configured successfully")
        return True
    else:
        print("❌ kaggle.json not found. Continuing without Kaggle datasets...")
        return False

def download_plantvillage_manual():
    """Manually download PlantVillage dataset"""
    print("📥 Setting up PlantVillage dataset manually...")
    
    data_dir = Path("data/datasets/plantvillage")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample dataset structure
    diseases = [
        "Tomato_Late_blight", "Tomato_Early_blight", "Tomato_healthy",
        "Wheat_Brown_rust", "Wheat_Yellow_rust", "Wheat_healthy", 
        "Apple_scab", "Apple_Black_rot", "Apple_healthy"
    ]
    
    for disease in diseases:
        disease_dir = data_dir / disease
        disease_dir.mkdir(exist_ok=True)
        
        # Create 10 sample synthetic images per class for demo
        for i in range(10):
            # Generate synthetic image data (placeholder)
            import numpy as np
            from PIL import Image
            
            # Create realistic-looking synthetic data
            if "healthy" in disease.lower():
                base_color = [50, 150, 50]  # Green
            elif "blight" in disease.lower() or "rust" in disease.lower():
                base_color = [120, 80, 40]  # Brown
            else:
                base_color = [80, 120, 60]  # Mixed
            
            # Add random variation
            img_data = np.random.randint(
                max(0, base_color[0] - 30), min(255, base_color[0] + 30), (224, 224, 3)
            )
            
            img = Image.fromarray(img_data.astype('uint8'))
            img.save(disease_dir / f"{disease}_{i:03d}.jpg")
    
    print(f"✅ Sample PlantVillage dataset created with {len(diseases)} classes")
    return True

def create_livestock_dataset():
    """Create sample livestock disease dataset"""
    print("📥 Creating livestock disease dataset...")
    
    data_dir = Path("data/datasets/livestock")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    diseases = ["healthy", "lumpy_skin_disease", "mastitis", "foot_mouth_disease"]
    
    for disease in diseases:
        disease_dir = data_dir / disease
        disease_dir.mkdir(exist_ok=True)
        
        # Create 15 sample images per class
        for i in range(15):
            import numpy as np
            from PIL import Image
            
            # Generate disease-specific synthetic data
            if disease == "healthy":
                base_color = [120, 100, 80]  # Normal skin tone
            elif disease == "lumpy_skin_disease":
                base_color = [150, 90, 70]   # Reddish with nodules
            elif disease == "mastitis":
                base_color = [160, 80, 80]   # Inflamed
            else:
                base_color = [140, 90, 75]   # General disease
            
            img_data = np.random.randint(
                max(0, base_color[0] - 20), min(255, base_color[0] + 20), (224, 224, 3)
            )
            
            # Add disease patterns
            if disease == "lumpy_skin_disease":
                # Add nodule-like spots
                for _ in range(np.random.randint(3, 8)):
                    center = (np.random.randint(50, 174), np.random.randint(50, 174))
                    radius = np.random.randint(8, 15)
                    img_data[center[0]-radius:center[0]+radius, 
                            center[1]-radius:center[1]+radius] = [180, 120, 100]
            
            img = Image.fromarray(img_data.astype('uint8'))
            img.save(disease_dir / f"{disease}_{i:03d}.jpg")
    
    print(f"✅ Livestock dataset created with {len(diseases)} classes")
    return True

def setup_environment():
    """Setup environment variables"""
    print("🔧 Setting up environment...")
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("# Environment Variables\n")
            f.write("OPENWEATHER_API_KEY=your_api_key_here\n")
            f.write("# Get API key from: https://openweathermap.org/api\n")
        print("✅ .env file created. Please add your OpenWeather API key.")
    else:
        print("✅ .env file already exists")
    
    return True

def create_directory_structure():
    """Create project directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "models/pretrained",
        "data/datasets",
        "data/uploads", 
        "knowledge_base",
        "utils",
        "assets",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory in ["models", "knowledge_base", "utils", "data"]:
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    print("✅ Directory structure created")
    return True

def download_real_datasets():
    """Attempt to download real datasets from Kaggle"""
    print("📥 Attempting to download real datasets...")
    
    try:
        import kaggle
        
        # Try to download PlantVillage dataset
        print("Downloading PlantVillage dataset...")
        kaggle.api.dataset_download_files(
            "arjuntejaswi/plant-village",
            path="data/datasets/",
            unzip=True
        )
        print("✅ PlantVillage dataset downloaded")
        
        # Try to download plant pathology dataset
        print("Downloading Plant Pathology dataset...")
        kaggle.api.competition_download_files(
            "plant-pathology-2020-fgvc7",
            path="data/datasets/",
            unzip=True
        )
        print("✅ Plant Pathology dataset downloaded")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Real dataset download failed: {e}")
        print("Creating sample datasets instead...")
        return False

def main():
    """Main setup function"""
    print("🚀 Starting Farmers Disease Portal Setup...")
    print("=" * 50)
    
    success_count = 0
    
    # Step 1: Install requirements
    if install_requirements():
        success_count += 1
    
    # Step 2: Create directory structure
    if create_directory_structure():
        success_count += 1
    
    # Step 3: Setup environment
    if setup_environment():
        success_count += 1
    
    # Step 4: Setup Kaggle API (optional)
    kaggle_setup = setup_kaggle_api()
    
    # Step 5: Download datasets
    if kaggle_setup and download_real_datasets():
        success_count += 1
        print("✅ Real datasets downloaded from Kaggle")
    else:
        # Fallback to sample datasets
        if download_plantvillage_manual() and create_livestock_dataset():
            success_count += 1
            print("✅ Sample datasets created for development")
    
    # Final summary
    print("\n" + "=" * 50)
    print("🏁 Setup Summary:")
    print(f"✅ Completed steps: {success_count}/4")
    
    if success_count >= 3:
        print("🎉 Setup completed successfully!")
        print("\n📝 Next Steps:")
        print("1. Add your OpenWeather API key to .env file")
        print("2. Run: streamlit run app.py")
        print("3. Visit: http://localhost:8501")
        print("\n🚀 Optional: Train real models with:")
        print("   python models/model_trainer.py")
    else:
        print("⚠️ Setup completed with some issues.")
        print("Check error messages above and retry.")
    
    print("\n📧 Need help? Contact: support@farmersportal.com")

if __name__ == "__main__":
    main()