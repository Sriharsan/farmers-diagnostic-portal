"""
Plant Disease Classification Model
Lightweight CNN for mobile deployment
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import pickle
import warnings
warnings.filterwarnings('ignore')

class PlantDiseaseClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = [
            'Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
            'Corn_Cercospora_leaf_spot', 'Corn_Common_rust', 'Corn_Northern_Leaf_Blight', 'Corn_healthy',
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
            'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites', 'Tomato_Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato_mosaic_virus', 'Tomato_healthy', 'Wheat_Brown_rust', 'Wheat_Yellow_rust', 'Wheat_healthy'
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        if model_path:
            self.load_model(model_path)
        else:
            self.model = self._create_model()
    
    def _create_model(self):
        """Create MobileNetV2 model for plant disease classification"""
        model = mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, len(self.classes))
        )
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor
    
    def predict(self, image, top_k=3):
        """Predict plant disease from image"""
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Mock prediction for demo (replace with actual model inference)
            # In real implementation: 
            # with torch.no_grad():
            #     outputs = self.model(input_tensor)
            #     probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Mock results for demo
            mock_predictions = np.random.rand(len(self.classes))
            mock_predictions = mock_predictions / mock_predictions.sum()  # normalize
            
            # Get top predictions
            top_indices = np.argsort(mock_predictions)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'disease': self.classes[idx],
                    'confidence': float(mock_predictions[idx]),
                    'crop': self.classes[idx].split('_')[0],
                    'condition': '_'.join(self.classes[idx].split('_')[1:])
                })
            
            return results
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return [{'disease': 'Unknown', 'confidence': 0.0, 'crop': 'Unknown', 'condition': 'Error'}]
    
    def get_disease_info(self, disease_name):
        """Get additional information about the disease"""
        disease_info = {
            'Tomato_Late_blight': {
                'symptoms': ['Dark brown spots on leaves', 'White fungal growth underneath leaves', 'Rapid spreading'],
                'causes': 'Phytophthora infestans fungus, favored by cool, wet conditions',
                'treatment': 'Apply copper-based fungicide, remove affected plant parts immediately',
                'prevention': 'Ensure good air circulation, avoid overhead watering, use resistant varieties',
                'severity': 'High'
            },
            'Wheat_Brown_rust': {
                'symptoms': ['Orange-brown pustules on leaves', 'Yellowing of leaves', 'Reduced grain filling'],
                'causes': 'Puccinia triticina fungus, spread by wind',
                'treatment': 'Apply triazole fungicides, use resistant wheat varieties',
                'prevention': 'Crop rotation, field sanitation, early sowing',
                'severity': 'Medium'
            },
            'Apple_scab': {
                'symptoms': ['Dark, velvety spots on leaves and fruit', 'Premature leaf drop', 'Fruit cracking'],
                'causes': 'Venturia inaequalis fungus, thrives in wet conditions',
                'treatment': 'Apply preventive fungicides, prune for air circulation',
                'prevention': 'Choose resistant varieties, proper pruning, rake fallen leaves',
                'severity': 'Medium'
            }
        }
        
        return disease_info.get(disease_name, {
            'symptoms': ['Consult agricultural expert for detailed symptoms'],
            'causes': 'Multiple factors - environmental, pathogenic, or nutritional',
            'treatment': 'Seek professional agricultural advice',
            'prevention': 'Follow integrated pest management practices',
            'severity': 'Unknown'
        })
    
    def save_model(self, path):
        """Save trained model"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load pre-trained model"""
        self.model = self._create_model()
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
        except:
            print("Model file not found, using mock predictions")
            self.model = self._create_model()

# Utility functions for image processing
def enhance_image_quality(image):
    """Enhance image quality for better prediction"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB if BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply histogram equalization
    if len(image.shape) == 3:
        # For color images, apply to each channel
        enhanced = np.zeros_like(image)
        for i in range(3):
            enhanced[:,:,i] = cv2.equalizeHist(image[:,:,i])
    else:
        enhanced = cv2.equalizeHist(image)
    
    return enhanced

def extract_leaf_region(image):
    """Extract leaf/plant region from background (basic implementation)"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to HSV for better plant detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define green color range for plants
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    classifier = PlantDiseaseClassifier()
    
    # Test with dummy data
    print("Plant Disease Classifier initialized successfully!")
    print(f"Number of supported diseases: {len(classifier.classes)}")
    
    # Mock test
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    results = classifier.predict(dummy_image)
    
    print("\nSample prediction results:")
    for result in results:
        print(f"Disease: {result['disease']}, Confidence: {result['confidence']:.2%}")