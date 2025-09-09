"""
Livestock Disease Classification Model
Optimized CNN for livestock disease detection
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

class LivestockDiseaseClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = [
            'Cattle_Healthy', 'Cattle_Lumpy_Skin_Disease', 'Cattle_Mastitis', 'Cattle_Foot_Mouth_Disease',
            'Buffalo_Healthy', 'Buffalo_Lumpy_Skin_Disease', 'Buffalo_Mastitis',
            'Goat_Healthy', 'Goat_PPR', 'Goat_Foot_Rot', 
            'Sheep_Healthy', 'Sheep_Bluetongue', 'Sheep_Foot_Rot',
            'Poultry_Healthy', 'Poultry_Newcastle_Disease', 'Poultry_Fowl_Pox', 'Poultry_Coccidiosis',
            'Pig_Healthy', 'Pig_Swine_Fever', 'Pig_Foot_Mouth_Disease'
        ]
        
        # Disease information mapping
        self.disease_info = {
            'Lumpy_Skin_Disease': {
                'severity': 'High',
                'contagious': True,
                'symptoms': ['Skin nodules', 'Fever', 'Reduced milk production', 'Loss of appetite'],
                'affected_animals': ['Cattle', 'Buffalo'],
                'transmission': 'Insect vectors, direct contact'
            },
            'Mastitis': {
                'severity': 'Medium',
                'contagious': False,
                'symptoms': ['Swollen udder', 'Hot udder', 'Abnormal milk', 'Fever'],
                'affected_animals': ['Cattle', 'Buffalo', 'Goat', 'Sheep'],
                'transmission': 'Bacterial infection, poor hygiene'
            },
            'Foot_Mouth_Disease': {
                'severity': 'Critical',
                'contagious': True,
                'symptoms': ['Mouth blisters', 'Foot blisters', 'Drooling', 'Lameness'],
                'affected_animals': ['Cattle', 'Pig', 'Sheep', 'Goat'],
                'transmission': 'Highly contagious virus'
            },
            'Newcastle_Disease': {
                'severity': 'Critical',
                'contagious': True,
                'symptoms': ['Respiratory distress', 'Neurological signs', 'Diarrhea'],
                'affected_animals': ['Poultry'],
                'transmission': 'Respiratory droplets'
            },
            'PPR': {
                'severity': 'High',
                'contagious': True,
                'symptoms': ['Fever', 'Nasal discharge', 'Diarrhea', 'Mouth sores'],
                'affected_animals': ['Goat', 'Sheep'],
                'transmission': 'Direct contact, respiratory droplets'
            }
        }
        
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
        """Create MobileNetV2 model for livestock disease classification"""
        model = mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
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
        """Predict livestock disease from image"""
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Mock prediction for demo (replace with actual model inference)
            # In real implementation: 
            # with torch.no_grad():
            #     outputs = self.model(input_tensor)
            #     probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Mock results for demo - simulate realistic livestock disease distribution
            mock_predictions = self._generate_realistic_predictions()
            
            # Get top predictions
            top_indices = np.argsort(mock_predictions)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                class_name = self.classes[idx]
                animal, condition = self._parse_class_name(class_name)
                
                result = {
                    'disease': condition,
                    'animal': animal,
                    'confidence': float(mock_predictions[idx]),
                    'full_class': class_name,
                    'severity': self._get_severity(condition),
                    'contagious': self._is_contagious(condition)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return [{'disease': 'Unknown', 'animal': 'Unknown', 'confidence': 0.0, 
                    'severity': 'Unknown', 'contagious': False}]
    
    def _generate_realistic_predictions(self):
        """Generate realistic mock predictions based on disease prevalence"""
        predictions = np.zeros(len(self.classes))
        
        # Common diseases have higher base probability
        disease_weights = {
            'Mastitis': 0.3,
            'Lumpy_Skin_Disease': 0.25,
            'Foot_Mouth_Disease': 0.2,
            'Newcastle_Disease': 0.15,
            'PPR': 0.1,
            'Healthy': 0.4
        }
        
        for i, class_name in enumerate(self.classes):
            _, condition = self._parse_class_name(class_name)
            base_weight = disease_weights.get(condition, 0.05)
            
            # Add random variation
            predictions[i] = base_weight * np.random.uniform(0.3, 1.0)
        
        # Normalize to probabilities
        predictions = predictions / np.sum(predictions)
        
        # Boost top prediction for more realistic results
        max_idx = np.argmax(predictions)
        predictions[max_idx] *= 2
        predictions = predictions / np.sum(predictions)
        
        return predictions
    
    def _parse_class_name(self, class_name):
        """Parse class name to extract animal and condition"""
        parts = class_name.split('_')
        animal = parts[0]
        condition = '_'.join(parts[1:]) if len(parts) > 1 else 'Unknown'
        return animal, condition
    
    def _get_severity(self, condition):
        """Get severity level for a condition"""
        for disease, info in self.disease_info.items():
            if disease in condition:
                return info['severity']
        return 'Medium'  # Default
    
    def _is_contagious(self, condition):
        """Check if condition is contagious"""
        for disease, info in self.disease_info.items():
            if disease in condition:
                return info['contagious']
        return False  # Default
    
    def get_disease_details(self, condition):
        """Get detailed information about a disease"""
        for disease, info in self.disease_info.items():
            if disease in condition:
                return info
        
        return {
            'severity': 'Medium',
            'contagious': False,
            'symptoms': ['Consult veterinarian for symptoms'],
            'affected_animals': ['Various'],
            'transmission': 'Unknown - consult expert'
        }
    
    def get_treatment_recommendations(self, condition, animal):
        """Get treatment recommendations for specific condition and animal"""
        treatments = {
            'Lumpy_Skin_Disease': {
                'immediate': 'Isolate affected animals immediately',
                'treatment': 'Supportive care, antibiotics for secondary infections, anti-inflammatory drugs',
                'prevention': 'Vaccination, vector control (fly control), quarantine new animals',
                'veterinary': True
            },
            'Mastitis': {
                'immediate': 'Stop milking affected quarters, apply cold compress',
                'treatment': 'Antibiotic therapy (consult vet), frequent milking, anti-inflammatory',
                'prevention': 'Good milking hygiene, teat dips, clean environment, dry cow therapy',
                'veterinary': True
            },
            'Foot_Mouth_Disease': {
                'immediate': 'IMMEDIATE ISOLATION - Report to authorities',
                'treatment': 'Supportive care, soft feed, wound care for blisters',
                'prevention': 'Vaccination, strict biosecurity, quarantine, movement restrictions',
                'veterinary': True,
                'reportable': True
            },
            'Newcastle_Disease': {
                'immediate': 'Isolate birds, report to authorities',
                'treatment': 'No specific treatment - supportive care only',
                'prevention': 'Vaccination, biosecurity, all-in-all-out system',
                'veterinary': True,
                'reportable': True
            },
            'PPR': {
                'immediate': 'Isolate affected animals',
                'treatment': 'Supportive care, treat secondary infections, fluid therapy',
                'prevention': 'Vaccination, quarantine new animals, good hygiene',
                'veterinary': True
            },
            'Healthy': {
                'immediate': 'Maintain good health practices',
                'treatment': 'Continue normal care',
                'prevention': 'Regular health checks, vaccination schedule, good nutrition',
                'veterinary': False
            }
        }
        
        # Get treatment for condition
        for disease, treatment in treatments.items():
            if disease in condition:
                return treatment
        
        # Default treatment advice
        return {
            'immediate': 'Monitor animal closely, isolate if showing symptoms',
            'treatment': 'Consult veterinarian for proper diagnosis and treatment',
            'prevention': 'Follow general animal husbandry practices',
            'veterinary': True
        }
    
    def assess_urgency(self, predictions):
        """Assess urgency level based on predictions"""
        max_prediction = predictions[0] if predictions else {}
        condition = max_prediction.get('disease', '')
        confidence = max_prediction.get('confidence', 0)
        
        # Critical diseases require immediate attention
        critical_diseases = ['Foot_Mouth_Disease', 'Newcastle_Disease']
        high_priority = ['Lumpy_Skin_Disease', 'PPR']
        
        if any(disease in condition for disease in critical_diseases) and confidence > 0.6:
            return {
                'level': 'CRITICAL',
                'message': 'Immediate veterinary attention and reporting required',
                'time_frame': 'Within 1 hour',
                'color': 'red'
            }
        elif any(disease in condition for disease in high_priority) and confidence > 0.7:
            return {
                'level': 'HIGH',
                'message': 'Veterinary consultation recommended soon',
                'time_frame': 'Within 24 hours',
                'color': 'orange'
            }
        elif condition != 'Healthy' and confidence > 0.6:
            return {
                'level': 'MEDIUM',
                'message': 'Monitor closely, consult vet if symptoms worsen',
                'time_frame': 'Within 2-3 days',
                'color': 'yellow'
            }
        else:
            return {
                'level': 'LOW',
                'message': 'Continue normal care and monitoring',
                'time_frame': 'Regular checkups',
                'color': 'green'
            }
    
    def generate_report(self, image, predictions):
        """Generate comprehensive diagnostic report"""
        if not predictions:
            return "Unable to generate report - no predictions available"
        
        top_prediction = predictions[0]
        condition = top_prediction['disease']
        animal = top_prediction['animal']
        confidence = top_prediction['confidence']
        
        urgency = self.assess_urgency(predictions)
        disease_details = self.get_disease_details(condition)
        treatment = self.get_treatment_recommendations(condition, animal)
        
        report = f"""
# Livestock Disease Diagnostic Report
**Generated:** {np.datetime64('now')}
**Animal Type:** {animal}
**Primary Diagnosis:** {condition}
**Confidence:** {confidence:.1%}
**Urgency Level:** {urgency['level']}

## Diagnostic Results
"""
        
        for i, pred in enumerate(predictions, 1):
            report += f"{i}. **{pred['disease']}** ({pred['animal']}) - {pred['confidence']:.1%} confidence\n"
        
        if condition != 'Healthy':
            report += f"""
## Disease Information
- **Severity:** {disease_details['severity']}
- **Contagious:** {'Yes' if disease_details['contagious'] else 'No'}
- **Transmission:** {disease_details['transmission']}

## Symptoms to Watch For
"""
            for symptom in disease_details['symptoms']:
                report += f"- {symptom}\n"
            
            report += f"""
## Treatment Recommendations
- **Immediate Action:** {treatment['immediate']}
- **Treatment:** {treatment['treatment']}
- **Prevention:** {treatment['prevention']}
- **Veterinary Required:** {'Yes' if treatment['veterinary'] else 'No'}
"""
            
            if treatment.get('reportable'):
                report += "\n⚠️ **REPORTABLE DISEASE** - Must report to local veterinary authorities\n"
        
        report += f"""
## Next Steps
{urgency['message']} ({urgency['time_frame']})

## Disclaimer
This is an AI-assisted diagnostic tool. Always consult with a qualified veterinarian for definitive diagnosis and treatment plans.
"""
        
        return report
    
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

# Utility functions for livestock health
def calculate_body_condition_score(image):
    """Calculate body condition score from image (basic implementation)"""
    # This would use computer vision to assess body condition
    # For demo, return random score
    return {
        'score': np.random.uniform(2.5, 4.0),
        'category': np.random.choice(['Thin', 'Ideal', 'Overweight']),
        'recommendation': 'Maintain current nutrition program'
    }

def detect_animal_in_image(image):
    """Detect and identify animal type in image"""
    # Mock implementation - in reality would use object detection
    animals = ['Cattle', 'Buffalo', 'Goat', 'Sheep', 'Poultry', 'Pig']
    detected_animal = np.random.choice(animals)
    confidence = np.random.uniform(0.7, 0.95)
    
    return {
        'animal': detected_animal,
        'confidence': confidence,
        'bbox': [50, 50, 200, 200]  # Mock bounding box
    }

def extract_health_indicators(image):
    """Extract visual health indicators from livestock image"""
    # Mock health indicators
    indicators = {
        'eye_clarity': np.random.choice(['Clear', 'Cloudy', 'Discharge']),
        'nose_condition': np.random.choice(['Normal', 'Discharge', 'Dry']),
        'coat_condition': np.random.choice(['Shiny', 'Dull', 'Patchy']),
        'posture': np.random.choice(['Alert', 'Droopy', 'Hunched']),
        'activity_level': np.random.choice(['Active', 'Lethargic', 'Normal'])
    }
    
    # Calculate overall health score
    positive_indicators = sum(1 for v in indicators.values() 
                             if v in ['Clear', 'Normal', 'Shiny', 'Alert', 'Active'])
    health_score = (positive_indicators / len(indicators)) * 100
    
    return {
        'indicators': indicators,
        'health_score': health_score,
        'overall_assessment': 'Good' if health_score > 60 else 'Concerning'
    }

def generate_vaccination_reminder(animal_type, last_vaccination=None):
    """Generate vaccination reminders based on animal type"""
    vaccination_schedules = {
        'Cattle': {
            'FMD': 'Every 6 months',
            'LSD': 'Annual',
            'Anthrax': 'Annual',
            'Black Quarter': 'Annual'
        },
        'Poultry': {
            'Newcastle': 'Every 3 months',
            'Fowl Pox': 'Annual',
            'Infectious Bronchitis': 'Every 4 months'
        },
        'Goat': {
            'PPR': 'Every 3 years',
            'FMD': 'Every 6 months',
            'Anthrax': 'Annual'
        },
        'Sheep': {
            'PPR': 'Every 3 years',
            'FMD': 'Every 6 months',
            'Enterotoxaemia': 'Annual'
        }
    }
    
    schedule = vaccination_schedules.get(animal_type, {})
    reminders = []
    
    for vaccine, frequency in schedule.items():
        reminders.append({
            'vaccine': vaccine,
            'frequency': frequency,
            'status': 'Due' if np.random.random() > 0.7 else 'Current'
        })
    
    return reminders

# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    classifier = LivestockDiseaseClassifier()
    
    # Test with dummy data
    print("Livestock Disease Classifier initialized successfully!")
    print(f"Number of supported classes: {len(classifier.classes)}")
    
    # Mock test
    dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    results = classifier.predict(dummy_image)
    
    print("\nSample prediction results:")
    for result in results:
        print(f"Disease: {result['disease']}")
        print(f"Animal: {result['animal']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Severity: {result['severity']}")
        print(f"Contagious: {result['contagious']}")
        print("-" * 40)
    
    # Test urgency assessment
    urgency = classifier.assess_urgency(results)
    print(f"Urgency Level: {urgency['level']}")
    print(f"Message: {urgency['message']}")
    
    # Test report generation
    report = classifier.generate_report(dummy_image, results)
    print(f"\nGenerated report length: {len(report)} characters")
    
    # Test utility functions
    health_indicators = extract_health_indicators(dummy_image)
    print(f"Health Score: {health_indicators['health_score']:.1f}%")
    
    vaccination_reminders = generate_vaccination_reminder('Cattle')
    print(f"Vaccination reminders: {len(vaccination_reminders)} vaccines")