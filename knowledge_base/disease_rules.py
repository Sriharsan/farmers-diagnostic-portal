"""
Knowledge-Based Disease Reasoning System
Rule-based diagnosis and treatment recommendations
"""

import yaml
import json
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DiagnosisResult:
    disease_name: str
    confidence: float
    symptoms_matched: List[str]
    treatment: str
    prevention: str
    severity: str
    additional_info: str = ""

class DiseaseKnowledgeBase:
    def __init__(self):
        self.plant_rules = self._load_plant_rules()
        self.livestock_rules = self._load_livestock_rules()
        self.symptom_patterns = self._create_symptom_patterns()
    
    def _load_plant_rules(self) -> Dict:
        """Load plant disease rules and treatments"""
        return {
            'tomato_late_blight': {
                'symptoms': ['dark spots', 'brown spots', 'black spots', 'white fungal growth', 
                           'leaf yellowing', 'rapid spreading', 'fruit rotting', 'wilting'],
                'keywords': ['blight', 'fungus', 'dark', 'spots', 'mold'],
                'causes': ['Phytophthora infestans', 'wet conditions', 'poor air circulation'],
                'treatment': 'Apply copper-based fungicide (2-3g/L). Remove affected parts immediately. Improve drainage.',
                'prevention': 'Plant resistant varieties. Ensure good spacing. Avoid overhead irrigation. Apply preventive fungicide.',
                'severity': 'High',
                'crops': ['tomato', 'potato'],
                'confidence_base': 0.85
            },
            
            'wheat_rust': {
                'symptoms': ['orange pustules', 'yellow pustules', 'rust colored spots', 
                           'leaf yellowing', 'stunted growth', 'reduced yield'],
                'keywords': ['rust', 'orange', 'yellow', 'pustules', 'spots'],
                'causes': ['Puccinia species', 'wind dispersal', 'moderate temperature'],
                'treatment': 'Apply triazole fungicides. Use resistant wheat varieties. Early harvest if severe.',
                'prevention': 'Crop rotation. Plant resistant varieties. Monitor weather conditions. Field sanitation.',
                'severity': 'Medium',
                'crops': ['wheat', 'barley', 'oats'],
                'confidence_base': 0.80
            },
            
            'apple_scab': {
                'symptoms': ['velvety spots', 'dark spots on leaves', 'fruit cracking', 
                           'premature leaf drop', 'olive colored lesions'],
                'keywords': ['scab', 'velvety', 'dark', 'cracking', 'lesions'],
                'causes': ['Venturia inaequalis', 'wet spring weather', 'poor air circulation'],
                'treatment': 'Apply protective fungicides. Prune affected branches. Rake fallen leaves.',
                'prevention': 'Choose resistant cultivars. Proper pruning for air flow. Fall sanitation.',
                'severity': 'Medium',
                'crops': ['apple', 'pear'],
                'confidence_base': 0.75
            },
            
            'corn_blight': {
                'symptoms': ['gray lesions', 'boat shaped lesions', 'yellowing', 
                           'brown margins', 'leaf death'],
                'keywords': ['blight', 'gray', 'lesions', 'boat', 'brown'],
                'causes': ['Exserohilum turcicum', 'high humidity', 'moderate temperature'],
                'treatment': 'Apply strobilurin fungicides. Remove crop residue. Plant resistant hybrids.',
                'prevention': 'Crop rotation. Residue management. Use resistant varieties. Proper spacing.',
                'severity': 'Medium',
                'crops': ['corn', 'maize', 'sorghum'],
                'confidence_base': 0.78
            }
        }
    
    def _load_livestock_rules(self) -> Dict:
        """Load livestock disease rules and treatments"""
        return {
            'lumpy_skin_disease': {
                'symptoms': ['skin nodules', 'lumps', 'fever', 'reduced milk', 
                           'loss of appetite', 'difficulty swallowing'],
                'keywords': ['lumpy', 'nodules', 'lumps', 'skin', 'fever'],
                'causes': ['LSD virus', 'insect vectors', 'direct contact'],
                'treatment': 'Supportive care. Antibiotics for secondary infections. Anti-inflammatory drugs. Isolation.',
                'prevention': 'Vaccination. Vector control. Quarantine new animals. Good hygiene.',
                'severity': 'High',
                'animals': ['cattle', 'buffalo', 'cow', 'bull'],
                'confidence_base': 0.90
            },
            
            'foot_mouth_disease': {
                'symptoms': ['mouth blisters', 'foot blisters', 'drooling', 'lameness', 
                           'fever', 'loss of appetite', 'difficulty eating'],
                'keywords': ['fmd', 'blisters', 'mouth', 'foot', 'lameness', 'drooling'],
                'causes': ['FMD virus', 'highly contagious', 'direct contact'],
                'treatment': 'Supportive care. Soft feed. Isolation. Anti-inflammatory. Report to authorities.',
                'prevention': 'Vaccination. Quarantine. Disinfection. Movement restriction. Biosecurity.',
                'severity': 'Critical',
                'animals': ['cattle', 'pig', 'sheep', 'goat'],
                'confidence_base': 0.95
            },
            
            'mastitis': {
                'symptoms': ['swollen udder', 'hot udder', 'reduced milk', 'abnormal milk', 
                           'blood in milk', 'fever', 'loss of appetite'],
                'keywords': ['mastitis', 'udder', 'swollen', 'milk', 'infection'],
                'causes': ['Bacterial infection', 'poor milking hygiene', 'udder injury'],
                'treatment': 'Antibiotics (consult vet). Frequent milking. Anti-inflammatory. Supportive care.',
                'prevention': 'Good milking hygiene. Teat dips. Dry cow therapy. Clean environment.',
                'severity': 'Medium',
                'animals': ['cattle', 'buffalo', 'goat', 'sheep'],
                'confidence_base': 0.85
            },
            
            'newcastle_disease': {
                'symptoms': ['respiratory distress', 'coughing', 'sneezing', 'diarrhea', 
                           'neurological signs', 'twisted neck', 'paralysis', 'death'],
                'keywords': ['newcastle', 'respiratory', 'cough', 'neurological', 'paralysis'],
                'causes': ['Newcastle virus', 'highly contagious', 'respiratory transmission'],
                'treatment': 'No specific treatment. Supportive care. Isolation. Report to authorities.',
                'prevention': 'Vaccination. Biosecurity. Quarantine. Disinfection. All-in-all-out.',
                'severity': 'Critical',
                'animals': ['poultry', 'chicken', 'duck', 'turkey'],
                'confidence_base': 0.92
            }
        }
    
    def _create_symptom_patterns(self) -> Dict:
        """Create regex patterns for symptom matching"""
        patterns = {}
        
        # Common symptom patterns
        patterns['spots'] = r'\b(spot|spots|lesion|lesions|mark|marks)\b'
        patterns['yellowing'] = r'\b(yellow|yellowing|chlorosis)\b'
        patterns['wilting'] = r'\b(wilt|wilting|drooping|sagging)\b'
        patterns['fungal'] = r'\b(fungus|mold|mould|fungal|white growth)\b'
        patterns['fever'] = r'\b(fever|temperature|hot|pyrexia)\b'
        patterns['swelling'] = r'\b(swollen|swelling|enlarged|inflammation)\b'
        
        return patterns
    
    def diagnose(self, symptoms: List[str], category: str, crop_animal: str = None) -> List[DiagnosisResult]:
        """Main diagnosis function using rule-based reasoning"""
        
        if category.lower() == 'plant':
            rules = self.plant_rules
        else:
            rules = self.livestock_rules
        
        # Normalize symptoms
        normalized_symptoms = [s.lower().strip() for s in symptoms if s.strip()]
        symptom_text = ' '.join(normalized_symptoms)
        
        results = []
        
        for disease_id, rule in rules.items():
            confidence = self._calculate_confidence(symptom_text, rule, crop_animal)
            
            if confidence > 0.3:  # Minimum threshold
                matched_symptoms = self._find_matched_symptoms(normalized_symptoms, rule['symptoms'])
                
                result = DiagnosisResult(
                    disease_name=self._format_disease_name(disease_id),
                    confidence=confidence,
                    symptoms_matched=matched_symptoms,
                    treatment=rule['treatment'],
                    prevention=rule['prevention'],
                    severity=rule['severity'],
                    additional_info=f"Caused by: {', '.join(rule['causes'])}"
                )
                results.append(result)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:3]  # Return top 3 matches
    
    def _calculate_confidence(self, symptom_text: str, rule: Dict, crop_animal: str = None) -> float:
        """Calculate confidence score for a disease rule"""
        base_confidence = rule.get('confidence_base', 0.5)
        
        # Check keyword matches
        keyword_matches = 0
        for keyword in rule.get('keywords', []):
            if keyword.lower() in symptom_text:
                keyword_matches += 1
        
        # Check symptom matches
        symptom_matches = 0
        for symptom in rule.get('symptoms', []):
            if any(word in symptom_text for word in symptom.lower().split()):
                symptom_matches += 1
        
        # Calculate weighted score
        keyword_weight = 0.4
        symptom_weight = 0.6
        
        if len(rule.get('keywords', [])) > 0:
            keyword_score = keyword_matches / len(rule['keywords'])
        else:
            keyword_score = 0
        
        if len(rule.get('symptoms', [])) > 0:
            symptom_score = symptom_matches / len(rule['symptoms'])
        else:
            symptom_score = 0
        
        confidence = base_confidence * (keyword_weight * keyword_score + symptom_weight * symptom_score)
        
        # Boost confidence if crop/animal matches
        if crop_animal:
            target_list = rule.get('crops', []) + rule.get('animals', [])
            if any(crop_animal.lower() in target.lower() for target in target_list):
                confidence *= 1.2
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _find_matched_symptoms(self, user_symptoms: List[str], rule_symptoms: List[str]) -> List[str]:
        """Find which symptoms matched the rule"""
        matches = []
        for user_symptom in user_symptoms:
            for rule_symptom in rule_symptoms:
                if any(word in user_symptom for word in rule_symptom.lower().split()):
                    matches.append(user_symptom)
                    break
        return matches
    
    def _format_disease_name(self, disease_id: str) -> str:
        """Format disease ID to readable name"""
        return disease_id.replace('_', ' ').title()
    
    def get_treatment_details(self, disease_name: str, category: str) -> Dict:
        """Get detailed treatment information"""
        rules = self.plant_rules if category == 'plant' else self.livestock_rules
        disease_id = disease_name.lower().replace(' ', '_')
        
        if disease_id in rules:
            rule = rules[disease_id]
            return {
                'immediate_action': self._extract_immediate_action(rule['treatment']),
                'full_treatment': rule['treatment'],
                'prevention': rule['prevention'],
                'severity': rule['severity'],
                'follow_up': self._get_follow_up_advice(rule['severity'])
            }
        
        return {'error': 'Disease not found in knowledge base'}
    
    def _extract_immediate_action(self, treatment: str) -> str:
        """Extract immediate action from treatment description"""
        sentences = treatment.split('.')
        return sentences[0] + '.' if sentences else treatment
    
    def _get_follow_up_advice(self, severity: str) -> str:
        """Get follow-up advice based on severity"""
        if severity.lower() == 'critical':
            return "Contact veterinarian or agricultural expert immediately. Report to local authorities if required."
        elif severity.lower() == 'high':
            return "Monitor closely. Consult expert if no improvement in 2-3 days."
        else:
            return "Monitor progress. Consult expert if condition worsens."
    
    def add_community_remedy(self, disease: str, remedy: str, farmer_name: str, effectiveness: int):
        """Add community-contributed remedy"""
        # In a real system, this would save to database
        community_remedy = {
            'disease': disease,
            'remedy': remedy,
            'farmer': farmer_name,
            'effectiveness': effectiveness,
            'timestamp': str(pd.Timestamp.now()),
            'verified': False
        }
        return community_remedy

# Utility functions
def fuzzy_match_symptoms(user_input: str, known_symptoms: List[str], threshold: float = 0.7) -> List[str]:
    """Fuzzy match user symptoms with known symptoms"""
    from difflib import SequenceMatcher
    
    matches = []
    for known_symptom in known_symptoms:
        ratio = SequenceMatcher(None, user_input.lower(), known_symptom.lower()).ratio()
        if ratio >= threshold:
            matches.append((known_symptom, ratio))
    
    return [match[0] for match in sorted(matches, key=lambda x: x[1], reverse=True)]

# Example usage
if __name__ == "__main__":
    kb = DiseaseKnowledgeBase()
    
    # Test plant diagnosis
    plant_symptoms = ["dark spots on leaves", "yellowing", "rapid spreading"]
    plant_results = kb.diagnose(plant_symptoms, "plant", "tomato")
    
    print("Plant Disease Diagnosis:")
    for result in plant_results:
        print(f"Disease: {result.disease_name}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Treatment: {result.treatment}")
        print("-" * 50)
    
    # Test livestock diagnosis  
    livestock_symptoms = ["skin nodules", "fever", "reduced milk production"]
    livestock_results = kb.diagnose(livestock_symptoms, "livestock", "cattle")
    
    print("\nLivestock Disease Diagnosis:")
    for result in livestock_results:
        print(f"Disease: {result.disease_name}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Treatment: {result.treatment}")
        print("-" * 50)