"""
Image Processing Utilities
Optimized for mobile images and disease detection
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
from typing import Tuple, Optional, Union

class ImageProcessor:
    def __init__(self):
        self.target_size = (224, 224)
        self.max_file_size = 5 * 1024 * 1024  # 5MB limit
    
    def validate_image(self, image_file) -> bool:
        """Validate uploaded image file"""
        try:
            if image_file.size > self.max_file_size:
                return False
            
            # Try to open image
            img = Image.open(image_file)
            img.verify()
            return True
        except:
            return False
    
    def preprocess_for_diagnosis(self, image_input: Union[Image.Image, np.ndarray, bytes]) -> np.ndarray:
        """Main preprocessing pipeline for disease diagnosis"""
        
        # Convert input to PIL Image
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        else:
            image = image_input
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply enhancement pipeline
        enhanced_image = self._enhance_image(image)
        
        # Resize for model
        resized_image = enhanced_image.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(resized_image) / 255.0
        
        return img_array
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better diagnosis"""
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance color saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        # Slight sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=2))
        
        return image
    
    def detect_plant_region(self, image: Image.Image) -> Image.Image:
        """Detect and crop plant/leaf region from background"""
        img_array = np.array(image)
        
        # Convert to HSV for better plant detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define green color ranges for plants
        lower_green1 = np.array([25, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        
        lower_green2 = np.array([40, 40, 40])
        upper_green2 = np.array([80, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean mask with morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (main plant region)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2*padding)
            h = min(img_array.shape[0] - y, h + 2*padding)
            
            # Crop to plant region
            cropped = img_array[y:y+h, x:x+w]
            return Image.fromarray(cropped)
        
        return image
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Simple background removal for cleaner diagnosis"""
        img_array = np.array(image)
        
        # Convert to HSV
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Create mask for plant regions
        lower_bound = np.array([25, 30, 30])
        upper_bound = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Apply Gaussian blur to soften edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Create 3-channel mask
        mask_3d = np.dstack([mask, mask, mask]) / 255.0
        
        # Apply mask
        result = img_array * mask_3d
        
        # Add white background
        white_bg = np.ones_like(img_array) * 255
        result = result + white_bg * (1 - mask_3d)
        
        return Image.fromarray(result.astype(np.uint8))
    
    def augment_for_training(self, image: Image.Image) -> list:
        """Generate augmented images for training (if needed)"""
        augmented_images = [image]
        
        # Rotation
        augmented_images.append(image.rotate(15, fillcolor='white'))
        augmented_images.append(image.rotate(-15, fillcolor='white'))
        
        # Brightness adjustment
        enhancer = ImageEnhance.Brightness(image)
        augmented_images.append(enhancer.enhance(1.3))
        augmented_images.append(enhancer.enhance(0.7))
        
        # Contrast adjustment
        enhancer = ImageEnhance.Contrast(image)
        augmented_images.append(enhancer.enhance(1.3))
        
        return augmented_images
    
    def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (150, 150)) -> str:
        """Create base64 encoded thumbnail for display"""
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.LANCZOS)
        
        buffer = io.BytesIO()
        thumbnail.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def analyze_image_quality(self, image: Image.Image) -> dict:
        """Analyze image quality metrics"""
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        
        # Calculate metrics
        blur_score = cv2.Laplacian(img_array, cv2.CV_64F).var()
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Determine quality
        quality_score = 0
        issues = []
        
        if blur_score < 100:
            issues.append("Image appears blurry")
        else:
            quality_score += 25
        
        if brightness < 50:
            issues.append("Image too dark")
        elif brightness > 200:
            issues.append("Image too bright")
        else:
            quality_score += 25
        
        if contrast < 30:
            issues.append("Low contrast")
        else:
            quality_score += 25
        
        # Check if image is too small
        if min(image.size) < 200:
            issues.append("Image resolution too low")
        else:
            quality_score += 25
        
        return {
            'score': quality_score,
            'blur_score': blur_score,
            'brightness': brightness,
            'contrast': contrast,
            'issues': issues,
            'recommendation': self._get_quality_recommendation(quality_score)
        }
    
    def _get_quality_recommendation(self, score: int) -> str:
        """Get recommendation based on quality score"""
        if score >= 75:
            return "✅ Good quality image for diagnosis"
        elif score >= 50:
            return "⚠️ Acceptable quality, but could be improved"
        else:
            return "❌ Poor quality - please retake photo with better lighting and focus"

# Utility functions
def compress_image(image: Image.Image, max_size_kb: int = 500) -> Image.Image:
    """Compress image to reduce file size"""
    quality = 95
    while True:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        size_kb = len(buffer.getvalue()) / 1024
        
        if size_kb <= max_size_kb or quality <= 10:
            break
        quality -= 5
    
    buffer.seek(0)
    return Image.open(buffer)

def create_side_by_side_comparison(original: Image.Image, processed: Image.Image) -> Image.Image:
    """Create side-by-side comparison image"""
    width, height = original.size
    
    # Create new image with double width
    comparison = Image.new('RGB', (width * 2, height), 'white')
    
    # Paste images side by side
    comparison.paste(original, (0, 0))
    comparison.paste(processed, (width, 0))
    
    return comparison

def extract_color_features(image: Image.Image) -> dict:
    """Extract color-based features for disease detection"""
    img_array = np.array(image)
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Calculate color statistics
    features = {
        'avg_red': np.mean(img_array[:,:,0]),
        'avg_green': np.mean(img_array[:,:,1]),
        'avg_blue': np.mean(img_array[:,:,2]),
        'avg_hue': np.mean(hsv[:,:,0]),
        'avg_saturation': np.mean(hsv[:,:,1]),
        'avg_value': np.mean(hsv[:,:,2]),
        'green_percentage': np.sum((hsv[:,:,0] >= 25) & (hsv[:,:,0] <= 85)) / (img_array.shape[0] * img_array.shape[1]) * 100,
        'brown_percentage': np.sum((hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 25)) / (img_array.shape[0] * img_array.shape[1]) * 100
    }
    
    return features

# Example usage
if __name__ == "__main__":
    processor = ImageProcessor()
    
    # Test with dummy image
    dummy_img = Image.new('RGB', (300, 300), color='green')
    
    # Test preprocessing
    processed = processor.preprocess_for_diagnosis(dummy_img)
    print(f"Processed image shape: {processed.shape}")
    
    # Test quality analysis
    quality = processor.analyze_image_quality(dummy_img)
    print(f"Quality score: {quality['score']}")
    print(f"Recommendation: {quality['recommendation']}")