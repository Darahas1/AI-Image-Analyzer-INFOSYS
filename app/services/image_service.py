from PIL import Image, ImageEnhance
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from config.config import BLIP_MODEL

class ImageProcessor:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained(BLIP_MODEL)
        self.model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
        
    def preprocess_image(self, image):
        """
        Preprocess image for better analysis
        Args:
            image (PIL.Image): Input image
        Returns:
            PIL.Image: Preprocessed image
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image quality
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            return image
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")

    def validate_image_quality(self, image):
        """
        Validate image quality metrics
        Args:
            image (PIL.Image): Input image
        Returns:
            dict: Quality metrics
        """
        try:
            # Convert image to numpy array
            img_array = np.array(image)
            
            # Calculate basic metrics
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            resolution = image.size
            
            # Define quality thresholds
            quality_metrics = {
                'brightness': brightness,
                'contrast': contrast,
                'resolution': resolution,
                'is_valid': True,
                'issues': []
            }
            
            # Check brightness
            if brightness < 30:
                quality_metrics['issues'].append('Image too dark')
            elif brightness > 225:
                quality_metrics['issues'].append('Image too bright')
                
            # Check contrast
            if contrast < 20:
                quality_metrics['issues'].append('Low contrast')
                
            # Check resolution
            min_resolution = 200 * 200
            if resolution[0] * resolution[1] < min_resolution:
                quality_metrics['issues'].append('Resolution too low')
                
            quality_metrics['is_valid'] = len(quality_metrics['issues']) == 0
            return quality_metrics
            
        except Exception as e:
            raise ValueError(f"Error validating image quality: {str(e)}")

    def generate_alt_text(self, image):
        """
        Generate alt text for an image using BLIP model
        Args:
            image (PIL.Image): Input image
        Returns:
            str: Generated alt text
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Check image quality
            quality_metrics = self.validate_image_quality(processed_image)
            if not quality_metrics['is_valid']:
                print(f"Warning: Image quality issues detected: {quality_metrics['issues']}")
            
            # Generate alt text using BLIP
            inputs = self.processor(processed_image, return_tensors="pt")
            out = self.model.generate(**inputs)
            alt_text = self.processor.decode(out[0], skip_special_tokens=True)
            
            return alt_text
            
        except Exception as e:
            return f"Error generating alt text: {str(e)}"

# Create singleton instance
image_processor = ImageProcessor() 