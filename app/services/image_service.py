from PIL import Image, ImageEnhance
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
import torch
from config.config import BLIP_MODEL
from collections import Counter
import warnings

# Suppress warnings about unused weights
warnings.filterwarnings(
    "ignore", 
    message="Some weights of the model checkpoint.*were not used"
)

class ImageProcessor:
    def __init__(self):
        # Initialize BLIP for general image description
        self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
        
        # Initialize DETR for object detection
        self.detr_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
        self.detr_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
        
        # Move models to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blip_model = self.blip_model.to(self.device)
        self.detr_model = self.detr_model.to(self.device)

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

    def detect_objects(self, image):
        """
        Detect objects in the image using DETR
        Args:
            image (PIL.Image): Input image
        Returns:
            list: List of detected objects with confidence scores
        """
        try:
            # Prepare image for object detection
            inputs = self.detr_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Perform object detection
            outputs = self.detr_model(**inputs)

            # Convert outputs to probabilities
            probas = outputs.logits.softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.7  # Confidence threshold

            # Convert target sizes
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            postprocessed_outputs = self.detr_processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=0.7
            )[0]

            # Extract detected objects
            objects = []
            for score, label in zip(postprocessed_outputs['scores'], postprocessed_outputs['labels']):
                if score > 0.7:  # Confidence threshold
                    object_name = self.detr_model.config.id2label[label.item()]
                    confidence = score.item()
                    objects.append({
                        'name': object_name,
                        'confidence': round(confidence * 100, 2)
                    })

            # Remove duplicates while keeping highest confidence
            unique_objects = {}
            for obj in objects:
                name = obj['name']
                if name not in unique_objects or obj['confidence'] > unique_objects[name]['confidence']:
                    unique_objects[name] = obj

            return list(unique_objects.values())

        except Exception as e:
            print(f"Error in object detection: {str(e)}")
            return []

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
            inputs = self.blip_processor(processed_image, return_tensors="pt")
            out = self.blip_model.generate(**inputs)
            alt_text = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return alt_text
            
        except Exception as e:
            return f"Error generating alt text: {str(e)}"

    def generate_alt_text_general(self, image):
        """
        Generate comprehensive image analysis for general purpose
        Args:
            image (PIL.Image): Input image
        Returns:
            dict: Analysis results containing description, objects, and colors
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Check image quality
            quality_metrics = self.validate_image_quality(processed_image)
            if not quality_metrics['is_valid']:
                print(f"Warning: Image quality issues detected: {quality_metrics['issues']}")
            
            # Generate description using BLIP
            inputs = self.blip_processor(processed_image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                min_length=30,
                temperature=0.7
            )
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Detect objects using DETR
            detected_objects = self.detect_objects(processed_image)
            
            # Sort objects by confidence
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Format objects for display
            objects = [f"{obj['name']} ({obj['confidence']}%)" for obj in detected_objects[:5]]
            
            # Extract colors using numpy
            img_array = np.array(processed_image)
            pixels = img_array.reshape(-1, 3)
            
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Convert colors to hex format
            colors = []
            for color in kmeans.cluster_centers_:
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0]), int(color[1]), int(color[2])
                )
                colors.append(hex_color)
            
            return {
                'description': description,
                'objects': objects,
                'colors': colors
            }
            
        except Exception as e:
            print(f"Error in generate_alt_text_general: {str(e)}")
            raise

# Create singleton instance
image_processor = ImageProcessor() 
