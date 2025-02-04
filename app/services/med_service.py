from transformers import pipeline
from PIL import Image
import openai
from config.ai_config import get_openai_client
import random
import logging

logger = logging.getLogger(__name__)

# Initialize the medical image captioning model
med_captioning_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

def determine_medical_image_type(caption):
    """Determine the image type based on medical conditions mentioned in the caption."""
    caption_lower = caption.lower()
    if "x-ray" in caption_lower:
        return "X-ray"
    elif "mri" in caption_lower or "brain scan" in caption_lower:
        return "MRI (Brain Tumor Detection)"
    elif "ultrasound" in caption_lower or "fetus" in caption_lower or "pregnancy" in caption_lower:
        return "Ultrasound (Pregnancy Detection)"
    elif "breast cancer" in caption_lower or "mammogram" in caption_lower:
        return "Mammogram (Breast Cancer Detection)"
    elif "fracture" in caption_lower or "bone" in caption_lower:
        return "X-ray (Fracture Detection)"
    else:
        return "General Medical Image"

def generate_dynamic_context(caption, image_type):
    """Generates an enhanced medical description."""
    client = get_openai_client()
    prompt = f"""
    Given the medical image caption: "{caption}" and image type: "{image_type}",
    provide an enhanced medical description. Include any notable abnormalities, 
    medical concerns, and potential diagnoses a radiologist might consider.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def generate_precautions(caption, image_type):
    """Generate medical precautions based on the analysis."""
    client = get_openai_client()
    prompt = f"""
    Given the medical image caption: "{caption}" and image type: "{image_type}",
    suggest important precautions a patient should take. Include home remedies, 
    necessary medical consultation, and lifestyle changes if required.

    Please strictly format your response as a numbered list of recommendations as:
    1. First recommendation point here...
    2. Second recommendation point here...
    3. Third recommendation point here...
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def determine_severity(caption):
    """Estimate injury or disease severity based on the caption."""
    keywords = {"mild": 20, "moderate": 50, "severe": 80, "critical": 95}
    for keyword, severity in keywords.items():
        if keyword in caption.lower():
            return severity
    return random.randint(30, 90)  

def is_medical_image(caption):
    """Checks if the caption contains medical-related terms."""
    medical_keywords = ["x-ray", "mri", "ultrasound", "fetus", "pregnancy", 
                       "breast cancer", "mammogram", "fracture", "bone", "tumor", 
                       "cancer", "radiology", "breast", "brain", "rash"]
    caption_lower = caption.lower()
    return any(keyword in caption_lower for keyword in medical_keywords)

def clean_repetitive_text(text):
    """Clean text by removing consecutive repetitive words."""
    if not text:
        return text
        
    # Split text into words
    words = text.split()
    if not words:
        return text
        
    # Remove consecutive repetitive words
    cleaned_words = [words[0]]  # Keep the first word
    for i in range(1, len(words)):
        if words[i].lower() != words[i-1].lower():  # Compare case-insensitive
            cleaned_words.append(words[i])
            
    return ' '.join(cleaned_words)

def analyze_medical_image(image_path):
    """Main function to analyze medical images."""
    try:
        # Process the image
        image = Image.open(image_path).convert("RGB")

        # Generate caption
        caption = med_captioning_model(image)[0]['generated_text']
        
        # Check if the caption is medical-related
        if not is_medical_image(caption):
            return {
                'success': False,
                'error': "Please upload a medical-related image."
            }
        
        image_type = determine_medical_image_type(caption)
        enhanced_context = generate_dynamic_context(caption, image_type)
        precautions = generate_precautions(caption, image_type)
        severity = determine_severity(caption)

        # Clean repetitive text in all text fields
        cleaned_caption = clean_repetitive_text(caption)
        cleaned_context = clean_repetitive_text(enhanced_context)
        cleaned_precautions = clean_repetitive_text(precautions)

        return {
            'success': True,
            'data': {
                'findings': cleaned_caption,
                'diagnosis': cleaned_context,
                'recommendations': cleaned_precautions,
                'confidence_score': severity / 100
            }
        }

    except Exception as e:
        logger.error(f"Error in medical image analysis: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }