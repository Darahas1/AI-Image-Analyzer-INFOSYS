from nltk.sentiment.vader import SentimentIntensityAnalyzer
from config.ai_config import get_openai_client, format_success_response, format_error_response, GPT_CONFIG
import logging
import re
from itertools import groupby
from app.services.image_service import image_processor

logger = logging.getLogger(__name__)

def generate_context(alt_text):
    """
    Generates context from alt text using OpenAI.
    Args:
        alt_text (str): Alt text to generate context from
    Returns:
        dict: Response containing generated context
    """
    try:
        # Clean the alt text first
        cleaned_alt_text = clean_text(alt_text)
        
        client = get_openai_client()
        prompt = f"""
        Generate a clear and concise description for this image.
        Avoid any repetition or redundant phrases.
        
        Original description: {cleaned_alt_text}
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at creating clear, enhanced image descriptions without repetition."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        context = response.choices[0].message.content.strip()
        return format_success_response({'context': context})

    except Exception as e:
        logger.error(f"Error generating context: {str(e)}")
        return format_error_response(str(e), 'CONTEXT_GENERATION_ERROR')

def enhance_context(context):
    """
    Enhances the context with additional details.
    Args:
        context (str): Original context to enhance
    Returns:
        dict: Response containing enhanced context
    """
    try:
        openai = get_openai_client()
        prompt = f"""Enhance this context with more descriptive details while maintaining accuracy:

Original: {context}

Requirements:
1. Add sensory details
2. Include specific measurements or technical details if applicable
3. Maintain factual accuracy
4. Keep the enhanced version under 100 words"""

        response = openai.chat.completions.create(
            model=GPT_CONFIG["model"],
            messages=[
                {"role": "system", "content": "You are a detail-oriented writer that enhances descriptions while maintaining accuracy."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        enhanced = response.choices[0].message.content.strip()
        return format_success_response({'enhanced_context': enhanced})
    except Exception as e:
        return format_error_response(
            error_message=f"Error enhancing context: {str(e)}",
            error_code="CONTEXT_ENHANCEMENT_ERROR"
        )

def clean_text(text, remove_duplicates=True, remove_repetitive_chars=True):
    """Unified text cleaning function"""
    if not text:
        return text
        
    # Split into words
    words = text.split()
    
    if remove_duplicates:
        # Remove consecutive duplicate words
        words = [next(group) for key, group in groupby(words)]
    
    # Join words back together
    cleaned_text = ' '.join(words)
    
    if remove_repetitive_chars:
        # Remove repetitive characters within words
        cleaned_text = re.sub(r'(.)\1+', r'\1', cleaned_text)
    
    return cleaned_text

def social_media_caption(context):
    """Generate engaging social media caption"""
    try:
        client = get_openai_client()
        
        prompt = f"""
        Create an engaging social media caption for this image.
        Make it conversational, include relevant emojis, and keep it under 200 characters.
        Avoid any word repetition or stuttering patterns.
        
        Image context: {context}
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a social media expert who creates engaging captions. Always use clear, concise language without repetition."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        caption = response.choices[0].message.content.strip()
        # Clean any potential repetitions
        caption = clean_text(caption)
        return format_success_response({'caption': caption})

    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        return format_error_response(str(e), 'CAPTION_GENERATION_ERROR')

def analyze_sentiment(text):
    """
    Analyzes sentiment of text using VADER.
    Args:
        text (str): Text to analyze
    Returns:
        dict: Response containing sentiment analysis
    """
    try:
        if not text:
            return format_error_response(
                error_message="No text provided for sentiment analysis",
                error_code="EMPTY_TEXT_ERROR"
            )

        try:
            analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            return format_error_response(
                error_message="Error initializing sentiment analyzer. Please ensure NLTK data is properly installed.",
                error_code="SENTIMENT_INIT_ERROR"
            )

        try:
            scores = analyzer.polarity_scores(text)
        except Exception as e:
            logger.error(f"Error calculating sentiment scores: {str(e)}")
            return format_error_response(
                error_message="Error calculating sentiment scores",
                error_code="SENTIMENT_CALCULATION_ERROR"
            )
        
        # Determine sentiment category
        compound = scores['compound']
        if compound >= 0.05:
            category = 'Positive'
        elif compound <= -0.05:
            category = 'Negative'
        else:
            category = 'Neutral'
            
        return format_success_response({
            'sentiment': {
                'score': compound,
                'category': category,
                'details': scores
            }
        })
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return format_error_response(
            error_message=f"Error analyzing sentiment: {str(e)}",
            error_code="SENTIMENT_ANALYSIS_ERROR"
        )

def analyze_medical_image(image, alt_text):
    """
    Analyzes medical image and generates detailed report.
    """
    try:
        print("Starting medical image analysis") # Debug log
        
        if not image or not alt_text:
            return format_error_response(
                error_message="Image and alt text are required for analysis",
                error_code="MISSING_INPUT"
            )

        # First get base image description using BLIP
        base_description = image_processor.generate_alt_text(image)
        print(f"Base description: {base_description}") # Debug log
        
        openai = get_openai_client()
        prompt = f"""Analyze this medical image and provide a detailed medical report:

Base Image Description: {base_description}
Additional Context: {alt_text}

Provide a comprehensive analysis in this exact format:

1. Key Findings:
[Provide detailed observations about visible anatomical structures, tissue characteristics, and any notable patterns]

2. Potential Observations:
[List possible medical interpretations, noting any concerning patterns or areas needing attention]

3. Recommendations:
[Suggest specific follow-up actions, additional tests, or monitoring protocols]

Use precise medical terminology and maintain professional objectivity."""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert medical imaging specialist. Provide detailed, professional analysis using medical terminology. Be thorough but avoid making definitive diagnoses."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3  # Lower temperature for more consistent output
        )
        
        analysis = response.choices[0].message.content.strip()
        
        # Parse sections with improved handling
        sections = {
            'key findings': '',
            'potential observations': '',
            'recommendations': ''
        }
        
        current_section = None
        current_content = []
        
        for line in analysis.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if '1. Key Findings:' in line:
                current_section = 'key findings'
                continue
            elif '2. Potential Observations:' in line:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'potential observations'
                current_content = []
                continue
            elif '3. Recommendations:' in line:
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'recommendations'
                current_content = []
                continue
                
            if current_section:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content)
            
        # Calculate confidence score based on content quality
        confidence_score = 0.5  # Base score
        
        # Adjust based on content length and quality
        for section, content in sections.items():
            if content:
                # Add points for section length
                words = len(content.split())
                confidence_score += min(0.1, words / 200)
                
                # Add points for medical terminology
                medical_terms = len([word for word in content.lower().split() 
                                  if any(term in word for term in ['tissue', 'anatomy', 'structure', 'pattern', 'density'])])
                confidence_score += min(0.1, medical_terms / 10)
        
        # Cap confidence score
        confidence_score = min(0.95, max(0.3, confidence_score))
            
        print(f"OpenAI response: {analysis}") # Debug log
        
        return format_success_response({
            'findings': sections['key findings'].strip(),
            'diagnosis': sections['potential observations'].strip(),
            'recommendations': sections['recommendations'].strip(),
            'confidence_score': confidence_score
        })
        
    except Exception as e:
        print(f"Error in analyze_medical_image: {str(e)}") # Debug log
        logger.error(f"Error analyzing medical image: {str(e)}")
        return format_error_response(
            error_message=f"Error analyzing medical image: {str(e)}",
            error_code="MEDICAL_ANALYSIS_ERROR"
        )

def generate_hashtags(text):
    """Generate relevant hashtags from the text"""
    try:
        client = get_openai_client()
        
        prompt = f"""
        Generate 8-10 relevant, trending hashtags for this social media post.
        Make them specific, engaging, and properly formatted with # symbol.
        Each hashtag should be a single word or compound words without spaces.
        
        Text: {text}
        
        Example format:
        #Photography #Nature #Wildlife #Beautiful
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a social media expert who creates engaging, relevant hashtags."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )

        # Extract hashtags from response
        hashtags_text = response.choices[0].message.content.strip()
        # Split by spaces and filter valid hashtags
        hashtags = [tag.strip() for tag in hashtags_text.split() if tag.strip().startswith('#')]
        
        # Ensure we have at least some hashtags
        if not hashtags:
            hashtags = ["#Photography", "#Social", "#Content"]  # Default fallback hashtags
            
        logger.info(f"Generated hashtags: {hashtags}")  # Add logging
        return format_success_response({'hashtags': hashtags})

    except Exception as e:
        logger.error(f"Error generating hashtags: {str(e)}")
        return format_error_response(str(e), 'HASHTAG_GENERATION_ERROR')
