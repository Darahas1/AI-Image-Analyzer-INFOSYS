from config.ai_config import get_openai_client, format_success_response, format_error_response, GPT_CONFIG
import openai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_seo_description(context, alt_text):
    """
    Generates a detailed product description and SEO title with improved formatting.
    
    Args:
        context (str): Context of the image
        alt_text (str): Generated alt text of the image
        
    Returns:
        dict: Contains formatted description, SEO title, and sections
    """
    try:
        # Validate inputs
        if not context or not alt_text:
            raise ValueError("Context and alt_text are required")

        # Generate description
        description = _generate_description(context, alt_text)
        
        # Extract sections
        sections = _extract_sections(description)
        
        # Generate SEO title
        seo_title = _generate_seo_title(context, alt_text)
        
        # Generate keywords
        keywords = extract_keywords(description + " " + seo_title)

        response_data = {
            'seo_title': seo_title,
            'sections': sections,
            'keywords': keywords
        }
        
        # Debug log
        logger.info(f"Generated SEO content successfully")
        
        return format_success_response(response_data)
        
    except Exception as e:
        logger.error(f"Error in generate_seo_description: {str(e)}")
        return format_error_response(
            error_message=f"Error generating SEO content: {str(e)}",
            error_code="SEO_GENERATION_ERROR"
        )

def _generate_description(context, alt_text):
    """Helper function to generate the product description"""
    description_prompt = f"""Based on this image context and alt text, generate a comprehensive product description:

Context: {context}
Alt Text: {alt_text}

Please provide detailed information in this exact format, ensuring each bullet point is a complete, detailed sentence:

About:
• Begin with the product's primary visual or design feature and its direct user benefit
• Follow with the main performance or functionality feature and its practical application
• Include the product's unique selling point with a specific use case example
• Highlight a user comfort, convenience, or safety feature that enhances daily use
• End with the most impressive capability and its real-world benefit

Technical: (If Applicable)
• Detail primary performance metrics with exact numbers (e.g., power, speed, capacity, efficiency)
• Specify all relevant physical specifications (dimensions, weight, materials, display/size metrics)
• Include operational specifications (battery life, power usage, runtime, capacity, etc.)
• List storage, memory, or capacity specifications with exact measurements
• Detail connectivity, compatibility, or technical standards compliance

Additional:
• Begin with the most innovative or unique feature that sets this product apart
• Include any smart features, automation, or advanced technologies
• List included accessories, attachments, or complementary items
• Highlight customization options, adjustability, or versatility features
• End with compatibility features and integration capabilities"""

    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert product content writer specializing in SEO-optimized descriptions. Your strengths include:
                    - Adapting technical detail to product category
                    - Using precise specifications and measurements
                    - Converting features into clear user benefits
                    - Maintaining consistent professional terminology
                    - Following exact formatting requirements
                    - Prioritizing search-relevant information
                    - Including category-specific key metrics
                    - Using industry-standard naming conventions
                    - Highlighting relevant certification standards"""
                },
                {"role": "user", "content": description_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in _generate_description: {str(e)}")
        raise

def _generate_seo_title(context, alt_text):
    """Helper function to generate the SEO title"""
    title_prompt = f"""Create a highly optimized product title following this format:
    [Brand Name] [Model/Series] [Identifier], [Primary Spec] ([Value/Rating]), [Secondary Spec], [Capacity/Size] ([Color/Material], [Key Feature]) [Additional Info]

    Use this context:
    {context}
    {alt_text}

    Requirements:
    1. Include brand and complete model information
    2. List 2-3 key specifications with values
    3. Include relevant certifications or ratings
    4. Add color/material and a key feature in parentheses
    5. End with an important additional feature
    6. Use proper technical terminology
    7. Include measurements with units
    8. Keep length between 50-65 characters
    9. Use commas and parentheses for separation
    10. Match format of relevant category example"""

    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a product listing specialist who excels at:
                    - Creating category-appropriate product titles
                    - Including critical specifications
                    - Using proper technical terminology
                    - Following exact formatting requirements
                    - Maintaining optimal title length strictly (50-65 characters)
                    - Using industry-standard abbreviations
                    - Highlighting key features and certifications
                    - Adapting to different product categories
                    - Ensuring proper specification ordering"""
                },
                {"role": "user", "content": title_prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in _generate_seo_title: {str(e)}")
        raise

def _extract_sections(description):
    """Helper function to extract sections from the description"""
    try:
        sections = {}
        current_section = None
        current_content = []
        
        for line in description.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.lower().startswith(('about:', 'technical:', 'additional:')):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.split(':')[0].lower()
                current_content = []
            else:
                current_content.append(line)
                
        if current_section:
            sections[current_section] = '\n'.join(current_content)
            
        return sections
    except Exception as e:
        logger.error(f"Error extracting sections: {str(e)}")
        return {
            'about': '',
            'technical': '',
            'additional': ''
        }

def extract_keywords(text):
    """
    Extract key phrases from text for SEO keywords
    
    Args:
        text (str): Input text to extract keywords from
        
    Returns:
        list: Top 10 keywords
    """
    if not text:
        return []

    # Expanded stop words list
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over',
        'after', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should'
    }
    
    # Convert to lowercase and split
    words = text.lower().split()
    
    # Filter out stop words and short words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Get frequency distribution
    from collections import Counter
    keyword_freq = Counter(keywords)
    
    # Return top 10 most common keywords
    return [word for word, _ in keyword_freq.most_common(10)] 