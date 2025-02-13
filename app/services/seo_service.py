from config.ai_config import get_openai_client, format_success_response, format_error_response, GPT_CONFIG
import openai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_seo_description(context, alt_text):
    """
    Generates a detailed product description and SEO title with improved formatting.
    """
    try:
        # Validate inputs
        if not context or not alt_text:
            raise ValueError("Context and alt_text are required")

        client = get_openai_client()
        
        # Generate SEO title
        title_prompt = f"""Create a highly optimized product title (50-65 characters) based on this context:
        Context: {context}
        Alt Text: {alt_text}

        Requirements:
        1. Include key product features
        2. Use relevant keywords
        3. Be concise but descriptive
        4. Include brand if mentioned
        5. Add key specifications
        """

        title_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an SEO expert specializing in product titles."},
                {"role": "user", "content": title_prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )

        seo_title = title_response.choices[0].message.content.strip()

        # Generate detailed description sections
        description_prompt = f"""Based on this context and alt text, generate a comprehensive product description:

        Context: {context}
        Alt Text: {alt_text}

        Format the response in these exact sections:

        About:
        • [Key features and benefits]
        • [Main selling points]
        • [User benefits]

        Technical:
        • [Specifications and measurements]
        • [Materials and construction]
        • [Performance metrics]

        Additional:
        • [Unique features]
        • [Use cases]
        • [Compatibility]
        """

        description_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a product description expert focusing on SEO-optimized content."},
                {"role": "user", "content": description_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        description = description_response.choices[0].message.content.strip()
        
        # Extract sections
        sections = _extract_sections(description)
        
        # Generate keywords
        keywords = extract_keywords(description + " " + seo_title)

        return format_success_response({
            'seo_title': seo_title,
            'sections': sections,
            'keywords': keywords
        })

    except Exception as e:
        logger.error(f"Error in generate_seo_description: {str(e)}")
        return format_error_response(str(e))

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
        list: Top 10 relevant product keywords
    """
    try:
        if not text:
            return []

        client = get_openai_client()
        
        prompt = f"""Extract the most relevant product-focused keywords from this text. 
        Focus on:
        1. Product name/model
        2. Key technical specifications
        3. Main features
        4. Product category
        5. Brand names (if any)
        6. Materials or components
        7. Use cases
        8. Target audience/purpose
        
        Text: {text}
        
        Return only the most relevant 8-10 keywords, separated by commas. 
        Each keyword should be specific and valuable for product search.
        Avoid generic words like 'that', 'this', 'your', 'you'.
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an SEO expert specializing in product keyword extraction. Extract only specific, product-relevant keywords."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3  # Lower temperature for more focused results
        )

        # Process the response
        keywords_text = response.choices[0].message.content.strip()
        keywords = [k.strip() for k in keywords_text.split(',')]
        
        # Filter out any remaining generic terms or short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over',
            'after', 'this', 'that', 'these', 'those', 'your', 'you', 'device',
            'features', 'unique', 'quality', 'best'
        }
        
        # Filter and clean keywords
        cleaned_keywords = []
        for keyword in keywords:
            # Convert to lowercase for comparison
            keyword = keyword.lower().strip()
            # Check if keyword is valid
            if (len(keyword) > 2 and  # More than 2 characters
                keyword not in stop_words and  # Not a stop word
                not keyword.isdigit() and  # Not just a number
                not any(char.isdigit() for char in keyword[:2])):  # Doesn't start with number
                cleaned_keywords.append(keyword)

        return cleaned_keywords[:10]  # Return top 10 keywords

    except Exception as e:
        logger.error(f"Error in keyword extraction: {str(e)}")
        return [] 
