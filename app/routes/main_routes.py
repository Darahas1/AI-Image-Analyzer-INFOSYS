from flask import Blueprint, request, jsonify, render_template, send_file, current_app
from werkzeug.utils import secure_filename
import os
import tempfile
from PIL import Image
from gtts import gTTS
from datetime import datetime
import logging

from app.utils.file_utils import allowed_file, validate_image
from app.services.image_service import image_processor
from app.services.text_service import (
    generate_context,
    enhance_context,
    social_media_caption,
    generate_hashtags,
)
from app.services.advanced_image_service import AdvancedImageProcessor
from app.services.seo_service import generate_seo_description
from config.config import UPLOAD_FOLDER

logger = logging.getLogger(__name__)

# Define allowed extensions for medical images
ALLOWED_MEDICAL_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'dcm'}

main = Blueprint('main', __name__)

@main.route('/')
def landing():
    return render_template('landing.html')

@main.route('/social-media', methods=['GET', 'POST'])
def social_media():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No image file provided'
                }), 400
            
            file = request.files['image']
            
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No selected file'
                }), 400
                
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type. Please upload a PNG, JPG, JPEG, or GIF'
                }), 400
            
            # Create temporary file
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                
                # Process the image
                image = Image.open(filepath)
                alt_text = image_processor.generate_alt_text(image)
                
                # Generate context
                context_result = generate_context(alt_text)
                if not context_result['success']:
                    raise ValueError(context_result['error'])
                
                # Generate caption
                caption_result = social_media_caption(context_result['data']['context'])
                if not caption_result['success']:
                    raise ValueError(caption_result['error'])
                
                # Generate hashtags
                hashtags_result = generate_hashtags(context_result['data']['context'])
                if not hashtags_result['success']:
                    logger.error(f"Hashtag generation failed: {hashtags_result.get('error')}")
                    hashtags = []  # Fallback to empty list
                else:
                    hashtags = hashtags_result['data']['hashtags']
                
                return jsonify({
                    'success': True,
                    'data': {
                        'alt_text': alt_text,
                        'caption': caption_result['data']['caption'],
                        'hashtags': hashtags
                    }
                })
                
            finally:
                # Clean up temporary file
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        logger.error(f"Error removing temporary file: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    return render_template('social_media.html')

@main.route('/seo', methods=['GET', 'POST'])
def seo():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No image file provided',
                    'code': 'NO_IMAGE'
                }), 400
            
            file = request.files['image']
            
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No selected file',
                    'code': 'EMPTY_FILE'
                }), 400
                
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type. Please upload a PNG, JPG, JPEG, or GIF',
                    'code': 'INVALID_TYPE'
                }), 400
            
            # Validate image
            if not validate_image(file.stream):
                return jsonify({
                    'success': False,
                    'error': 'Invalid image file',
                    'code': 'INVALID_IMAGE'
                }), 400
                
            # Save and process image
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            try:
                # Process the image
                image = Image.open(filepath)
                alt_text = image_processor.generate_alt_text(image)
                
                # Generate initial context
                context_result = generate_context(alt_text)
                if not context_result['success']:
                    raise ValueError(context_result['error'])
                
                # Generate SEO content
                seo_result = generate_seo_description(context_result['data']['context'], alt_text)
                if not seo_result['success']:
                    raise ValueError(seo_result['error'])
                
                return jsonify({
                    'success': True,
                    'data': seo_result['data']
                })
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'Error processing image. Please try again.',
                    'code': 'PROCESSING_ERROR'
                }), 500
            
            finally:
                # Clean up uploaded file
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception as e:
                    logger.error(f"Error removing file: {str(e)}")
        
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'An unexpected error occurred. Please try again.',
                'code': 'SERVER_ERROR'
            }), 500
            
    return render_template('seo.html')

@main.route('/general', methods=['GET'])
def general():
    return render_template('general.html')

@main.route('/api/analyze/general', methods=['POST'])
def analyze_general():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, JPEG, or GIF'}), 400
        
        # Validate image
        if not validate_image(file.stream):
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Save and process image
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Process the image using the new general analysis method
            image = Image.open(filepath)
            result = image_processor.generate_alt_text_general(image)
            
            # Return formatted response matching frontend expectations
            return jsonify({
                'description': result['description'],
                'objects': result['objects'],
                'colors': result['colors']
            })
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': 'Error processing image. Please try again.'}), 500
        
        finally:
            # Clean up uploaded file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.error(f"Error removing file: {str(e)}")
    
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

@main.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Temporary file for the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts = gTTS(text=text, lang='en')
        tts.save(temp_file.name)
        
        return send_file(
            temp_file.name,
            mimetype='audio/mp3',
            as_attachment=True,
            download_name='speech.mp3'
        )
    except Exception as e:
        print(f"Error generating speech: {str(e)}")  # Add logging
        return jsonify({'error': 'Error generating speech. Please try again.'}), 500

@main.route('/medical-image-analysis', methods=['GET'])
def medical_analysis():
    """Route for rendering the medical analysis page"""
    return render_template('medical.html')

@main.route('/api/analyze-medical-image', methods=['POST'])
def analyze_medical_image_route():
    """API endpoint for processing medical images"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file'
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload a valid medical image file.'
            }), 400

        # Create temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            # Process the image using the medical service
            from app.services.med_service import analyze_medical_image
            result = analyze_medical_image(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing medical image: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }), 500
            
        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Server error in medical analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred'
        }), 500

@main.route('/image-analyzer', methods=['GET', 'POST'])
def image_analyzer():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No image file provided',
                    'code': 'NO_INPUT'
                }), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No selected file',
                    'code': 'EMPTY_FILE'
                }), 400
                
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type. Please upload a PNG, JPG, JPEG, or GIF',
                    'code': 'INVALID_TYPE'
                }), 400
            
            # Save and process image
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            try:
                # Process the image using image_service
                image = Image.open(filepath)
                alt_text = image_processor.generate_alt_text(image)
                
                # Generate enhanced context using text_service
                enhanced_result = enhance_context(alt_text)
                if not enhanced_result['success']:
                    raise ValueError(enhanced_result['error'])
                
                enhanced_context = enhanced_result['data']['enhanced_context']
                
                return jsonify({
                    'success': True,
                    'data': {
                        'alt_text': alt_text,
                        'context': enhanced_context
                    }
                })
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'Error processing image. Please try again.',
                    'code': 'PROCESSING_ERROR'
                }), 500
            
            finally:
                # Clean up uploaded file
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception as e:
                    logger.error(f"Error removing file: {str(e)}")
        
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'An unexpected error occurred. Please try again.',
                'code': 'SERVER_ERROR'
            }), 500
            
    return render_template('image_analyzer.html')

@main.route('/api/social-media/analyze', methods=['POST'])
def analyze_social_media():
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided',
                'error_code': 'NO_FILE'
            }), 400

        file = request.files['image']
        
        # Check if file was actually selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected',
                'error_code': 'NO_FILE'
            }), 400

        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Supported formats: PNG, JPEG, GIF',
                'error_code': 'INVALID_FILE_TYPE'
            }), 400

        # Create temporary file
        temp_path = None
        try:
            # Save uploaded file temporarily
            temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(temp_path)

            # Process the image using your existing services
            processor = AdvancedImageProcessor()
            processor.load_image(temp_path)

            # Generate base description
            alt_text = processor.generate_image_context()

            # Generate enhanced description for social media
            enhanced_text = processor.generate_enhanced_text(alt_text)

            # Generate hashtags from the context
            hashtags = generate_hashtags(enhanced_text)

            return jsonify({
                'success': True,
                'data': {
                    'alt_text': alt_text,
                    'caption': enhanced_text,
                    'hashtags': hashtags
                }
            })

        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error in social media analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'PROCESSING_ERROR'
        }), 500

@main.route('/advanced-analysis', methods=['GET', 'POST'])
def advanced_analysis():
    try:
        if request.method == 'POST':
            if 'image' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No image file provided'
                }), 400
            
            file = request.files['image']
            
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No selected file'
                }), 400
            
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG'
                }), 400
            
            # Save and process image
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Initialize processors
                image = Image.open(filepath)
                processor = AdvancedImageProcessor()
                processor.load_image(filepath)
                
                # Generate base description
                alt_text = image_processor.generate_alt_text(image)
                context_result = generate_context(alt_text)
                
                if not context_result['success']:
                    raise Exception(context_result['error'])
                
                # Generate enhanced description
                enhanced_result = processor.generate_enhanced_text(context_result['data']['context'])
                
                # Analyze colors
                hist_fig, pie_fig, color_data = processor.analyze_colors()
                
                # Convert matplotlib figures to base64
                import io
                import base64
                
                def fig_to_base64(fig):
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    return base64.b64encode(buf.getvalue()).decode('utf-8')
                
                # Get sentiment analysis
                sentiment_result = processor.sentiment_analysis(enhanced_result)
                
                return jsonify({
                    'success': True,
                    'data': {
                        'description': enhanced_result,
                        'color_analysis': {
                            'histogram': fig_to_base64(hist_fig),
                            'pie_chart': fig_to_base64(pie_fig),
                            'dominant_colors': color_data['dominant_colors'],
                            'color_percentages': color_data['percentages']
                        },
                        'sentiment': {
                            'score': float(sentiment_result['Confidence'].values[0]),
                            'label': sentiment_result['Sentiment'].values[0]
                        }
                    }
                })
                
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Error processing image: {str(e)}'
                }), 500
            
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        return render_template('advanced_analysis.html')
        
    except Exception as e:
        logger.error(f"Error in advanced analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500 
