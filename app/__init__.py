from flask import Flask
from flask_cors import CORS
from config.config import MAX_CONTENT_LENGTH, UPLOAD_FOLDER
import os
from app.utils.init_utils import initialize_nltk, initialize_ml_dependencies
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    try:
        # Initialize dependencies
        logger.info("Initializing dependencies...")
        initialize_ml_dependencies()
        initialize_nltk()
        logger.info("Dependencies initialization complete")

        app = Flask(__name__, 
                   template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
                   static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'))
        
        # Enable CORS
        CORS(app)
        
        # Ensure upload folder exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Configure upload folder
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
        
        # Register blueprints
        from app.routes.main_routes import main
        app.register_blueprint(main)
        
        return app
    except Exception as e:
        logger.error(f"Error creating app: {str(e)}")
        raise 
