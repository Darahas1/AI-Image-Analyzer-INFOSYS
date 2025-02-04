import imghdr
from config.config import ALLOWED_EXTENSIONS

def allowed_file(filename, allowed_extensions=None):
    """
    Check if a filename has an allowed extension.
    Args:
        filename (str): Name of the file to check
        allowed_extensions (set): Optional set of allowed extensions. If None, uses default image extensions.
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    if not filename:
        return False
        
    # Default allowed extensions for images
    DEFAULT_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Use provided extensions or default to image extensions
    extensions = allowed_extensions if allowed_extensions is not None else DEFAULT_EXTENSIONS
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extensions

def validate_image(input_data):
    """
    Validate if the input is a valid image.
    Args:
        input_data: Either a file stream or a PIL Image object
    Returns:
        str: The image format extension if valid, None otherwise
    """
    try:
        if hasattr(input_data, 'read'):  # If it's a file stream
            header = input_data.read(512)
            input_data.seek(0)
            format = imghdr.what(None, header)
        else:  # If it's a PIL Image
            format = input_data.format.lower() if input_data.format else None
            
        if not format:
            return None
            
        # Normalize format name
        if format == 'jpeg':
            format = 'jpg'
        return '.' + format
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error validating image: {str(e)}")
        return None 