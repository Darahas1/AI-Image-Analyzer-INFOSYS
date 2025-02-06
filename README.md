# INFOSYS Image Analyzer

A powerful Flask-based web application that leverages AI to analyze images, providing features like alt text generation, SEO descriptions, social media content generation, medical image analysis, and advanced color analysis.

## Key Features

### 1. General Image Analysis
- Real-time object detection and scene understanding
- Detailed visual element descriptions
- Advanced color analysis and pattern recognition
- Automated alt text generation for accessibility

### 2. Advanced Image Analysis
- Deep learning-powered visual analysis
- Comprehensive color detection and palette generation
- Enhanced AI descriptions with contextual understanding
- Detailed sentiment analysis of image content
- Pattern and texture recognition
- Visual composition analysis, etc


### 3. Medical Image Analysis
- Support for DICOM, TIFF, PNG, JPEG formats
- AI-assisted preliminary medical image interpretation
- Detailed anatomical structure identification
- **Important**: Not for diagnostic use - educational purposes only
- Confidence scoring system for analysis reliability

### 4. SEO Content Generator
- AI-powered product descriptions
- SEO-optimized title generation
- Smart keyword extraction and analysis
- Content optimization recommendations
- Engagement metrics analysis

### 5. Social Media Tools
- Platform-specific caption generation
- Trending hashtag suggestions
- Engagement optimization strategies
- Sentiment analysis and tone recommendations

## Technical Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- CUDA-compatible GPU (optional, for enhanced performance)
- Internet connection for API services

## Project Structure

```
.
├── app/
│   ├── routes/
│   │   └── main_routes.py      # Route handlers
│   ├── services/
│   │   ├── advanced_image_service.py  # Advanced image processing
│   │   ├── image_service.py    # Basic image processing
│   │   ├── seo_service.py      # SEO content generation
│   │   ├── med_service.py      # Medical image analysis
│   │   └── text_service.py     # Text processing and analysis
│   └── utils/
│       ├── file_utils.py       # File handling utilities
│       └── init_utils.py       # Initialization utilities
├── config/
│   ├── ai_config.py           # AI service configuration
│   └── config.py              # Application configuration
├── templates/                 # HTML templates
├── static/                   # Static assets
├── uploads/                  # Uploaded files (created automatically)
├── requirements.txt          # Python dependencies
└── run.py                   # Application entry point
```

1. **Environment Setup**
   ```bash
   # Clone repository
   git clone https://github.com/Darahas1/AI-Image-Analyzer-INFOSYS.git

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration**
   ```bash
   # Create .env file
   cp example.env .env
   
   # Edit .env file with your API keys
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Initialize NLTK Data**
   ```python
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

5. **Launch Application**
   ```bash
   python run.py
   ```

## Usage Guide

### Web Interface
- Access the application at `http://localhost:5000`
- Navigate to specific tools using the top navigation menu
- Upload images through drag-and-drop or file selection
- View analysis results in real-time

### API Integration
```python
import requests

# Example: General Image Analysis
response = requests.post(
    'http://localhost:5000/api/analyze/general',
    files={'image': open('image.jpg', 'rb')}
)

# Example: SEO Content Generation
response = requests.post(
    'http://localhost:5000/api/seo',
    files={'image': open('product.jpg', 'rb')}
)
```

## Available Routes

- `/` - Landing page with feature overview
- `/image-analyzer` - Basic image analysis
- `/advanced-analysis` - Advanced image analysis with color detection
- `/medical-image-analysis` - Medical image analysis
- `/social-media` - Social media content generation
- `/seo` - SEO optimization tools
- `/general` - General image analysis

## Security Considerations

1. **API Key Protection**
   - Never commit API keys to version control
   - Use environment variables for sensitive data
   - Rotate API keys periodically

2. **File Upload Security**
   - File type validation
   - Size limitations
   - Secure file handling

3. **Data Privacy**
   - No medical images are stored
   - Temporary file cleanup
   - Secure data transmission

## Error Handling

Common error scenarios and solutions:

1. **Installation Issues**
   - Verify Python version compatibility
   - Check virtual environment activation
   - Confirm all dependencies are installed

2. **Runtime Errors**
   - Validate API key configuration
   - Check NLTK data installation
   - Verify file permissions

3. **Processing Errors**
   - Confirm supported image formats
   - Check file size limits
   - Ensure stable internet connection

## Development Guidelines

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use descriptive variable names
   - Include docstrings for functions and classes

2. **Testing**
   ```bash
   # Run all tests
   pytest

   # Run specific test category
   pytest tests/test_image_analysis.py
   ```

3. **Contributing**
   - Fork the repository
   - Create feature branch
   - Submit pull request with tests
   - Follow code review process

## Performance Optimization

- GPU acceleration when available
- Caching for frequent requests
- Optimized image processing
- Efficient API usage


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [BLIP](https://github.com/salesforce/BLIP) - Image captioning
- [OpenAI](https://openai.com/) - GPT models
- [NLTK](https://www.nltk.org/) - Text processing
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [PyTorch](https://pytorch.org/) - Machine learning
- [Facebook DETR](https://github.com/facebookresearch/detr) - Object detection



