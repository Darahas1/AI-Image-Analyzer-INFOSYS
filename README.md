# INFOSYS Image Analyzer

A powerful Flask-based web application that leverages AI to analyze images, providing features like alt text generation, SEO descriptions, medical image analysis, and advanced color analysis.

## Features

- 🖼️ **Image Analysis**
  - Alt text generation using BLIP model
  - Context generation using GPT-3.5
  - Enhanced descriptions using GPT-4
  - Color analysis and distribution
  - Sentiment analysis

- 🏥 **Medical Image Analysis**
  - Detailed medical findings
  - Diagnostic observations
  - Professional recommendations
  - Confidence scoring

- 📱 **Social Media Tools**
  - Caption generation
  - Hashtag suggestions
  - Engagement optimization
  - Sentiment analysis

- 🔍 **SEO Tools**
  - SEO-optimized descriptions
  - Product listing optimization
  - Keyword extraction
  - Technical specifications

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

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)
- OpenAI API key
- Git (for cloning the repository)

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd infosys-image-analyzer
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```bash
   # Create .env file
   cp example.env .env
   
   # Edit .env file with your OpenAI API key
   OPENAI_API_KEY=your-api-key-here
   ```

5. **Initialize NLTK Data**
   ```python
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

## Running the Application

1. **Start the Flask Server**
   ```bash
   python run.py
   ```

2. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - The application will be running with all features available

## Available Routes

- `/` - Landing page with feature overview
- `/image-analyzer` - Basic image analysis
- `/advanced-analysis` - Advanced image analysis with color detection
- `/medical-image-analysis` - Medical image analysis
- `/social-media` - Social media content generation
- `/seo` - SEO optimization tools
- `/general` - General image analysis

## Development Guidelines

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use descriptive variable names
   - Add docstrings to functions and classes

2. **Error Handling**
   - Implement proper try-except blocks
   - Return meaningful error messages
   - Log errors appropriately

3. **Testing**
   - Write unit tests for new features
   - Test edge cases
   - Ensure proper error handling

## Troubleshooting

1. **Installation Issues**
   - Ensure Python 3.8+ is installed
   - Check virtual environment activation
   - Verify all dependencies are installed

2. **Runtime Errors**
   - Check OpenAI API key configuration
   - Verify NLTK data installation
   - Ensure proper file permissions

3. **Image Processing Issues**
   - Verify supported image formats
   - Check image file size limits
   - Ensure proper file uploads directory permissions

## Security Considerations

1. **API Keys**
   - Never commit API keys to version control
   - Use environment variables for sensitive data
   - Rotate API keys periodically

2. **File Uploads**
   - Validate file types
   - Limit file sizes
   - Sanitize file names

3. **User Input**
   - Validate all user inputs
   - Sanitize data before processing
   - Implement proper error handling

## Contributing

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes
   ```bash
   git commit -m "Add your feature description"
   ```
4. Push to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [BLIP](https://github.com/salesforce/BLIP) for image captioning
- [OpenAI](https://openai.com/) for GPT models
- [NLTK](https://www.nltk.org/) for sentiment analysis
- [Flask](https://flask.palletsprojects.com/) for web framework
- [Plotly](https://plotly.com/) for data visualization


