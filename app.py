from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from pathlib import Path
import logging
from werkzeug.utils import secure_filename
import google.generativeai as genai

# Import the PDFPaperAnalyzer class from your existing code
from pdf_analyzer import PDFPaperAnalyzer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Default expertise prompt
DEFAULT_EXPERTISE = """You are an Assistant Professor in the Department of Computer Science at the University of Maryland, College Park, specializing in AI for security and large language models (LLMs) for code generation. Your research aims to make AI coding assistants like GitHub Copilot and Cursor generate secure code, benchmark their cybersecurity capabilities, and develop LLM agents that can automatically detect and patch vulnerabilities. You lead CMSC818I, a role-playing seminar course on LLMs, Security, and Privacy, where students examine emerging threats and safeguards in AI systems through stakeholder simulations.
Your recent work includes the creation of PrimeVul, a large-scale benchmark dataset for vulnerability detection in C/C++ code, used by Google DeepMind to evaluate Gemini 1.5 Pro. In your ICSE 2025 paper, you demonstrate that even state-of-the-art models still struggle to identify common vulnerability types. Your ICLR 2025 workshop paper highlights why web-based LLM agents are more susceptible to attacks than standalone models, exposing architectural weaknesses in agent design. You have also contributed to the field of malware detection, showing in SaTML 2025 that ML-based behavioral models remain highly vulnerable to evasion and lack generalizability across malware variants.
Previously, your work in USENIX Security 2023 introduced a continual learning framework for Android malware detection that adapts to new threats over time without catastrophic forgetting. At CCS 2021, you proposed learning classifiers with verified robustness guarantees against adversarial attacks, earning Best Paper Award Runner-Up. You also developed DiverseVul, a vulnerability dataset presented at RAID 2023, focused on minimizing label noise and improving dataset diversity. You are currently recruiting motivated PhD students and postdocs who are excited to work at the intersection of LLMs, program analysis, and secure software engineering."""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_expertise_prompt(biography, api_key):
    """
    Use Gemini to generate an expertise prompt based on the user's biography.
    """
    try:
        genai.configure(api_key=api_key)
        
        prompt = f"""Based on the following academic biography, create a detailed reviewer expertise description 
        that will be used to review research papers from this person's perspective. The description should be 
        written in second person (using "You are...") and include their research areas, methodologies, 
        theoretical frameworks, and any specific interests mentioned. Make it comprehensive and professional.
        
        Biography:
        {biography}
        
        Generate a reviewer expertise description (200-300 words):"""
        
        response = genai.generate_content(
            model="models/gemini-2.0-flash",
            contents=[{"parts": [{"text": prompt}]}],
            generation_config={"temperature": 0.7, "max_output_tokens": 500}
        )
        
        expertise = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                expertise += part.text
                
        return expertise.strip()
        
    except Exception as e:
        logger.error(f"Error generating expertise prompt: {str(e)}")
        # Fallback to a generic expertise if generation fails
        return f"""You are an academic reviewer with expertise based on the following background: {biography}. 
        You bring your unique perspective and experience to evaluate research papers in your field."""

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    try:
        # Check if file is in request
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400
        
        file = request.files['pdf']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF files are allowed'}), 400
        
        # Get biography from form data (optional)
        biography = request.form.get('biography', '').strip()
        
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Get API key from environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            os.remove(temp_path)
            return jsonify({'error': 'GEMINI_API_KEY not configured'}), 500
        
        try:
            # Use default expertise if no biography provided, otherwise generate from biography
            if len(biography) == 0:
                logger.info("Using default expertise prompt")
                expertise = DEFAULT_EXPERTISE
            else:
                # Generate expertise prompt based on biography
                logger.info("Generating expertise prompt from biography")
                expertise = generate_expertise_prompt(biography, api_key)
                logger.info(f"Generated expertise: {expertise[:100]}...")
            
            # Run analysis with the expertise
            logger.info("Running PDF analysis")
            analyzer = PDFPaperAnalyzer(temp_path, api_key, expertise)
            result = analyzer.run()
            
            # Return the review
            return jsonify({'review': result}), 200
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return jsonify({'error': f'Error during analysis: {str(e)}'}), 500
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        logger.error(f"Error in analyze_pdf: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)