from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import json
import os
import time
import logging
import subprocess
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from elevenlabs import ElevenLabs
#from elevenlabs.client import ApiError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcode API keys (replace with environment variables in production)
GOOGLE_API_KEY = "AIzaSyBIysFdLs3hEmd5GOiOIbLoCphWOEckLgw"  # Replace with your Gemini API key
ELEVENLABS_API_KEY = "sk_5ba950b061ec7cb7871f72ee6996d36bcacc32f5095f97b7"  # Replace with your new ElevenLabs API key
ELEVENLABS_VOICE_FEMALE = "L0yTtpRXzdyzQlzALhgD"  # Rachel (female, for index.html)
ELEVENLABS_VOICE_MALE = "oTQK6KgOJHp8UGGZjwUu"  # Adam (male, for quiz.html)
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Fallback: Rachel

# Validate API keys
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the code!")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY is not set in the code!")
if not ELEVENLABS_VOICE_FEMALE or not ELEVENLABS_VOICE_MALE:
    logger.warning("Voice IDs not set, using default voice ID: %s", DEFAULT_VOICE_ID)

# Configure ElevenLabs API with retry
valid_voice_ids = [DEFAULT_VOICE_ID]
voice_names = {ELEVENLABS_VOICE_FEMALE: "Rachel", ELEVENLABS_VOICE_MALE: "Adam", DEFAULT_VOICE_ID: "Rachel"}
try:
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    for attempt in range(3):
        try:
            voices = elevenlabs_client.voices.get_all()
            valid_voice_ids = [voice.voice_id for voice in voices.voices]
            voice_names = {voice.voice_id: voice.name for voice in voices.voices}
            logger.info("Valid voice IDs: %s", valid_voice_ids)
            logger.info("Voice names: %s", voice_names)
            break
        except ApiError as e:
            logger.error("ElevenLabs API initialization failed (attempt %d): %s, status_code: %s", attempt + 1, str(e), getattr(e, 'status_code', 'unknown'))
            if e.status_code == 401:
                logger.error("Invalid ElevenLabs API key. Please regenerate at https://elevenlabs.io/.")
                break
            elif attempt == 2:
                logger.error("Max retries reached for ElevenLabs API initialization")
                break
            time.sleep(2)
except Exception as e:
    logger.error("Error initializing ElevenLabs client: %s", str(e))
    logger.warning("Using default voice ID %s (Rachel) without validation due to API error", DEFAULT_VOICE_ID)

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# Create audio folder if not exists
os.makedirs("static/audio", exist_ok=True)

# Fallback data
FALLBACK_DATA = {
    "products": [
        {
            "name": "UVMed Tinted Sunscreen Gel",
            "description": "A lightweight, non-comedogenic sunscreen for acne-prone skin with broad-spectrum UV protection.",
            "price": "$25",
            "ingredients": "Zinc oxide, Niacinamide",
            "category": "Skincare"
        },
        {
            "name": "360 Block Sunscreen Cream",
            "description": "A moisturizing sunscreen that protects against UV rays and blue light, ideal for acne-prone and sensitive skin.",
            "ingredients": "Zinc oxide, moisturizing agents",
            "category": "Skincare"
        }
    ],
    "services": [
        {
            "name": "iDoc Academy",
            "description": "Training for dermatologists with online courses and hands-on workshops.",
            "features": "Online courses, hands-on workshops, certification programs"
        },
        {
            "name": "BrandYou",
            "description": "Private label service for dermatologists to create custom skin and hair products.",
            "features": "Formulation, packaging, supply chain support"
        }
    ],
    "pages": {
        "https://ubiksolutions.com/about": {
            "title": "About Us",
            "meta_description": "UBIK Solutions provides innovative dermatology solutions.",
            "category": "Company",
            "sections": [
                {"header": "Our Mission", "content": "To advance dermatology through innovation and quality products."},
                {"header": "Global Presence", "content": "UBIK Solutions exports dermatology products to over 20 countries, including the US, Europe, and Asia."},
                {"header": "Contact Information", "content": "Reach us at Office No.407, 4th Floor, Imperial Heights Tower-B, 150 FT Ring Road, Rajkot, Gujarat, India, 360005. Email: info@ubiksolution.com. Phone: +91 91045 69103."}
            ]
        }
    }
}

# Load ubik_data.json for evaluation
def load_data():
    try:
        with open('ubik_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("ubik_data.json not found, using fallback data")
        return FALLBACK_DATA

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            logger.error("Invalid or missing JSON payload: %s", data)
            return jsonify({'error': 'No text provided'}), 400

        text = data.get('text', '')
        voice_type = data.get('voice', 'female')
        voice_id = ELEVENLABS_VOICE_FEMALE if voice_type == 'female' else ELEVENLABS_VOICE_MALE

        if not text:
            logger.error("Empty text provided for audio generation")
            return jsonify({'error': 'No text provided'}), 400

        logger.info("Requested voice_type: %s, selected voice_id: %s (%s)", 
                    voice_type, voice_id, voice_names.get(voice_id, 'Unknown'))

        if voice_id not in valid_voice_ids:
            logger.error("Voice ID %s (%s) not available in valid_voice_ids: %s", 
                         voice_id, voice_names.get(voice_id, 'Unknown'), valid_voice_ids)
            return jsonify({'error': f'Voice ID {voice_id} ({voice_names.get(voice_id, "Unknown")}) not available. Check ElevenLabs dashboard.'}), 400

        model = "eleven_multilingual_v2"
        logger.info("Generating audio for text: %s..., voice_id: %s (%s), model: %s", 
                    text[:50], voice_id, voice_names.get(voice_id, 'Unknown'), model)
        
        audio_stream = elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model,
            output_format="mp3_44100_128",
            voice_settings={"stability": 0.5, "similarity_boost": 0.5, "speed": 0.8}
        )
        logger.info("Audio generated successfully for voice_id: %s (%s)", 
                    voice_id, voice_names.get(voice_id, 'Unknown'))
        return Response(audio_stream, mimetype='audio/mpeg', headers={"Content-Disposition": "inline"})
    except Exception as e:
        logger.error("Error generating audio: %s", str(e))
        return jsonify({'error': f'Failed to generate audio: {str(e)}'}), 500

@app.route('/')
def serve_index():
    logger.info("Serving index.html from static folder")
    return send_from_directory('static', 'index.html')

@app.route('/quiz')
def serve_quiz():
    logger.info("Serving quiz.html from static folder")
    return send_from_directory('static', 'quiz.html')

@app.route('/quiz-instruction')
def serve_quiz_instruction():
    logger.info("Serving quiz-instruction.html from static folder")
    return send_from_directory('static', 'quiz-instruction.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').lower()
        logger.info("User message: %s", message)
        
        scraped_data = load_data()
        chunks = []
        for product in scraped_data.get('products', []):
            chunks.append(f"Product: {product.get('name', '')}. Description: {product.get('description', '')}. Ingredients: {product.get('ingredients', '')}")
        for service in scraped_data.get('services', []):
            chunks.append(f"Service: {service.get('name', '')}. Description: {service.get('description', '')}")
        for url, page in scraped_data.get('pages', {}).items():
            for section in page.get('sections', []):
                chunks.append(f"{section.get('header', '')}: {section.get('content', '')}")

        logger.info("Loaded %d text chunks", len(chunks))
        logger.debug("Sample chunks: %s", chunks[:2])

        def find_relevant_chunk(query, chunks):
            try:
                vectorizer = TfidfVectorizer().fit_transform([query] + chunks)
                similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
                best_idx = similarities.argmax()
                best_score = similarities[best_idx]
                logger.info("Best chunk similarity score: %f", best_score)
                return chunks[best_idx] if best_score > 0.2 else None
            except Exception as e:
                logger.error("Error in find_relevant_chunk: %s", str(e))
                return None

        relevant_chunk = find_relevant_chunk(message, chunks)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = (
            f"You are a knowledgeable assistant for UBIK Solutions, a dermatology company. "
            f"Provide a concise answer (1-2 sentences) to the user's question based on this information: "
            f"{relevant_chunk or 'No specific information available.'} "
            f"If insufficient, use fallback data: {json.dumps(FALLBACK_DATA)}. "
            f"For specific topics (e.g., global presence, contact information, acne-prone skin, products, services, mission), "
            f"provide a detailed answer with all relevant details. "
            f"Question: {message}"
        )
        try:
            response = model.generate_content(prompt)
            reply = response.text.strip() if hasattr(response, 'text') and response.text else "No response generated."
        except ResourceExhausted as e:
            logger.error("Gemini API quota exceeded: %s", str(e))
            return jsonify({'error': 'Gemini API quota exceeded. Please try again later or upgrade your plan at https://ai.google.dev/gemini-api/docs/rate-limits.'}), 429
        except Exception as e:
            logger.error("Error generating content: %s", str(e))
            return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500
        
        # Add professional emojis and handle specific topics
        if 'global presence' in message:
            reply = "UBIK Solutions exports dermatology products to over 20 countries, including the US, Europe, and Asia, ensuring high-quality skincare solutions meet global demands. 🌍"
        elif 'contact information' in message:
            reply = "Reach UBIK Solutions at Office No.407, 4th Floor, Imperial Heights Tower-B, 150 FT Ring Road, Rajkot, Gujarat, India, 360005; Email: info@ubiksolution.com; Phone: +91 91045 69103. 📞"
        elif 'acne-prone skin' in message or 'acne prone skin' in message:
            reply = "UBIK Solutions offers UVMed Tinted Sunscreen Gel ($25, Zinc oxide, Niacinamide) and 360 Block Sunscreen Cream, both non-comedogenic and ideal for acne-prone skin. 🧴"
        elif 'products' in message:
            reply = "UBIK Solutions provides UVMed Tinted Sunscreen Gel ($25, Zinc oxide, Niacinamide) and 360 Block Sunscreen Cream, designed for acne-prone and sensitive skin. 🧴"
        elif 'services' in message:
            reply = "UBIK Solutions offers iDoc Academy for dermatologist training and BrandYou for custom skin/hair product creation, including formulation and packaging support. 📚"
        elif 'mission' in message:
            reply = "UBIK Solutions’ mission is to advance dermatology through innovative, high-quality products for global skincare needs. 📋"
        else:
            reply = f"{reply} 📋"

        logger.info("Generated response: %s", reply[:100])
        return jsonify({'reply': reply})
    except Exception as e:
        logger.error("Error in /api/chat: %s", str(e))
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/questions', methods=['GET'])
def get_questions():
    try:
        questions = [
            "What is the mission of UBIK Solutions?",
            "Which products does UBIK Solutions offer for acne-prone skin?",
            "What is the global presence of UBIK Solutions?",
            "How can I contact UBIK Solutions?",
            "What services does UBIK Solutions provide for dermatologists?"
        ]
        return jsonify(questions)
    except Exception as e:
        logger.error("Error in /api/questions: %s", str(e))
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_answer():
    try:
        data = request.get_json()
        question = data.get('question', '').lower()
        answer = data.get('answer', '').lower()
        logger.info("Evaluating question: %s, answer: %s", question, answer)

        model = genai.GenerativeModel('gemini-1.5-flash')
        scraped_data = load_data()
        prompt = (
            f"Evaluate the following answer for the question: '{question}'. "
            f"Answer: '{answer}'. "
            f"Provide detailed feedback and a score (0 to 1) based on accuracy and relevance, "
            f"using information from: {json.dumps(scraped_data)}. "
            f"If the answer is 'SKIPPED', assign a score of 0.0 and note it was skipped. "
            f"For correct answers, assign 0.8 or higher; for partially correct answers, assign 0.5; "
            f"for incorrect answers, assign 0.3 or lower."
        )
        try:
            response = model.generate_content(prompt)
            feedback = response.text.strip() if hasattr(response, 'text') and response.text else "No feedback generated."
        except ResourceExhausted as e:
            logger.error("Gemini API quota exceeded: %s", str(e))
            return jsonify({'error': 'Gemini API quota exceeded. Please try again later or upgrade your plan at https://ai.google.dev/gemini-api/docs/rate-limits.'}), 429
        score = 0.8 if 'correct' in feedback.lower() else 0.5 if answer != 'skipped' else 0.0
        return jsonify({'error': None, 'feedback': feedback, 'score': score})
    except Exception as e:
        logger.error("Error in /api/evaluate: %s", str(e))
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        if not data or 'answers' not in data:
            logger.error("Invalid or missing JSON payload: %s", data)
            return jsonify({'error': 'No answers provided'}), 400

        answers = data.get('answers', {})
        questions = [
            "What is the mission of UBIK Solutions?",
            "Which products does UBIK Solutions offer for acne-prone skin?",
            "What is the global presence of UBIK Solutions?",
            "How can I contact UBIK Solutions?",
            "What services does UBIK Solutions provide for dermatologists?"
        ]
        total_questions = len(questions)
        results = []

        model = genai.GenerativeModel('gemini-1.5-flash')
        scraped_data = load_data()
        total_score = 0.0

        for i, question in enumerate(questions):
            answer = answers.get(str(i), 'SKIPPED').lower()
            prompt = (
                f"Evaluate the following answer for the question: '{question}'. "
                f"Answer: '{answer}'. "
                f"Provide detailed feedback and a score (0 to 1) based on accuracy and relevance, "
                f"using information from: {json.dumps(scraped_data)}. "
                f"If the answer is 'SKIPPED', assign a score of 0.0 and note it was skipped. "
                f"For correct answers, assign 0.8 or higher; for partially correct answers, assign 0.5; "
                f"for incorrect answers, assign 0.3 or lower."
            )
            try:
                response = model.generate_content(prompt)
                feedback = response.text.strip() if hasattr(response, 'text') and response.text else "No feedback generated."
            except ResourceExhausted as e:
                logger.error("Gemini API quota exceeded: %s", str(e))
                return jsonify({'error': 'Gemini API quota exceeded. Please try again later or upgrade your plan at https://ai.google.dev/gemini-api/docs/rate-limits.'}), 429
            score = 0.8 if 'correct' in feedback.lower() else 0.5 if answer != 'skipped' else 0.0
            total_score += score
            results.append({
                'question': question,
                'answer': answer,
                'feedback': feedback,
                'score': score
            })

        average_score = total_score / total_questions if total_questions > 0 else 0.0

        # Generate HTML report
        html_content = """
        <div style='font-family: Roboto, sans-serif; padding: 1rem;'>
            <h2>UBIK Solutions Quiz Summary Report</h2>
            <p>Generated on July 22, 2025, 10:03 AM IST</p>
            <p>This report summarizes the answers provided for the UBIK Solutions quiz, evaluated against the company’s data from <code>ubik_data.json</code> using the Gemini API. Each answer is scored from 0 to 1 based on accuracy and relevance.</p>
            <table style='width: 100%; border-collapse: collapse; margin-top: 1rem;'>
                <tr style='background-color: #E91E63; color: white;'>
                    <th style='padding: 8px; border: 1px solid #ddd;'>Question</th>
                    <th style='padding: 8px; border: 1px solid #ddd;'>Answer</th>
                    <th style='padding: 8px; border: 1px solid #ddd;'>Feedback</th>
                    <th style='padding: 8px; border: 1px solid #ddd;'>Score</th>
                </tr>
"""
        for result in results:
            question = result['question'].replace("'", "'").replace('"', '"')
            answer = result['answer'].replace("'", "'").replace('"', '"')
            feedback = result['feedback'].replace("'", "'").replace('"', '"')
            score = result['score']
            html_content += f"""
                <tr>
                    <td style='padding: 8px; border: 1px solid #ddd;'>{question}</td>
                    <td style='padding: 8px; border: 1px solid #ddd;'>{answer}</td>
                    <td style='padding: 8px; border: 1px solid #ddd;'>{feedback}</td>
                    <td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{score}</td>
                </tr>
"""
        html_content += f"""
            </table>
            <h3 style='margin-top: 1rem;'>Overall Score</h3>
            <p>The average score is {average_score:.2f}.</p>
        </div>
"""

        logger.info("Generated HTML report for %d answers", len(answers))
        return Response(html_content, mimetype='text/html')
    except Exception as e:
        logger.error("Error generating report: %s", str(e))
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/download-report', methods=['POST'])
def download_report():
    try:
        data = request.get_json()
        if not data or 'answers' not in data:
            logger.error("Invalid or missing JSON payload for PDF: %s", data)
            return jsonify({'error': 'No answers provided'}), 400

        answers = data.get('answers', {})
        questions = [
            "What is the mission of UBIK Solutions?",
            "Which products does UBIK Solutions offer for acne-prone skin?",
            "What is the global presence of UBIK Solutions?",
            "How can I contact UBIK Solutions?",
            "What services does UBIK Solutions provide for dermatologists?"
        ]
        total_questions = len(questions)
        total_score = 0.0
        results = []

        model = genai.GenerativeModel('gemini-1.5-flash')
        scraped_data = load_data()

        for i, question in enumerate(questions):
            answer = answers.get(str(i), 'SKIPPED').lower()
            prompt = (
                f"Evaluate the following answer for the question: '{question}'. "
                f"Answer: '{answer}'. "
                f"Provide detailed feedback and a score (0 to 1) based on accuracy and relevance, "
                f"using information from: {json.dumps(scraped_data)}. "
                f"If the answer is 'SKIPPED', assign a score of 0.0 and note it was skipped. "
                f"For correct answers, assign 0.8 or higher; for partially correct answers, assign 0.5; "
                f"for incorrect answers, assign 0.3 or lower."
            )
            try:
                response = model.generate_content(prompt)
                feedback = response.text.strip() if hasattr(response, 'text') and response.text else "No feedback generated."
            except ResourceExhausted as e:
                logger.error("Gemini API quota exceeded: %s", str(e))
                return jsonify({'error': 'Gemini API quota exceeded. Please try again later or upgrade your plan at https://ai.google.dev/gemini-api/docs/rate-limits.'}), 429
            score = 0.8 if 'correct' in feedback.lower() else 0.5 if answer != 'skipped' else 0.0
            total_score += score
            results.append({
                'question': question,
                'answer': answer,
                'feedback': feedback,
                'score': score
            })

        average_score = total_score / total_questions if total_questions > 0 else 0.0

        # Generate LaTeX content
        latex_content = r"""
\documentclass[a4paper,12pt]{article}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{parskip}
\setlength{\parindent}{0pt}
\setlength{\parskip}{1em}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
\usepackage{titling}
\usepackage{xcolor}
\usepackage{enumitem}
\setlist[itemize]{leftmargin=*}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}

\title{UBIK Solutions Quiz Summary Report}
\author{AI Assistant}
\date{July 22, 2025}

\begin{document}

\maketitle
\section*{Summary Report for UBIK Solutions Quiz}
This report summarizes the answers provided for the five questions in the UBIK Solutions quiz, evaluated against the company’s data using the Gemini API. Each answer is scored from 0 to 1 based on accuracy and relevance, using information from \texttt{ubik_data.json}. Generated on July 22, 2025, 10:03 AM IST.

\section*{Detailed Results}
\begin{longtable}{p{0.3\textwidth} p{0.3\textwidth} p{0.3\textwidth} p{0.1\textwidth}}
\toprule
\textbf{Question} & \textbf{Answer} & \textbf{Feedback} & \textbf{Score} \\
\midrule
\endhead
"""
        for result in results:
            question = result['question'].replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')
            answer = result['answer'].replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')
            feedback = result['feedback'].replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')
            score = result['score']
            latex_content += f"{question} & {answer} & {feedback} & {score} \\\\\n\\midrule\n"

        average_score = total_score / total_questions if total_questions > 0 else 0.0
        latex_content += r"""
\bottomrule
\end{longtable}

\section*{Overall Score}
The average score across all questions is calculated as follows: \\
\[\frac{""" + f"{'+'.join(str(r['score']) for r in results)}{{{total_questions}}} = {average_score:.2f}" + r"""\]

\end{document}
"""

        # Write LaTeX to temporary file and compile to PDF
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as tex_file:
            tex_file.write(latex_content.encode('utf-8'))
            tex_file_path = tex_file.name

        pdf_path = tex_file_path.replace('.tex', '.pdf')
        try:
            subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', tex_file_path], check=True)
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            os.unlink(tex_file_path)
            for ext in ['.aux', '.log', '.out', '.pdf']:
                try:
                    os.unlink(tex_file_path.replace('.tex', ext))
                except FileNotFoundError:
                    pass
            return Response(pdf_data, mimetype='application/pdf', headers={"Content-Disposition": "attachment;filename=quiz_summary_report.pdf"})
        except subprocess.CalledProcessError as e:
            logger.error("Error compiling LaTeX to PDF: %s", str(e))
            return jsonify({'error': 'Failed to generate PDF report'}), 500
    except Exception as e:
        logger.error("Error in /api/download-report: %s", str(e))
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)