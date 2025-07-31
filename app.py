from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Set Gemini API key
api_key = "AIzaSyAdgG1y_NsrJ6EjPe_ya8UR79dUCPDup6c"
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set.")
genai.configure(api_key=api_key)

# Set ElevenLabs API key
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY is not set.")

# Load UBIK reference data from JSON
with open('ubik_data.json', 'r', encoding='utf-8') as f:
    ubik_info = json.load(f)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Spelling correction for internal use
def correct_spelling(user_input):
    input_lower = user_input.lower()
    corrections = {
        'ubeek': 'UBIK', 'ubiik': 'UBIK', 'youbik': 'UBIK',
        'ethiglo': 'EthiGlo', 'ethi glo': 'EthiGlo', 'ethiglow': 'EthiGlo', 'ethigloo': 'EthiGlo',
        'sisonext': 'SisoNext',
        'tehnology': 'technology', 'wat': 'what', 'prodacts': 'products', 'soultion': 'solution'
    }
    for wrong, right in corrections.items():
        if wrong in input_lower:
            user_input = user_input.replace(wrong, right)
    return user_input.capitalize()

# Static routes
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/quiz-instruction')
def quiz_instruction():
    return send_from_directory('static', 'quiz-instruction.html')

@app.route('/quiz')
def quiz():
    return send_from_directory('static', 'quiz.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

# Quiz question generation
@app.route('/api/quiz-questions', methods=['GET'])
def get_quiz_questions():
    try:
        questions = ubik_info.get('questions', [
            {
                "question": "What is the main ingredient in UBIK Solutions' Sebogel?",
                "options": ["Salicylic Acid", "Hyaluronic Acid", "Retinol", "Vitamin C"],
                "correct": "Salicylic Acid",
                "reference": "Sebogel contains Salicylic Acid and Niacinamide for anti-acne treatment."
            },
            {
                "question": "Which product is part of UBIK Solutions' Anti-Ageing line?",
                "options": ["Sebogel", "Reti K Cream", "EthiGlo Serum", "SisoNext Cleanser"],
                "correct": "Reti K Cream",
                "reference": "Reti K Cream with Encapsulated Retinol is a key Anti-Ageing product."
            },
            {
                "question": "What technology does UBIK Solutions use for dermatology?",
                "options": ["Blockchain", "AI", "Virtual Reality", "Quantum Computing"],
                "correct": "AI",
                "reference": "UBIK Solutions leverages AI for dermatology applications and services."
            },
            {
                "question": "What is a key feature of UBIK Solutions' iDoc Academy?",
                "options": ["Gaming", "Dermatology Training", "Financial Planning", "Cooking Classes"],
                "correct": "Dermatology Training",
                "reference": "iDoc Academy provides specialized dermatology training programs."
            },
            {
                "question": "Which brand is a subsidiary of UBIK Solutions?",
                "options": ["Nike", "EthiGlo", "Apple", "Samsung"],
                "correct": "EthiGlo",
                "reference": "EthiGlo is a subsidiary brand under UBIK Solutions."
            }
        ])
        return jsonify(questions[:5])
    except Exception as e:
        print("Error generating quiz questions:", e)
        return jsonify([
            {
                "question": "What is the main ingredient in UBIK Solutions' Sebogel?",
                "options": ["Salicylic Acid", "Hyaluronic Acid", "Retinol", "Vitamin C"],
                "correct": "Salicylic Acid",
                "reference": "Sebogel contains Salicylic Acid and Niacinamide for anti-acne treatment."
            },
            {
                "question": "Which product is part of UBIK Solutions' Anti-Ageing line?",
                "options": ["Sebogel", "Reti K Cream", "EthiGlo Serum", "SisoNext Cleanser"],
                "correct": "Reti K Cream",
                "reference": "Reti K Cream with Encapsulated Retinol is a key Anti-Ageing product."
            },
            {
                "question": "What technology does UBIK Solutions use for dermatology?",
                "options": ["Blockchain", "AI", "Virtual Reality", "Quantum Computing"],
                "correct": "AI",
                "reference": "UBIK Solutions leverages AI for dermatology applications and services."
            },
            {
                "question": "What is a key feature of UBIK Solutions' iDoc Academy?",
                "options": ["Gaming", "Dermatology Training", "Financial Planning", "Cooking Classes"],
                "correct": "Dermatology Training",
                "reference": "iDoc Academy provides specialized dermatology training programs."
            },
            {
                "question": "Which brand is a subsidiary of UBIK Solutions?",
                "options": ["Nike", "EthiGlo", "Apple", "Samsung"],
                "correct": "EthiGlo",
                "reference": "EthiGlo is a subsidiary brand under UBIK Solutions."
            }
        ])

# Answer evaluation
@app.route('/api/evaluate', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    question = data.get('question', '')
    answer = data.get('answer', '')
    options = data.get('options', [])

    print(f"Evaluating question: {question}")
    print(f"Answer: {answer}")
    print(f"Options: {options}")

    try:
        context_data = json.dumps(ubik_info, indent=2)
        correct_answer = next((q['correct'] for q in ubik_info.get('questions', []) if q['question'] == question), None)
        reference = next((q['reference'] for q in ubik_info.get('questions', []) if q['question'] == question), 'Refer to UBIK Solutions’ official resources.')

        if answer == 'SKIPPED':
            return jsonify({
                'feedback': 'Question skipped.',
                'score': 0.0,
                'reference': reference
            })

        prompt = f"""
        You are an official representative of UBIK Solutions.
        Reference data about UBIK:
        {context_data}
        Evaluate the following answer to the multiple-choice question: '{question}'
        Options: {options}
        Correct answer: {correct_answer}
        User answer: '{answer}'
        Provide feedback on the correctness and relevance of the answer. 
        Return a JSON object with 'feedback' (string), 'score' (float, 0 to 1), and 'reference' (string).
        Score 1.0 for correct answer, 0.0 for incorrect or skipped, 0.5 for partially relevant answers.
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        print("Gemini API response:", response.text)

        try:
            evaluation = json.loads(response.text.strip())
            if not isinstance(evaluation, dict) or 'feedback' not in evaluation or 'score' not in evaluation:
                raise ValueError("Invalid format")
            evaluation['score'] = float(evaluation['score'])
            if not 0 <= evaluation['score'] <= 1:
                evaluation['score'] = 0.0
                evaluation['feedback'] = "Invalid score returned; please try again."
            evaluation['reference'] = reference
        except Exception as e:
            print("Parse error:", e)
            evaluation = {
                'feedback': "Unable to evaluate due to format issue. Please try again with a clear answer.",
                'score': 0.0,
                'reference': reference
            }

        return jsonify(evaluation)
    except Exception as e:
        print("Evaluation error:", e)
        return jsonify({
            'feedback': "Error evaluating answer. Please try again later.",
            'score': 0.0,
            'reference': reference
        })

# Text-to-speech endpoint for ElevenLabs
@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice_id = data.get('voice_id', 'pNInz6obpgDQGcFmaJgB')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        response = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
            headers={
                'xi-api-key': ELEVENLABS_API_KEY,
                'Content-Type': 'application/json'
            },
            json={
                'text': text,
                'model_id': 'eleven_multilingual_v2',
                'voice_settings': {
                    'stability': 0.5,
                    'similarity_boost': 0.75
                }
            }
        )
        if response.status_code != 200:
            return jsonify({'error': response.text}), response.status_code
        return response.content, 200, {'Content-Type': 'audio/mpeg'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Chatbot API
@app.route('/api/chat', methods=['POST'])
def chatbot_reply():
    data = request.get_json()
    msg = data.get("message", "")
    user_message = msg["text"] if isinstance(msg, dict) and "text" in msg else msg
    print("User message:", user_message)

    corrected_message = correct_spelling(user_message)
    context_data = json.dumps(ubik_info, indent=2)

    prompt = f"""
    You are UBIK AI, a professional assistant with comprehensive knowledge about UBIK Solutions.
    UBIK's website: https://www.ubiksolution.com
    Reference data about UBIK:
    {context_data}
    User asked: "{corrected_message}"
    Answer the user's question based on the reference data. Keep answers short, clear, and relevant.
    If the question is unrelated to UBIK, politely state you don't have enough info.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        reply = response.text.strip()
        if not reply or "sorry" in reply.lower() or "not enough" in reply.lower():
            reply = "I'm not sure I have enough information to answer that. Could you please clarify or ask something else? 😊"
    except Exception as e:
        print("Chat Error:", e)
        reply = "Oops! Something went wrong while processing your question. Please try again. 😕"
    print("Sending reply:", reply)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)