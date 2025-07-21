from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import json
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import re
import google.generativeai as genai
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

load_dotenv()

# Configure Gemini and ElevenLabs API keys
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables!")

elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
if not elevenlabs_api_key:
    raise ValueError("ELEVENLABS_API_KEY is not set in the environment variables!")

genai.configure(api_key=api_key)
try:
    elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key)
except Exception as e:
    print(f"Error initializing ElevenLabs client: {str(e)}")
    raise

app = Flask(__name__, static_folder='static')
CORS(app)

# Create audio folder if not exists (for other static files)
os.makedirs("static/audio", exist_ok=True)

@app.route('/api/get-elevenlabs-key', methods=['GET'])
def get_elevenlabs_key():
    return jsonify({'api_key': os.getenv("ELEVENLABS_API_KEY")})

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            print("Error: Invalid or missing JSON payload")
            return jsonify({'error': 'No text provided'}), 400

        text = data.get('text', '')
        voice_type = data.get('voice', 'female')  # 'female' or 'male'

        if not text:
            print("Error: Empty text provided for audio generation")
            return jsonify({'error': 'No text provided'}), 400

        voice_id = os.getenv("ELEVENLABS_VOICE_FEMALE") if voice_type == 'female' else os.getenv("ELEVENLABS_VOICE_MALE")
        if not voice_id:
            print(f"Error: No voice ID found for {voice_type}")
            return jsonify({'error': f'No voice ID configured for {voice_type}'}), 400

        model = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

        print(f"Streaming audio with voice_id: {voice_id}, model: {model}")
        audio_stream = elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model,
            output_format="mp3_44100_128",
            voice_settings={"stability": 0.5, "similarity_boost": 0.5, "speed": 0.8}
        )
        return Response(audio_stream, mimetype='audio/mpeg')
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return jsonify({'error': f'Failed to generate audio: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chatbot_reply():
    try:
        # Validate JSON input
        data = request.get_json(silent=True)
        if not data:
            print("Error: Invalid or missing JSON payload")
            return jsonify({"error": "Invalid JSON payload"}), 400

        msg = data.get("message", "")
        user_message = msg["text"] if isinstance(msg, dict) and "text" in msg else msg
        if not user_message:
            print("Error: Empty message provided")
            return jsonify({"error": "No message provided"}), 400

        print("User message:", user_message)

        # Handle conversational phrases
        user_message_lower = user_message.lower().strip()
        if user_message_lower in conversational_phrases:
            print(f"Matched conversational phrase: {user_message_lower}")
            reply = conversational_phrases[user_message_lower]
            return jsonify({"reply": reply})

        for phrase in conversational_phrases:
            if SequenceMatcher(None, user_message_lower, phrase).ratio() >= 0.8:
                print(f"Fuzzy matched conversational phrase: {user_message_lower} -> {phrase}")
                reply = conversational_phrases[phrase]
                return jsonify({"reply": reply})

        # Process UBIK-related or general queries
        if is_related_to_ubik(user_message):
            normalized_message = normalize_to_ubik(user_message)
            context = find_relevant_chunk(normalized_message, text_chunks)
            print(f"Selected context chunk: {context[:200]}")

            prompt = f"""
            You are a helpful chatbot assistant for UBIK Solutions. Respond based on the provided context:

            Context:
            {context}

            Question: {normalized_message}

            Respond:
            - In a short, clear, and meaningful way.
            - Stay professional but friendly.
            - Use 1 or 2 relevant emojis in your response.
            - If no relevant information is found, say: "I couldn't find specific details on that. Please ask another question about UBIK Solutions! 😊"
            """
        else:
            prompt = f"""
            You are a friendly, conversational chatbot assistant. Respond to the following user input in a natural, engaging way, as if continuing a casual conversation. Keep the tone professional but friendly, and use 1 or 2 relevant emojis:

            User input: {user_message}

            Respond:
            - Keep the response short and relevant.
            - Avoid restricting to UBIK Solutions unless relevant.
            - Use 1 or 2 emojis to match the tone.
            """

        # Generate response with Gemini
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            print("Gemini API response:", response.text)
            reply = response.text.strip()
            if not reply or "sorry" in reply.lower() or "cannot" in reply.lower() or "not enough" in reply.lower():
                reply = "Hmm, not sure about that one! What's next? 😄"
            return jsonify({"reply": reply})
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return jsonify({"reply": "Hmm, not sure about that one! What's next? 😄"})

    except Exception as e:
        print(f"Unexpected error in /api/chat: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# [Rest of app.py remains unchanged]
def load_scraped_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            raise ValueError("Scraped data file is empty")
        sections = content.split('\n\n--- ')
        scraped_text = ""
        for section in sections:
            if section.strip():
                lines = section.split('\n', 1)
                if len(lines) > 1:
                    scraped_text += lines[1] + "\n"
                else:
                    scraped_text += lines[0] + "\n"
        return scraped_text
    except Exception as e:
        print(f"Error loading scraped data: {e}")
        return ""

def chunk_text(text, chunk_size=1500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def normalize_to_ubik(message):
    variations = ['ubiq', 'ubikx', 'ubikz', 'ubiqs', 'ubiksolution', 'ubiksolutionz', 'ubique']
    for v in variations:
        message = re.sub(rf"\b{v}\b", "UBIK Solutions", message, flags=re.IGNORECASE)
    return message

def is_related_to_ubik(text):
    text = text.lower()
    keywords = ['ubik', 'ubiq', 'ubikx', 'ubikz', 'ubiqs', 'ubiksolution', 'ubik solutions', 'ubiksolutionz', 'dermatology', 'cosmetology', 'skin', 'hair', 'haircare', 'scalp', 'idoc', 'brandyou', 'vistaderm', 'ubique']
    for word in re.findall(r'\w+', text):
        for keyword in keywords:
            ratio = SequenceMatcher(None, word, keyword).ratio()
            if ratio >= 0.6:
                print(f"Matched keyword: {word} -> {keyword} (ratio: {ratio})")
                return True
    print(f"No relevant keywords found in: {text}")
    return False

def find_relevant_chunk(question, chunks):
    if not chunks:
        return "No relevant information available."
    vectorizer = TfidfVectorizer().fit_transform([question] + chunks)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    best_index = similarities.argmax()
    return chunks[best_index]

print("⏳ Loading and parsing UBIK Solutions website data...")
scraped_text = load_scraped_data("ubik_scraped_content.txt")
if not scraped_text:
    print("⚠️ Failed to load scraped data, using fallback.")
text_chunks = chunk_text(scraped_text)
print(f"✅ Loaded {len(text_chunks)} text chunks from scraped data.")

conversational_phrases = {
    'okay': 'Got it! 😊',
    'ok': 'Alright, cool! 😎',
    'yeah': 'Cool, what’s next? 🤗',
    'yes': 'Alright, anything else I can help with? 😊',
    'yep': 'Nice, what’s up? 👍',
    'sure': 'No problem, what’s on your mind? 😄',
    'thanks': 'You’re welcome! 😊',
    'thank you': 'Happy to help! What’s next? 😄',
    'hi': 'Hey there! How can I assist you today? 😊',
    'hello': 'Hi! Ready to chat about UBIK Solutions or anything else? 😄',
    'hey': 'Yo! What’s up? 😎',
    'good morning': 'Morning! How can I make your day even better? ☀️',
    'good evening': 'Evening! What’s on your mind tonight? 🌙',
    'bye': 'See ya later! 😊',
    'goodbye': 'Catch you next time! 👋',
    'thank u': 'No prob, happy to help! 😄',
    'tell me more': 'Sure thing! What do you want to dive into? 🤓',
    'what about': 'Ooh, tell me more about that! 😊',
    'how does it work': 'Great question! What specifically are you curious about? 🤔',
    'what else': 'Plenty more to explore! What’s next on your list? 😄'
}

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

@app.route('/api/questions', methods=['GET'])
def get_questions():
    try:
        prompt = f"""
        Based on the following content about UBIK Solutions, generate 5 open-ended quiz questions that test understanding of their services, mission, or processes. Each question should start with 'How', 'What', or 'Why' and focus on key aspects of UBIK Solutions. Return the questions as a JSON array, e.g., ["question1", "question2", "question3", "question4", "question5"].
        
        Content:
        {scraped_text}
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        print("Quiz API response:", response.text)
        questions = json.loads(response.text.strip()) if response.text.strip().startswith('[') else [
            "How does UBIK Solutions leverage AI for dermatology applications?",
            "What are the key services offered by UBIK Solutions?",
            "Why is UBIK Solutions' approach to data processing unique?",
            "How does UBIK Solutions ensure the accuracy of their AI models?",
            "What is the mission of UBIK Solutions in the healthcare industry?"
        ]
        return jsonify(questions[:5])
    except Exception as e:
        print("Error generating questions:", e)
        return jsonify([
            "How does UBIK Solutions leverage AI for dermatology applications?",
            "What are the key services offered by UBIK Solutions?",
            "Why is UBIK Solutions' approach to data processing unique?",
            "How does UBIK Solutions ensure the accuracy of their AI models?",
            "What is the mission of UBIK Solutions in the healthcare industry?"
        ])

@app.route('/api/evaluate', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    question = data.get('question', '')
    answer = data.get('answer', '')
    prompt = f"""
    Evaluate the following answer to the question: '{question}'
    Answer: '{answer}'
    
    Provide feedback on the correctness and completeness of the answer. If the answer is 'SKIPPED', note that no answer was provided. Return a JSON object with 'feedback' (string) and 'score' (float, 0 to 1).
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        print("Evaluation API response:", response.text)
        evaluation = json.loads(response.text.strip()) if response.text.strip().startswith('{') else {
            'feedback': 'Answer received but could not be evaluated properly.' if answer != 'SKIPPED' else 'No answer provided (skipped).',
            'score': 0.8 if answer != 'SKIPPED' else 0.0
        }
        return jsonify(evaluation)
    except Exception as e:
        print("Error evaluating answer:", e)
        return jsonify({'feedback': 'Error processing answer.', 'score': 0.0})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)