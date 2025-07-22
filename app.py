from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API key
api_key = "AIzaSyAdjup1wdoRP0GrOFuixnfxt9AgepmWR_8"  # Replace with your actual Gemini API key
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set!")

genai.configure(api_key=api_key)

app = Flask(__name__, static_folder='static')
CORS(app)

# Load JSON data
def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data:
            raise ValueError(f"{file_path} is empty")
        print(f"Successfully loaded {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

# Process JSON data into text chunks
def process_json_data(data, chunk_size=1500):
    text = ""
    for key, value in data.items():
        if isinstance(value, str):
            text += f"{key}: {value}\n"
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    text += " ".join([f"{k}: {v}" for k, v in item.items()]) + "\n"
        elif isinstance(value, dict):
            text += f"{key}: " + " ".join([f"{k}: {v}" for k, v in value.items()]) + "\n"
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"Processed {len(chunks)} chunks from JSON data")
    return chunks

# Load and process JSON files
print("⏳ Loading and parsing UBIK Solutions JSON data...")
ubik_data = load_json_data("ubik_data.json")
ubik_product_details = load_json_data("ubik_product_details.json")
if not ubik_data or not ubik_product_details:
    print("⚠️ Failed to load JSON data, using fallback.")
text_chunks = process_json_data(ubik_data)
product_text_chunks = process_json_data(ubik_product_details)
print(f"✅ Loaded {len(text_chunks)} general text chunks and {len(product_text_chunks)} product text chunks.")

# Normalize UBIK variations
def normalize_to_ubik(message):
    variations = ['ubiq', 'ubit', 'ubikx', 'ubikz', 'ubiqs', 'ubiksolution', 'ubiksolutionz', 'ubique']
    for v in variations:
        message = re.sub(rf"\b{v}\b", "UBIK Solutions", message, flags=re.IGNORECASE)
    return message

# Check if query is related to UBIK or products
def is_related_to_ubik(text):
    text = text.lower()
    keywords = ['ubik', 'ubiq', 'ubikx', 'ubikz', 'ubiqs', 'ubiksolution', 'ubik solutions', 'ubiksolutionz', 'dermatology', 'cosmetology', 'skin', 'hair', 'haircare', 'scalp', 'idoc', 'brandyou', 'vistaderm', 'ubique', 'anti-acne', 'anti-ageing', 'anti-fungal']
    for word in re.findall(r'\w+', text):
        for keyword in keywords:
            ratio = SequenceMatcher(None, word, keyword).ratio()
            if ratio >= 0.6:
                print(f"Matched keyword: {word} -> {keyword} (ratio: {ratio})")
                return True
    print(f"No relevant keywords found in: {text}")
    return False

# Check if query is product-related
def is_product_related(text):
    text = text.lower()
    product_keywords = ['anti-acne', 'anti-ageing', 'anti-fungal', 'product', 'cream', 'serum', 'ingredients', 'price', 'cost', 'how much', 'sebogel', 'benzonext', 'aczee', 'sebollic', 'saligly', 'sebonia', 'sefpil', 'o wash', 'cutishine', 'acmed', 'actreat', 'reti k']
    for keyword in product_keywords:
        if keyword in text:
            return True
    return False

# Check if query asks for price
def is_price_query(text):
    text = text.lower()
    price_keywords = ['price', 'cost', 'how much', 'how expensive', 'what is the price', 'what does it cost']
    for keyword in price_keywords:
        if keyword in text:
            return True
    return False

# Find relevant chunk for general queries
def find_relevant_chunk(question, chunks):
    if not chunks:
        print("No chunks available for evaluation")
        return "No relevant information available."
    vectorizer = TfidfVectorizer().fit_transform([question] + chunks)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    best_index = similarities.argmax()
    print(f"Selected chunk {best_index} with similarity {similarities[best_index]}")
    return chunks[best_index]

# Find specific product by name
def find_product_by_name(query, product_details):
    query = query.lower()
    for category in product_details.get('product_categories', []):
        for product in category.get('products', []):
            if product.get('name', '').lower().find(query) != -1 or SequenceMatcher(None, query, product.get('name', '').lower()).ratio() >= 0.8:
                return product, category.get('name')
    return None, None

# Format product details for response
def format_product_details(product, category, query):
    if not product:
        return "I couldn't find specific details on that product. Please ask about another UBIK Solutions product! 😊"
    
    name = product.get('name', 'Unknown')
    price = product.get('price', 'N/A')
    description = product.get('description', 'No description available.')
    ingredients = product.get('ingredients', 'N/A')
    url = product.get('url', '#')
    
    # Define default usage and safety info for products
    usage_info = (
        "1. Wash and dry your face or affected area thoroughly.\n"
        "2. Take a small amount of the product on your fingertips.\n"
        "3. Apply gently to the affected areas, massaging in a circular motion until absorbed.\n"
        "4. Use twice or thrice daily, or as directed by a dermatologist.\n"
        "5. Avoid contact with eyes, mouth, or mucous membranes, and perform a patch test before full application."
    )
    safety_info = (
        "- Not recommended for children under 3 years of age.\n"
        "- Consult a dermatologist before use if pregnant or breastfeeding.\n"
        "- May cause mild skin irritation; reduce application frequency if dryness or peeling occurs.\n"
        "- Store in a cool, dry place away from direct sunlight."
    )
    
    # Customize benefits based on category and description
    benefits = [
        "Reduces acne, blackheads, and whiteheads effectively.",
        "Controls excess oil production for a clearer complexion.",
        "Soothes inflammation and redness caused by acne.",
        "Non-comedogenic and paraben-free, suitable for sensitive skin."
    ] if category == "Anti-Acne" else [
        "Reduces visible signs of aging like wrinkles and fine lines.",
        "Enhances skin elasticity and firmness.",
        "Diminishes age spots and promotes even skin tone.",
        "Deeply hydrates and strengthens the skin barrier."
    ]

    # Conditionally include price if requested
    key_details = [
        f"- **Product Name**: {name}",
        f"- **Category**: {category}",
        f"- **Description**: {description}",
        f"- **Ingredients**: {ingredients}",
        f"- **URL**: [{name}]({url})"
    ]
    if is_price_query(query):
        key_details.insert(1, f"- **Price**: {price}")

    response = (
        f"**{name} Overview**  \n"
        f"{name} is a highly effective {category.lower()} product from UBIK Solutions, designed to {description.split('.')[0].lower()}. "
        f"It’s crafted to deliver radiant, healthy skin with a lightweight, non-greasy formula. 😊\n\n"
        
        f"**Key Details**  \n"
        f"{chr(10).join(key_details)}  \n\n"
        
        f"**Additional Benefits**  \n"
        f"{chr(10).join(f'- {benefit}' for benefit in benefits)}  \n\n"
        
        f"**How to Use**  \n"
        f"{usage_info}  \n\n"
        
        f"**Safety Information**  \n"
        f"{safety_info}  \n\n"
        
        f"**Anything Specific You’d Like to Know?**  \n"
        f"Is there a particular aspect of {name} you’re curious about, such as its effectiveness for specific skin types, comparison with other UBIK products, or tips for incorporating it into your skincare routine? Let me know! 🤔"
    )
    return response

# Conversational responses for common phrases
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
    'hello': 'Hi! Ready to chat about UBIK Solutions or our products? 😄',
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

# Serve UI pages
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

# Quiz APIs
@app.route('/api/questions', methods=['GET'])
def get_questions():
    try:
        prompt = f"""
        You are an official representative of UBIK Solutions. Generate 5 open-ended quiz questions based on the following data about UBIK Solutions and its products. Focus on services, mission, processes, or product details (e.g., Anti-Acne, Anti-Ageing products). Each question should start with 'How', 'What', or 'Why'. Return the questions as a JSON array, e.g., ["question1", "question2", ...]. Do not mention 'customer reviews' or speculative information; base questions solely on the provided data.

        General Data:
        {json.dumps(ubik_data, indent=2)}

        Product Data:
        {json.dumps(ubik_product_details, indent=2)}
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        print("Quiz API response:", response.text)
        questions = json.loads(response.text.strip()) if response.text.strip().startswith('[') else [
            "How does UBIK Solutions leverage AI for dermatology applications?",
            "What are the key services offered by UBIK Solutions?",
            "Why is UBIK Solutions' approach to data processing unique?",
            "What are the main ingredients in UBIK Solutions' Anti-Acne products?",
            "How do UBIK Solutions' Anti-Ageing products benefit the skin?"
        ]
        return jsonify(questions[:5])
    except Exception as e:
        print("Error generating questions:", e)
        return jsonify([
            "How does UBIK Solutions leverage AI for dermatology applications?",
            "What are the key services offered by UBIK Solutions?",
            "Why is UBIK Solutions' approach to data processing unique?",
            "What are the main ingredients in UBIK Solutions' Anti-Acne products?",
            "How do UBIK Solutions' Anti-Ageing products benefit the skin?"
        ])

@app.route('/api/evaluate', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    question = data.get('question', '')
    answer = data.get('answer', '')
    
    # Log input for debugging
    print(f"Evaluating question: {question}")
    print(f"Answer: {answer}")
    
    # Check if JSON data is available
    if not text_chunks or not product_text_chunks:
        print("JSON data not loaded")
        return jsonify({
            'feedback': 'Error: JSON data not available. Please ensure ubik_data.json and ubik_product_details.json are loaded.',
            'score': 0.0
        })

    # Handle skipped answers
    if answer == 'SKIPPED':
        print("Answer skipped")
        return jsonify({
            'feedback': 'No answer provided (skipped). Please provide an answer to demonstrate your knowledge.',
            'score': 0.0
        })

    # Determine question type for specific feedback
    is_product_question = is_product_related(question)
    is_anti_ageing = 'anti-ageing' in question.lower() or 'aging' in question.lower()
    is_anti_acne = 'anti-acne' in question.lower() or 'ingredients' in question.lower()
    is_service_question = not is_product_question

    # Define feedback templates based on question type
    feedback_templates = {
        'service_correct': "Your answer is mostly correct and includes relevant details about UBIK Solutions' services. To improve, add specifics like how iDoc Academy supports dermatologists or AI-driven dermatology applications.",
        'service_partial': "Your answer contains some relevant points but is incomplete. Include more details about UBIK’s mission, services, or iDoc Academy to strengthen your response.",
        'service_incorrect': "Your answer does not fully address the question. Review UBIK Solutions’ mission and services, such as iDoc Academy or AI-driven dermatology solutions, for more information.",
        'anti_acne_correct': "Your answer is mostly correct and includes relevant details about Anti-Acne products. To improve, specify ingredients like Salicylic Acid or Niacinamide used in products like Sebogel.",
        'anti_acne_partial': "Your answer mentions relevant aspects of Anti-Acne products but is incomplete. Include specific ingredients or benefits, such as those in Sebogel, to strengthen your response.",
        'anti_acne_incorrect': "Your answer does not fully address the question. Review product details like Anti-Acne ingredients (e.g., Salicylic Acid, Niacinamide in Sebogel) for more information.",
        'anti_ageing_correct': "Your answer is mostly correct and includes relevant details about Anti-Ageing products. To improve, specify benefits or ingredients like Encapsulated Retinol in Reti K Cream.",
        'anti_ageing_partial': "Your answer mentions relevant aspects of Anti-Ageing products but is incomplete. Include specific benefits or ingredients, such as those in Reti K Cream, to strengthen your response.",
        'anti_ageing_incorrect': "Your answer does not fully address the question. Review product details like Anti-Ageing ingredients (e.g., Encapsulated Retinol in Reti K Cream) for more information."
    }

    # Simplified prompt for evaluation
    prompt = f"""
    You are an official representative of UBIK Solutions. Evaluate the following answer to the question: '{question}'
    Answer: '{answer}'
    
    Provide feedback on the correctness and completeness of the answer based solely on the provided data. Do not mention 'customer reviews' or speculative information. Return a JSON object with 'feedback' (string) and 'score' (float, 0 to 1). For correct or partially correct answers, provide specific feedback referencing the relevant data (e.g., iDoc Academy, product ingredients). For incorrect or incomplete answers, suggest improvements tied to the data. Assign a partial score (0.5–0.8) if the answer contains relevant keywords, even if incomplete.

    General Data:
    {json.dumps(ubik_data, indent=2)}

    Product Data:
    {json.dumps(ubik_product_details, indent=2)}
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        print("Gemini API raw response:", response.text)
        
        # Attempt to parse response as JSON
        try:
            evaluation = json.loads(response.text.strip())
            if not isinstance(evaluation, dict) or 'feedback' not in evaluation or 'score' not in evaluation:
                raise ValueError("Invalid response format")
            # Validate score
            evaluation['score'] = float(evaluation['score'])
            if not 0 <= evaluation['score'] <= 1:
                evaluation['score'] = 0.0
                evaluation['feedback'] = "Invalid score returned; please try again."
        except (json.JSONDecodeError, ValueError) as e:
            print("Error parsing Gemini response:", e)
            # Fallback evaluation logic
            relevant_chunks = product_text_chunks if is_product_question else text_chunks
            relevant_chunk = find_relevant_chunk(question, relevant_chunks)
            
            # Calculate similarity
            vectorizer = TfidfVectorizer().fit_transform([answer.lower(), relevant_chunk.lower()])
            similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()[0]
            score = min(max(similarity * 1.5, 0.0), 1.0)  # Scale similarity for better scoring
            
            # Boost score for relevant keywords
            keywords = ['idoc', 'academy', 'ai', 'dermatology', 'anti-acne', 'anti-ageing', 'salicylic', 'niacinamide', 'retinol', 'acid']
            answer_lower = answer.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in answer_lower)
            if keyword_matches > 0:
                score = min(score + (keyword_matches * 0.25), 1.0)  # Increased boost for keywords
            
            print(f"Similarity score: {score}, Keyword matches: {keyword_matches}")
            
            # Generate feedback based on question type and score
            if is_anti_ageing:
                feedback = feedback_templates['anti_ageing_correct'] if score >= 0.8 else feedback_templates['anti_ageing_partial'] if score >= 0.5 else feedback_templates['anti_ageing_incorrect']
            elif is_anti_acne:
                feedback = feedback_templates['anti_acne_correct'] if score >= 0.8 else feedback_templates['anti_acne_partial'] if score >= 0.5 else feedback_templates['anti_acne_incorrect']
            else:
                feedback = feedback_templates['service_correct'] if score >= 0.8 else feedback_templates['service_partial'] if score >= 0.5 else feedback_templates['service_incorrect']
            
            evaluation = {'feedback': feedback, 'score': score}
        
        print("Evaluation result:", evaluation)
        return jsonify(evaluation)
    except Exception as e:
        print("Evaluation error:", str(e))
        # Enhanced fallback for any errors
        relevant_chunks = product_text_chunks if is_product_question else text_chunks
        relevant_chunk = find_relevant_chunk(question, relevant_chunks)
        
        vectorizer = TfidfVectorizer().fit_transform([answer.lower(), relevant_chunk.lower()])
        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()[0]
        score = min(max(similarity * 1.5, 0.0), 1.0)
        
        # Boost score for relevant keywords
        keywords = ['idoc', 'academy', 'ai', 'dermatology', 'anti-acne', 'anti-ageing', 'salicylic', 'niacinamide', 'retinol', 'acid']
        answer_lower = answer.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in answer_lower)
        if keyword_matches > 0:
            score = min(score + (keyword_matches * 0.25), 1.0)
        
        print(f"Fallback similarity score: {score}, Keyword matches: {keyword_matches}")
        
        # Generate feedback based on question type
        if is_anti_ageing:
            feedback = feedback_templates['anti_ageing_correct'] if score >= 0.8 else feedback_templates['anti_ageing_partial'] if score >= 0.5 else feedback_templates['anti_ageing_incorrect']
        elif is_anti_acne:
            feedback = feedback_templates['anti_acne_correct'] if score >= 0.8 else feedback_templates['anti_acne_partial'] if score >= 0.5 else feedback_templates['anti_acne_incorrect']
        else:
            feedback = feedback_templates['service_correct'] if score >= 0.8 else feedback_templates['service_partial'] if score >= 0.5 else feedback_templates['service_incorrect']
        
        return jsonify({
            'feedback': feedback,
            'score': score
        })

# Chatbot API
@app.route('/api/chat', methods=['POST'])
def chatbot_reply():
    data = request.get_json()
    msg = data.get("message", "")
    user_message = msg["text"] if isinstance(msg, dict) and "text" in msg else msg
    print("User message:", user_message)

    # Check for exact conversational phrases
    user_message_lower = user_message.lower().strip()
    if user_message_lower in conversational_phrases:
        print(f"Matched conversational phrase: {user_message_lower}")
        return jsonify({"reply": conversational_phrases[user_message_lower]})

    # Fuzzy matching for conversational phrases
    for phrase in conversational_phrases:
        if SequenceMatcher(None, user_message_lower, phrase).ratio() >= 0.8:
            print(f"Fuzzy matched conversational phrase: {user_message_lower} -> {phrase}")
            return jsonify({"reply": conversational_phrases[phrase]})

    # Check if it's UBIK-related
    if is_related_to_ubik(user_message):
        normalized_message = normalize_to_ubik(user_message)
        # Handle product-related queries
        if is_product_related(normalized_message):
            # Try to find a specific product match
            product, category = find_product_by_name(normalized_message, ubik_product_details)
            if product:
                reply = format_product_details(product, category, normalized_message)
            else:
                context = find_relevant_chunk(normalized_message, product_text_chunks)
                prompt = f"""
                You are an official representative of UBIK Solutions. Respond to the following product-related question based solely on the provided data. Do not mention 'customer reviews' or speculative information. Respond in a short, clear, and professional manner, using 1 or 2 relevant emojis. If no relevant information is found, say: "I couldn't find specific details on that. Please ask another question about UBIK Solutions' products! 😊"

                Product Data:
                {json.dumps(ubik_product_details, indent=2)}

                Question: {normalized_message}
                """
                try:
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)
                    print("Gemini API response:", response.text)
                    reply = response.text.strip()
                    if not reply or "sorry" in reply.lower() or "cannot" in reply.lower() or "not enough" in reply.lower():
                        reply = "I couldn't find specific details on that. Please ask another question about UBIK Solutions' products! 😊"
                except Exception as e:
                    print("Chat Error:", e)
                    reply = "I couldn't find specific details on that. Please ask another question about UBIK Solutions' products! 😊"
        else:
            context = find_relevant_chunk(normalized_message, text_chunks)
            prompt = f"""
            You are an official representative of UBIK Solutions. Respond to the following question based solely on the provided data. Do not mention 'customer reviews' or speculative information. Respond in a short, clear, and professional manner, using 1 or 2 relevant emojis. If no relevant information is found, say: "I couldn't find specific details on that. Please ask another question about UBIK Solutions! 😊"

            General Data:
            {json.dumps(ubik_data, indent=2)}

            Question: {normalized_message}
            """
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                print("Gemini API response:", response.text)
                reply = response.text.strip()
                if not reply or "sorry" in reply.lower() or "cannot" in reply.lower() or "not enough" in reply.lower():
                    reply = "I couldn't find specific details on that. Please ask another question about UBIK Solutions! 😊"
            except Exception as e:
                print("Chat Error:", e)
                reply = "I couldn't find specific details on that. Please ask another question about UBIK Solutions! 😊"
    else:
        # Free-flowing conversational response
        prompt = f"""
        You are a friendly, conversational chatbot for UBIK Solutions. Respond to the following user input in a natural, engaging way, as if continuing a casual conversation. Keep the tone professional but friendly, and use 1 or 2 relevant emojis. Do not mention 'customer reviews' or speculative information.

        User input: {user_message}
        """
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            print("Gemini API response:", response.text)
            reply = response.text.strip()
            if not reply or "sorry" in reply.lower() or "cannot" in reply.lower() or "not enough" in reply.lower():
                reply = "Hmm, not sure about that one! What's next? 😄"
        except Exception as e:
            print("Chat Error:", e)
            reply = "Hmm, not sure about that one! What's next? 😄"

    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)