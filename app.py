from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk
import json

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load your custom dataset from a text file
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n\n')
    dataset = []
    for line in lines:
        if line.strip():
            try:
                question_part, response_part = line.split('\nResponse: ')
                question = question_part.replace('Question: ', '').strip()
                response = response_part.strip()
                dataset.append({'question': question, 'response': response})
            except ValueError as e:
                print(f"Skipping line due to error: {e}")
                continue
    return dataset

dataset = load_dataset("data.txt")

# Extract user questions and model responses from the dataset
user_questions = [item['question'] for item in dataset]
model_responses = [item['response'] for item in dataset]

if not user_questions or not model_responses:
    raise ValueError("No valid user questions and model responses found in the dataset.")

vectorizer = TfidfVectorizer().fit(user_questions)
vectors = vectorizer.transform(user_questions).toarray()

lemmatizer = WordNetLemmatizer()

def preprocess_input(user_input):
    tokens = user_input.strip().lower().split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

greeting_messages = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings", "howdy"
]

def is_greeting(message):
    preprocessed_message = preprocess_input(message)
    return preprocessed_message in greeting_messages

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    if is_greeting(user_input):
        return jsonify({"response": "Hello! How can I assist you today?"})

    preprocessed_input = preprocess_input(user_input)
    user_vector = vectorizer.transform([preprocessed_input]).toarray()
    similarities = cosine_similarity(user_vector, vectors)
    most_similar_index = similarities.argmax()
    most_similar_score = similarities[0, most_similar_index]
    similarity_threshold = 0.5

    if most_similar_score < similarity_threshold:
        return jsonify({"response": "I'm sorry, I don't have relevant information for your query."})

    response = model_responses[most_similar_index]
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
