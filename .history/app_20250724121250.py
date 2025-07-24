import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Allow all origins

# Load your dataset
dataset = pd.read_csv("combined_knowledge_base.csv")
dataset.columns = dataset.columns.str.strip()

# Ensure 'Question' and 'Answer' columns exist
if 'Question' not in dataset.columns or 'Answer' not in dataset.columns:
    raise ValueError("CSV must contain 'Question' and 'Answer' columns")

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 2]
    return ' '.join(tokens)

# Preprocess questions
dataset["Cleaned_Question"] = dataset["Question"].apply(clean_text)

# Vectorize questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(dataset["Cleaned_Question"])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('message')

    if not user_input:
        return jsonify({'response': 'Please enter a message.'}), 400

    cleaned_input = clean_text(user_input)
    user_vector = vectorizer.transform([cleaned_input])

    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_idx = similarities.argmax()

    response = dataset.iloc[best_match_idx]['Answer']
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=800)
