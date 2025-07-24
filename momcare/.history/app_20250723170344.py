from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import joblib
import torch
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins by default

# Load model (INT8 quantized or regular)
model = joblib.load("momcare.pkl")  # Pre-trained SentenceTransformer

# Load dataset
dataset = pd.read_csv("combined_knowledge_base.csv")
dataset.columns = dataset.columns.str.strip()

# Ensure required columns exist
if 'Question' not in dataset.columns or 'Answer' not in dataset.columns:
    raise ValueError("CSV must contain 'Question' and 'Answer' columns")

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 2]
    return ' '.join(tokens)

# Clean and embed all questions
dataset["Cleaned_Question"] = dataset["Question"].apply(clean_text)
question_embeddings = model.encode(dataset["Cleaned_Question"].tolist(), convert_to_tensor=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('input')

    if not user_input:
        return jsonify({'response': "No input received. Please enter a question."}), 400

    cleaned_input = clean_text(user_input)
    user_embedding = model.encode(cleaned_input, convert_to_tensor=True)

    similarities = util.cos_sim(user_embedding, question_embeddings)
    best_match_idx = torch.argmax(similarities).item()
    response = dataset.iloc[best_match_idx]['Answer']

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
