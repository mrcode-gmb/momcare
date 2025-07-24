import os
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import joblib
import torch
import re
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("momcare_model")


app = Flask(__name__)
CORS(app)  # This will allow all origins by default
# Load your model
model = joblib.load("momcare.pkl")  # Use model path or name if it's a SentenceTransformer

# Load dataset and fix column names
dataset = pd.read_csv("combined_knowledge_base.csv")
dataset.columns = dataset.columns.str.strip()  # << Fix column name issue

# Ensure 'Question' and 'Response' columns exist
if 'Question' not in dataset.columns or 'Answer' not in dataset.columns:
    raise ValueError("CSV must contain 'Question' and 'Response' columns")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 2]
    return ' '.join(tokens)

# Preprocess questions
dataset["Cleaned_Question"] = dataset["Question"].apply(clean_text)
question_embeddings = model.encode(dataset["Cleaned_Question"].tolist(), convert_to_tensor=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('input')

    if not user_input:
        return jsonify({'response': 'I can oo.'}), 400

    cleaned_input = clean_text(user_input)
    user_embedding = model.encode(cleaned_input, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.cos_sim(user_embedding, question_embeddings)
    best_match_idx = torch.argmax(similarities).item()

    response = dataset.iloc[best_match_idx]['Answer']
    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # use $PORT or default to 5000 locally
    app.run(host="0.0.0.0", port=port)