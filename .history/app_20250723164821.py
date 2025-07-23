from flask import Flask, render_template, request, jsonify
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import pandas as pd
import torch
import re
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

# Load ONNX quantized model
model_path = "onnx_model"  # folder where your quantized ONNX model is stored
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ORTModelForFeatureExtraction.from_pretrained(model_path)

# Load dataset
dataset = pd.read_csv("combined_knowledge_base.csv")
dataset.columns = dataset.columns.str.strip()

if 'Question' not in dataset.columns or 'Answer' not in dataset.columns:
    raise ValueError("CSV must contain 'Question' and 'Answer' columns")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 2]
    return ' '.join(tokens)

# Helper: Get embeddings using ONNX model
def get_embedding(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Precompute question embeddings
dataset["Cleaned_Question"] = dataset["Question"].apply(clean_text)
question_embeddings = get_embedding(dataset["Cleaned_Question"].tolist())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('input')
    cleaned_input = clean_text(user_input)
    user_embedding = get_embedding([cleaned_input])  # list input

    # Compute cosine similarity
    sims = cosine_similarity(user_embedding, question_embeddings)[0]
    best_idx = np.argmax(sims)

    response = dataset.iloc[best_idx]['Answer']
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
