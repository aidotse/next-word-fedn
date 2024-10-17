import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

from train.models import NextWordLSTM, NextWordGRU

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_next_word(model, sequence, idx_to_word):
    model.eval()
    sequence = torch.tensor(sequence).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(sequence)
        predicted_idx = torch.argmax(output, dim=1).item()
    return idx_to_word[predicted_idx]

def load_model(model_type):
    global loaded_word_to_idx, loaded_model
    model_load_path = f'train/{model_type}/model.pth'
    vocab_load_path = f'train/{model_type}/vocabulary.json'

    with open(vocab_load_path, 'r') as f:
        loaded_word_to_idx = json.load(f)

    loaded_model = torch.load(model_load_path, map_location=device)
    loaded_model.eval()

@app.route('/model-type', methods=['POST'])
def choose_model():
    model_type = request.json.get('model_type')
    
    global loaded_word_to_idx, loaded_model
    model_load_path = f'train/{model_type}/model.pth'
    vocab_load_path = f'train/{model_type}/vocabulary.json'

    with open(vocab_load_path, 'r') as f:
        loaded_word_to_idx = json.load(f)

    loaded_model = torch.load(model_load_path, map_location=device)
    loaded_model.eval()
    
    print(f"Model {model_type} loaded successfully")
    
    return jsonify({'message': 'Model loaded successfully'})

def generate_text(seed_text, num_words=10):
    words = seed_text.split()
    indata = [loaded_word_to_idx.get(word.lower(), loaded_word_to_idx.get('<UNK>', 0)) for word in words]
    words.append(predict_next_word(loaded_model, indata, {v: k for k, v in loaded_word_to_idx.items()}))

    return ' '.join(words)


@app.route('/generate', methods=['POST', 'OPTIONS', 'GET'])
def autocomplete_word():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
    else:
        data = request.json
        seed_text = data.get('seed_text', '')
        num_words = data.get('num_words', 1) # does nothing now
        
        generated_text = generate_text(seed_text, num_words)
        
        response = jsonify({'generated_text': generated_text})
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

model_type = 'gru_onehot'
load_model(model_type)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
