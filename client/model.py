import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import csv
import subprocess
import threading

import torch.nn as nn

class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, repetition_penalty=1.0):
        super(NextWordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.repetition_penalty = repetition_penalty
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last LSTM cell
        out = self.fc(lstm_out)
        
        if self.repetition_penalty != 1.0:
            for i in range(out.size(0)):
                for token in x[i]:
                    out[i, token] /= self.repetition_penalty
        
        return out

class NextWordGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout_prob=0.3, repetition_penalty=1.0):
        super(NextWordGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.repetition_penalty = repetition_penalty
    
    def forward(self, x):
        x = self.embedding(x)
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out[:, -1, :]) 
        out = self.fc(gru_out)
        
        if self.repetition_penalty != 1.0:
            for i in range(out.size(0)):
                for token in x[i]:
                    out[i, token] /= self.repetition_penalty
        
        return out

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
    model_load_path = f'model.pth'
    vocab_load_path = f'vocabulary.json'

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
        
        # Save the input text in the data directory as CSV
        with open('data/history.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([seed_text, generated_text[len(seed_text):].strip()])
        
        response = jsonify({'generated_text': generated_text})
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

def run_bash_script():
    subprocess.run(['node', 'svelte/build/index.js'])

model_type = 'gru_onehot'
load_model(model_type)

if __name__ == '__main__':
    # Start the bash script in a separate thread
    bash_thread = threading.Thread(target=run_bash_script)
    bash_thread.start()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)