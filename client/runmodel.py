import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import csv
import subprocess
import threading
from fedn import APIClient
import torch.nn as nn
from transformers import BertTokenizer
from model import load_model_inference

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
client = APIClient(host="fedn.scaleoutsystems.com/ai-sweden-young-talent-2024-vua-fedn-reducer", token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzMzNDc5NDEwLCJpYXQiOjE3MzA4ODc0MTAsImp0aSI6IjI0YTVjMzE0NjQ2OTQxZWY4YWJiZjJkMjBjYWViNjYxIiwidXNlcl9pZCI6NjEyLCJjcmVhdG9yIjoibWFra2EiLCJyb2xlIjoiYWRtaW4iLCJwcm9qZWN0X3NsdWciOiJhaS1zd2VkZW4teW91bmctdGFsZW50LTIwMjQtdnVhIn0.EajGSiKsQt9gMEPb9b2vnqa0A9zZlkdtou8tRjCiyjo", secure=True, verify=True)


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
        lstm_out = lstm_out[:, -1, :]
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
def predict_next_word(model, sequence, idx_to_word, word_to_idx, seq_length=6):
    model.eval()
    
    padded_sequence = sequence[-seq_length:]
    print(word_to_idx)
    padded_sequence += [word_to_idx['[PAD]']] * (seq_length - len(padded_sequence))
    
    sequence_tensor = torch.tensor(padded_sequence).unsqueeze(0).to(device)
    lengths = torch.tensor([min(len(sequence), seq_length)], dtype=torch.int64).cpu()
    
    with torch.no_grad():
        output = model(sequence_tensor, lengths)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = torch.argmax(output, dim=1).item()
        top_indices = torch.topk(probabilities, k=3, dim=1).indices[0]
        top_probs = torch.topk(probabilities, k=3, dim=1).values[0]
        top_words = [idx_to_word[idx.item()] for idx in top_indices]
    return top_words[0], top_words, [f"{prob.item():.3f}" for prob in top_probs]
    with torch.no_grad():
        output = model(sequence)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_indices = torch.topk(probabilities, k=3, dim=1).indices[0]
        top_probs = torch.topk(probabilities, k=3, dim=1).values[0]
        top_words = [idx_to_word[idx.item()] for idx in top_indices]
    return top_words[0], top_words, [f"{prob.item():.3f}" for prob in top_probs]

def load_model():
    global loaded_model
    model_load_path = 'bert3.npz'
    #client.download_model("e0415099-5474-4910-8494-cb5f995eb9e4", path=model_load_path)
    loaded_model = load_model_inference(model_load_path)

def update_model():
    global loaded_model
    model_load_path = 'bert.npz'
    current_models = client.get_models()
    print(current_models)
    latest = current_models['result'][0]['model']
    client.download_model(latest, path=model_load_path)
    loaded_model = load_model_inference(model_load_path)
    return True
    
def generate_text(seed_text, num_words=10):

    encoded = tokenizer.encode(seed_text.lower(), add_special_tokens=False)
    indata = encoded
    

    idx_to_word = {idx: word for word, idx in tokenizer.vocab.items()}

    top_word_id, top_3_ids, prob = predict_next_word(loaded_model, indata,  idx_to_word, tokenizer.vocab)
    
    result_text = seed_text + " " + top_word_id

   
    
    return result_text, top_word_id, top_3_ids, prob


@app.route('/generate', methods=['POST', 'OPTIONS', 'GET'])
def autocomplete_word():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
    else:
        data = request.json
        seed_text = data.get('seed_text', '')
        num_words = data.get('num_words', 1)
        
        generated_text, top_word, top_3, prob = generate_text(seed_text, num_words)
        
        with open('data/history.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([seed_text])
        
        response = jsonify({
            'generated_text': generated_text,
            'top_3': top_3,
            'prob': prob
        })
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

@app.route('/update-model', methods=['POST'])
def update_model_endpoint():
    try:
        success = update_model()
        return jsonify({'success': success, 'message': 'Model updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def run_node_script():
    subprocess.run(['node', 'svelte/build/index.js'])

    

if __name__ == '__main__':
    
    load_model()
    
    node_thread = threading.Thread(target=run_node_script)

    node_thread.start()
 
    
    app.run(debug=True, host='0.0.0.0', port=5000)