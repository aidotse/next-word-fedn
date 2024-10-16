import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from torch.utils.data import Dataset
from torch import nn
import json

app = Flask(__name__)
CORS(app)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_next_word(model, sequence, word_to_idx, idx_to_word):
    model.eval()
    sequence = torch.tensor(sequence).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(sequence)
        predicted_idx = torch.argmax(output, dim=1).item()
    return idx_to_word[predicted_idx]


# Load the saved model and vocabulary
model_load_path = 'train/gru_onehot/model.pth'
vocab_load_path = 'train/gru_onehot/vocabulary.json'


# Create custom Dataset
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, target = self.sequences[idx]
        return torch.tensor(sequence), torch.tensor(target)

# Define GRU model
class NextWordGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(NextWordGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]  # Take the output of the last GRU cell
        out = self.fc(gru_out)
        return out

# Load the vocabulary
with open(vocab_load_path, 'r') as f:
    loaded_word_to_idx = json.load(f)

# Define model parameters
embed_size = 128
hidden_size = 256
num_layers = 2

# Create a new model instance with the same architecture
loaded_model = NextWordGRU(len(loaded_word_to_idx), embed_size, hidden_size, num_layers).to(device)

# Load the saved state dict
loaded_model.load_state_dict(torch.load(model_load_path, map_location=device))
loaded_model.eval()  # Set the model to evaluation mode

print("Model and vocabulary loaded successfully.")

# Function to generate text
def generate_text(seed_text, num_words=10):
    words = seed_text.split()
    for _ in range(num_words):
        sequence = [loaded_word_to_idx.get(word, loaded_word_to_idx.get('<UNK>', 0)) for word in words[-3:]]
        next_word = predict_next_word(loaded_model, sequence, loaded_word_to_idx, {v: k for k, v in loaded_word_to_idx.items()})
        words.append(next_word)
    return ' '.join(words)

# Test the loaded model
seed_text = "The quick brown"
generated_text = generate_text(seed_text)
print(f"Generated text: {generated_text}")

@app.route('/generate', methods=['POST', 'OPTIONS', 'GET'])
def autocomplete_word():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
    else:
        data = request.json
        seed_text = data.get('seed_text', '')
        num_words = data.get('num_words', 1)
        
        generated_text = generate_text(seed_text, num_words)
        
        response = jsonify({'generated_text': generated_text})
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
