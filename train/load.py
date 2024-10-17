import torch
import json

from models import NextWordGRU, NextWordLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_save_path = 'train/gru_onehot/model.pth'
vocab_save_path = 'train/gru_onehot/vocabulary.json'

with open(vocab_save_path, 'r') as f:
    word_to_idx = json.load(f)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

model = torch.load(model_save_path)

def predict_next_word(model, sequence, word_to_idx, idx_to_word):
    model.eval()
    sequence = torch.tensor(sequence).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(sequence)
        predicted_idx = torch.argmax(output, dim=1).item()
    return idx_to_word[predicted_idx]

print("Model and vocabulary loaded successfully.")

indata = [word_to_idx[words] for words in "hello how are".split()]
print(predict_next_word(model, indata, word_to_idx, idx_to_word))