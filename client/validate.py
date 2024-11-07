import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer
import re
from collections import Counter
from fedn.utils.helpers.helpers import save_metrics
from model import load_model_train


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_model(model_load_path):
    model_load_path = f'bert.pth'
    loaded_model = torch.load(model_load_path, map_location=device)
    loaded_model.eval()
    return loaded_model

def validate(in_model_path, out_json_path, data_path=None):

    file_path = 'data/validate.csv'
    df = pd.read_csv(file_path)

    texts = []
    for _, row in df.iterrows():
        prompt = row['data']
        text = prompt
        texts.append(text)
    print(f"Loaded {len(texts)} text samples from CSV.")

    word_to_idx, tokenized_texts = build_vocab(texts)
    sequences = encode_sequences(tokenized_texts, word_to_idx, seq_length=4)

    train_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
    train_dataset = TextDataset(train_sequences)
    test_dataset = TextDataset(test_sequences)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = load_model_train(in_model_path)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    training_loss, training_accuracy = validate_model(model, train_loader, criterion)
    test_loss, test_accuracy = validate_model(model, test_loader, criterion)

    report = {
        "training_loss": training_loss.item(),
        "training_accuracy": training_accuracy.item(),
        "test_loss": test_loss.item(),
        "test_accuracy": test_accuracy.item()
    }

    save_metrics(report, out_json_path)

def validate_model(model, loader, criterion):
    model.eval()  
    total_loss = 0
    total_correct = 0
    with torch.no_grad():  
        for sequences, targets in loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()

    avg_loss = torch.tensor(total_loss / len(loader))
    avg_acc = torch.tensor(total_correct / len(loader.dataset))
    return avg_loss, avg_acc

class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, target = self.sequences[idx]
        return torch.tensor(sequence), torch.tensor(target)

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = re.sub(r'[.,:]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '[MASK]', text)
    return text

def tokenize_text(text):
    return clean_text(text).split()

def build_vocab(texts):
    tokenized_texts = [tokenize_text(text) for text in texts]
    all_tokens = [token for text in tokenized_texts for token in text]
    token_counts = Counter(all_tokens)
    sorted_tokens = sorted(token_counts, key=token_counts.get, reverse=True)
    
    bert_tokens = tokenizer.vocab.keys()
    word_to_idx = {token: tokenizer.vocab[token] for token in bert_tokens}
    return word_to_idx, tokenized_texts

def encode_sequences(tokenized_texts, word_to_idx, seq_length=6):
    sequences = []
    for tokens in tokenized_texts:
        if len(tokens) < seq_length:
            continue
        for i in range(seq_length, len(tokens)):
            seq = tokens[i-seq_length:i]
            target = tokens[i]
            encoded_seq = [word_to_idx.get(word, word_to_idx['[UNK]']) for word in seq]
            encoded_target = word_to_idx.get(target, word_to_idx['[UNK]'])
            sequences.append((encoded_seq, encoded_target))
    return sequences

if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])