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
    model_load_path = f'bert4.pth'
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

    word_to_idx, tokenized_texts = build_vocab(texts, tokenizer)
    sequences, lengths, targets = encode_sequences(tokenized_texts, word_to_idx)

    train_sequences, val_sequences, train_lengths, val_lengths, train_targets, val_targets = train_test_split(
    sequences, lengths, targets, test_size=0.2, random_state=42)

    train_dataset = TextDataset(train_sequences, train_lengths, train_targets)
    val_dataset = TextDataset(val_sequences, val_lengths, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    model = load_model_train(in_model_path)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    training_loss, training_accuracy = validate_model(model, train_loader, criterion)
    test_loss, test_accuracy = validate_model(model, val_loader, criterion)

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
         for b, (sequences, lengths, targets) in enumerate(loader):
            sequences = sequences.to(device)
            lengths = lengths.cpu()  # Lengths need to be on CPU for pack_padded_sequence
            targets = targets.to(device)
            outputs = model(sequences, lengths)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()

    avg_loss = torch.tensor(total_loss / len(loader))
    avg_acc = torch.tensor(total_correct / len(loader.dataset))
    return avg_loss, avg_acc

class TextDataset(Dataset):
    def __init__(self, sequences, lengths, targets):
        self.sequences = sequences
        self.lengths = lengths
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.lengths[idx]), torch.tensor(self.targets[idx])

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', 'name', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
def tokenize_text(text, tokenizer):
    cleaned_text = clean_text(text)
    tokens = tokenizer.tokenize(cleaned_text)
    return tokens

def build_vocab(texts, tokenizer):
    tokenized_texts = [tokenize_text(text, tokenizer) for text in texts]
    
    all_tokens = [token for text in tokenized_texts for token in text]
    token_counts = Counter(all_tokens)
    sorted_tokens = sorted(token_counts, key=token_counts.get, reverse=True)
    if sorted_tokens:
        print(f"Most common tokens: {sorted_tokens[:1]}")
    else:
        print("No tokens found")
    
    word_to_idx = tokenizer.vocab
    return word_to_idx, tokenized_texts


def process_sequence(tokens, word_to_idx, max_seq_length, padding_idx, unk_idx):
    if not tokens:
        return [], [], []
        
    encoded_tokens = [word_to_idx.get(word, unk_idx) for word in tokens]
    sequences = []
    lengths = []
    targets = []

    # Process from the first token to allow shorter sequences
    for i in range(1, len(encoded_tokens)):
        # Take the previous tokens as sequence, up to max_seq_length
        sequence = encoded_tokens[max(0, i - max_seq_length):i]
        target = encoded_tokens[i]
        
        # Pad sequence if needed
        if len(sequence) < max_seq_length:
            padding = [padding_idx] * (max_seq_length - len(sequence))
            sequence = padding + sequence
        
        sequences.append(sequence)
        lengths.append(min(len(sequence), max_seq_length))
        targets.append(target)
        
    return sequences, lengths, targets


def encode_sequences(tokenized_texts, word_to_idx, max_seq_length=25, padding_token='[PAD]', unk_token='[UNK]'):
    word_to_idx.setdefault(padding_token, len(word_to_idx))
    word_to_idx.setdefault(unk_token, len(word_to_idx))
    
    padding_idx = word_to_idx[padding_token]
    unk_idx = word_to_idx[unk_token]

    sequences, lengths, targets = [], [], []
    for text in tokenized_texts:
        seq, length, target = process_sequence(text, word_to_idx, max_seq_length, padding_idx, unk_idx)
        sequences.extend(seq)
        lengths.extend(length)
        targets.extend(target)

    if sequences:
        print(f"Number of sequences generated: {len(sequences)}")
        print(f"Sample sequence: {sequences[0]}")
        print(f"Sample target: {targets[0]}")
    else:
        print("No sequences generated")
        
    return sequences, lengths, targets
if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])