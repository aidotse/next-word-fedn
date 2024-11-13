import os
from math import floor
from torch.utils.data import DataLoader, Dataset
import torch
import requests
import pandas as pd
from transformers import BertTokenizer
import re
from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(out_dir="data"):
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Load CSV data
    file_path = f'{out_dir}/history.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = re.sub(r'[.,:]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '[MASK]', text)
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

    for i in range(max_seq_length, len(encoded_tokens)):
        sequence = encoded_tokens[i - max_seq_length:i]
        target = encoded_tokens[i]
        
        if len(sequence) < max_seq_length:
            sequence += [padding_idx] * (max_seq_length - len(sequence))
        
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

class TextDataset(Dataset):
    def __init__(self, sequences, lengths, targets):
        self.sequences = sequences
        self.lengths = lengths
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.lengths[idx]), torch.tensor(self.targets[idx])

def load_data(data_path=None, is_train=True):

    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/history.csv")

    df = pd.read_csv(data_path)
    texts = df['data'].tolist()
    print(f"loaded data from file: {data_path} ")
    print(f"Number of texts loaded: {len(texts)}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    word_to_idx, tokenized_texts = build_vocab(texts, tokenizer)
    sequences, lengths, targets = encode_sequences(tokenized_texts, word_to_idx)
    
    if not sequences:
        raise ValueError("No sequences were generated from the input texts")

    train_sequences, val_sequences, train_lengths, val_lengths, train_targets, val_targets = train_test_split(
        sequences, lengths, targets, test_size=0.2, random_state=42)

    train_dataset = TextDataset(train_sequences, train_lengths, train_targets)
    val_dataset = TextDataset(val_sequences, val_lengths, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader if is_train else val_loader

def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return result

def process_sequence(tokens, word_to_idx, max_seq_length, padding_idx, unk_idx):
    if not tokens:
        return [], [], []
        
    encoded_tokens = [word_to_idx.get(word, unk_idx) for word in tokens]
    sequences = []
    lengths = []
    targets = []

    # Changed from max_seq_length to 1 to allow shorter sequences
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

if __name__ == "__main__":
    load_data('./data/history.csv', True)