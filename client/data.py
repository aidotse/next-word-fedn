import os
from math import floor

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
    text = text.lower().strip()
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = re.sub(r'[.,:]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '[MASK]', text)
    return text

def tokenize_text(text):
    return clean_text(text).split()

def build_vocab(texts, tokenizer):
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

class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, target = self.sequences[idx]
        return torch.tensor(sequence), torch.tensor(target)

def load_data(data_path, is_train=True):

    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/history.csv")

    df = pd.read_csv(data_path)
    texts = [row['prompt'] for _, row in df.iterrows()]
    print(f"loaded data from file: {data_path} ")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    word_to_idx, tokenized_texts = build_vocab(texts, tokenizer)
    sequences = encode_sequences(tokenized_texts, word_to_idx, seq_length=4)
    
    # Split into train/test
    train_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
    
    if is_train:
        dataset = TextDataset(train_sequences)
    else:
        dataset = TextDataset(test_sequences)
        
    X = torch.stack([item[0] for item in dataset])
    y = torch.stack([item[1] for item in dataset])
    
    return X, y

def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return result

def split(out_dir="data"):
    n_splits = int(os.environ.get("FEDN_NUM_DATA_SPLITS", 2))

    if not os.path.exists(f"{out_dir}/clients"):
        os.mkdir(f"{out_dir}/clients")

    X_train, y_train = load_data(None, is_train=True)
    X_test, y_test = load_data(None, is_train=False)

    data = {
        "x_train": splitset(X_train, n_splits),
        "y_train": splitset(y_train, n_splits),
        "x_test": splitset(X_test, n_splits),
        "y_test": splitset(y_test, n_splits),
    }

    for i in range(n_splits):
        subdir = f"{out_dir}/clients/{str(i+1)}"
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        torch.save(
            {
                "x_train": data["x_train"][i],
                "y_train": data["y_train"][i],
                "x_test": data["x_test"][i],
                "y_test": data["y_test"][i],
            },
            f"{subdir}/data.pt",
        )

if __name__ == "__main__":
    if not os.path.exists(abs_path + "/data/clients/1"):
        get_data()
        split()