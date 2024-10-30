import os
from math import floor
import torchvision
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer
import re
from collections import Counter
import pandas as pd
import numpy as np

bertTokens = BertTokenizer.from_pretrained('bert-base-uncased').vocab
bertTokens_with_numbers = {token: index for index, token in enumerate(bertTokens)}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


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
    
    top_tokens_mapped_to_bert = []
    for token in sorted_tokens:
        bert_token = tokenizer.tokenize(token)
        for bert_token in bert_token:
            top_tokens_mapped_to_bert.append(bert_token)
    
    top_tokens_mapped_to_bert = list(set(top_tokens_mapped_to_bert))
    print(len(top_tokens_mapped_to_bert))
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


def load_data(data_path, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/clients/1/mnist.pt")

    data = torch.load(data_path, weights_only=True)

    if is_train:
        X = data["x_train"]
        y = data["y_train"]
    else:
        X = data["x_test"]
        y = data["y_test"]

    # Normalize
    X = X / 255

    return X, y


def splitset(dataset, parts):
    """Split a dataset into parts.
    
    :param dataset: List of data samples
    :param parts: Number of parts to split into
    :return: List of dataset parts
    """
    n = len(dataset)
    local_n = floor(n / parts)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n : (i + 1) * local_n])
    return result


def get_data(out_dir="data"):
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Load data from CSV
    file_path = os.path.join(out_dir, 'history.csv')
    df = pd.read_csv(file_path)
    
    texts = []
    for _, row in df.iterrows():
        prompt = row['prompt']
        text = prompt
        texts.append(text)
    print(f"Loaded {len(texts)} text samples from CSV.")
    
    return texts

def split(texts, out_dir="data"):
    n_splits = 1

    if not os.path.exists(f"{out_dir}/clients"):
        os.mkdir(f"{out_dir}/clients")

    word_to_idx, tokenized_texts = build_vocab(texts)
    sequences = encode_sequences(tokenized_texts, word_to_idx, seq_length=4)
    
    x_sequences = [seq[:-1] for seq in sequences]
    y_sequences = [seq[-1] for seq in sequences]
    
    print(x_sequences[0])
    print(y_sequences[0])
    data = {
        "x_train": splitset(x_sequences, n_splits),
        "y_train": splitset(y_sequences, n_splits), 
        "x_test": splitset(x_sequences, n_splits),
        "y_test": splitset(y_sequences, n_splits)
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
            f"{subdir}/mnist.pt",
        )

if __name__ == "__main__":
    if not os.path.exists(abs_path + "/data/clients/1"):
        data = get_data()
        split(data)