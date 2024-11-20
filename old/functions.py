import re
from collections import Counter
import pandas as pd

# Tokenize the text
def tokenize_text(text):
    return text.split()

# Build a vocabulary and tokenize the dataset
def build_vocab(texts):
    
    # Create a DataFrame with all texts
    df = pd.DataFrame({'text': texts})
    
    # Apply regex operations using pandas
    df['text'] = df['text'].replace({
        r'https?://\S+': '',  # Remove URLs (including http:// and https://)
        r'htp\S+': '',  # Remove URLs
        r'@\w+': '',  # Remove usernames
        r'[^a-zA-Z\s]': '',  # Remove special characters and numbers
        r'[.,!:;]': '',  # Remove punctuation
        '&quot;': '',  # Remove &quot; from the data
        r'(.)\1{2,}': r'\1',  # Replace more than 3 consecutive characters with 1
        r'\bname\b': ''  # Remove standalone "name"
    }, regex=True)
    
    df['text'] = df['text'].str.lower().str.strip()
    
    df['words'] = df['text'].str.split()
    
    tokenized_texts = df['words'].tolist()
    
    all_words = [word for text in tokenized_texts for word in text]
    word_counts = Counter(all_words)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    
    # Create a mapping from word to index
    word_to_idx = {word: idx+1 for idx, word in enumerate(sorted_words)}
    word_to_idx['<PAD>'] = 0  # Padding index
    return word_to_idx, tokenized_texts

# Convert sequences of words to sequences of integers
def encode_sequences(tokenized_texts, word_to_idx, seq_length=4):
    sequences = []
    for tokens in tokenized_texts:
        if len(tokens) < seq_length:
            continue
        for i in range(seq_length, len(tokens)):
            seq = tokens[i-seq_length:i]  # Input sequence of words
            target = tokens[i]  # Target word (next word)
            encoded_seq = [word_to_idx[word] for word in seq]
            encoded_target = word_to_idx[target]
            sequences.append((encoded_seq, encoded_target))
    return sequences