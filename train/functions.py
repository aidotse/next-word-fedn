import re
from collections import Counter

# Preprocessing: Clean and Tokenize Text Data
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = text.lower().strip()  # Lowercase and strip whitespaces
    return text

# Tokenize the text
def tokenize_text(text):
    return text.split()

# Build a vocabulary and tokenize the dataset
def build_vocab(texts):
    tokenized_texts = [tokenize_text(clean_text(text)) for text in texts]
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