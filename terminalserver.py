import torch
import json
import argparse
from train.models import NextWordLSTM, NextWordGRU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_next_word(model, sequence, word_to_idx, idx_to_word):
    model.eval()
    sequence = torch.tensor(sequence).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(sequence)
        predicted_idx = torch.argmax(output, dim=1).item()
    return idx_to_word[predicted_idx]

def load_model_and_vocab(model_path, vocab_path):
    with open(vocab_path, 'r') as f:
        loaded_word_to_idx = json.load(f)
    
    loaded_model = torch.load(model_path, map_location=device)
    loaded_model.eval()
    
    print("Model and vocabulary loaded successfully.")
    return loaded_model, loaded_word_to_idx

def generate_text(seed_text, num_words, model, word_to_idx):
    words = seed_text.split()
    for _ in range(num_words):
        sequence = [word_to_idx.get(word, word_to_idx.get('<UNK>', 0)) for word in words[-3:]]
        next_word = predict_next_word(model, sequence, word_to_idx, {v: k for k, v in word_to_idx.items()})
        words.append(next_word)
    return ' '.join(words)

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained model.")
    parser.add_argument("--model", default='train/gru_onehot/model.pth', help="Path to the trained model")
    parser.add_argument("--vocab", default='train/gru_onehot/vocabulary.json', help="Path to the vocabulary file")
    parser.add_argument("--words", type=int, default=10, help="Number of words to generate")
    
    args = parser.parse_args()
    
    model, word_to_idx = load_model_and_vocab(args.model, args.vocab)
    
    while True:
        seed_text = input("Enter seed text (or 'quit' to exit): ")
        if seed_text.lower() == 'quit':
            break
        generated_text = generate_text(seed_text, args.words, model, word_to_idx)
        print(f"Generated text: {generated_text}")

if __name__ == '__main__':
    main()
