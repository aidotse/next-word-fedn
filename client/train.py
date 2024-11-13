import math
import os
import sys
import torch
import torch.nn as nn
from model import load_model_train, save_model
from data import load_data
from fedn.utils.helpers.helpers import save_metadata

# Set the path and device
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train(in_model_path, out_model_path, data_path=None, batch_size=4, epochs=2, lr=0.01):
    train_loader = load_data(data_path, is_train=True)
    
    model = load_model_train(in_model_path)
    model = model.to(device)

    model.dropout = nn.Dropout(p=0.3)
    model.train() 

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n_batches = len(train_loader)
    min_loss_threshold = 0.0001

    for e in range(epochs):
        total_loss = 0

        for b, (sequences, lengths, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            lengths = lengths.cpu()  # Lengths need to be on CPU for pack_padded_sequence
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if b % 100 == 0:
                current_loss = total_loss / (b + 1)
                print(f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {current_loss:.4f}")
                if current_loss < min_loss_threshold:
                    print(f"Loss {current_loss:.4f} below threshold {min_loss_threshold}. Stopping early to prevent overfitting.")
                    metadata = {
                        "num_examples": len(train_loader.dataset),
                        "batch_size": batch_size,
                        "epochs": e + (b/n_batches),
                        "lr": lr
                    }
                    save_metadata(metadata, out_model_path)
                    save_model(model, out_model_path)
                    return

        avg_loss = total_loss / n_batches
        print(f"Epoch {e}/{epochs-1} completed | Average Loss: {avg_loss:.4f}")
        if avg_loss < min_loss_threshold:
            print(f"Average loss {avg_loss:.4f} below threshold {min_loss_threshold}. Stopping early to prevent overfitting.")
            break

    metadata = {
        "num_examples": len(train_loader.dataset),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr
    }

    save_metadata(metadata, out_model_path)
    save_model(model, out_model_path)


if __name__ == "__main__":

    train(sys.argv[1], sys.argv[2])
