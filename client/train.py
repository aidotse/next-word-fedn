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

def train(in_model_path, out_model_path, data_path=None, batch_size=1, epochs=5, lr=0.002):
    x_train, y_train = load_data(data_path)
    
    model = load_model_train(in_model_path)
    model = model.to(device)

    model.dropout = nn.Dropout(p=0.3)
    model.train() 

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n_batches = max(1, int(math.ceil(len(x_train) / batch_size)))

    for e in range(epochs):
        total_loss = 0

        for b in range(n_batches):
            batch_x = x_train[b * batch_size : (b + 1) * batch_size].to(device)
            batch_y = y_train[b * batch_size : (b + 1) * batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if b % 100 == 0:
                print(f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")

        avg_loss = total_loss / n_batches
        print(f"Epoch {e}/{epochs-1} completed | Average Loss: {avg_loss:.4f}")

    metadata = {
        "num_examples": len(x_train),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr
    }

    save_metadata(metadata, out_model_path)
    save_model(model, out_model_path)


if __name__ == "__main__":

    train(sys.argv[1], sys.argv[2])
