import collections
import torch
from fedn.utils.helpers.helpers import get_helper
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size=30522, embed_size=128, hidden_size=256, num_layers=1, repetition_penalty=1.1):
        super(NextWordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.repetition_penalty = repetition_penalty
    
    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (ht, ct) = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        lstm_out = lstm_out[torch.arange(lstm_out.size(0)), lengths - 1]
        out = self.fc(lstm_out)



        if self.repetition_penalty != 1.0:
            for i in range(out.size(0)):
                for j in range(lengths[i].item()):
                    token_id = x[i, j].argmax().item()
                    if token_id != 0:  # Check for padding token
                        out[i, token_id] -= torch.log(torch.tensor(self.repetition_penalty, device=out.device))

        
        return out

def compile_model():
    return NextWordLSTM().to(device)

def save_model(model, out_path):
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


def load_parameters(model_path):
    model = compile_model()
    parameters_np = helper.load(model_path)

    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

def load_pth_weights(model_path):
    model = compile_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def convert_pth_to_npz(pth_file_path, npz_file_path):
    model = load_pth_weights(pth_file_path)
    
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    helper.save(parameters_np, npz_file_path)
    print(f"Converted {pth_file_path} to {npz_file_path}.")

def load_model_train(model_path):
    model = load_parameters(model_path)
    model = model.train()
    return model
    
def load_model_eval(model_path):
    model = load_parameters(model_path)
    model = model.eval()
    return model
def load_model_inference(model_path):
    model = load_parameters(model_path)
    model = model.eval()
    return model

if __name__ == "__main__":
    print("file exists")
