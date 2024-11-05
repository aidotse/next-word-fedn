import torch
from  model import NextWordLSTM

model_path = "bert.pth"

model = torch.load(model_path, map_location=torch.device('cpu'))

if not isinstance(model, NextWordLSTM):
    model = NextWordLSTM()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

torch.save(model.state_dict(), "bert_weights.pth")

