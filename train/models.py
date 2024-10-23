import torch.nn as nn

class NextWordLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, repetition_penalty=1.0):
        super(NextWordLSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.repetition_penalty = repetition_penalty
        
        self.embedding = nn.Embedding(1, embed_size)  # Placeholder embedding
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Placeholder output layer
    
    def forward(self, x):
        if self.embedding.num_embeddings <= x.max().item():
            self._extend_embedding(x.max().item() + 1)
        if self.fc.out_features <= x.max().item():
            self._extend_output(x.max().item() + 1)
        
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last LSTM cell
        out = self.fc(lstm_out)
        
        if self.repetition_penalty != 1.0:
            for i in range(out.size(0)):
                for token in x[i]:
                    out[i, token] /= self.repetition_penalty
        
        return out
    
    def _extend_embedding(self, new_size):
        old_embedding = self.embedding
        self.embedding = nn.Embedding(new_size, self.embed_size)
        self.embedding.weight.data[:old_embedding.num_embeddings] = old_embedding.weight.data
    
    def _extend_output(self, new_size):
        old_fc = self.fc
        self.fc = nn.Linear(self.hidden_size, new_size)
        self.fc.weight.data[:old_fc.out_features] = old_fc.weight.data
        self.fc.bias.data[:old_fc.out_features] = old_fc.bias.data

class NextWordGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout_prob=0.3, repetition_penalty=1.0):
        super(NextWordGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.repetition_penalty = repetition_penalty
    
    def forward(self, x):
        x = self.embedding(x)
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out[:, -1, :]) 
        out = self.fc(gru_out)
        
        if self.repetition_penalty != 1.0:
            for i in range(out.size(0)):
                for token in x[i]:
                    out[i, token] /= self.repetition_penalty
        
        return out