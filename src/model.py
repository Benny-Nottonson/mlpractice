from torch.nn import Module, Linear, ReLU, Sequential, Embedding, CrossEntropyLoss, LSTM, Dropout
from torch.optim import AdamW

from src.constants import PRIME, LEARNING_RATE, WEIGHT_DECAY
from src.experiments.layers import ScoreSpaceEnsembleLayer

class Model(Module):
    def __init__(self, p=PRIME, embed_dim=128, hidden_size=256, output_size=PRIME):
        super(Model, self).__init__()
        self.loss = CrossEntropyLoss()
        self.embed = Embedding(p, embed_dim)
        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.model = Sequential(
            Linear(hidden_size, hidden_size),
            ReLU(),
            ScoreSpaceEnsembleLayer(hidden_size),
            Linear(hidden_size, output_size)
        )
        self.optimizer = AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    def forward(self, x):
        e = self.embed(x)
        _, (h, _) = self.lstm(e)
        return self.model(h[-1])
