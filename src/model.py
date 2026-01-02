from torch.nn import Module, Linear, ReLU, Sequential, CrossEntropyLoss, Embedding
from torch.optim import AdamW

from src.constants import PRIME, LEARNING_RATE, WEIGHT_DECAY

class Model(Module):
    def __init__(self, p=PRIME, embed_dim=128, hidden_size=256, output_size=PRIME):
        super(Model, self).__init__()
        self.loss = CrossEntropyLoss()
        self.embed = Embedding(p, embed_dim)
        self.model = Sequential(
            Linear(embed_dim * 2, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, output_size)
        )
        self.optimizer = AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    def forward(self, x):
        e = self.embed(x)
        e = e.view(e.size(0), -1)
        return self.model(e)