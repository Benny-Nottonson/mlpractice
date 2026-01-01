from torch.nn import Module, Linear, ReLU, Sequential, CrossEntropyLoss, Embedding

class Model(Module):
    def __init__(self, p=97, embed_dim=128, hidden_size=256, output_size=97):
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
    
    def forward(self, x):
        e = self.embed(x)
        e = e.view(e.size(0), -1)
        return self.model(e)