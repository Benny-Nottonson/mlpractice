from torch.nn import Module, Linear, Tanh, Sequential, MSELoss

class Model(Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
        super(Model, self).__init__()
        self.loss = MSELoss()
        self.model = Sequential(
            Linear(input_size, hidden_size),
            Tanh(),
            Linear(hidden_size, hidden_size),
            Tanh(),
            Linear(hidden_size, hidden_size),
            Tanh(),
            Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)