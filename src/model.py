from torch.nn import Module, Linear, ReLU, Sequential, Embedding, CrossEntropyLoss, LSTM, Dropout, Conv2d, MaxPool2d, Flatten, BatchNorm2d, AdaptiveAvgPool2d
from torch.optim import AdamW

from src.constants import PRIME, LEARNING_RATE, WEIGHT_DECAY

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
            Linear(hidden_size, output_size)
        )
        self.optimizer = AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    def forward(self, x):
        e = self.embed(x)
        _, (h, _) = self.lstm(e)
        return self.model(h[-1])

class ImageModel(Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(ImageModel, self).__init__()
        self.loss = CrossEntropyLoss(label_smoothing=0.1)
        self.model = Sequential(
            Conv2d(in_channels, 64, kernel_size=7, stride=4, padding=3, bias=True),
            ReLU(inplace=True),

            Conv2d(64, 128, kernel_size=1, bias=True),
            ReLU(inplace=True),

            Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=True),
            ReLU(inplace=True),

            Conv2d(128, 160, kernel_size=1, bias=True),
            ReLU(inplace=True),

            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(160, num_classes, bias=True)
        )

        self.optimizer = AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True, betas=(0.9, 0.95))

    def forward(self, x):
        return self.model(x)
