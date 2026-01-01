from torch.utils.data import Dataset
from torch import linspace
from math import pi

from .constants import DATASET_SIZE

class FunctionalDataset(Dataset):
    def __init__(self, func):
        self.x = linspace(0, 2 * pi, DATASET_SIZE).unsqueeze(1)
        self.y = func(self.x)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]