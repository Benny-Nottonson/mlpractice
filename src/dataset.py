from torch.utils.data import Dataset
from torch import arange, stack

class FunctionalDataset(Dataset):
    def __init__(self, func, p=97):
        self.p = p
        a = arange(p)
        self.x = stack([a.repeat(p), a.repeat_interleave(p)], dim=1)
        self.y = func(self.x[:, 0], self.x[:, 1], p)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]