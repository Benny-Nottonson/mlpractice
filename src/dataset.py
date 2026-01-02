from torch.utils.data import DataLoader, Dataset, random_split
from torch import arange, stack

from src.constants import PRIME, TRAIN_SPLIT, BATCH_SIZE

class FunctionalDataset(Dataset):
    def __init__(self, func, p=PRIME):
        self.p = p
        a = arange(p)
        self.x = stack([a.repeat(p), a.repeat_interleave(p)], dim=1)
        self.y = func(self.x[:, 0], self.x[:, 1], p)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def get_loaders(self):
        train_size = int(TRAIN_SPLIT * len(self))
        test_size = len(self) - train_size
        train, test = random_split(self, [train_size, test_size])
        train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test, batch_size=BATCH_SIZE)
        return train_loader, test_loader