from torch.utils.data import DataLoader, Dataset, random_split
from torch import arange, stack
from torchvision import datasets, transforms

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


class MNISTDataset:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def get_loaders(self):
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=self.transform
        )
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=self.transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        return train_loader, test_loader


class FashionMNISTDataset:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    
    def get_loaders(self):
        train_dataset = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=self.transform
        )
        test_dataset = datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=self.transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        return train_loader, test_loader


class CIFAR10Dataset:
    def __init__(self):
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    def get_loaders(self):
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=self.transform_train
        )
        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=self.transform_test
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        return train_loader, test_loader