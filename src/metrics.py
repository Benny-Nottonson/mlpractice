from dataclasses import dataclass

@dataclass
class EpochMetrics:
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float