from dataclasses import dataclass
from torch import cat

def compute_metrics(losses, preds, targets):
    loss_avg = sum(losses) / len(losses)
    preds = cat(preds)
    targets = cat(targets)
    acc = (preds == targets).float().mean().item()
    return loss_avg, acc

@dataclass
class EpochMetrics:
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float