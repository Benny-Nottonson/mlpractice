from torch import no_grad
from src.constants import DEVICE

def train_epoch(model, train_loader):
    model.train()
    losses, preds, targets = [], [], []
    
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.optimizer.zero_grad()
        outputs = model(x)
        batch_loss = model.loss(outputs, y)
        batch_loss.backward()
        model.optimizer.step()
        
        losses.append(batch_loss.item())
        preds.append(outputs.argmax(dim=1))
        targets.append(y)
    
    return losses, preds, targets

def evaluate(model, test_loader):
    model.eval()
    losses, preds, targets = [], [], []
    
    with no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            losses.append(model.loss(outputs, y).item())
            preds.append(outputs.argmax(dim=1))
            targets.append(y)
    
    return losses, preds, targets