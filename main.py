from torch import no_grad, compile, cat
from tqdm import tqdm

from src import EPOCHS, DEVICE, FunctionalDataset, Model, EpochMetrics, plot_results

def function(a, b, p):
    return (a + b) % p

if __name__ == "__main__":
    model = Model().to(DEVICE)
    train_loader, test_loader = FunctionalDataset(func=function).get_loaders()
    metrics = []

    with tqdm(range(EPOCHS)) as bar:
        for epoch in bar:
            model.train()
            train_losses, train_preds, train_targets = [], [], []
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                model.optimizer.zero_grad()
                outputs = model(x)
                batch_loss = model.loss(outputs, y)
                batch_loss.backward()
                model.optimizer.step()
                train_losses.append(batch_loss.item())
                train_preds.append(outputs.argmax(dim=1))
                train_targets.append(y)
            
            train_loss_avg = sum(train_losses) / len(train_losses)
            train_preds = cat(train_preds)
            train_targets = cat(train_targets)
            train_acc = (train_preds == train_targets).float().mean().item()

            model.eval()
            test_losses, test_preds, test_targets = [], [], []
            with no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    outputs = model(x)
                    test_losses.append(model.loss(outputs, y).item())
                    test_preds.append(outputs.argmax(dim=1))
                    test_targets.append(y)
            
            test_loss_avg = sum(test_losses) / len(test_losses)
            test_preds = cat(test_preds)
            test_targets = cat(test_targets)
            test_acc = (test_preds == test_targets).float().mean().item()
            
            epoch_metrics = EpochMetrics(
                train_loss=train_loss_avg,
                train_acc=train_acc,
                test_loss=test_loss_avg,
                test_acc=test_acc
            )
            metrics.append(epoch_metrics)
            bar.set_description(f"Σ={epoch_metrics.train_loss:.3f} α={epoch_metrics.train_acc:.2f} Σ'={epoch_metrics.test_loss:.3f} α'={epoch_metrics.test_acc:.2f}")

    plot_results(metrics)