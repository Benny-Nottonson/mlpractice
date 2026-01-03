from tqdm import tqdm

from src.constants import EPOCHS, DEVICE
from src.dataset import FunctionalDataset
from src.model import Model
from src.metrics import EpochMetrics, compute_metrics
from src.train import train_epoch, evaluate
from src.plotting import plot_results

if __name__ == "__main__":
    model = Model().to(DEVICE)
    train_loader, test_loader = FunctionalDataset(func=lambda a, b, p: (a + b) % p).get_loaders()
    metrics = []

    with tqdm(range(EPOCHS)) as bar:
        for epoch in bar:
            train_losses, train_preds, train_targets = train_epoch(model, train_loader)
            train_loss_avg, train_acc = compute_metrics(train_losses, train_preds, train_targets)
            
            test_losses, test_preds, test_targets = evaluate(model, test_loader)
            test_loss_avg, test_acc = compute_metrics(test_losses, test_preds, test_targets)
            
            bar.set_description(f"Σ={train_loss_avg:.3f} α={train_acc:.2f} Σ'={test_loss_avg:.3f} α'={test_acc:.2f}")
            metrics.append(EpochMetrics(
                train_loss=train_loss_avg,
                train_acc=train_acc,
                test_loss=test_loss_avg,
                test_acc=test_acc
            ))

    plot_results(metrics)