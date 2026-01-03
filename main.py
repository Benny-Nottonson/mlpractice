from tqdm import tqdm

from src.constants import DEVICE
from src.dataset import MNISTDataset, FashionMNISTDataset, CIFAR10Dataset
from src.model import ImageModel
from src.metrics import EpochMetrics, compute_metrics
from src.train import train_epoch, evaluate
from src.plotting import plot_results

def run_experiment(dataset_name, dataset_class, num_classes, in_channels, epochs=10, learning_rate=1e-3):
    print(f"\n{'='*60}")
    print(f"Training on {dataset_name}")
    print(f"{'='*60}")
    
    model = ImageModel(num_classes=num_classes, in_channels=in_channels, learning_rate=learning_rate).to(DEVICE)
    train_loader, test_loader = dataset_class().get_loaders()
    metrics = []

    with tqdm(range(epochs), desc=dataset_name) as bar:
        for epoch in bar:
            train_losses, train_preds, train_targets = train_epoch(model, train_loader)
            train_loss_avg, train_acc = compute_metrics(train_losses, train_preds, train_targets)
            
            test_losses, test_preds, test_targets = evaluate(model, test_loader)
            test_loss_avg, test_acc = compute_metrics(test_losses, test_preds, test_targets)
            
            bar.set_description(f"{dataset_name} - Train Loss: {train_loss_avg:.3f} | Train Acc: {train_acc:.2%} | Test Loss: {test_loss_avg:.3f} | Test Acc: {test_acc:.2%}")
            metrics.append(EpochMetrics(
                train_loss=train_loss_avg,
                train_acc=train_acc,
                test_loss=test_loss_avg,
                test_acc=test_acc
            ))
    
    print(f"\nFinal {dataset_name} Results:")
    print(f"  Train Accuracy: {metrics[-1].train_acc:.2%}")
    print(f"  Test Accuracy:  {metrics[-1].test_acc:.2%}")
    
    return metrics

if __name__ == "__main__":
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    
    datasets = [
        ("MNIST", MNISTDataset, 10, 1),
        ("Fashion MNIST", FashionMNISTDataset, 10, 1),
        ("CIFAR-10", CIFAR10Dataset, 10, 3),
    ]
    
    all_metrics = {}
    
    for dataset_name, dataset_class, num_classes, in_channels in datasets:
        metrics = run_experiment(
            dataset_name=dataset_name,
            dataset_class=dataset_class,
            num_classes=num_classes,
            in_channels=in_channels,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE
        )
        all_metrics[dataset_name] = metrics
    
    print(f"\n{'='*60}")
    print("Summary of All Experiments")
    print(f"{'='*60}")
    for dataset_name, metrics in all_metrics.items():
        print(f"{dataset_name:15s} - Final Test Accuracy: {metrics[-1].test_acc:.2%}")
    
    for dataset_name, metrics in all_metrics.items():
        plot_results(metrics, title=f"{dataset_name} Training Results")