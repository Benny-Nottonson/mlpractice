from torch import no_grad
from tqdm import tqdm

from src import EPOCHS, FunctionalDataset, Model, EpochMetrics, plot_results

def function(a, b, p):
    return (a + b) % p

if __name__ == "__main__":
    model = Model()
    train_loader, test_loader = FunctionalDataset(func=function).get_loaders()
    metrics = []

    with tqdm(range(EPOCHS)) as bar:
        for epoch in bar:
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for x, y in train_loader:
                x, y = x, y
                model.optimizer.zero_grad()
                outputs = model(x)
                batch_loss = model.loss(outputs, y)
                batch_loss.backward()
                model.optimizer.step()
                train_loss += batch_loss.item()
                train_correct += (outputs.argmax(dim=1) == y).sum().item()
                train_total += y.size(0)
            train_loss_avg = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            with no_grad():
                for x, y in test_loader:
                    x, y = x, y
                    outputs = model(x)
                    test_loss += model.loss(outputs, y).item()
                    test_correct += (outputs.argmax(dim=1) == y).sum().item()
                    test_total += y.size(0)
            test_loss_avg = test_loss / len(test_loader)
            test_acc = test_correct / test_total
            
            epoch_metrics = EpochMetrics(
                train_loss=train_loss_avg,
                train_acc=train_acc,
                test_loss=test_loss_avg,
                test_acc=test_acc
            )
            metrics.append(epoch_metrics)
            bar.set_description(f"Σ={epoch_metrics.train_loss:.3f} α={epoch_metrics.train_acc:.2f} Σ'={epoch_metrics.test_loss:.3f} α'={epoch_metrics.test_acc:.2f}")

    plot_results(metrics)