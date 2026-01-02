from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch import no_grad, device, cuda
from torch.nn.functional import cross_entropy
from tqdm import tqdm

from src import FunctionalDataset, Model, plot_results
from src import DATASET_SIZE, TRAIN_SPLIT, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY

def function(a, b, p):
    return (a + b) % p

if __name__ == "__main__":
    dev = device("cuda" if cuda.is_available() else "cpu")

    dataset = FunctionalDataset(func=function)
    train, test = random_split(dataset, [int(TRAIN_SPLIT * DATASET_SIZE), DATASET_SIZE - int(TRAIN_SPLIT * DATASET_SIZE)])
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE)

    model = Model().to(dev)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_losses = []
    test_losses = []
    excl_losses = []
    train_accs = []
    test_accs = []
    excl_accs = []

    with tqdm(range(EPOCHS)) as bar:
        for epoch in bar:
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            for x, y in train_loader:
                x, y = x.to(dev), y.to(dev)
                optimizer.zero_grad()
                outputs = model(x)
                batch_loss = model.loss(outputs, y)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                epoch_correct += (outputs.argmax(dim=1) == y).sum().item()
                epoch_total += y.size(0)
            train_losses.append(epoch_loss / len(train_loader))
            train_accs.append(epoch_correct / epoch_total)

            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            excl_loss = 0.0
            excl_correct = 0
            excl_total = 0
            with no_grad():
                for x, y in test_loader:
                    x, y = x.to(dev), y.to(dev)
                    outputs = model(x)
                    test_loss += model.loss(outputs, y).item()
                    test_correct += (outputs.argmax(dim=1) == y).sum().item()
                    test_total += y.size(0)
                for x, y in train_loader:
                    x, y = x.to(dev), y.to(dev)
                    outputs = model(x)
                    preds = outputs.argmax(dim=1)
                    wrong = preds != y
                    if wrong.sum() > 0:
                        excl_loss += cross_entropy(outputs[wrong], y[wrong]).item()
                        excl_total += 1
                    excl_correct += wrong.sum().item()
            test_losses.append(test_loss / len(test_loader))
            test_accs.append(test_correct / test_total)
            excl_losses.append(excl_loss / max(excl_total, 1))
            excl_accs.append(1 - (excl_correct / epoch_total))
            bar.set_description(f"Σ={train_losses[-1]:.3f} α={train_accs[-1]:.2f} Σ'={test_losses[-1]:.3f} α'={test_accs[-1]:.2f}")

    plot_results(train_losses, test_losses, excl_losses, train_accs, test_accs, excl_accs)