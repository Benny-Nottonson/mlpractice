from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch import sin, no_grad, linspace
from tqdm import tqdm

from src import FunctionalDataset, Model, plot_results
from src import DATASET_SIZE, TRAIN_SPLIT, BATCH_SIZE, EPOCHS, LEARNING_RATE

if __name__ == "__main__":
    dataset = FunctionalDataset(func=sin)
    train, test = random_split(dataset, [int(TRAIN_SPLIT * DATASET_SIZE), DATASET_SIZE - int(TRAIN_SPLIT * DATASET_SIZE)])
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE)

    model = Model()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []

    model.train()
    with tqdm(range(EPOCHS)) as bar:
        for epoch in bar:
            epoch_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                outputs = model(x)
                batch_loss = model.loss(outputs, y)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
            epoch_loss /= len(train_loader)
            losses.append(epoch_loss)
            bar.set_description(f"Σ={epoch_loss:.6f}")

    model.eval()
    test_loss = 0.0
    with no_grad():
        for x, y in test_loader:
            outputs = model(x)
            test_loss += model.loss(outputs, y).item()

    mse = test_loss / len(test_loader)
    acc = 1 / (1 + mse)
    print(f"μ={mse:.8f}  α={acc:.8f}")

    xs = linspace(dataset.x.min(), dataset.x.max(), DATASET_SIZE).unsqueeze(1)
    with no_grad():
        ys_pred = model(xs)
    ys_true = sin(xs)

    plot_results(losses, xs, ys_true, ys_pred)