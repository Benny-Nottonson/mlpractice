from matplotlib.pyplot import figure, subplot, plot, legend, xlabel, ylabel, yscale, title, show, tight_layout

def plot_results(train_losses, test_losses, train_accs, test_accs):
    figure(figsize=(12, 5))

    subplot(1, 2, 1)
    plot(train_losses, label="Σ")
    plot(test_losses, label="Σ'")
    yscale("log")
    xlabel("Epoch")
    ylabel("Σ")
    title("Loss")
    legend()

    subplot(1, 2, 2)
    plot(train_accs, label="α")
    plot(test_accs, label="α'")
    xlabel("Epoch")
    ylabel("α")
    title("Accuracy")
    legend()

    tight_layout()
    show()