from matplotlib.pyplot import figure, subplot, plot, legend, xlabel, ylabel, yscale, title, show, tight_layout

def plot_results(metrics):
    figure(figsize=(12, 5))

    subplot(1, 2, 1)
    plot([m.train_loss for m in metrics], label="Σ")
    plot([m.test_loss for m in metrics], label="Σ'")
    yscale("log")
    xlabel("Epoch")
    ylabel("Σ")
    title("Loss")
    legend()

    subplot(1, 2, 2)
    plot([m.train_acc for m in metrics], label="α")
    plot([m.test_acc for m in metrics], label="α'")
    xlabel("Epoch")
    ylabel("α")
    title("Accuracy")
    legend()

    tight_layout()
    show()