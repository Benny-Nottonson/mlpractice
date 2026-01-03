from matplotlib.pyplot import figure, subplot, plot, legend, xlabel, ylabel, yscale, title, show, tight_layout, suptitle

def plot_results(metrics, title=None):
    figure(figsize=(12, 5))

    subplot(1, 2, 1)
    plot([m.train_loss for m in metrics], label="Train Loss")
    plot([m.test_loss for m in metrics], label="Test Loss")
    yscale("log")
    xlabel("Epoch")
    ylabel("Loss")
    legend()

    subplot(1, 2, 2)
    plot([m.train_acc for m in metrics], label="Train Acc")
    plot([m.test_acc for m in metrics], label="Test Acc")
    xlabel("Epoch")
    ylabel("Accuracy")
    legend()

    if title:
        suptitle(title, fontsize=14, fontweight='bold')
    
    tight_layout()
    show()