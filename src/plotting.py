from matplotlib.pyplot import figure, subplot, plot, legend, title, show, tight_layout

def plot_results(losses, xs, ys_true, ys_pred):
    figure(figsize=(12, 4))

    subplot(1, 2, 1)
    plot(losses)
    title("Σ")
    subplot(1, 2, 2)
    plot(xs.numpy(), ys_true.numpy(), label="f")
    plot(xs.numpy(), ys_pred.numpy(), label="ŷ")
    legend()

    tight_layout()
    show()