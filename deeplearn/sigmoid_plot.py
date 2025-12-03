import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == '__main__':
    xs = np.linspace(-10, 10, 400)
    ys = sigmoid(xs)
    ys_der = ys * (1 - ys)

    plt.figure(figsize=(8,4))
    plt.plot(xs, ys, label='sigmoid(x)', color='#1f77b4')
    plt.plot(xs, ys_der, label="sigmoid'(x)", color='#ff7f0e')
    plt.axvline(0, color='#999', linewidth=0.8)
    plt.title('Sigmoid and its derivative')
    plt.xlabel('x')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('sigmoid_plot.png', dpi=150)
    plt.show()
