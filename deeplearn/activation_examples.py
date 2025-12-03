"""activation_examples.py
示例：使用 NumPy 实现常见激活函数并对前向计算速度做简单对比。
运行方式（在仓库的 deeplearn 目录下）：
    pip install numpy
    python activation_examples.py
会打印每个激活在随机输入上的均值、标准差以及前向计算耗时。
"""

import time
import numpy as np

R = np.random.RandomState(0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))

def selu(x, lam=1.0507009873554805, alpha=1.6732632423543772):
    return lam * np.where(x > 0, x, alpha * (np.exp(x) - 1.0))

def swish(x):
    return x * sigmoid(x)

# approximate GELU using tanh approximation
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x):
    # assumes x is (N, C)
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

activations = [
    ("sigmoid", sigmoid),
    ("tanh", tanh),
    ("relu", relu),
    ("leaky_relu", leaky_relu),
    ("elu", elu),
    ("selu", selu),
    ("swish", swish),
    ("gelu", gelu),
]


def benchmark_vector(activation, x, n_repeat=50):
    # warmup
    y = activation(x)
    t0 = time.time()
    for _ in range(n_repeat):
        y = activation(x)
    t1 = time.time()
    return (t1 - t0) / n_repeat, y


def main():
    print("Activation examples and quick benchmark (NumPy)")
    # large vector input
    x = R.randn(1000000).astype(np.float32)

    for name, fn in activations:
        t_avg, y = benchmark_vector(fn, x)
        print(f"{name:10} | mean={y.mean(): .6f} std={y.std(): .6f} time_avg={t_avg*1000: .3f} ms")

    # softmax benchmark with 2D input
    x2 = R.randn(10000, 100).astype(np.float32)
    t0 = time.time()
    for _ in range(10):
        _ = softmax(x2)
    t1 = time.time()
    print(f"softmax   | shape={x2.shape} time_avg={(t1-t0)/10*1000: .3f} ms")

    print("\n示例说明：")
    print("- Sigmoid/Tanh 在大数值上趋于饱和，导数接近 0，会导致梯度消失（深层网络不建议在隐藏层大量使用）。")
    print("- ReLU 计算开销小，实用性强，但可能会出现死亡 ReLU；LeakyReLU/ELU/SELU 能缓解此问题。")
    print("- Swish/GELU 在现代 Transformer/NLP 模型中效果好，但计算成本较高。")

if __name__ == '__main__':
    main()
