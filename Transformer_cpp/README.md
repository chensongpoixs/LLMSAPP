# Transformer C++ 实现

本目录包含使用 C++ 和 libtorch (PyTorch C++ API) 实现的 Transformer 架构。

## 架构概述

**重要说明：当前实现是 Decoder-Only 架构（GPT风格），不是完整的 Encoder-Decoder Transformer。**

### 架构类型

- **Decoder-Only（当前实现）**：只包含解码器层，使用 Masked Self-Attention
  - 适合任务：语言模型、文本生成（自回归）
  - 特点：使用掩码防止看到未来信息
  - 示例：GPT、GPT-2、GPT-3、ChatGPT

- **Encoder-Decoder（未实现）**：包含编码器和解码器
  - 适合任务：机器翻译、摘要生成
  - 特点：编码器处理输入，解码器生成输出，之间有 Cross-Attention
  - 示例：原始 Transformer、BERT（只有Encoder）、T5

### 当前实现

本实现采用 GPT (Generative Pre-trained Transformer) 风格，即只使用解码器（Decoder）部分的自回归模型。

### 主要组件

1. **Multi-Head Attention** (`MultiHeadAttention`): 多头自注意力机制
2. **Feed Forward Network** (`FeedForwardNetwork`): 前馈神经网络
3. **Transformer Block** (`TransformerBlock`): Transformer 块（包含注意力、前馈网络、残差连接）
4. **GELU Activation** (`NewGELU`): GELU 激活函数
5. **GPT Model** (`GPTModel`): 完整的 GPT 模型

### 架构流程

```
输入序列 (Token IDs)
    ↓
[1. Token Embedding] 将 token ID 转换为向量
    ↓
[2. Position Embedding] 添加位置信息
    ↓
[3. 嵌入融合] token_embeds + position_embeds
    ↓
[4. Dropout] 防止过拟合
    ↓
[5. Transformer Blocks] (重复 N 次，通常 N = 12)
    │
    ├─ Layer Norm 1
    ├─ Multi-Head Self-Attention
    ├─ Dropout
    ├─ 残差连接
    ├─ Layer Norm 2
    ├─ Feed Forward Network
    ├─ Dropout
    └─ 残差连接
    ↓
[6. 最终层归一化]
    ↓
[7. 输出头] 线性投影到词汇表大小
    ↓
输出 Logits (每个位置对应词汇表中每个词的概率)
```

## 依赖要求

### 必需依赖

1. **libtorch**: PyTorch C++ API
   - 下载地址: https://pytorch.org/get-started/locally/
   - 选择 Stable, C++, 以及您的平台（Windows/Linux/macOS）
   - 解压后设置 `CMAKE_PREFIX_PATH` 指向 libtorch 目录

2. **CMake**: 3.18 或更高版本
   - Windows: https://cmake.org/download/
   - Linux: `sudo apt-get install cmake` (Ubuntu/Debian)
   - macOS: `brew install cmake`

3. **C++17 或更高版本的编译器**
   - Windows: Visual Studio 2019 或更高版本（使用 MSVC）
   - Linux: GCC 7+ 或 Clang 5+
   - macOS: Xcode 10+ 或 Clang 5+

### 可选依赖

- **CUDA**: 如果使用 GPU 加速（需要 CUDA 版本的 libtorch）

## 编译步骤

### Windows

1. 下载 libtorch (CPU 或 CUDA 版本)
   ```powershell
   # 例如下载到 D:\libtorch
   ```

2. 创建构建目录
   ```powershell
   mkdir build
   cd build
   ```

3. 运行 CMake（设置 libtorch 路径）
   ```powershell
   cmake -DCMAKE_PREFIX_PATH=D:\libtorch ..
   ```

4. 编译
   ```powershell
   cmake --build . --config Release
   ```

5. 运行示例
   ```powershell
   .\Release\Transformer_inference.exe
   ```

### Linux/macOS

1. 下载 libtorch
   ```bash
   # 例如下载到 ~/libtorch
   wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
   unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
   ```

2. 创建构建目录
   ```bash
   mkdir build
   cd build
   ```

3. 运行 CMake
   ```bash
   cmake -DCMAKE_PREFIX_PATH=~/libtorch ..
   ```

4. 编译
   ```bash
   make -j4
   ```

5. 运行示例
   ```bash
   ./Transformer_inference
   ```

## 可执行文件

编译后会生成三个可执行文件：

1. **`Transformer_inference`**: 推理程序，用于文本生成
   - 加载训练好的模型
   - 根据提示词生成文本
   - 支持 CPU/GPU 选择

2. **`Transformer_train`**: 训练程序，用于模型训练
   - 从文本文件加载数据
   - 执行训练循环（前向传播、反向传播、参数更新）
   - 保存模型检查点
   - 支持 CPU/GPU 选择

3. **`Transformer_model_info`**: 模型信息工具，用于分析和显示模型参数
   - 加载保存的模型文件
   - 显示模型结构和所有参数信息
   - 统计参数数量、内存占用等
   - 可保存模型信息到文件

## 文件说明

### 核心文件

- **`GPTModel.h/cpp`**: GPT 模型实现
- **`MultiHeadAttention.h/cpp`**: 多头注意力机制
- **`TransformerBlock.h/cpp`**: Transformer 块
- **`FeedForwardNetwork.h/cpp`**: 前馈神经网络
- **`NewGELU.h/cpp`**: GELU 激活函数
- **`ModelConfig.h`**: 模型配置结构体

### 工具文件

- **`inference.cpp`**: 推理程序，使用 Generator 类进行文本生成
- **`train.cpp`**: 训练程序，使用 Trainer 类进行模型训练
- **`model_info.cpp`**: 模型信息工具，使用 ModelAnalyzer 类分析模型
- **`Generator.h/cpp`**: 文本生成器类
- **`Trainer.h/cpp`**: 训练器类
- **`ModelAnalyzer.h/cpp`**: 模型分析器类
- **`TrainingUtils.h/cpp`**: 训练工具函数
- **`TextDataset.h/cpp`**: 文本数据集加载器
- **`Logger.h/cpp`**: 日志模块

### 构建文件

- **`CMakeLists.txt`**: CMake 构建配置文件

## 使用示例

### 基本使用

```cpp
#include "Transformer.h"
#include <torch/torch.h>

int main() {
    // 1. 创建模型配置
    ModelConfig cfg;
    cfg.vocab_size = 50257;
    cfg.max_seq_length = 1024;
    cfg.embedding_dim = 768;
    cfg.n_heads = 12;
    cfg.n_layers = 12;
    cfg.drop_rate = 0.1;
    cfg.qkv_bias = false;
    
    // 2. 创建模型
    auto model = std::make_shared<GPTModel>(cfg);
    model->eval();  // 设置为评估模式
    
    // 3. 创建输入（batch_size=2, seq_len=10）
    auto input = torch::randint(cfg.vocab_size, {2, 10}, 
                                 torch::TensorOptions().dtype(torch::kLong));
    
    // 4. 前向传播
    torch::NoGradGuard no_grad;  // 禁用梯度计算
    auto output = model->forward(input);
    
    // 5. 输出形状: (batch_size, seq_len, vocab_size)
    std::cout << "Output shape: " << output.sizes() << std::endl;
    
    return 0;
}
```

### 模型配置

`ModelConfig` 结构体包含以下配置项：

- `vocab_size`: 词汇表大小（默认: 50257，GPT-2 标准）
- `max_seq_length`: 最大序列长度（默认: 1024）
- `embedding_dim`: 嵌入向量维度（默认: 768，GPT-2 标准）
- `n_heads`: 注意力头数量（默认: 12）
- `n_layers`: Transformer 层数（默认: 12）
- `drop_rate`: Dropout 比率（默认: 0.1）
- `qkv_bias`: Q、K、V 线性层是否使用偏置（默认: false）

### 模型信息工具使用

`Transformer_model_info` 工具用于分析和显示模型的所有参数信息：

```bash
# 基本使用（分析默认模型文件）
./Transformer_model_info

# 指定模型文件
./Transformer_model_info --model runs/train/exp0/weights/best.pth

# 保存模型信息到文件
./Transformer_model_info --model transformer_model_full.pth --output model_info.txt
```

工具功能：
- 显示模型结构（按模块分组）
- 显示所有参数的详细信息（形状、元素数量、统计信息）
- 显示参数摘要（总参数量、内存占用、可训练参数等）
- 按模块统计参数分布
- 可保存模型信息到文本文件

### GPU 支持

如果需要使用 GPU：

1. 下载 CUDA 版本的 libtorch
2. 确保 CUDA 和 cuDNN 已正确安装
3. 在代码中将张量移动到 GPU：

```cpp
// 创建模型并移动到 GPU
auto model = std::make_shared<GPTModel>(cfg);
model->to(torch::kCUDA);

// 输入也移动到 GPU
auto input = torch::randint(cfg.vocab_size, {2, 10}, 
                             torch::TensorOptions()
                                 .device(torch::kCUDA)
                                 .dtype(torch::kLong));
```

## 性能说明

### 参数量

对于默认配置（GPT-2 标准）：
- 词汇表: 50257
- 嵌入维度: 768
- 注意力头数: 12
- 层数: 12
- 最大序列长度: 1024

参数量约为 **124M**（与 GPT-2 Small 相同）

### 内存使用

- CPU 推理: 约 500 MB RAM
- GPU 推理: 取决于 GPU 显存（通常需要 2-4 GB）

### 推理速度

- CPU (单线程): 约 10-50 tokens/秒（取决于 CPU）
- GPU: 约 100-1000 tokens/秒（取决于 GPU）

## 注意事项

1. **libtorch 路径**: 确保 CMake 能够找到 libtorch，设置正确的 `CMAKE_PREFIX_PATH`
2. **C++ 标准**: 需要 C++17 或更高版本
3. **内存管理**: libtorch 使用智能指针管理内存，通常不需要手动释放
4. **线程安全**: 模型本身不是线程安全的，多线程使用需要加锁
5. **数值稳定性**: 使用 float32 精度，对于大模型可能需要 float64

## 与 Python 版本的对应关系

本 C++ 实现与 `MyTransformer.py` 中的 Python 实现功能相同：

- `GPTModel` ↔ `MyGPTModel`
- `TransformerBlock` ↔ `MyTransformerBlock`
- `MultiHeadAttention` ↔ `MultiHeadAttention`
- `FeedForwardNetwork` ↔ `FeedForwardNetwork`
- `NewGELU` ↔ `NewGELU`

模型结构、参数和计算逻辑完全一致，可以互用模型权重（需要额外的序列化/反序列化代码）。

## 许可证

与项目主目录的许可证相同。

## 参考资源

- [PyTorch C++ API 文档](https://pytorch.org/cppdocs/)
- [libtorch 教程](https://pytorch.org/tutorials/advanced/cpp_frontend.html)
- [Transformer 原始论文](https://arxiv.org/abs/1706.03762)
- [GPT-2 论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

