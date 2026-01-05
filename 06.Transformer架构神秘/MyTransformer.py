# '''
# Docstring for 06.Transformer架构神秘.Transformer
# '''
# from turtle import forward
# import torch;
# import torch.nn as nn;
# import math;

# from multiheadattention import maskselfattention;



# # from MySelfAttention import MultiHeadAttention

# # transformater 实现
# class TestTransformer(nn.Moudle):
#     def __init__(self, cfg):
#         super().__init__();
#         #  词元 嵌入向量
#         self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
#         #  位置嵌入向量
#         self.position_embedding = nn.Embedding(cfg["max_seq_length"], cfg["embedding_dim"])
#         # 
#         self.transformer_blocks = nn.Sequential(
#             *[MyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
#         )
#         # 自己实现归一化操作
#         #self.layer_norm = LayerNorm()
#         # torch实现层归一化性能比较高
#         self.layer_norm = nn.LayerNorm(cfg["embedding_dim"])
#         # 输出头
#         self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)
#         self.drop = nn.Dropout(cfg["drop_rate"])
    
#     def forward(self, x):
#         #x它是一个矩阵，每一行是段训练数据（也就是一句话）
#         #x不是文字，而是文字所对应的token ID 串
#         #所以，x中包括了多行训练数据，称为一个批量
#         #它的列表示，每一段训练数据的长度
#         batch_size, seq_len = x.shape

#         #1. batch_size; 2. seq_len; 3. embedding_dim
#         token_embeds = self.token_embedding(x) #token_embeds 是一个三维的矩阵

#         #position_embedding结果是一个二维矩阵
#         #每一行表示arange生成的字符
#         #而每一行的列数是由embedding_dim决定的，GPT-2中是768
#         postion_embeds = self.position_embedding(torch.arange(seq_len, device=x.device))

#         #广播机制（batch_size, seq_len, embedding_dim), (batch_size, seq_len, embedding_dim)
#         x = token_embeds + postion_embeds

#         #防止过拟合
#         x = self.drop(x)

#         #(batch_size, seq_len, embedding_dim)
#         x = self.transformer_blocks(x)

#         x = self.layer_norm(x)

#         logits = self.out_head(x)

#         return logits


# class NewGELU(nn.Module):
#     def forward(self, x):
#         return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# # 前馈网络
# class FeedForwardNetwork(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         # 输出是原来4倍
#         self.layers = nn.Sequential(
#             nn.Linear(cfg["embedding_dim"], 4*cfg["embedding_dim"]), # 输出
#             NewGELU(),
#             #nn.GELU(),
#             nn.Linear(4*cfg["embedding_dim"], cfg["embedding_dim"]) # 输入
#         )
    
#     def forward(self, x):
#         return self.layers(x)

# class MyTransformerBlock(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         # MY_GPT_CONFIG = {
#         #   "vocab_size": 50257,    #词汇表大小
#         #   "max_seq_length": 1024, #每一句训练数据的最大长度
#         #   "embedding_dim": 768,   #嵌入向量的维度
#         #   "n_heads": 12,          #注意力头个数
#         #   "n_layers": 12,         #Transformer 层数
#         #   "drop_rate": 0.1,       #Dropout rate
#         #   "qkv_bias": False       #bias
#         # }
#         self.mha = maskselfattention(
#             d_in=cfg["embedding_dim"], 
#             d_out=cfg["embedding_dim"],
#             num_heads=cfg["n_heads"],
#             drop_rate=cfg["drop_rate"],
#             mask_matrix_len=cfg["max_seq_length"],
#             qkv_bias=cfg["qkv_bias"])
#         self.ffn = FeedForwardNetwork(cfg)
#         self.norm_1 = nn.LayerNorm(cfg["embedding_dim"])
#         self.norm_2 = nn.LayerNorm(cfg["embedding_dim"])
#         self.dropout = nn.Dropout(cfg["drop_rate"])
    
#     def forward(self, x):
#         old_x = x
#         #################################
#         x = self.norm_1(x)
#         x = self.mha(x)
#         x = self.dropout(x)
#         #################################
#         #残差
#         x = x + old_x
#         old_x = x #为后面的残差做准备
#         #####################################
#         x = self.norm_2(x)
#         x = self.ffn(x)
#         x = self.dropout(x)
#         ####################################
#         x = x + old_x
#         return x
        

# # 层归一化
# class LayerNorm(nn.Module):
#     def __init__(self, embedding_dim):
#         super().__init__();
#         self.eps = 1e-5;
#         self.gamma = nn.Parameter(torch.ones(embedding_dim, embedding_dim));
#         self.beta = nn.parameter(torch.zeros(embedding_dim, embedding_dim));

#     def forward(self, x):
#         # 归一化公式
#         mean = x.mean(dim=-1, keepdim=True); # (3, 8, 1)
#         # 方差
#         var = x.var(dim=-1, keepdim=True);

#         norm_x = (x -mean) / torch.sqrt(var + self.eps);

#         return norm_x * self.gamma + self.beta;



"""
Transformer 架构实现 - GPT 风格模型

本文件实现了一个完整的 Transformer 架构，采用 GPT（Generative Pre-trained Transformer）风格，
即只使用解码器（Decoder）部分的自回归模型。

架构流程图：
================================================================================
输入序列 (Token IDs)
    ↓
[1. 输入嵌入层]
    ├─ Token Embedding: 将 token ID 转换为向量 (vocab_size → embedding_dim)
    └─ Position Embedding: 添加位置信息 (max_seq_length → embedding_dim)
    ↓
[2. 嵌入融合]
    └─ x = token_embeds + position_embeds  (广播机制)
    ↓
[3. Dropout] (防止过拟合)
    ↓
[4. Transformer Blocks] (重复 N 次，通常 N = 12)
    │
    ├─ [Block 1]
    │   ├─ [Layer Norm 1]
    │   ├─ [Multi-Head Self-Attention] (掩码注意力，防止看到未来信息)
    │   ├─ [Dropout]
    │   ├─ [残差连接] x = x + old_x
    │   ├─ [Layer Norm 2]
    │   ├─ [Feed Forward Network]
    │   │   ├─ Linear(embedding_dim → 4×embedding_dim)
    │   │   ├─ GELU 激活函数
    │   │   └─ Linear(4×embedding_dim → embedding_dim)
    │   ├─ [Dropout]
    │   └─ [残差连接] x = x + old_x
    │
    ├─ [Block 2] (相同结构)
    │   ...
    │
    └─ [Block N] (相同结构)
    ↓
[5. 最终层归一化]
    ↓
[6. 输出头]
    └─ Linear(embedding_dim → vocab_size)
    ↓
输出 Logits (每个位置对应词汇表中每个词的概率)
================================================================================

关键组件说明：
1. Multi-Head Self-Attention: 多头自注意力机制，允许模型关注序列中不同位置的信息
2. Feed Forward Network: 两层全连接网络，扩展维度后压缩，增加模型表达能力
3. Layer Normalization: 层归一化，稳定训练过程
4. Residual Connection: 残差连接，缓解梯度消失，允许训练更深的网络
5. Dropout: 随机失活，防止过拟合

数据维度变化：
- 输入: (batch_size, seq_len) - Token IDs
- 嵌入后: (batch_size, seq_len, embedding_dim)
- 经过 Transformer Blocks: (batch_size, seq_len, embedding_dim)
- 输出: (batch_size, seq_len, vocab_size) - Logits
"""

import torch
import torch.nn as nn
import math

from MySelfAttention import MultiHeadAttention


class MyGPTModel(nn.Module):
    """
    GPT 风格的 Transformer 模型
    
    这是一个只使用解码器（Decoder）部分的 Transformer 模型，采用自回归方式生成文本。
    模型结构：输入嵌入 → N 个 Transformer 块 → 层归一化 → 输出头
    
    Args:
        cfg (dict): 模型配置字典，包含以下键：
            - vocab_size: 词汇表大小
            - max_seq_length: 最大序列长度
            - embedding_dim: 嵌入向量维度（通常为 768）
            - n_heads: 注意力头数量（通常为 12）
            - n_layers: Transformer 层数（通常为 12）
            - drop_rate: Dropout 比率（通常为 0.1）
            - qkv_bias: 是否在 Q、K、V 线性层使用偏置
    
    Forward 流程：
        1. Token Embedding: 将 token IDs 转换为嵌入向量
        2. Position Embedding: 添加位置编码
        3. 嵌入融合: token_embeds + position_embeds
        4. Dropout: 防止过拟合
        5. Transformer Blocks: 通过 N 个 Transformer 块处理
        6. Layer Norm: 最终归一化
        7. Output Head: 线性投影到词汇表大小
    """
    def __init__(self, cfg):
        """
        初始化 GPT 模型
        
        组件说明：
        - token_embedding: 词元嵌入层，将 token ID (0~vocab_size-1) 映射到 embedding_dim 维向量
        - position_embedding: 位置嵌入层，为每个位置 (0~max_seq_length-1) 生成 embedding_dim 维向量
        - transformer_blocks: N 个 Transformer 块的序列，每个块包含注意力机制和前馈网络
        - layer_norm: 最终层归一化，稳定输出
        - out_head: 输出头，将 embedding_dim 维向量投影到 vocab_size 维（词汇表大小）
        - drop: Dropout 层，训练时随机将部分神经元置零，防止过拟合
        """
        super().__init__()
        # 词元嵌入：将离散的 token ID 转换为连续的向量表示
        # 输入: (batch_size, seq_len) → 输出: (batch_size, seq_len, embedding_dim)
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        
        # 位置嵌入：为序列中的每个位置添加位置信息
        # 输入: (seq_len,) → 输出: (seq_len, embedding_dim)
        self.position_embedding = nn.Embedding(cfg["max_seq_length"], cfg["embedding_dim"])
        
        # ========== Transformer 块序列 ==========
        # nn.Sequential: 按顺序执行多个模块的容器
        # 
        # 工作原理：
        # 1. Sequential 是一个容器类，将多个模块按顺序组织
        # 2. 前向传播时，输入会依次通过每个模块
        # 3. 每个模块的输出作为下一个模块的输入
        # 
        # 语法说明：
        # - *[MyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        #   * 是解包操作符，将列表展开为多个参数
        #   [MyTransformerBlock(cfg) for _ in range(cfg["n_layers"])] 
        #   创建 n_layers 个 Transformer 块的列表
        #   例如：n_layers=12 会创建 12 个独立的 Transformer 块
        # 
        # 执行流程：
        # 输入 x → Block 1 → Block 2 → ... → Block N → 输出
        # 
        # 每个块的结构：
        # - 多头自注意力 (Multi-Head Self-Attention)
        # - 前馈网络 (Feed Forward Network)
        # - 残差连接 (Residual Connection)
        # - 层归一化 (Layer Normalization)
        # 
        # 为什么使用 Sequential：
        # 1. 代码简洁：一行代码定义多个层
        # 2. 自动管理：PyTorch 自动管理模块的顺序执行
        # 3. 易于扩展：可以轻松调整层数
        # 
        # 等价写法（不使用 Sequential）：
        # self.block_1 = MyTransformerBlock(cfg)
        # self.block_2 = MyTransformerBlock(cfg)
        # ...
        # self.block_N = MyTransformerBlock(cfg)
        # 然后在 forward 中手动调用每个块
        self.transformer_blocks = nn.Sequential(
            *[MyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # 最终层归一化：在所有 Transformer 块之后进行归一化
        # 有助于稳定训练和改善梯度流
        self.layer_norm = nn.LayerNorm(cfg["embedding_dim"])
        
        # 输出头：将隐藏状态投影到词汇表大小
        # 输出每个位置对应词汇表中每个词的概率（logits）
        # bias=False: 通常不使用偏置，因为嵌入层已经包含了偏置信息
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)
        
        # Dropout：训练时随机失活，防止过拟合
        self.drop = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        """
        前向传播过程
        
        Args:
            x: 输入 token IDs，形状为 (batch_size, seq_len)
               - batch_size: 批次大小，一次处理多少个样本
               - seq_len: 序列长度，每个样本的 token 数量
               - 每个元素是 token 在词汇表中的 ID (0 ~ vocab_size-1)
        
        Returns:
            logits: 输出 logits，形状为 (batch_size, seq_len, vocab_size)
                   每个位置对应词汇表中每个词的未归一化概率分数
        
        流程详解：
        ========================================================================
        步骤 1: Token Embedding (词元嵌入)
        ------------------------------------------------------------------------
        输入: x (batch_size, seq_len) - Token IDs
        操作: 通过嵌入查找表将每个 token ID 转换为 embedding_dim 维向量
        输出: token_embeds (batch_size, seq_len, embedding_dim)
        
        示例: token_id=5 → [0.1, -0.3, 0.8, ..., 0.2] (embedding_dim 维向量)
        
        ========================================================================
        步骤 2: Position Embedding (位置嵌入)
        ------------------------------------------------------------------------
        输入: 位置索引 [0, 1, 2, ..., seq_len-1]
        操作: 为每个位置生成唯一的 embedding_dim 维向量
        输出: position_embeds (seq_len, embedding_dim)
        
        作用: 让模型知道每个 token 在序列中的位置信息
        注意: 位置嵌入是固定的，不随输入内容变化
        
        ========================================================================
        步骤 3: 嵌入融合 (Embedding Fusion)
        ------------------------------------------------------------------------
        操作: x = token_embeds + position_embeds
        机制: 广播机制自动扩展 position_embeds 到 (batch_size, seq_len, embedding_dim)
        输出: x (batch_size, seq_len, embedding_dim)
        
        原理: 将语义信息（token embedding）和位置信息（position embedding）相加
             这样每个 token 的表示既包含其含义，也包含其位置
        
        ========================================================================
        步骤 4: Dropout (随机失活)
        ------------------------------------------------------------------------
        操作: 训练时随机将部分元素置零，测试时保持不变
        作用: 防止过拟合，提高模型泛化能力
        输出: x (batch_size, seq_len, embedding_dim)
        
        ========================================================================
        步骤 5: Transformer Blocks (Transformer 块序列)
        ------------------------------------------------------------------------
        操作: 通过 N 个 Transformer 块处理
        每个块包含:
            - 多头自注意力 (Multi-Head Self-Attention)
            - 前馈网络 (Feed Forward Network)
            - 残差连接 (Residual Connection)
            - 层归一化 (Layer Normalization)
        输出: x (batch_size, seq_len, embedding_dim)
        
        作用: 逐步提取和整合序列中的信息，建立 token 之间的依赖关系
        
        ========================================================================
        步骤 6: 最终层归一化 (Final Layer Normalization)
        ------------------------------------------------------------------------
        操作: 对最后一个维度进行归一化
        作用: 稳定输出，改善数值稳定性
        输出: x (batch_size, seq_len, embedding_dim)
        
        ========================================================================
        步骤 7: 输出头 (Output Head)
        ------------------------------------------------------------------------
        操作: 线性投影到词汇表大小
        输入: x (batch_size, seq_len, embedding_dim)
        输出: logits (batch_size, seq_len, vocab_size)
        
        含义: 每个位置的 logits 表示该位置预测每个词的概率分数
             后续通过 softmax 可以转换为概率分布
        ========================================================================
        """
        # 获取输入维度
        # x 形状: (batch_size, seq_len)
        # - batch_size: 批次中样本数量
        # - seq_len: 每个样本的序列长度（token 数量）
        batch_size, seq_len = x.shape

        # ========== 步骤 1: Token Embedding ==========
        # 将 token ID 转换为嵌入向量
        # 输入: (batch_size, seq_len) - Token IDs
        # 输出: (batch_size, seq_len, embedding_dim) - 嵌入向量
        token_embeds = self.token_embedding(x)

        # ========== 步骤 2: Position Embedding ==========
        # 为序列中的每个位置生成位置嵌入
        # 
        # torch.arange 详解：
        # ========================================================================
        # torch.arange(start, end, step, device) 生成一个从 start 到 end-1 的序列
        # 
        # 参数说明：
        # - seq_len: 序列长度，例如 10
        # - device: 指定张量所在的设备（CPU 或 GPU），必须与输入 x 在同一设备
        # 
        # 返回值：
        # - 生成一个一维张量：[0, 1, 2, 3, ..., seq_len-1]
        # - 形状: (seq_len,)
        # 
        # 示例：
        # 如果 seq_len = 5，则 torch.arange(5) 生成：
        # tensor([0, 1, 2, 3, 4])
        # 
        # 如果 seq_len = 10，则 torch.arange(10) 生成：
        # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # 
        # 为什么使用 torch.arange：
        # 1. 生成位置索引：每个位置对应一个唯一的索引（0, 1, 2, ...）
        # 2. 位置嵌入查找：通过索引在 position_embedding 中查找对应的位置向量
        # 3. 设备一致性：device=x.device 确保位置索引与输入数据在同一设备（CPU/GPU）
        # 
        # 工作流程：
        # 1. torch.arange(seq_len, device=x.device) 
        #    → 生成位置索引 [0, 1, 2, ..., seq_len-1]
        # 2. self.position_embedding(位置索引)
        #    → 通过嵌入层查找每个位置对应的位置向量
        #    → 输出: (seq_len, embedding_dim)
        # 
        # 例如：seq_len=3, embedding_dim=768
        # 位置索引: [0, 1, 2]
        # 位置嵌入查找:
        #   - 位置 0 → 查找 position_embedding[0] → (768,) 维向量
        #   - 位置 1 → 查找 position_embedding[1] → (768,) 维向量
        #   - 位置 2 → 查找 position_embedding[2] → (768,) 维向量
        # 最终输出: (3, 768) - 3 个位置，每个位置 768 维向量
        # ========================================================================
        postion_embeds = self.position_embedding(torch.arange(seq_len, device=x.device))

        # ========== 步骤 3: 嵌入融合 ==========
        # 将词元嵌入和位置嵌入相加
        # 广播机制: (batch_size, seq_len, embedding_dim) + (seq_len, embedding_dim)
        #          → (batch_size, seq_len, embedding_dim)
        x = token_embeds + postion_embeds

        # ========== 步骤 4: Dropout ==========
        # 训练时随机失活，防止过拟合
        x = self.drop(x)

        # ========== 步骤 5: Transformer Blocks ==========
        # 通过 N 个 Transformer 块处理
        # 每个块包含: 多头自注意力 + 前馈网络 + 残差连接 + 层归一化
        x = self.transformer_blocks(x)

        # ========== 步骤 6: 最终层归一化 ==========
        # 稳定输出，改善数值稳定性
        x = self.layer_norm(x)

        # ========== 步骤 7: 输出头 ==========
        # 线性投影到词汇表大小，得到 logits
        # 输出: (batch_size, seq_len, vocab_size)
        logits = self.out_head(x)

        return logits

class NewGELU(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) 激活函数
    
    GELU 是 Transformer 中常用的激活函数，相比 ReLU 更平滑。
    公式: GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    特点:
    - 平滑可导，有利于梯度流动
    - 在负值区域也有输出（虽然很小），保留更多信息
    - 在 GPT、BERT 等模型中广泛使用
    
    Args:
        x: 输入张量，任意形状
    
    Returns:
        输出张量，形状与输入相同
    """
    def forward(self, x):
        # GELU 激活函数实现
        # 使用 tanh 近似，计算效率高且数值稳定
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class FeedForwardNetwork(nn.Module):
    """
    前馈神经网络 (Feed Forward Network, FFN)
    
    Transformer 中的前馈网络是一个两层全连接网络，采用"扩展-压缩"结构：
    1. 第一层：扩展维度 (embedding_dim → 4×embedding_dim)
    2. 激活函数：GELU
    3. 第二层：压缩维度 (4×embedding_dim → embedding_dim)
    
    结构流程：
    ========================================================================
    输入: x (batch_size, seq_len, embedding_dim)
         ↓
    [Linear 1] 扩展维度
         ↓
    (batch_size, seq_len, 4×embedding_dim)
         ↓
    [GELU] 激活函数
         ↓
    [Linear 2] 压缩维度
         ↓
    输出: (batch_size, seq_len, embedding_dim)
    ========================================================================
    
    设计原理:
    - 扩展维度增加模型表达能力
    - 通常扩展 4 倍（embedding_dim → 4×embedding_dim）
    - 每个位置独立处理，不依赖其他位置的信息
    - 与注意力机制互补：注意力关注"哪里"，FFN 关注"什么"
    
    Args:
        cfg (dict): 配置字典，需要包含 "embedding_dim"
    """
    def __init__(self, cfg):
        super().__init__()
        # ========== 前馈网络层序列 ==========
        # nn.Sequential: 按顺序执行多个层的容器
        # 
        # Sequential 的工作原理：
        # 1. 将多个模块按顺序组织成一个整体
        # 2. 前向传播时，数据依次通过每个模块
        # 3. 每个模块的输出自动作为下一个模块的输入
        # 
        # 本前馈网络的结构（3 层）：
        # Layer 1: Linear (扩展维度)
        #    ↓
        # Layer 2: GELU (激活函数)
        #    ↓
        # Layer 3: Linear (压缩维度)
        # 
        # 数据流：
        # 输入 (batch_size, seq_len, embedding_dim)
        #    ↓ [Linear 1]
        # (batch_size, seq_len, 4×embedding_dim)
        #    ↓ [GELU]
        # (batch_size, seq_len, 4×embedding_dim) [激活后]
        #    ↓ [Linear 2]
        # 输出 (batch_size, seq_len, embedding_dim)
        # 
        # 为什么使用 Sequential：
        # 1. 代码简洁：一行定义整个网络结构
        # 2. 自动管理：PyTorch 自动处理层之间的数据传递
        # 3. 易于理解：结构清晰，一目了然
        # 
        # 等价写法（不使用 Sequential）：
        # self.linear1 = nn.Linear(cfg["embedding_dim"], 4*cfg["embedding_dim"])
        # self.activation = NewGELU()
        # self.linear2 = nn.Linear(4*cfg["embedding_dim"], cfg["embedding_dim"])
        # 然后在 forward 中：x = self.linear2(self.activation(self.linear1(x)))
        self.layers = nn.Sequential(
            # ========== 第一层：扩展维度 ==========
            # 作用：增加模型表达能力，从 embedding_dim 扩展到 4×embedding_dim
            # 输入: (batch_size, seq_len, embedding_dim)
            # 输出: (batch_size, seq_len, 4×embedding_dim)
            # 
            # 设计原理：
            # - 扩展维度允许模型学习更复杂的特征表示
            # - 通常扩展 4 倍（这是 Transformer 的标准做法）
            # - 例如：embedding_dim=768 → 扩展到 3072
            nn.Linear(cfg["embedding_dim"], 4*cfg["embedding_dim"]),
            
            # ========== 第二层：激活函数 ==========
            # 作用：引入非线性，使模型能够学习复杂的模式
            # 输入: (batch_size, seq_len, 4×embedding_dim)
            # 输出: (batch_size, seq_len, 4×embedding_dim) [激活后]
            # 
            # GELU 特点：
            # - 平滑可导，有利于梯度流动
            # - 在负值区域也有输出（虽然很小），保留更多信息
            # - 在 GPT、BERT 等模型中广泛使用
            # 
            # 也可以使用 PyTorch 内置的 GELU: nn.GELU()
            NewGELU(),
            
            # ========== 第三层：压缩维度 ==========
            # 作用：将扩展后的维度压缩回原始维度
            # 输入: (batch_size, seq_len, 4×embedding_dim)
            # 输出: (batch_size, seq_len, embedding_dim)
            # 
            # 设计原理：
            # - 保持输入输出维度一致，便于残差连接
            # - 压缩过程进一步提取和整合特征
            # - 与第一层形成"扩展-压缩"结构
            nn.Linear(4*cfg["embedding_dim"], cfg["embedding_dim"])
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, embedding_dim)
        
        Returns:
            输出张量 (batch_size, seq_len, embedding_dim)
        """
        return self.layers(x)

class MyTransformerBlock(nn.Module):
    """
    Transformer 块 (Transformer Block)
    
    这是 Transformer 的核心组件，每个块包含两个子层：
    1. 多头自注意力层 (Multi-Head Self-Attention)
    2. 前馈神经网络层 (Feed Forward Network)
    
    每个子层都采用 Pre-LayerNorm 架构：
    - 先进行层归一化
    - 再执行子层操作
    - 最后添加残差连接
    
    块结构流程图：
    ========================================================================
    输入: x (batch_size, seq_len, embedding_dim)
         ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 子层 1: 多头自注意力 (Multi-Head Self-Attention)            │
    ├─────────────────────────────────────────────────────────────┤
    │ 1. Layer Norm 1                                             │
    │    └─ 归一化输入，稳定训练                                  │
    │ 2. Multi-Head Self-Attention                                │
    │    └─ 计算注意力分数，整合序列信息                          │
    │    └─ 掩码机制：防止看到未来信息（自回归模型）              │
    │ 3. Dropout                                                  │
    │ 4. 残差连接: x = x + old_x                                  │
    │    └─ 缓解梯度消失，允许训练更深的网络                      │
    └─────────────────────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 子层 2: 前馈神经网络 (Feed Forward Network)                 │
    ├─────────────────────────────────────────────────────────────┤
    │ 1. Layer Norm 2                                             │
    │ 2. Feed Forward Network                                     │
    │    └─ 扩展维度 → GELU → 压缩维度                            │
    │ 3. Dropout                                                  │
    │ 4. 残差连接: x = x + old_x                                  │
    └─────────────────────────────────────────────────────────────┘
         ↓
    输出: x (batch_size, seq_len, embedding_dim)
    ========================================================================
    
    配置示例:
    MY_GPT_CONFIG = {
        "vocab_size": 50257,      # 词汇表大小（GPT-2）
        "max_seq_length": 1024,   # 最大序列长度
        "embedding_dim": 768,     # 嵌入向量维度
        "n_heads": 12,            # 注意力头数量
        "n_layers": 12,           # Transformer 层数
        "drop_rate": 0.1,         # Dropout 比率
        "qkv_bias": False         # Q、K、V 是否使用偏置
    }
    
    Args:
        cfg (dict): 配置字典，包含模型超参数
    """
    def __init__(self, cfg):
        super().__init__()
        # 多头自注意力层
        # 作用: 让模型关注序列中不同位置的信息，建立 token 之间的依赖关系
        # 特点: 使用掩码机制，防止看到未来信息（自回归模型要求）
        self.mha = MultiHeadAttention(
            d_in=cfg["embedding_dim"],           # 输入维度
            d_out=cfg["embedding_dim"],           # 输出维度
            num_heads=cfg["n_heads"],            # 注意力头数量
            drop_rate=cfg["drop_rate"],          # Dropout 比率
            mask_matrix_len=cfg["max_seq_length"], # 掩码矩阵长度
            qkv_bias=cfg["qkv_bias"]             # Q、K、V 是否使用偏置
        )
        
        # 前馈神经网络层
        # 作用: 对每个位置独立进行非线性变换，增加模型表达能力
        self.ffn = FeedForwardNetwork(cfg)
        
        # 层归一化 1: 用于注意力子层之前
        # 作用: 归一化输入，稳定训练过程，改善梯度流
        self.norm_1 = nn.LayerNorm(cfg["embedding_dim"])
        
        # 层归一化 2: 用于前馈网络子层之前
        self.norm_2 = nn.LayerNorm(cfg["embedding_dim"])
        
        # Dropout: 随机失活，防止过拟合
        self.dropout = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, embedding_dim)
        
        Returns:
            输出张量 (batch_size, seq_len, embedding_dim)
        
        详细流程：
        ========================================================================
        子层 1: 多头自注意力
        ------------------------------------------------------------------------
        1. 保存输入用于残差连接: old_x = x
        2. 层归一化: x = LayerNorm(x)
        3. 多头自注意力: x = MultiHeadAttention(x)
           - 计算 Q、K、V
           - 计算注意力分数: Scores = QK^T / √d_k
           - 应用掩码（防止看到未来信息）
           - Softmax 归一化
           - 加权求和: Attention = softmax(Scores) × V
        4. Dropout: x = Dropout(x)
        5. 残差连接: x = x + old_x
           - 作用: 保留原始信息，缓解梯度消失
        
        ========================================================================
        子层 2: 前馈神经网络
        ------------------------------------------------------------------------
        1. 保存当前状态: old_x = x
        2. 层归一化: x = LayerNorm(x)
        3. 前馈网络: x = FFN(x)
           - Linear(embedding_dim → 4×embedding_dim)
           - GELU 激活
           - Linear(4×embedding_dim → embedding_dim)
        4. Dropout: x = Dropout(x)
        5. 残差连接: x = x + old_x
        
        ========================================================================
        """
        # ========== 子层 1: 多头自注意力 ==========
        # 保存输入用于残差连接
        old_x = x
        
        # Pre-LayerNorm: 先归一化，再执行注意力
        x = self.norm_1(x)
        
        # 多头自注意力：计算序列中 token 之间的依赖关系
        # 掩码机制确保自回归特性（不能看到未来信息）
        x = self.mha(x)
        
        # Dropout: 训练时随机失活
        x = self.dropout(x)
        
        # 残差连接：x = x + old_x
        # 作用：
        # 1. 保留原始信息，防止信息丢失
        # 2. 缓解梯度消失问题，允许训练更深的网络
        # 3. 使网络更容易学习恒等映射
        x = x + old_x
        
        # ========== 子层 2: 前馈神经网络 ==========
        # 保存当前状态用于残差连接
        old_x = x
        
        # Pre-LayerNorm: 先归一化，再执行前馈网络
        x = self.norm_2(x)
        
        # 前馈网络：对每个位置独立进行非线性变换
        # 扩展维度 → 激活 → 压缩维度
        x = self.ffn(x)
        
        # Dropout: 训练时随机失活
        x = self.dropout(x)
        
        # 残差连接：x = x + old_x
        x = x + old_x
        
        return x