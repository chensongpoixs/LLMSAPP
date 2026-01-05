"""
自注意力机制 (Self-Attention) 实现

本文件实现了多个版本的自注意力机制，从基础版本到完整的多头注意力机制。

自注意力机制核心思想：
================================================================================
自注意力机制允许序列中的每个位置关注序列中的所有位置（包括自己），
通过计算位置之间的相关性来整合信息。

核心公式：
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

其中：
- Q (Query): 查询向量，表示"我想知道什么"
- K (Key): 键向量，表示"我有什么信息"
- V (Value): 值向量，表示"实际的信息内容"
- d_k: 键向量的维度，用于缩放点积，防止梯度消失

工作流程：
================================================================================
输入: x (batch_size, seq_len, embedding_dim)
    ↓
[1. 线性投影] 生成 Q, K, V
    ├─ Q = x × W_q  (查询向量)
    ├─ K = x × W_k  (键向量)
    └─ V = x × W_v  (值向量)
    ↓
[2. 计算注意力分数]
    └─ Scores = Q × K^T  (点积，计算相似度)
    ↓
[3. 缩放和归一化]
    ├─ Scaled_Scores = Scores / √d_k  (防止梯度爆炸)
    └─ Attention_Weights = softmax(Scaled_Scores)  (归一化为概率分布)
    ↓
[4. 加权求和]
    └─ Output = Attention_Weights × V  (根据权重聚合值向量)
    ↓
输出: (batch_size, seq_len, d_out)
================================================================================

掩码机制（用于自回归模型）：
================================================================================
掩码的作用：防止模型在预测时看到未来的信息（自回归特性）

掩码矩阵示例（seq_len=4）：
    [0, 1, 1, 1]  ← 位置 0 只能看到自己
    [0, 0, 1, 1]  ← 位置 1 只能看到位置 0 和 1
    [0, 0, 0, 1]  ← 位置 2 只能看到位置 0, 1, 2
    [0, 0, 0, 0]  ← 位置 3 可以看到所有位置

1 表示需要掩码（设为 -∞），0 表示可见

应用方式：
attn_score.masked_fill_(mask == 1, -torch.inf)
→ 被掩码的位置在 softmax 后概率为 0
================================================================================

多头注意力（Multi-Head Attention）：
================================================================================
多头注意力将注意力机制并行执行多次，每次使用不同的权重矩阵，
允许模型从不同的"视角"关注信息。

流程：
1. 生成 Q, K, V (d_out 维)
2. 分割成 num_heads 个头，每个头 d_k = d_out / num_heads 维
3. 每个头独立计算注意力
4. 拼接所有头的输出
5. 通过输出投影层整合
================================================================================
"""

import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    """
    自注意力机制 - 版本 1（基础实现）
    
    这是最基础的自注意力实现，使用 nn.Parameter 定义权重矩阵。
    适用于理解自注意力机制的核心原理。
    
    架构流程：
    ========================================================================
    输入: x (batch_size, seq_len, d_in)
         ↓
    [线性投影] 通过权重矩阵生成 Q, K, V
         ├─ Q = x @ W_q  (batch_size, seq_len, d_out)
         ├─ K = x @ W_k  (batch_size, seq_len, d_out)
         └─ V = x @ W_v  (batch_size, seq_len, d_out)
         ↓
    [计算注意力分数]
         └─ attn_score = Q @ K^T  (batch_size, seq_len, seq_len)
         ↓
    [缩放和归一化]
         ├─ scaled_score = attn_score / √d_k
         └─ attn_weight = softmax(scaled_score, dim=-1)
         ↓
    [加权求和]
         └─ context_vec = attn_weight @ V  (batch_size, seq_len, d_out)
         ↓
    输出: context_vec (batch_size, seq_len, d_out)
    ========================================================================
    
    Args:
        d_in: 输入维度（embedding_dim）
        d_out: 输出维度（通常等于 d_in，但可以不同）
    
    注意：
    - 使用 nn.Parameter 需要手动初始化权重
    - 没有偏置项
    - 没有 Dropout
    - 没有掩码机制
    """
    def __init__(self, d_in, d_out):
        """
        初始化自注意力层
        
        Args:
            d_in: 输入维度，例如 768（GPT-2 的 embedding_dim）
            d_out: 输出维度，例如 64（可以自定义）
        
        权重矩阵说明：
        - W_q: 查询权重矩阵 (d_in, d_out)
        - W_k: 键权重矩阵 (d_in, d_out)
        - W_v: 值权重矩阵 (d_in, d_out)
        
        注意：使用 torch.rand() 随机初始化，范围 [0, 1)
        """
        super().__init__()
        # 查询权重矩阵：将输入投影到查询空间
        self.W_q = nn.Parameter(torch.rand(d_in, d_out))
        # 键权重矩阵：将输入投影到键空间
        self.W_k = nn.Parameter(torch.rand(d_in, d_out))
        # 值权重矩阵：将输入投影到值空间
        self.W_v = nn.Parameter(torch.rand(d_in, d_out))
        self.d_out = d_out

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, d_in)
        
        Returns:
            context_vec: 上下文向量 (batch_size, seq_len, d_out)
        
        详细步骤：
        ========================================================================
        步骤 1: 生成 Q, K, V
        ------------------------------------------------------------------------
        操作: q = x @ W_q, k = x @ W_k, v = x @ W_v
        说明: @ 是矩阵乘法的简写，等价于 torch.matmul()
        维度: 
            - 输入: (batch_size, seq_len, d_in)
            - 权重: (d_in, d_out)
            - 输出: (batch_size, seq_len, d_out)
        
        步骤 2: 计算注意力分数
        ------------------------------------------------------------------------
        操作: attn_score = q @ k.T
        说明: 计算每个位置对每个位置的相似度
        维度:
            - q: (batch_size, seq_len, d_out)
            - k.T: (batch_size, d_out, seq_len) [转置后]
            - attn_score: (batch_size, seq_len, seq_len)
        
        含义: attn_score[i][j] 表示位置 i 对位置 j 的注意力分数
        
        步骤 3: 缩放和归一化
        ------------------------------------------------------------------------
        操作: 
            1. scaled_score = attn_score / √d_k
            2. attn_weight = softmax(scaled_score, dim=-1)
        
        缩放原因:
            - d_k 较大时，点积结果可能很大，导致 softmax 梯度接近 0
            - 除以 √d_k 可以稳定梯度，这是 Transformer 论文中的标准做法
        
        Softmax 作用:
            - 将注意力分数转换为概率分布
            - 每行的和等于 1
            - 值越大，注意力权重越高
        
        维度: (batch_size, seq_len, seq_len)
        
        步骤 4: 加权求和
        ------------------------------------------------------------------------
        操作: context_vec = attn_weight @ v
        说明: 根据注意力权重对值向量进行加权求和
        维度:
            - attn_weight: (batch_size, seq_len, seq_len)
            - v: (batch_size, seq_len, d_out)
            - context_vec: (batch_size, seq_len, d_out)
        
        含义: 每个位置的输出是序列中所有位置值的加权平均
        ========================================================================
        """
        # ========== 步骤 1: 生成 Q, K, V ==========
        # @ 是矩阵乘法的简写，等价于 torch.matmul()
        # 将输入 x 通过不同的权重矩阵投影到查询、键、值空间
        q = x @ self.W_q  # (batch_size, seq_len, d_out)
        k = x @ self.W_k  # (batch_size, seq_len, d_out)
        v = x @ self.W_v  # (batch_size, seq_len, d_out)
        
        # ========== 步骤 2: 计算注意力分数 ==========
        # Q @ K^T: 计算每个位置对每个位置的相似度
        # 结果: (batch_size, seq_len, seq_len)
        # attn_score[i][j] 表示位置 i 对位置 j 的注意力分数
        attn_score = q @ k.T
        
        # ========== 步骤 3: 缩放和归一化 ==========
        # 缩放: 除以 √d_k，防止点积结果过大导致梯度消失
        # k.shape[-1] 是 d_out，即 d_k
        # softmax: 将分数转换为概率分布（每行和为 1）
        attn_weight = torch.softmax(attn_score / k.shape[-1]**0.5, dim=-1)
        
        # ========== 步骤 4: 加权求和 ==========
        # 根据注意力权重对值向量进行加权求和
        # 每个位置的输出是序列中所有位置值的加权平均
        context_vec = attn_weight @ v
        
        return context_vec

class MySelfAttention_v2(nn.Module):
    """
    自注意力机制 - 版本 2（使用 Linear 层）
    
    这是改进版本，使用 nn.Linear 替代 nn.Parameter，具有以下优势：
    1. 自动初始化权重（Xavier/Kaiming 初始化）
    2. 可选择是否使用偏置项
    3. 代码更简洁，更符合 PyTorch 最佳实践
    
    架构流程与版本 1 相同，但使用 Linear 层实现。
    
    Args:
        d_in: 输入维度
        d_out: 输出维度
        qkv_bias: 是否在 Q、K、V 线性层使用偏置项（默认 False）
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        """
        初始化自注意力层
        
        Args:
            d_in: 输入维度
            d_out: 输出维度
            qkv_bias: 是否使用偏置项
                     - True: 使用偏置，增加模型表达能力
                     - False: 不使用偏置（GPT-2 等模型的标准做法）
        
        优势：
        - nn.Linear 自动使用合适的权重初始化方法
        - 可以选择是否使用偏置项
        - 代码更简洁，更易维护
        """
        super().__init__()
        # 使用 Linear 层替代 Parameter，自动管理权重和偏置
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out
    
    def forward(self, x):
        """
        前向传播
        
        流程与版本 1 相同，但使用 Linear 层的方法调用方式。
        
        Args:
            x: 输入张量 (batch_size, seq_len, d_in)
        
        Returns:
            context_vec: 上下文向量 (batch_size, seq_len, d_out)
        """
        # 通过 Linear 层生成 Q, K, V
        q = self.W_q(x)  # (batch_size, seq_len, d_out)
        k = self.W_k(x)  # (batch_size, seq_len, d_out)
        v = self.W_v(x)  # (batch_size, seq_len, d_out)
        
        # 计算注意力分数
        attn_score = q @ k.T  # (batch_size, seq_len, seq_len)
        
        # 缩放和归一化
        attn_weight = torch.softmax(attn_score / k.shape[-1]**0.5, dim=-1)
        
        # 加权求和
        context_vec = attn_weight @ v  # (batch_size, seq_len, d_out)
        
        return context_vec  
    
class MaskSelfAttention(nn.Module):
    """
    带掩码的自注意力机制 (Masked Self-Attention)
    
    这是用于自回归模型（如 GPT）的自注意力实现，包含掩码机制和 Dropout。
    
    掩码机制说明：
    ========================================================================
    掩码的作用：防止模型在预测时看到未来的信息，确保自回归特性
    
    掩码矩阵生成：
    - torch.triu(ones, diagonal=1): 生成上三角矩阵
    - diagonal=1: 对角线以上的元素为 1，对角线及以下为 0
    
    示例（seq_len=4）：
    mask = [
        [0, 1, 1, 1],  ← 位置 0 只能看到自己（mask[0][0]=0）
        [0, 0, 1, 1],  ← 位置 1 只能看到位置 0 和 1
        [0, 0, 0, 1],  ← 位置 2 只能看到位置 0, 1, 2
        [0, 0, 0, 0]   ← 位置 3 可以看到所有位置
    ]
    
    应用方式：
    - 将 mask=1 的位置设为 -∞
    - 经过 softmax 后，这些位置的概率变为 0
    - 从而阻止模型看到未来信息
    ========================================================================
    
    架构流程：
    ========================================================================
    输入: x (batch_size, seq_len, d_in)
         ↓
    [生成 Q, K, V]
         ↓
    [计算注意力分数] attn_score = Q @ K^T
         ↓
    [应用掩码] 将未来位置设为 -∞
         ↓
    [缩放和归一化] softmax(attn_score / √d_k)
         ↓
    [Dropout] 防止过拟合
         ↓
    [加权求和] context_vec = attn_weight @ V
         ↓
    输出: (batch_size, seq_len, d_out)
    ========================================================================
    
    Args:
        d_in: 输入维度
        d_out: 输出维度
        mask_matrix_len: 掩码矩阵的最大长度（通常等于 max_seq_length）
        drop_rate: Dropout 比率
        qkv_bias: 是否在 Q、K、V 线性层使用偏置项
    """
    def __init__(self, d_in, d_out, mask_matrix_len, drop_rate, qkv_bias=False):
        """
        初始化带掩码的自注意力层
        
        Args:
            d_in: 输入维度
            d_out: 输出维度
            mask_matrix_len: 掩码矩阵的最大长度
            drop_rate: Dropout 比率（通常 0.1）
            qkv_bias: 是否使用偏置项
        """
        super().__init__()
        # Q, K, V 线性投影层
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out
        
        # Dropout: 训练时随机失活，防止过拟合
        self.dropout = nn.Dropout(drop_rate)
        
        # 注册掩码矩阵为缓冲区（不参与梯度更新，但会随模型保存/加载）
        # torch.triu(ones, diagonal=1): 生成上三角矩阵
        # diagonal=1: 对角线以上的元素为 1，对角线及以下为 0
        # 例如 4x4 矩阵：
        #   [0, 1, 1, 1]
        #   [0, 0, 1, 1]
        #   [0, 0, 0, 1]
        #   [0, 0, 0, 0]
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(mask_matrix_len, mask_matrix_len), diagonal=1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_length, embedding_dim)
        
        Returns:
            context_vec: 上下文向量 (batch_size, seq_length, d_out)
        
        详细步骤：
        ========================================================================
        步骤 1: 生成 Q, K, V
        ------------------------------------------------------------------------
        通过线性投影生成查询、键、值向量
        维度: (batch_size, seq_length, d_out)
        
        步骤 2: 计算注意力分数
        ------------------------------------------------------------------------
        操作: attn_score = q @ k.transpose(1, 2)
        
        为什么使用 transpose(1, 2) 而不是 .T:
        - 对于三维张量 (batch_size, seq_len, d_out)
        - .T 会转置所有维度，得到 (d_out, seq_len, batch_size) [错误]
        - transpose(1, 2) 只交换维度 1 和 2，得到 (batch_size, d_out, seq_len) [正确]
        
        维度变化:
            q: (batch_size, seq_length, d_out)
            k.transpose(1, 2): (batch_size, d_out, seq_length)
            attn_score: (batch_size, seq_length, seq_length)
        
        步骤 3: 应用掩码
        ------------------------------------------------------------------------
        操作: attn_score.masked_fill_(mask, -torch.inf)
        
        masked_fill_ 说明:
        - 带下划线表示原地操作（in-place），直接修改原张量
        - 不带下划线会返回新张量（masked_fill）
        
        掩码应用:
        - mask[i][j] == 1 的位置（未来位置）被设为 -∞
        - 经过 softmax 后，这些位置的概率变为 0
        - 从而阻止模型看到未来信息
        
        步骤 4: 缩放和归一化
        ------------------------------------------------------------------------
        操作: softmax(attn_score / √d_k, dim=-1)
        - 除以 √d_k 防止梯度消失
        - softmax 转换为概率分布
        
        步骤 5: Dropout
        ------------------------------------------------------------------------
        操作: 训练时随机将部分注意力权重置零
        作用: 增加模型泛化能力，防止过拟合
        
        步骤 6: 加权求和
        ------------------------------------------------------------------------
        操作: context_vec = attn_weight @ v
        根据注意力权重对值向量进行加权求和
        ========================================================================
        """
        batch_size, seq_length, embedding_dim = x.shape
        
        # ========== 步骤 1: 生成 Q, K, V ==========
        q = self.W_q(x)  # (batch_size, seq_length, d_out)
        k = self.W_k(x)  # (batch_size, seq_length, d_out)
        v = self.W_v(x)  # (batch_size, seq_length, d_out)
        
        # ========== 步骤 2: 计算注意力分数 ==========
        # 对于三维张量，不能直接使用 .T（会转置所有维度）
        # 需要使用 transpose(1, 2) 只交换维度 1 和 2
        # q: (batch_size, seq_length, d_out)
        # k.transpose(1, 2): (batch_size, d_out, seq_length)
        # 结果: (batch_size, seq_length, seq_length)
        attn_score = q @ k.transpose(1, 2)
        
        # ========== 步骤 3: 应用掩码 ==========
        # masked_fill_: 原地操作，直接修改原张量
        # masked_fill: 返回新张量（不修改原张量）
        # 
        # 操作说明:
        # - self.mask.bool(): 将掩码转换为布尔类型
        # - [:seq_length, :seq_length]: 根据实际序列长度截取掩码
        # - -torch.inf: 将被掩码的位置设为负无穷
        # 
        # 效果:
        # - 未来位置（mask=1）的注意力分数变为 -∞
        # - 经过 softmax 后，这些位置的概率变为 0
        # - 从而阻止模型看到未来信息
        attn_score.masked_fill_(self.mask.bool()[:seq_length, :seq_length], -torch.inf)
        
        # ========== 步骤 4: 缩放和归一化 ==========
        # 除以 √d_k 防止点积结果过大
        # softmax 将分数转换为概率分布（每行和为 1）
        attn_weight = torch.softmax(attn_score / k.shape[-1]**0.5, dim=-1)
        
        # ========== 步骤 5: Dropout ==========
        # 训练时随机将部分注意力权重置零，增加泛化能力
        attn_weight = self.dropout(attn_weight)
        
        # ========== 步骤 6: 加权求和 ==========
        # 根据注意力权重对值向量进行加权求和
        context_vec = attn_weight @ v  # (batch_size, seq_length, d_out)
        
        return context_vec  

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    
    这是 Transformer 的核心组件，将注意力机制并行执行多次，
    每次使用不同的权重矩阵，允许模型从不同的"视角"关注信息。
    
    多头注意力原理：
    ========================================================================
    核心思想：将 d_out 维的 Q、K、V 分割成 num_heads 个头，每个头独立计算注意力
    
    维度分配：
    - 总维度: d_out (例如 768)
    - 头数量: num_heads (例如 12)
    - 每个头的维度: head_dim = d_out / num_heads (例如 768 / 12 = 64)
    
    优势：
    1. 并行计算：多个头可以并行处理，提高效率
    2. 多视角：每个头关注不同的信息模式
    3. 表达能力：比单头注意力更强大
    
    架构流程：
    ========================================================================
    输入: x (batch_size, seq_len, d_in)
         ↓
    [生成 Q, K, V] 通过线性投影
         ↓ (batch_size, seq_len, d_out)
    [分割成多个头]
         ↓ (batch_size, seq_len, num_heads, head_dim)
    [调整维度顺序] 将 num_heads 移到第 2 维
         ↓ (batch_size, num_heads, seq_len, head_dim)
    [计算注意力] 每个头独立计算
         ↓ (batch_size, num_heads, seq_len, seq_len)
    [应用掩码] 防止看到未来信息
         ↓
    [Softmax] 归一化为概率分布
         ↓
    [Dropout] 防止过拟合
         ↓
    [加权求和] 每个头独立计算
         ↓ (batch_size, num_heads, seq_len, head_dim)
    [拼接所有头] 调整维度顺序并拼接
         ↓ (batch_size, seq_len, d_out)
    [输出投影] 通过线性层整合
         ↓
    输出: (batch_size, seq_len, d_out)
    ========================================================================
    
    Args:
        d_in: 输入维度
        d_out: 输出维度（必须能被 num_heads 整除）
        mask_matrix_len: 掩码矩阵的最大长度
        drop_rate: Dropout 比率
        num_heads: 注意力头的数量（例如 12）
        qkv_bias: 是否在 Q、K、V 线性层使用偏置项
    """
    def __init__(self, d_in, d_out, 
                 mask_matrix_len, drop_rate, num_heads, qkv_bias=False):
        """
        初始化多头注意力层
        
        Args:
            d_in: 输入维度
            d_out: 输出维度（必须能被 num_heads 整除）
            mask_matrix_len: 掩码矩阵的最大长度
            drop_rate: Dropout 比率
            num_heads: 注意力头的数量
            qkv_bias: 是否使用偏置项
        
        注意：
        - d_out 必须能被 num_heads 整除
        - 例如：d_out=768, num_heads=12 → head_dim=64
        """
        super().__init__()
        # Q, K, V 线性投影层（投影到 d_out 维）
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out
        
        # Dropout: 防止过拟合
        self.dropout = nn.Dropout(drop_rate)
        
        # 多头注意力参数
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度
        
        # 输出投影层：整合所有头的输出
        self.output = nn.Linear(d_out, d_out)
        
        # 注册掩码矩阵
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(mask_matrix_len, mask_matrix_len), diagonal=1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_length, embedding_dim)
        
        Returns:
            context_vec: 上下文向量 (batch_size, seq_length, d_out)
        
        详细步骤：
        ========================================================================
        步骤 1: 生成 Q, K, V
        ------------------------------------------------------------------------
        通过线性投影生成查询、键、值向量
        维度: (batch_size, seq_length, d_out)
        
        步骤 2: 分割成多个头
        ------------------------------------------------------------------------
        操作: view(batch_size, seq_length, num_heads, head_dim)
        
        示例: d_out=768, num_heads=12, head_dim=64
        - 输入: (batch_size, seq_length, 768)
        - 输出: (batch_size, seq_length, 12, 64)
        
        含义: 将 768 维分成 12 个头，每个头 64 维
        
        步骤 3: 调整维度顺序
        ------------------------------------------------------------------------
        操作: transpose(1, 2)
        
        目的: 将 num_heads 移到第 2 维，便于并行计算
        - 输入: (batch_size, seq_length, num_heads, head_dim)
        - 输出: (batch_size, num_heads, seq_length, head_dim)
        
        步骤 4: 计算注意力分数
        ------------------------------------------------------------------------
        操作: attn_score = q @ k.transpose(2, 3)
        
        维度变化:
            q: (batch_size, num_heads, seq_length, head_dim)
            k.transpose(2, 3): (batch_size, num_heads, head_dim, seq_length)
            attn_score: (batch_size, num_heads, seq_length, seq_length)
        
        含义: 每个头独立计算注意力分数
        
        步骤 5: 应用掩码
        ------------------------------------------------------------------------
        将未来位置的注意力分数设为 -∞
        
        步骤 6: 缩放和归一化
        ------------------------------------------------------------------------
        softmax(attn_score / √head_dim, dim=-1)
        注意: 这里除以 √head_dim（不是 √d_out）
        
        步骤 7: Dropout
        ------------------------------------------------------------------------
        训练时随机失活
        
        步骤 8: 加权求和
        ------------------------------------------------------------------------
        操作: context_vec = attn_weight @ v
        维度: (batch_size, num_heads, seq_length, head_dim)
        
        步骤 9: 拼接所有头
        ------------------------------------------------------------------------
        操作:
            1. transpose(1, 2): (batch_size, seq_length, num_heads, head_dim)
            2. contiguous().view(): (batch_size, seq_length, d_out)
        
        contiguous(): 确保内存连续，view 操作才能成功
        
        步骤 10: 输出投影
        ------------------------------------------------------------------------
        通过线性层整合所有头的信息
        ========================================================================
        """
        batch_size, seq_length, embedding_dim = x.shape
        
        # ========== 步骤 1: 生成 Q, K, V ==========
        # 通过线性投影生成查询、键、值向量
        # 维度: (batch_size, seq_length, d_out)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # ========== 步骤 2: 分割成多个头 ==========
        # 将 d_out 维分成 num_heads 个头，每个头 head_dim 维
        # 例如: d_out=768, num_heads=12 → head_dim=64
        # 维度: (batch_size, seq_length, num_heads, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # ========== 步骤 3: 调整维度顺序 ==========
        # 将 num_heads 移到第 2 维，便于并行计算每个头
        # 维度: (batch_size, num_heads, seq_length, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ========== 步骤 4: 计算注意力分数 ==========
        # 每个头独立计算注意力分数
        # q: (batch_size, num_heads, seq_length, head_dim)
        # k.transpose(2, 3): (batch_size, num_heads, head_dim, seq_length)
        # 结果: (batch_size, num_heads, seq_length, seq_length)
        attn_score = q @ k.transpose(2, 3)
        
        # ========== 步骤 5: 应用掩码 ==========
        # 将未来位置的注意力分数设为 -∞
        # 注意: 掩码需要扩展到 num_heads 维度
        attn_score.masked_fill_(self.mask.bool()[:seq_length, :seq_length], -torch.inf)
        
        # ========== 步骤 6: 缩放和归一化 ==========
        # 除以 √head_dim（每个头的维度），不是 √d_out
        # softmax 将分数转换为概率分布
        attn_weight = torch.softmax(attn_score / k.shape[-1]**0.5, dim=-1)
        
        # ========== 步骤 7: Dropout ==========
        # 训练时随机失活，增加泛化能力
        attn_weight = self.dropout(attn_weight)
        
        # ========== 步骤 8: 加权求和 ==========
        # 每个头独立计算加权求和
        # attn_weight: (batch_size, num_heads, seq_length, seq_length)
        # v: (batch_size, num_heads, seq_length, head_dim)
        # 结果: (batch_size, num_heads, seq_length, head_dim)
        context_vec = (attn_weight @ v).transpose(1, 2)
        
        # ========== 步骤 9: 拼接所有头 ==========
        # transpose(1, 2): 将 num_heads 移回第 2 维
        # contiguous(): 确保内存连续（view 操作要求）
        # view(): 将多个头拼接成 d_out 维
        # 维度: (batch_size, seq_length, d_out)
        context_vec = context_vec.contiguous().view(batch_size, seq_length, self.d_out)
        
        # ========== 步骤 10: 输出投影 ==========
        # 通过线性层整合所有头的信息
        context_vec = self.output(context_vec)
        
        return context_vec  

# ========================================================================
# 测试代码示例
# ========================================================================
# 
# 自注意力机制的输入输出维度说明：
# ------------------------------------------------------------------------
# 输入: (batch_size, seq_length, embedding_dim)
# 输出: (batch_size, seq_length, d_out)
# 
# 注意：
# - Transformer Block 的输入是三维的 (batch_size, seq_length, embedding_dim)
# - 自注意力机制的输出维度 d_out 通常等于 embedding_dim
# - 但也可以不同，例如用于降维或升维
# 
# 测试示例：
# ------------------------------------------------------------------------
# d_in = 768        # 输入维度（例如 GPT-2 的 embedding_dim）
# d_out = 64        # 输出维度（可以自定义）
# torch.manual_seed(123)  # 设置随机种子，保证结果可复现
# attn = SelfAttention_v1(d_in, d_out)
# 
# # 创建测试输入
# # inputs = [[0.1, -0.32], [0.06, -0.234]]  # 二维输入（已注释，不适用）
# # 实际使用时应该是三维输入: (batch_size, seq_length, embedding_dim)
# 
# # 示例输入（三维）:
# # batch_size = 2, seq_length = 3, embedding_dim = 768
# # inputs = torch.randn(2, 3, 768)
# # output = attn(inputs)  # 输出: (2, 3, 64)
# ========================================================================