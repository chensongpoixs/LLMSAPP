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



import torch
import torch.nn as nn
import math

from MySelfAttention import MultiHeadAttention

class MyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.position_embedding = nn.Embedding(cfg["max_seq_length"], cfg["embedding_dim"])
        self.transformer_blocks = nn.Sequential(
            *[MyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        #self.layer_norm = MyLayerNorm()
        self.layer_norm = nn.LayerNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)
        self.drop = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        #x它是一个矩阵，每一行是段训练数据（也就是一句话）
        #x不是文字，而是文字所对应的token ID 串
        #所以，x中包括了多行训练数据，称为一个批量
        #它的列表示，每一段训练数据的长度
        batch_size, seq_len = x.shape

        #1. batch_size; 2. seq_len; 3. embedding_dim
        token_embeds = self.token_embedding(x) #token_embeds 是一个三维的矩阵

        #position_embedding结果是一个二维矩阵
        #每一行表示arange生成的字符
        #而每一行的列数是由embedding_dim决定的，GPT-2中是768
        postion_embeds = self.position_embedding(torch.arange(seq_len, device=x.device))

        #广播机制（batch_size, seq_len, embedding_dim), (batch_size, seq_len, embedding_dim)
        x = token_embeds + postion_embeds

        #防止过拟合
        x = self.drop(x)

        #(batch_size, seq_len, embedding_dim)
        x = self.transformer_blocks(x)

        x = self.layer_norm(x)

        logits = self.out_head(x)

        return logits

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embedding_dim"], 4*cfg["embedding_dim"]),
            NewGELU(),
            #nn.GELU(),
            nn.Linear(4*cfg["embedding_dim"], cfg["embedding_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)

class MyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # MY_GPT_CONFIG = {
        #   "vocab_size": 50257,    #词汇表大小
        #   "max_seq_length": 1024, #每一句训练数据的最大长度
        #   "embedding_dim": 768,   #嵌入向量的维度
        #   "n_heads": 12,          #注意力头个数
        #   "n_layers": 12,         #Transformer 层数
        #   "drop_rate": 0.1,       #Dropout rate
        #   "qkv_bias": False       #bias
        # }
        self.mha = MultiHeadAttention(
            d_in=cfg["embedding_dim"],
            d_out=cfg["embedding_dim"],
            num_heads=cfg["n_heads"],
            drop_rate=cfg["drop_rate"],
            mask_matrix_len=cfg["max_seq_length"],
            qkv_bias=cfg["qkv_bias"])
        self.ffn = FeedForwardNetwork(cfg)
        self.norm_1 = nn.LayerNorm(cfg["embedding_dim"])
        self.norm_2 = nn.LayerNorm(cfg["embedding_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        old_x = x
        x = self.norm_1(x)
        x = self.mha(x)
        x = self.dropout(x)
        #残差
        x = x + old_x
        old_x = x #为后面的残差做准备
        x = self.norm_2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + old_x
        return x