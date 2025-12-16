'''
@author: chensong
@date: 2025-12-16
Docstring for 06.Transformer架构神秘.config
'''


GPT_CONFIG = {
    "vocab_size": 50257,    #词汇表大小
    "max_seq_length": 256, #每一句训练数据的最大长度
    "embedding_dim": 768,   #嵌入向量的维度
    "n_heads": 12,          #注意力头个数
    "n_layers": 12,         #Transformer 层数
    "drop_rate": 0.1,       #Dropout rate
    "qkv_bias": False       #bias
}