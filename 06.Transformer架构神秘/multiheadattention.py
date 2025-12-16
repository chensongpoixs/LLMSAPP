'''
Docstring for 06.Transformer架构神秘.multiheadattention
'''

import torch;
import torch.nn as nn;

class maskselfattention(nn.Module):
    def __init__(self, d_in, d_out, 
                 mask_matrix_len, drop_rate, 
                 num_heads,  qkv_bias = False):
        super().__init__();
        # 设置Query， Key， Value 随机值
        #  Y = aX + b # 去了 b qkv_bias设置false
        self.W_q = nn.Linear(d_in, d_out, qkv_bias) #nn.Parameter(torch.rand(d_in, d_out));
        self.W_k = nn.Linear(d_in, d_out, qkv_bias) #nn.Parameter(torch.rand(d_in, d_out));
        self.W_v = nn.Linear(d_in, d_out, qkv_bias) #nn.Parameter(torch.rand(d_in, d_out));
        self.d_out = d_out;

        self.dropout = nn.Dropout(drop_rate);
        self.num_heads = num_heads;
        # head 维度
        self.head_dim = d_out // num_heads;
        # 线性函数
        self.output = nn.Linear(d_out, d_out);
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(mask_matrix_len, mask_matrix_len), 0)
        );
    def forward(self, x):
        batch_size, seq_length, embedding_dim = x.shape;
        # 获取Q， K， V
        # 在Python '@'表示的就是使用matmul() , 它是一种简单写法
        # 点积 ==> 
        q = self.W_q(x);
        k =  self.W_k(x);
        v = self.W_v(x);

        # 头变4维啦 ==> embedding_dim ==> 
        q.view(batch_size, seq_length, self.num_heads, self.head_dim);
        k.view(batch_size, seq_length, self.num_heads, self.head_dim);
        v.view(batch_size, seq_length, self.num_heads, self.head_dim);

        #  1:seq_length, 2:num_heads 调换
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);
        # 注意力分数  # k：二维转三维度变化了 1：seq_length, 2：embedding_dim
        attn_score = q @ k.transpose(2, 3);
        # masked_fill作用遮掩矩阵， 替换
        attn_score.masked_fill_(self.mask.bool()[:seq_length, :seq_length],
                                -torch.inf  # 替换
                                );
        
        # -1 : 最后一位
        # 注意力权重
        attn_weight =  torch.softmax( attn_score / k.shape[-1] ** 0.5, # 开根号 ==> **0.5 
                      dim = -1); # 分词
        # drop 增加泛化性
        attn_weight = self.dropout(attn_weight);
        # 让注意力权重与v点积
        content_vec =( attn_weight @ v).transpose(1, 2);

        content_vec.contiguous().view(batch_size, seq_length, self.d_out);
        return content_vec;