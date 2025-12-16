'''
Docstring for 06.Transformer架构神秘.selfattention
# python version:  3.10.18
'''
#from numpy import matmul;
import torch;
import torch.nn as nn;

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__();
        # 设置Query， Key， Value 随机值
        self.W_q = nn.Parameter(torch.rand(d_in, d_out));
        self.W_k = nn.Parameter(torch.rand(d_in, d_out));
        self.W_v = nn.Parameter(torch.rand(d_in, d_out));
        self.d_out = d_out;
    def forward(self, x):
        # 获取Q， K， V
        # 在Python '@'表示的就是使用matmul() , 它是一种简单写法
        # 点积 ==> 
        q = x @ self.W_q;
        k = x @ self.W_k;
        v = x @ self.W_v;

        # 注意力分数
        attn_score = q @ k.T;
        # -1 : 最后一位
        # 注意力权重
        attn_weight =  torch.softmax( attn_score / k.shape[-1] ** 0.5, # 开根号 ==> **0.5 
                      dim = -1); # 分词
        
        # 让注意力权重与v点积
        content_vec = attn_weight @ v;
        return content_vec;




class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__();
        # 设置Query， Key， Value 随机值
        #  Y = aX + b # 去了 b qkv_bias设置false
        self.W_q = nn.Linear(d_in, d_out, qkv_bias) #nn.Parameter(torch.rand(d_in, d_out));
        self.W_k = nn.Linear(d_in, d_out, qkv_bias) #nn.Parameter(torch.rand(d_in, d_out));
        self.W_v = nn.Linear(d_in, d_out, qkv_bias) #nn.Parameter(torch.rand(d_in, d_out));
        self.d_out = d_out;
    def forward(self, x):
        # 获取Q， K， V
        # 在Python '@'表示的就是使用matmul() , 它是一种简单写法
        # 点积 ==> 
        q = self.W_q(x);
        k =  self.W_k(x);
        v = self.W_v(x);

        # 注意力分数
        attn_score = q @ k.T;
        # -1 : 最后一位
        # 注意力权重
        attn_weight =  torch.softmax( attn_score / k.shape[-1] ** 0.5, # 开根号 ==> **0.5 
                      dim = -1); # 分词
        
        # 让注意力权重与v点积
        content_vec = attn_weight @ v;
        return content_vec;




# test
# d_in = 768;
# d_out = 64;
# #  设置随机种子
# torch.manual_seed(12345);


# attn = SelfAttention_v1(d_in, d_out);

# inputs = [[0.1, 0.23], [0.23, 0.12]];

# content_vec = attn.forward(inputs);

# print("{}".format(content_vec));