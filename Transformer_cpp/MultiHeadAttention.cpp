/******************************************************************************
 *  Copyright (c) 2026 The Transformer project authors . All Rights Reserved.
 *
 *  Please visit https://chensongpoixs.github.io for detail
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 
				   Author: chensong
				   date:  2026-01-01

  多头注意力机制实现
  




 输赢不重要，答案对你们有什么意义才重要。

 光阴者，百代之过客也，唯有奋力奔跑，方能生风起时，是时代造英雄，英雄存在于时代。或许世人道你轻狂，可你本就年少啊。 看护好，自己的理想和激情。


 我可能会遇到很多的人，听他们讲好2多的故事，我来写成故事或编成歌，用我学来的各种乐器演奏它。
 然后还可能在一个国家遇到一个心仪我的姑娘，她可能会被我帅气的外表捕获，又会被我深邃的内涵吸引，在某个下雨的夜晚，她会全身淋透然后要在我狭小的住处换身上的湿衣服。
 3小时候后她告诉我她其实是这个国家的公主，她愿意向父皇求婚。我不得已告诉她我是穿越而来的男主角，我始终要回到自己的世界。
 然后我的身影慢慢消失，我看到她眼里的泪水，心里却没有任何痛苦，我才知道，原来我的心被丢掉了，我游历全世界的原因，就是要找回自己的本心。
 于是我开始有意寻找各种各样失去心的人，我变成一块砖头，一颗树，一滴水，一朵白云，去听大家为什么会失去自己的本心。
 我发现，刚出生的宝宝，本心还在，慢慢的，他们的本心就会消失，收到了各种黑暗之光的侵蚀。
 从一次争论，到嫉妒和悲愤，还有委屈和痛苦，我看到一只只无形的手，把他们的本心扯碎，蒙蔽，偷走，再也回不到主人都身边。
 我叫他本心猎手。他可能是和宇宙同在的级别 但是我并不害怕，我仔细回忆自己平淡的一生 寻找本心猎手的痕迹。
 沿着自己的回忆，一个个的场景忽闪而过，最后发现，我的本心，在我写代码的时候，会回来。
 安静，淡然，代码就是我的一切，写代码就是我本心回归的最好方式，我还没找到本心猎手，但我相信，顺着这个线索，我一定能顺藤摸瓜，把他揪出来。



  
 */

#include "MultiHeadAttention.h"
#include <torch/torch.h>
#include <cmath>
#include <stdexcept>

MultiHeadAttentionImpl::MultiHeadAttentionImpl(const ModelConfig& cfg)
    : d_out(cfg.embedding_dim),
      num_heads(cfg.n_heads),
      head_dim(cfg.embedding_dim / cfg.n_heads),
      W_q(torch::nn::LinearOptions(cfg.embedding_dim, cfg.embedding_dim).bias(cfg.qkv_bias)),
      W_k(torch::nn::LinearOptions(cfg.embedding_dim, cfg.embedding_dim).bias(cfg.qkv_bias)),
      W_v(torch::nn::LinearOptions(cfg.embedding_dim, cfg.embedding_dim).bias(cfg.qkv_bias)),
      output(torch::nn::LinearOptions(cfg.embedding_dim, cfg.embedding_dim)),
      dropout(torch::nn::DropoutOptions(cfg.drop_rate)) {
    
    // 注册模块
    register_module("W_q", W_q);
    register_module("W_k", W_k);
    register_module("W_v", W_v);
    register_module("output", output);
    register_module("dropout", dropout);
    
    // 创建掩码矩阵（上三角矩阵，防止看到未来信息）
    // mask[i][j] = 1 表示位置 i 不能看到位置 j（j > i）
     mask_tensor = torch::triu(torch::ones({cfg.max_seq_length, cfg.max_seq_length}), 1);
    register_buffer("mask", mask_tensor);
    
    // 确保 d_out 能被 num_heads 整除
    if (cfg.embedding_dim % cfg.n_heads != 0) {
        throw std::runtime_error("embedding_dim must be divisible by n_heads");
    }
}

torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor x) {
    // 输入: x (batch_size, seq_length, embedding_dim)
    auto batch_size = x.size(0);
    auto seq_length = x.size(1);
    
    // ========== 步骤 1: 生成 Q, K, V ==========
    // 通过线性投影生成查询、键、值向量
    // 维度: (batch_size, seq_length, d_out)
    auto q = W_q->forward(x);
    auto k = W_k->forward(x);
    auto v = W_v->forward(x);
    
    // ========== 步骤 2: 分割成多个头 ==========
    // 将 d_out 维分成 num_heads 个头，每个头 head_dim 维
    // 维度: (batch_size, seq_length, num_heads, head_dim)
    q = q.view({batch_size, seq_length, num_heads, head_dim});
    k = k.view({batch_size, seq_length, num_heads, head_dim});
    v = v.view({batch_size, seq_length, num_heads, head_dim});
    
    // ========== 步骤 3: 调整维度顺序 ==========
    // 将 num_heads 移到第 2 维，便于并行计算每个头
    // 维度: (batch_size, num_heads, seq_length, head_dim)
    q = q.transpose(1, 2);
    k = k.transpose(1, 2);
    v = v.transpose(1, 2);
    
    // ========== 步骤 4: 计算注意力分数 ==========
    // 每个头独立计算注意力分数
    // q @ k.transpose(-2, -1): (batch_size, num_heads, seq_length, seq_length)
    auto attn_score = torch::matmul(q, k.transpose(-2, -1));
    
    // ========== 步骤 5: 缩放和归一化 ==========
    // 除以 √head_dim，防止点积结果过大
    attn_score = attn_score / std::sqrt(static_cast<double>(head_dim));
    
    // ========== 步骤 6: 应用掩码 ==========
    // 将未来位置的注意力分数设为 -∞
    // 注意: 需要扩展掩码到 batch_size 和 num_heads 维度
    auto mask_expanded = mask_tensor.slice(0, 0, seq_length).slice(1, 0, seq_length);
    mask_expanded = mask_expanded.unsqueeze(0).unsqueeze(0); // (1, 1, seq_length, seq_length)
    attn_score.masked_fill_(mask_expanded.to(torch::kBool), -std::numeric_limits<double>::infinity());
    
    // ========== 步骤 7: Softmax 归一化 ==========
    // 将分数转换为概率分布
    auto attn_weight = torch::softmax(attn_score, -1);
    
    // ========== 步骤 8: Dropout ==========
    // 训练时随机失活，增加泛化能力
    attn_weight = dropout->forward(attn_weight);
    
    // ========== 步骤 9: 加权求和 ==========
    // 根据注意力权重对值向量进行加权求和
    // 维度: (batch_size, num_heads, seq_length, head_dim)
    at::Tensor context_vec = torch::matmul(attn_weight, v);
    
    // ========== 步骤 10: 拼接所有头 ==========
    // 将 num_heads 移回第 2 维并拼接
    // 维度: (batch_size, seq_length, num_heads, head_dim)
    context_vec = context_vec.transpose(1, 2);
    
    // 拼接成 d_out 维
    // 维度: (batch_size, seq_length, d_out)
    context_vec = context_vec.contiguous().view({batch_size, seq_length, d_out});
    
    // ========== 步骤 11: 输出投影 ==========
    // 通过线性层整合所有头的信息
    context_vec = output->forward(context_vec);
    
    return context_vec;
}
