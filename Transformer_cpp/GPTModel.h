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

 * GPT 模型 (Decoder-Only Transformer 模型)
 *
 * 这是 GPT (Generative Pre-trained Transformer) 模型实现，
 * 采用 GPT 风格架构，即只使用解码器（Decoder）部分的自回归模型。
 * 
 * 注意：这是 Decoder-Only 架构，不是完整的 Encoder-Decoder Transformer。
 * 
 * 架构类型：
 * - Decoder-Only: 只包含解码器层（当前实现）
 * - Encoder-Decoder: 包含编码器和解码器（未实现）
 * 
 * 当前实现特点：
 * - 使用 Masked Self-Attention（防止看到未来token）
 * - 适合自回归生成任务（如语言模型）
 * - 不包含 Encoder-Decoder Cross-Attention
 * 
 * 架构对比：
 * ┌─────────────────────────────────────────────────────────┐
 * │ Decoder-Only (当前实现)                                  │
 * │ ┌─────────────────────────────────────────────────┐   │
 * │ │ Input Embedding                                  │   │
 * │ │ Position Embedding                               │   │
 * │ │ ┌─────────────────────────────────────────────┐ │   │
 * │ │ │ Decoder Block (N层)                          │ │   │
 * │ │ │   - Masked Self-Attention                    │ │   │
 * │ │ │   - Feed Forward Network                     │ │   │
 * │ │ └─────────────────────────────────────────────┘ │   │
 * │ │ Output Head                                       │   │
 * │ └─────────────────────────────────────────────────┘   │
 * └─────────────────────────────────────────────────────────┘
 * 
 * ┌─────────────────────────────────────────────────────────┐
 * │ Encoder-Decoder (未实现)                                 │
 * │ ┌──────────────┐         ┌──────────────┐              │
 * │ │   Encoder    │         │   Decoder    │              │
 * │ │ ┌──────────┐ │         │ ┌──────────┐ │              │
 * │ │ │Encoder   │ │         │ │Decoder   │ │              │
 * │ │ │Block (N) │ │         │ │Block (N) │ │              │
 * │ │ │-Self-    │ │         │ │-Masked   │ │              │
 * │ │ │ Attention│ │         │ │ Self-    │ │              │
 * │ │ │-FFN      │ │         │ │ Attention│ │              │
 * │ │ └──────────┘ │         │ │-Cross-   │ │              │
 * │ └──────────────┘         │ │ Attention│ │              │
 * │                          │ │-FFN      │ │              │
 * │                          │ └──────────┘ │              │
 * │                          └──────────────┘              │
 * └─────────────────────────────────────────────────────────┘
 *
 * 架构流程：
 * 1. Token Embedding: 将 token IDs 转换为向量表示
 *    - 输入: (batch_size, seq_len) - Token IDs
 *    - 输出: (batch_size, seq_len, embedding_dim)
 *
 * 2. Position Embedding: 添加位置信息
 *    - 为每个位置生成唯一的嵌入向量
 *    - 输出: (seq_len, embedding_dim)
 *
 * 3. 嵌入融合: token_embeds + position_embeds
 *    - 广播机制合并词元和位置信息
 *
 * 4. Dropout: 防止过拟合
 *
 * 5. Transformer Blocks: 通过 N 个 Transformer 块处理（通常 N=12）
 *    - 每个块包含多头注意力和前馈网络
 *    - 采用残差连接和层归一化
 *
 * 6. Layer Norm: 最终层归一化
 *
 * 7. Output Head: 线性投影到词汇表大小
 *    - 输出: (batch_size, seq_len, vocab_size) - Logits
 *
 * 输入输出：
 * - 输入: (batch_size, seq_len) - Token IDs (整数，范围 0 到 vocab_size-1)
 * - 输出: (batch_size, seq_len, vocab_size) - Logits (未归一化的概率分数)


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

#ifndef GPT_MODEL_H
#define GPT_MODEL_H

#include <torch/torch.h>
#include "ModelConfig.h"
#include "TransformerBlock.h"

class GPTModel : public torch::nn::Module {
public:
    GPTModel(const ModelConfig& cfg);
    torch::Tensor forward(torch::Tensor x);

private:
    ModelConfig cfg;
    torch::nn::Embedding token_embedding;
    torch::nn::Embedding position_embedding;
    torch::nn::ModuleList transformer_blocks;
    torch::nn::LayerNorm layer_norm;
    torch::nn::Linear out_head;
    torch::nn::Dropout drop;
};

#endif // GPT_MODEL_H
