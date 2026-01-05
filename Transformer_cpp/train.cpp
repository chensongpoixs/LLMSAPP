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
 ******************************************************************************/
/*****************************************************************************
				   Author: chensong
				   date:  2026-01-01

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

 ******************************************************************************/
/**
 * Transformer 训练实现（包含反向传播和 GPU 支持）
 * 
 * 注意：当前实现是 Decoder-Only 架构（GPT风格），不是完整的 Encoder-Decoder Transformer
 * 
 * 架构说明：
 * - Decoder-Only: 只包含解码器层，使用 Masked Self-Attention
 * - 适合任务: 语言模型、文本生成（自回归）
 * - 不包含: Encoder层、Encoder-Decoder Cross-Attention
 * 
 * 本文件实现了完整的训练流程，包括：
 * 1. 前向传播：模型计算预测结果
 * 2. 损失计算：计算预测与真实标签的差异
 * 3. 反向传播：计算梯度
 * 4. 梯度裁剪：防止梯度爆炸
 * 5. 参数更新：使用优化器更新模型参数
 * 6. GPU 支持：自动检测并使用 GPU（如果可用）
 * 
 * 训练流程：
 * ┌─────────────────────────────────────────────────────────┐
 * │  1. 准备数据 (input_ids, target_ids)                    │
 * │  2. 前向传播: logits = model(input_ids)                 │
 * │  3. 计算损失: loss = criterion(logits, target_ids)      │
 * │  4. 反向传播: loss.backward()                           │
 * │  5. 梯度裁剪: clip_grad_norm_(model.parameters(), ...)  │
 * │  6. 参数更新: optimizer.step()                           │
 * │  7. 梯度清零: optimizer.zero_grad()                      │
 * └─────────────────────────────────────────────────────────┘
 * 
 * GPU 使用：
 * - 自动检测：如果 CUDA 可用，自动使用 GPU
 * - 命令行参数：--device cuda 或 --device cpu
 * - 示例：./Transformer_train --device cuda
 * 
 * 关键概念：
 * - 损失函数：交叉熵损失（CrossEntropyLoss），用于多分类任务
 * - 优化器：Adam 优化器，自适应学习率
 * - 学习率调度：可选，用于动态调整学习率
 * - 梯度裁剪：防止梯度爆炸，提高训练稳定性
 * - GPU 加速：使用 CUDA 加速训练，大幅提升训练速度
 */

#include "GPTModel.h"
#include "ModelConfig.h"
#include "Logger.h"
#include "TextDataset.h"
#include "Tiktoken.h"
#include "TrainingUtils.h"
#include "Trainer.h"
#include <torch/torch.h>
#include <memory>
#include <string>

int main(int argc, char* argv[]) {
    Logger::info("=== Transformer Training Example (with Backpropagation) ===");
    
    // ========================================================================
    // 0. Parse device parameters
    // ========================================================================
    torch::Device device = parse_device(argc, argv);
    
    Logger::info("Using device: {}", (device == torch::kCUDA ? "CUDA (GPU)" : "CPU"));
    if (device == torch::kCUDA) {
        Logger::info("  - Number of GPUs: {}", torch::cuda::device_count());
        if (torch::cuda::device_count() > 0) {
            try {
            //    Logger::info("  - Current GPU: {}", torch::cuda::current_device());
            //    Logger::info("  - GPU Name: {}", torch::cuda::getDeviceName(torch::cuda::current_device()));
            } catch (...) {
                Logger::warning("  - GPU Name: Unable to retrieve");
            }
        }
    }
    Logger::info("Usage:");
    Logger::info("  - Use GPU: {} --device cuda", argv[0]);
    Logger::info("  - Use CPU: {} --device cpu", argv[0]);
    
    // ========================================================================
    // 1. Create model configuration
    // ========================================================================
    ModelConfig cfg;
    cfg.vocab_size = 50257;        // Vocab size (GPT-2 standard)
    cfg.max_seq_length = 1024;     // Max sequence length
    cfg.embedding_dim = 768;       // Embedding dimension
    cfg.n_heads = 12;              // Number of attention heads
    cfg.n_layers = 12;             // Number of Transformer layers
    cfg.drop_rate = 0.1;           // Dropout rate
    cfg.qkv_bias = false;          // No bias
    
    Logger::info("Model Configuration:");
    Logger::info("  - Vocab Size: {}", cfg.vocab_size);
    Logger::info("  - Max Sequence Length: {}", cfg.max_seq_length);
    Logger::info("  - Embedding Dimension: {}", cfg.embedding_dim);
    Logger::info("  - Number of Attention Heads: {}", cfg.n_heads);
    Logger::info("  - Number of Transformer Layers: {}", cfg.n_layers);
    Logger::info("  - Dropout Rate: {}", cfg.drop_rate);
    
    // ========================================================================
    // 2. Create model
    // ========================================================================
    Logger::info("Creating GPT model...");
    std::shared_ptr<GPTModel> model = std::make_shared<GPTModel>(cfg);
    model->to(device);
    model->train();
    
    Logger::info("Model created successfully!");
    
    // Count parameters
    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }
    Logger::info("Total Parameters: {}", total_params);
    Logger::info("Total Parameters (MB): {:.2f}", (total_params * sizeof(float) / (1024.0 * 1024.0)));
    
    // ========================================================================
    // 3. Load training data
    // ========================================================================
    std::string data_file = "the-verdict.txt";
    Logger::info("Loading training data...");
    Logger::info("Data file: {}", data_file);
    
    int seq_len = 128;             // Sequence length
    
    // 创建 tiktoken 编码器（可以选择使用 GPT-2 或其他编码器）
    std::shared_ptr<Tiktoken> encoder = nullptr;
    
    // 选项1: 使用简单编码器（字符级，向后兼容）
    // encoder = tiktoken::create_simple_encoding();
    
    // 选项2: 使用 GPT-2 编码器（推荐，更好的 tokenization）
    encoder = tiktoken::create_gpt2_encoding();
    
    // 选项3: 从文件加载编码器
    // encoder = tiktoken::load_encoding_from_file("merges.txt", "gpt2");
    
    // 使用 tiktoken 编码器创建数据集
    TextDataset dataset(data_file, seq_len, encoder);
    
    // 更新词汇表大小以匹配编码器
    cfg.vocab_size = dataset.getVocabSize();
    Logger::info("Using vocab size from encoder: {}", cfg.vocab_size);
    
    if (!dataset.load()) {
        Logger::error("Failed to load data, exiting program");
        return -1;
    }
    
    Logger::info("Dataset loaded successfully!");
    Logger::info("Dataset size: {} samples", dataset.size());
    
    // ========================================================================
    // 4. Training parameters
    // ========================================================================
    int num_epochs = 10;           // Number of epochs
    int batch_size = 40;          // Batch size
    double learning_rate = 3e-4;   // Learning rate
    double weight_decay = 0.1;     // Weight decay
    double clip_grad_norm = 1.0;   // Gradient clipping
    
    Logger::info("Training Parameters:");
    Logger::info("  - Number of Epochs: {}", num_epochs);
    Logger::info("  - Batch Size: {}", batch_size);
    Logger::info("  - Sequence Length: {}", seq_len);
    Logger::info("  - Learning Rate: {}", learning_rate);
    Logger::info("  - Weight Decay: {}", weight_decay);
    Logger::info("  - Gradient Clipping: {}", clip_grad_norm);
    
    // ========================================================================
    // 5. Create trainer and start training
    // ========================================================================
    Trainer trainer(model, dataset, device, cfg);
    
    if (!trainer.train(num_epochs, batch_size, learning_rate, weight_decay, clip_grad_norm, data_file)) {
        Logger::error("Training failed!");
        return -1;
    }
    
    Logger::info("Program execution completed!");
    
    return 0;
}

