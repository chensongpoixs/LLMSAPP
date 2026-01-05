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
#include <torch/torch.h>
#include <c10/core/DeviceGuard.h>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <cstring>
#include <iomanip>
#include <limits>
#include <algorithm>

/**
 * 训练一个批次的数据
 * 
 * @param model: GPT 模型
 * @param input_ids: 输入 token IDs (batch_size, seq_len)
 * @param target_ids: 目标 token IDs (batch_size, seq_len)，用于计算损失
 * @param criterion: 损失函数（交叉熵损失）
 * @param optimizer: 优化器（Adam）
 * @param clip_grad_norm: 梯度裁剪的最大范数
 * @return 损失值
 */
float train_batch(
    std::shared_ptr<GPTModel> model,
    torch::Tensor input_ids,
    torch::Tensor target_ids,
    torch::nn::CrossEntropyLoss& criterion,
    torch::optim::Adam& optimizer,
    double clip_grad_norm = 1.0) {
    
    // ========== 步骤 1: 梯度清零 ==========
    // 清除上一次迭代的梯度，防止梯度累积
    optimizer.zero_grad();
    
    // ========== 步骤 2: 前向传播 ==========
    // 模型计算预测结果
    // 输入: input_ids (batch_size, seq_len)
    // 输出: logits (batch_size, seq_len, vocab_size)
    torch::Tensor logits = model->forward(input_ids);
    
    // ========== 步骤 3: 计算损失 ==========
    // 将 logits 和 target_ids 重塑为 2D 张量以计算损失
    // logits: (batch_size * seq_len, vocab_size)
    // targets: (batch_size * seq_len)
    int64_t batch_size = logits.size(0);
    int64_t seq_len = logits.size(1);
    int64_t vocab_size = logits.size(2);
    
    // 重塑 logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    logits = logits.view({-1, vocab_size});
    
    // 重塑 targets: (batch_size, seq_len) -> (batch_size * seq_len)
    target_ids = target_ids.view({-1});
    
    // 计算交叉熵损失
    // 对于每个位置，预测下一个 token 的概率分布
    torch::Tensor loss = criterion(logits, target_ids);
    
    // ========== 步骤 4: 反向传播 ==========
    // 计算所有参数的梯度
    // PyTorch 的 autograd 系统会自动计算梯度并存储在参数的 .grad 属性中
    loss.backward();
    
    // ========== 步骤 5: 梯度裁剪 ==========
    // 防止梯度爆炸，提高训练稳定性
    // 将所有参数的梯度范数裁剪到最大值
    torch::nn::utils::clip_grad_norm_(model->parameters(), clip_grad_norm);
    
    // ========== 步骤 6: 参数更新 ==========
    // 使用优化器根据梯度更新模型参数
    // 优化器会根据学习率和梯度更新每个参数的值
    optimizer.step();
    
    // 返回损失值（标量）
    return loss.item<float>();
}

/**
 * 验证一个批次的数据（不更新参数）
 * 
 * @param model: GPT 模型
 * @param input_ids: 输入 token IDs
 * @param target_ids: 目标 token IDs
 * @param criterion: 损失函数
 * @return 损失值
 */
float validate_batch(
    std::shared_ptr<GPTModel> model,
    torch::Tensor input_ids,
    torch::Tensor target_ids,
    torch::nn::CrossEntropyLoss& criterion) {
    
    // 设置为评估模式（不使用 Dropout）
    model->eval();
    
    // 禁用梯度计算，节省内存和计算
    torch::NoGradGuard no_grad;
    
    // 前向传播
    torch::Tensor logits = model->forward(input_ids);
    
    // 计算损失
    int64_t vocab_size = logits.size(2);
    logits = logits.view({-1, vocab_size});
    target_ids = target_ids.view({-1});
    torch::Tensor loss = criterion(logits, target_ids);
    
    // 恢复训练模式
    model->train();
    
    return loss.item<float>();
}

/**
 * 生成简单的训练数据（示例）
 * 
 * @param batch_size: 批次大小
 * @param seq_len: 序列长度
 * @param vocab_size: 词汇表大小
 * @param device: 设备（CPU 或 CUDA）
 * @return 输入和目标 token IDs
 */
std::pair<torch::Tensor, torch::Tensor> generate_dummy_data(
    int batch_size, 
    int seq_len, 
    int vocab_size,
    torch::Device device) {
    
    // 生成随机输入 token IDs，并移动到指定设备
    torch::Tensor input_ids = torch::randint(
        vocab_size, 
        {batch_size, seq_len}, 
        torch::TensorOptions().dtype(torch::kLong).device(device)
    );
    
    // 目标 token IDs：对于语言模型，目标是下一个 token
    // 这里简化为输入序列向右移动一位
    torch::Tensor target_ids = input_ids.clone();
    // 在实际应用中，target_ids 应该是 input_ids 的下一个 token
    // 这里仅作为示例，实际应该从数据集中加载
    
    return std::make_pair(input_ids, target_ids);
}

/**
 * 解析命令行参数，获取设备类型
 * 
 * @param argc: 参数数量
 * @param argv: 参数数组
 * @return 设备类型（torch::Device）
 */
torch::Device parse_device(int argc, char* argv[]) {
    torch::Device device = torch::kCPU;
    
    // 检查是否有 --device 或 -d 参数
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 || std::strcmp(argv[i], "-d") == 0) {
            if (i + 1 < argc) {
                std::string device_str = argv[i + 1];
                if (device_str == "cuda" || device_str == "gpu") {
                    // 检查 CUDA 是否可用
                    if (torch::cuda::is_available()) {
                        device = torch::kCUDA;
                        Logger::info("检测到 CUDA 可用，使用 GPU 训练");
                    } else {
                        Logger::warning("请求使用 GPU，但 CUDA 不可用，将使用 CPU");
                        device = torch::kCPU;
                    }
                } else if (device_str == "cpu") {
                    device = torch::kCPU;
                } else {
                    Logger::warning("未知设备类型 '{}'，使用默认设备 CPU", device_str);
                }
            }
            break;
        }
    }
    
    // 如果没有指定参数，自动检测并使用 GPU（如果可用）
    if (device == torch::kCPU && torch::cuda::is_available()) {
        Logger::info("检测到 CUDA 可用，自动使用 GPU 训练");
        Logger::info("提示: 使用 --device cpu 强制使用 CPU");
        device = torch::kCUDA;
    }
    
    return device;
}

int main(int argc, char* argv[]) {
    Logger::info("=== Transformer 训练示例（包含反向传播）===");
    
    // ========================================================================
    // 0. 解析设备参数
    // ========================================================================
    torch::Device device = parse_device(argc, argv);
    
    Logger::info("使用设备: {}", (device == torch::kCUDA ? "CUDA (GPU)" : "CPU"));
    if (device == torch::kCUDA) {
        Logger::info("  - GPU 数量: {}", torch::cuda::device_count());
        if (torch::cuda::device_count() > 0) {
            try {
            //    Logger::info("  - 当前 GPU: {}", torch::cuda::current_device());
            //    Logger::info("  - GPU 名称: {}", torch::cuda::getDeviceName(torch::cuda::current_device()));
            } catch (...) {
                Logger::warning("  - GPU 名称: 无法获取");
            }
        }
    }
    Logger::info("使用说明:");
    Logger::info("  - 使用 GPU: {} --device cuda", argv[0]);
    Logger::info("  - 使用 CPU: {} --device cpu", argv[0]);
    
    // ========================================================================
    // 2. 创建模型配置
    // ========================================================================
    ModelConfig cfg;
    cfg.vocab_size = 50257;        // 词汇表大小（GPT-2 标准）
    cfg.max_seq_length = 1024;     // 最大序列长度
    cfg.embedding_dim = 768;       // 嵌入向量维度
    cfg.n_heads = 12;              // 注意力头数量
    cfg.n_layers = 12;             // Transformer 层数
    cfg.drop_rate = 0.1;           // Dropout 比率
    cfg.qkv_bias = false;          // 不使用偏置
    
    Logger::info("模型配置:");
    Logger::info("  - 词汇表大小: {}", cfg.vocab_size);
    Logger::info("  - 最大序列长度: {}", cfg.max_seq_length);
    Logger::info("  - 嵌入维度: {}", cfg.embedding_dim);
    Logger::info("  - 注意力头数: {}", cfg.n_heads);
    Logger::info("  - Transformer 层数: {}", cfg.n_layers);
    Logger::info("  - Dropout 比率: {}", cfg.drop_rate);
    
    // ========================================================================
    // 3. 创建模型
    // ========================================================================
    Logger::info("创建 GPT 模型...");
    std::shared_ptr<GPTModel> model = std::make_shared<GPTModel>(cfg);
    
    // 将模型移动到指定设备（CPU 或 GPU）
    model->to(device);
    
    // 设置为训练模式（启用 Dropout）
    model->train();
    
    Logger::info("模型创建完成！");
    
    // 统计参数量
    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }
    Logger::info("总参数量: {}", total_params);
    Logger::info("总参数量 (MB): {:.2f}", (total_params * sizeof(float) / (1024.0 * 1024.0)));
    
    // ========================================================================
    // 4. 创建损失函数
    // ========================================================================
    // 交叉熵损失，用于多分类任务
    // ignore_index: 忽略的 token ID（通常用于 padding）
    torch::nn::CrossEntropyLoss criterion(
        torch::nn::CrossEntropyLossOptions().ignore_index(-1)
    );
    
    // 将损失函数移动到指定设备
    criterion->to(device);
    
    Logger::info("损失函数: CrossEntropyLoss");
    
    // ========================================================================
    // 5. 创建优化器
    // ========================================================================
    // Adam 优化器，自适应学习率
    double learning_rate = 3e-4;  // 学习率（GPT 常用值）
    double weight_decay = 0.1;    // 权重衰减（L2 正则化）
    
    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay)
    );
    
    Logger::info("优化器: Adam");
    Logger::info("  - 学习率: {}", learning_rate);
    Logger::info("  - 权重衰减: {}", weight_decay);
    
    // ========================================================================
    // 6. 加载训练数据
    // ========================================================================
    // 数据文件路径（相对于项目根目录）
    std::string data_file = "the-verdict.txt";
    
    Logger::info("加载训练数据...");
    Logger::info("数据文件: {}", data_file);
    
    int seq_len = 128;             // 序列长度
    TextDataset dataset(data_file, seq_len, cfg.vocab_size);
    
    if (!dataset.load()) {
        Logger::error("数据加载失败，退出程序");
        return -1;
    }
    
    Logger::info("数据集加载成功！");
    Logger::info("数据集大小: {} 个样本", dataset.size());
    
    // ========================================================================
    // 7. 训练参数
    // ========================================================================
    int num_epochs = 10;           // 训练轮数
    int batch_size = 50;             // 批次大小（根据数据集大小调整）
    double clip_grad_norm = 1.0;   // 梯度裁剪的最大范数
    
    // 计算每个epoch的批次数量
    int num_batches_per_epoch = static_cast<int>(dataset.size()) / batch_size;
    if (num_batches_per_epoch == 0) {
        num_batches_per_epoch = 1;
    }
    
    Logger::info("训练参数:");
    Logger::info("  - 训练轮数: {}", num_epochs);
    Logger::info("  - 批次大小: {}", batch_size);
    Logger::info("  - 序列长度: {}", seq_len);
    Logger::info("  - 每轮批次数量: {}", num_batches_per_epoch);
    Logger::info("  - 梯度裁剪: {}", clip_grad_norm);
    
    // ========================================================================
    // 8. 训练循环
    // ========================================================================
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("开始训练...");
    Logger::info("═══════════════════════════════════════════════════════════════");
    
    // 记录总体训练时间
    auto training_start = std::chrono::high_resolution_clock::now();
    
    // 记录最佳损失
    float best_loss = std::numeric_limits<float>::max();
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // 记录每个 epoch 的总损失
        float epoch_loss = 0.0f;
        float min_batch_loss = std::numeric_limits<float>::max();
        float max_batch_loss = std::numeric_limits<float>::min();
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Logger::info("Epoch [{}/{}]", (epoch + 1), num_epochs);
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        for (int batch_idx = 0; batch_idx < num_batches_per_epoch; ++batch_idx) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            // 从数据集中获取一个批次的数据
            auto [input_ids, target_ids] = dataset.getBatch(batch_size, device);
            
            // 训练一个批次
            float loss = train_batch(
                model, 
                input_ids, 
                target_ids, 
                criterion, 
                optimizer, 
                clip_grad_norm
            );
            
            epoch_loss += loss;
            min_batch_loss = std::min(min_batch_loss, loss);
            max_batch_loss = std::max(max_batch_loss, loss);
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                batch_end - batch_start
            );
            
            // 计算训练速度（样本/秒）
            int64_t samples_per_second = (batch_size * 1000) / (1L > batch_duration.count() ? 1L : batch_duration.count());//std::max(1L, batch_duration.count());
            
            // 计算进度百分比
            float progress = ((batch_idx + 1) * 100.0f) / num_batches_per_epoch;
            
            // 打印每个批次的详细信息
            std::ostringstream batch_info;
            batch_info << "  [" << std::fixed << std::setprecision(1) << progress << "%] "
                       << "Batch [" << std::setw(3) << (batch_idx + 1) << "/" << num_batches_per_epoch << "] | "
                       << "Loss: " << std::setw(8) << std::setprecision(6) << loss << " | "
                       << "Time: " << std::setw(5) << batch_duration.count() << " ms | "
                       << "Speed: " << std::setw(6) << samples_per_second << " samples/s";
            
            // 如果是第一个或最后一个批次，显示更多信息
            if (batch_idx == 0 || batch_idx == num_epochs - 1) {
                // 计算梯度范数（可选，可能较慢）
                double grad_norm = 0.0;
                for (const auto& param : model->parameters()) {
                    if (param.grad().defined()) {
                        grad_norm += param.grad().norm().item<double>() * param.grad().norm().item<double>();
                    }
                }
                grad_norm = std::sqrt(grad_norm);
                batch_info << " | Grad Norm: " << std::setw(8) << std::setprecision(4) << grad_norm;
            }
            
            Logger::info("{}", batch_info.str());
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            epoch_end - epoch_start
        );
        auto epoch_duration_sec = epoch_duration.count() / 1000.0;
        
        // 计算平均损失
        float avg_loss = epoch_loss / num_batches_per_epoch;
        
        // 更新最佳损失
        if (avg_loss < best_loss) {
            best_loss = avg_loss;
        }
        
        // 计算总样本数
        int64_t total_samples = batch_size * num_batches_per_epoch;
        double samples_per_second = total_samples / std::max(0.001, epoch_duration_sec);
        
        // 打印 Epoch 总结信息
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Logger::info("Epoch [{}/{}] 总结:", (epoch + 1), num_epochs);
        std::ostringstream avg_loss_str, min_loss_str, max_loss_str, best_loss_epoch_str;
        avg_loss_str << std::fixed << std::setprecision(6) << avg_loss;
        min_loss_str << std::fixed << std::setprecision(6) << min_batch_loss;
        max_loss_str << std::fixed << std::setprecision(6) << max_batch_loss;
        best_loss_epoch_str << std::fixed << std::setprecision(6) << best_loss;
        Logger::info("  ├─ 平均损失: {}", avg_loss_str.str());
        Logger::info("  ├─ 最小损失: {}", min_loss_str.str());
        Logger::info("  ├─ 最大损失: {}", max_loss_str.str());
        Logger::info("  ├─ 最佳损失: {}", best_loss_epoch_str.str());
        std::ostringstream time_str;
        time_str << std::fixed << std::setprecision(2) << epoch_duration_sec << " 秒 (" << epoch_duration.count() << " ms)";
        Logger::info("  ├─ 耗时: {}", time_str.str());
        Logger::info("  ├─ 处理样本: {} 个", total_samples);
        std::ostringstream speed_str;
        speed_str << std::fixed << std::setprecision(1) << samples_per_second << " 样本/秒";
        Logger::info("  └─ 训练速度: {}", speed_str.str());
    }
    
    // 计算总训练时间
    auto training_end = std::chrono::high_resolution_clock::now();
    auto total_training_time = std::chrono::duration_cast<std::chrono::seconds>(
        training_end - training_start
    );
    
    // 打印训练总结
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("训练完成！");
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("  ├─ 总训练轮数: {}", num_epochs);
    std::ostringstream best_loss_str;
    best_loss_str << std::fixed << std::setprecision(6) << best_loss;
    Logger::info("  ├─ 最佳损失: {}", best_loss_str.str());
    Logger::info("  ├─ 总训练时间: {} 秒", total_training_time.count());
    std::ostringstream avg_time_str;
    avg_time_str << std::fixed << std::setprecision(2) << (total_training_time.count() / static_cast<double>(num_epochs)) << " 秒";
    Logger::info("  └─ 平均每轮: {}", avg_time_str.str());
    
    // ========================================================================
    // 8. 验证（可选）
    // ========================================================================
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("执行验证...");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    auto val_start = std::chrono::high_resolution_clock::now();
    
    // 从数据集中获取验证数据
    auto [val_input_ids, val_target_ids] = dataset.getBatch(batch_size, device);
    
    // 验证
    float val_loss = validate_batch(
        model, 
        val_input_ids, 
        val_target_ids, 
        criterion
    );
    
    auto val_end = std::chrono::high_resolution_clock::now();
    auto val_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        val_end - val_start
    );
    
    std::ostringstream val_loss_str;
    val_loss_str << std::fixed << std::setprecision(6) << val_loss;
    Logger::info("  ├─ 验证损失: {}", val_loss_str.str());
    Logger::info("  └─ 验证耗时: {} ms", val_duration.count());
    
    Logger::info("程序执行完成！");
    
    // ========================================================================
    // 9. 推理测试
    // ========================================================================
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("开始推理测试...");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    model->eval();
    
    std::string prompt = "Hello World!!!";
    int max_new_tokens = 100;  // 最大生成token数量
    
    Logger::info("提示词: \"{}\"", prompt);
    Logger::info("最大生成长度: {} tokens", max_new_tokens);
    
    // 将提示词转换为token IDs
    std::vector<int64_t> prompt_tokens;
    for (char c : prompt) {
        unsigned char uc = static_cast<unsigned char>(c);
        int token_id = static_cast<int>(uc) % cfg.vocab_size;
        prompt_tokens.push_back(token_id);
    }
    
    Logger::info("提示词token序列长度: {}", prompt_tokens.size());
    
    // 准备输入tensor
    torch::Tensor input_ids = torch::zeros({1, (int32_t)prompt_tokens.size()}, 
        torch::TensorOptions().dtype(torch::kLong).device(device));
    
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        input_ids[0][i] = prompt_tokens[i];
    }
    
    // 生成文本（自回归生成）
    std::vector<int64_t> generated_tokens = prompt_tokens;
    generated_tokens.reserve(prompt_tokens.size() + max_new_tokens);
    
    Logger::info("开始生成...");
    
    torch::NoGradGuard no_grad;
    
    for (int i = 0; i < max_new_tokens; ++i) {
        // 当前序列长度
        int current_len = input_ids.size(1);
        
        // 如果序列太长，只保留最近的max_seq_length个token（滑动窗口）
        if (current_len > cfg.max_seq_length) {
            int start_idx = current_len - cfg.max_seq_length;
            input_ids = input_ids.slice(1, start_idx, current_len);
            current_len = cfg.max_seq_length;
        }
        
        // 前向传播：使用整个当前序列
        auto logits = model->forward(input_ids);
        
        // 获取最后一个位置的logits (vocab_size)
        // logits shape: (batch_size=1, seq_len, vocab_size)
        auto next_token_logits = logits[0][current_len - 1];
        
        // 使用温度采样（temperature sampling）增加随机性
        double temperature = 0.8;
        auto scaled_logits = next_token_logits / temperature;
        auto probs = torch::softmax(scaled_logits, 0);
        
        // 采样下一个token
        auto next_token = torch::multinomial(probs, 1).item<int64_t>();
        
        // 添加到生成序列
        generated_tokens.push_back(next_token);
        
        // 将新token添加到输入序列（用于下一次迭代）
        auto new_token_tensor = torch::tensor({{next_token}}, 
            torch::TensorOptions().dtype(torch::kLong).device(device));
        input_ids = torch::cat({input_ids, new_token_tensor}, 1);
        
        // 每10个token打印一次进度
        if ((i + 1) % 10 == 0) {
            Logger::info("已生成 {}/{} tokens", (i + 1), max_new_tokens);
        }
    }
    
    // 将token序列转换为文本
    std::string generated_text;
    for (int64_t token_id : generated_tokens) {
        char c = static_cast<char>(token_id);
        generated_text += c;
    }
    
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("推理结果:");
    Logger::info("原始提示词: \"{}\"", prompt);
    Logger::info("生成文本: \"{}\"", generated_text);
    Logger::info("生成token数量: {}", generated_tokens.size());
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // ========================================================================
    // 10. 保存模型和优化器
    // ========================================================================
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("保存模型和优化器...");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // ========================================================================
    // 方式1：保存完整模型（包括模型结构和参数）
    // ========================================================================
    std::string model_full_path = "transformer_model_full.pth";
    try {
        torch::save(model, model_full_path);
        Logger::info("✓ [方式1] 完整模型已保存到: {}", model_full_path);
        Logger::info("  说明: 包含模型结构和所有参数，可以直接加载使用");
    } catch (const std::exception& e) {
        Logger::error("✗ 保存完整模型失败: {}", e.what());
    }
    
    // ========================================================================
    // 方式2：保存模型参数（model_dict）
    // ========================================================================
    //std::string model_dict_path = "transformer_model_dict.pth";
    //try {
    //    // 获取模型参数字典
    //    auto model_dict = model->state_dict();
    //    torch::save(model_dict, model_dict_path);
    //    Logger::info("✓ [方式2] 模型参数字典（model_dict）已保存到: {}", model_dict_path);
    //    Logger::info("  说明: 只包含模型参数，加载时需要重新创建模型结构");
    //} catch (const std::exception& e) {
    //    Logger::error("✗ 保存模型参数字典失败: {}", e.what());
    //}
    //
    //// ========================================================================
    //// 方式3：保存优化器参数（optim_dict）
    //// ========================================================================
    //std::string optim_dict_path = "transformer_optim_dict.pth";
    //try {
    //    // 获取优化器状态字典
    //    auto optim_dict = optimizer.state_dict();
    //    torch::save(optim_dict, optim_dict_path);
    //    Logger::info("✓ [方式3] 优化器状态字典（optim_dict）已保存到: {}", optim_dict_path);
    //    Logger::info("  说明: 包含优化器状态（学习率、动量等），用于恢复训练");
    //} catch (const std::exception& e) {
    //    Logger::error("✗ 保存优化器状态字典失败: {}", e.what());
    //}
    
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("所有文件保存完成！");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("保存的文件列表:");
    Logger::info("  1. {} - 完整模型（模型结构 + 参数）", model_full_path);
    //Logger::info("  2. {} - 模型参数（model_dict）", model_dict_path);
    //Logger::info("  3. {} - 优化器参数（optim_dict）", optim_dict_path);
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("加载示例:");
    Logger::info("  加载完整模型:");
    Logger::info("    auto model = torch::load<GPTModel>(\"{}\");", model_full_path);
    /*Logger::info("  加载模型参数:");
    Logger::info("    auto model = std::make_shared<GPTModel>(cfg);");
    Logger::info("    auto model_dict = torch::load(\"{}\");", model_dict_path);
    Logger::info("    model->load_state_dict(model_dict);");
    Logger::info("  加载优化器参数:");
    Logger::info("    auto optim_dict = torch::load(\"{}\");", optim_dict_path);
    Logger::info("    optimizer.load_state_dict(optim_dict);");*/
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    //system("pause");
    return 0;
}

