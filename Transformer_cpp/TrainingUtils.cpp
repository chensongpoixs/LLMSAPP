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
 * Training Utilities Implementation
 * 
 * 训练相关的工具函数实现
 */

#include "TrainingUtils.h"
#include "Logger.h"
#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <cstring>

float train_batch(
    std::shared_ptr<GPTModel> model,
    torch::Tensor input_ids,
    torch::Tensor target_ids,
    torch::nn::CrossEntropyLoss& criterion,
    torch::optim::Adam& optimizer,
    double clip_grad_norm) {
    
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
                        Logger::info("CUDA detected, using GPU for training");
                    } else {
                        Logger::warning("GPU requested but CUDA not available, falling back to CPU");
                        device = torch::kCPU;
                    }
                } else if (device_str == "cpu") {
                    device = torch::kCPU;
                } else {
                    Logger::warning("Unknown device type '{}', using default CPU", device_str);
                }
            }
            break;
        }
    }
    
    // 如果没有指定参数，自动检测并使用 GPU（如果可用）
    if (device == torch::kCPU && torch::cuda::is_available()) {
        Logger::info("CUDA detected, automatically using GPU for training");
        Logger::info("Hint: Use --device cpu to force CPU usage");
        device = torch::kCUDA;
    }
    
    return device;
}

int get_max_exp_number(const std::string& train_dir) {
    namespace fs = std::filesystem;
    
    if (!fs::exists(train_dir) || !fs::is_directory(train_dir)) {
        return 0;  // 如果目录不存在，从exp开始
    }
    
    int max_num = 0;  // 默认是exp（编号0）
    
    try {
        for (const auto& entry : fs::directory_iterator(train_dir)) {
            if (entry.is_directory()) {
                std::string dir_name = entry.path().filename().string();
                
                // 检查是否是exp格式（exp, exp2, exp3, ...）
                if (dir_name == "exp") {
                    max_num = std::max(max_num, 1);  // exp对应编号1
                } else if (dir_name.length() > 3 && dir_name.substr(0, 3) == "exp") {
                    // 提取exp后面的数字
                    std::string num_str = dir_name.substr(3);
                    if (!num_str.empty()) {
                        try {
                            int num = std::stoi(num_str);
                            max_num = std::max(max_num, num);
                        } catch (const std::exception&) {
                            // 如果无法转换为数字，忽略
                            continue;
                        }
                    } else {
                        // exp（没有数字后缀）对应编号1
                        max_num = std::max(max_num, 1);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        Logger::warning("Error reading train directory: {}", e.what());
        return 0;
    }
    
    return max_num;
}

std::string get_exp_dir_name(int exp_num) {
    if (exp_num <= 1) {
        return "exp";
    } else {
        return "exp" + std::to_string(exp_num);
    }
}

bool save_model_checkpoint(std::shared_ptr<GPTModel> model, const std::string& save_path) {
    try {
        // 确保目录存在
        namespace fs = std::filesystem;
        fs::path path(save_path);
        fs::path dir = path.parent_path();
        
        if (!dir.empty() && !fs::exists(dir)) {
            fs::create_directories(dir);
        }
        
        // 保存模型
        torch::save(model, save_path);
        return true;
    } catch (const std::exception& e) {
        Logger::error("Failed to save model checkpoint: {}", e.what());
        return false;
    }
}

void save_training_config(const std::string& exp_dir, const ModelConfig& cfg, 
                         double learning_rate, double weight_decay, 
                         int batch_size, int num_epochs) {
    namespace fs = std::filesystem;
    std::string config_path = exp_dir + "/config.yaml";
    
    try {
        std::ofstream config_file(config_path);
        if (config_file.is_open()) {
            config_file << "# Training Configuration\n";
            config_file << "# Generated automatically during training\n\n";
            
            config_file << "# Model Configuration\n";
            config_file << "model:\n";
            config_file << "  vocab_size: " << cfg.vocab_size << "\n";
            config_file << "  max_seq_length: " << cfg.max_seq_length << "\n";
            config_file << "  embedding_dim: " << cfg.embedding_dim << "\n";
            config_file << "  n_heads: " << cfg.n_heads << "\n";
            config_file << "  n_layers: " << cfg.n_layers << "\n";
            config_file << "  drop_rate: " << cfg.drop_rate << "\n";
            config_file << "  qkv_bias: " << (cfg.qkv_bias ? "true" : "false") << "\n\n";
            
            config_file << "# Training Hyperparameters\n";
            config_file << "training:\n";
            config_file << "  learning_rate: " << learning_rate << "\n";
            config_file << "  weight_decay: " << weight_decay << "\n";
            config_file << "  batch_size: " << batch_size << "\n";
            config_file << "  num_epochs: " << num_epochs << "\n";
            
            config_file.close();
            Logger::debug("Training configuration saved to: {}", config_path);
        }
    } catch (const std::exception& e) {
        Logger::warning("Failed to save training configuration: {}", e.what());
    }
}

std::string find_latest_model(const std::string& train_dir) {
    namespace fs = std::filesystem;
    
    if (!fs::exists(train_dir) || !fs::is_directory(train_dir)) {
        Logger::warning("Training directory does not exist: {}", train_dir);
        return "";
    }
    
    // 获取最大实验编号
    int max_exp_num = get_max_exp_number(train_dir);
    
    // 从最大编号开始向下查找，找到第一个存在的模型文件
    for (int exp_num = max_exp_num; exp_num >= 1; --exp_num) {
        std::string exp_dir_name = get_exp_dir_name(exp_num);
        std::string exp_dir = train_dir + "/" + exp_dir_name;
        std::string weights_dir = exp_dir + "/weights";
        
        if (!fs::exists(exp_dir) || !fs::is_directory(exp_dir)) {
            continue;
        }
        
        if (!fs::exists(weights_dir) || !fs::is_directory(weights_dir)) {
            continue;
        }
        
        // 优先查找 best.pth
        std::string best_model = weights_dir + "/best.pth";
        if (fs::exists(best_model) && fs::is_regular_file(best_model)) {
            Logger::info("Found latest model (best): {}", best_model);
            return best_model;
        }
        
        // 如果没有 best.pth，查找 last.pth
        std::string last_model = weights_dir + "/last.pth";
        if (fs::exists(last_model) && fs::is_regular_file(last_model)) {
            Logger::info("Found latest model (last): {}", last_model);
            return last_model;
        }
    }
    
    Logger::warning("No model found in {}", train_dir);
    return "";
}

