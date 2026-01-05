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
 * Transformer C++ 使用示例（支持 CPU/GPU）
 * 
 * 演示如何使用 C++ 实现的 Transformer 模型进行前向传播（推理）。
 * 本示例展示了模型的创建、输入准备、前向传播和结果展示的完整流程。
 * 
 * GPU 使用：
 * - 自动检测：如果 CUDA 可用，自动使用 GPU
 * - 命令行参数：--device cuda 或 --device cpu
 * - 示例：./Transformer_demo --device cuda
 */

#include "GPTModel.h"
#include "ModelConfig.h"
#include "Logger.h"
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <string>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <vector>

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
                        Logger::info("检测到 CUDA 可用，使用 GPU 推理");
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
        Logger::info("检测到 CUDA 可用，自动使用 GPU 推理");
        Logger::info("提示: 使用 --device cpu 强制使用 CPU");
        device = torch::kCUDA;
    }
    
    return device;
}

int main(int argc, char * argv[]) 
{
    // ========================================================================
    // 0. 配置日志模块
    // ========================================================================
    Logger::getInstance().setLogLevel(LogLevel::DEBUG);
    Logger::getInstance().setShowTimestamp(true);
    Logger::getInstance().setShowLevel(true);
    
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Transformer C++ 示例（推理）");
    Logger::info("═══════════════════════════════════════════════════════════════");
    
    // ========================================================================
    // 1. 解析设备参数
    // ========================================================================
    torch::Device device = parse_device(argc, argv);
    
    Logger::info("使用设备: {}", (device == torch::kCUDA ? "CUDA (GPU)" : "CPU"));
    if (device == torch::kCUDA) {
        Logger::info("  - GPU 数量: {}", torch::cuda::device_count());
        if (torch::cuda::device_count() > 0) {
           // Logger::info("  - 当前 GPU: {}", torch::cuda::current_device());
            try {
           //     Logger::info("  - GPU 名称: {}", torch::cuda::getDeviceName(torch::cuda::current_device()));
            } catch (...) {
                Logger::warning("  - GPU 名称: 无法获取");
            }
        }
    }
    Logger::info("使用说明:");
    Logger::info("  - 使用 GPU: {} --device cuda", argv[0]);
    Logger::info("  - 使用 CPU: {} --device cpu", argv[0]);
    
    // ========================================================================
    // 1. 创建模型配置
    // ========================================================================
    ModelConfig cfg;
    cfg.vocab_size = 50257;        // 词汇表大小（GPT-2 标准）
    cfg.max_seq_length = 1024;     // 最大序列长度
    cfg.embedding_dim = 768;       // 嵌入向量维度（GPT-2 标准）
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
    // 2. 加载模型
    // ========================================================================
    std::string model_path = "transformer_model_full.pth";
    std::shared_ptr<GPTModel> model= std::make_shared<GPTModel>(cfg) ;
    
    Logger::info("尝试加载已保存的模型: {}", model_path);
    
    try {
        // 尝试加载完整模型
        torch::load(model, model_path);
        model->to(device);
        model->eval();
        Logger::info("✓ 成功加载完整模型: {}", model_path);
    } catch (const std::exception& e) {
        Logger::warning("无法加载模型文件: {}，错误: {}", model_path, e.what());
        Logger::info("创建新模型...");
        // 如果加载失败，创建新模型
        model = std::make_shared<GPTModel>(cfg);
        model->to(device);
        model->eval();
        Logger::info("新模型创建完成（未训练）");
    }
    
    // ========================================================================
    // 3. 推理：使用提示词生成文本
    // ========================================================================
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("开始推理...");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    std::string prompt = "hi are you ！！！";
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
    // 4. 模型参数量统计
    // ========================================================================
    Logger::info("=== 模型统计 ===");
    
    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }
    
    Logger::info("总参数量: {}", total_params);
    std::ostringstream params_mb;
    params_mb << std::fixed << std::setprecision(2) << (total_params * sizeof(float) / (1024.0 * 1024.0));
    Logger::info("总参数量 (MB): {}", params_mb.str());
    
    Logger::info("示例程序执行完成！");
    

    //system("pause");
    return 0;
}
