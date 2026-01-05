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
 * Transformer Inference Program
 * 
 * Transformer模型推理程序（支持 CPU/GPU）
 * 
 * 使用 Generator 类进行文本生成和推理
 * 
 * 架构说明：
 * - 当前实现是 Decoder-Only 架构（GPT风格）
 * - 只包含解码器层，使用 Masked Self-Attention
 * - 不是完整的 Encoder-Decoder Transformer
 * 
 * GPU 使用：
 * - 自动检测：如果 CUDA 可用，自动使用 GPU
 * - 命令行参数：--device cuda 或 --device cpu
 * - 示例：./Transformer_inference --device cuda
 */

#include "GPTModel.h"
#include "ModelConfig.h"
#include "Logger.h"
#include "Generator.h"
#include "TrainingUtils.h"
#include "Tiktoken.h"
#include <torch/torch.h>
#include <memory>
#include <string>
#include <iomanip>

int main(int argc, char * argv[]) 
{
    // ========================================================================
    // 0. Configure logger
    // ========================================================================
    Logger::getInstance().setLogLevel(LogLevel::DEBUG);
    Logger::getInstance().setShowTimestamp(true);
    Logger::getInstance().setShowLevel(true);
    
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Transformer C++ Inference Program");
    Logger::info("═══════════════════════════════════════════════════════════════");
    
    // ========================================================================
    // 1. Parse device parameters
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
    // 2. Create model configuration
    // ========================================================================
    ModelConfig cfg;
    cfg.vocab_size = 50257;        // Vocab size (GPT-2 standard)
    cfg.max_seq_length = 1024;     // Max sequence length
    cfg.embedding_dim = 768;       // Embedding dimension (GPT-2 standard)
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
    // 3. Create tiktoken encoder
    // ========================================================================
    std::shared_ptr<Tiktoken> encoder = nullptr;
    
    // 选项1: 使用简单编码器（字符级，向后兼容）
    // encoder = tiktoken::create_simple_encoding();
    
    // 选项2: 使用 GPT-2 编码器（推荐，更好的 tokenization）
    encoder = tiktoken::create_gpt2_encoding();
    
    // 选项3: 从文件加载编码器
    // encoder = tiktoken::load_encoding_from_file("merges.txt", "gpt2");
    
    // 更新词汇表大小以匹配编码器
    cfg.vocab_size = static_cast<int>(encoder->getVocabSize());
    Logger::info("Using vocab size from encoder: {}", cfg.vocab_size);
    
    // ========================================================================
    // 4. Create model and generator
    // ========================================================================
    // 查找runs/train目录下最新的模型文件
    std::string model_path = find_latest_model("runs/train");
    
    // 如果没有找到，回退到默认路径
    if (model_path.empty()) {
        model_path = "transformer_model_full.pth";
        Logger::info("No model found in runs/train, using default path: {}", model_path);
    } else {
        Logger::info("Using latest model from runs/train: {}", model_path);
    }
    
    std::shared_ptr<GPTModel> model = std::make_shared<GPTModel>(cfg);
    
    Logger::info("Attempting to load saved model: {}", model_path);
    
    Generator generator(model, device, cfg, encoder);
    
    if (!generator.loadModel(model_path)) {
        Logger::warning("Failed to load model, using untrained model");
    }
    
    // ========================================================================
    // 4. Generate text
    // ========================================================================
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("Starting inference...");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    std::string prompt = "How Are You ！！！";
    int max_new_tokens = 100;
    double temperature = 0.8;
    int top_k = 50;  // Top-K采样参数（只从概率最高的50个token中采样）
    
    Logger::info("Prompt: \"{}\"", prompt);
    Logger::info("Max Generation Length: {} tokens", max_new_tokens);
    Logger::info("Temperature: {}", temperature);
    Logger::info("Top-K: {}", top_k);
    
    // 使用Generator生成文本
    GenerationResult result = generator.generate(prompt, max_new_tokens, temperature, top_k);
    
    // 打印结果
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("Inference Results:");
    Logger::info("Original Prompt: \"{}\"", result.prompt);
    Logger::info("Generated Text: \"{}\"", result.generated_text);
    Logger::info("Number of Generated Tokens: {}", result.generated_tokens.size());
    std::ostringstream avg_speed_str;
    avg_speed_str << std::fixed << std::setprecision(2) << result.avg_tokens_per_second;
    Logger::info("Total Inference Time: {:.2f} seconds ({} ms)", result.total_time_sec, result.total_time_ms);
    Logger::info("Average Generation Speed: {} tokens/sec", avg_speed_str.str());
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // ========================================================================
    // 5. Model Parameter Statistics
    // ========================================================================
    Logger::info("=== Model Statistics ===");
    
    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }
    
    Logger::info("Total Parameters: {}", total_params);
    std::ostringstream params_mb;
    params_mb << std::fixed << std::setprecision(2) << (total_params * sizeof(float) / (1024.0 * 1024.0));
    Logger::info("Total Parameters (MB): {}", params_mb.str());
    
    Logger::info("Inference program execution completed!");
    
    return 0;
}

