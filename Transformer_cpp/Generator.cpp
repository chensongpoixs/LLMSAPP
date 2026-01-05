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
 * Generator Implementation
 * 
 * 文本生成器实现
 */

#include "Generator.h"
#include "Logger.h"
#include <torch/torch.h>
#include <sstream>
#include <iomanip>
#include <chrono>

Generator::Generator(std::shared_ptr<GPTModel> model,
                     torch::Device device,
                     const ModelConfig& cfg,
                     std::shared_ptr<Tiktoken> encoder)
    : model_(model), device_(device), cfg_(cfg), encoder_(encoder) {
    
    // 如果没有提供编码器，使用简单编码器
    if (!encoder_) {
        encoder_ = tiktoken::create_simple_encoding();
        Logger::info("Using simple encoding (character-level)");
    } else {
        Logger::info("Using tiktoken encoding: {}", encoder_->getName());
    }
    
    if (model_) {
        model_->to(device_);
        model_->eval();
    }
}

bool Generator::loadModel(const std::string& model_path) {
    try {
        torch::load(model_, model_path);
        model_->to(device_);
        model_->eval();
        return true;
    } catch (const std::exception& e) {
        Logger::error("Failed to load model: {}", e.what());
        return false;
    }
}

std::vector<int64_t> Generator::textToTokens(const std::string& text) {
    // 使用 tiktoken 编码器
    std::vector<uint32_t> encoded = encoder_->encode(text);
    std::vector<int64_t> tokens;
    tokens.reserve(encoded.size());
    for (uint32_t token : encoded) {
        tokens.push_back(static_cast<int64_t>(token));
    }
    return tokens;
}

std::string Generator::tokensToText(const std::vector<int64_t>& tokens) {
    // 使用 tiktoken 编码器
    std::vector<uint32_t> token_ids;
    token_ids.reserve(tokens.size());
    for (int64_t token : tokens) {
        if (token >= 0 && token <= std::numeric_limits<uint32_t>::max()) {
            token_ids.push_back(static_cast<uint32_t>(token));
        }
    }
    return encoder_->decode(token_ids);
}

GenerationResult Generator::generate(const std::string& prompt,
                                    int max_new_tokens,
                                    double temperature,
                                    int top_k) {
    GenerationResult result;
    result.prompt = prompt;
    
    // 将提示词转换为token IDs
    std::vector<int64_t> prompt_tokens = textToTokens(prompt);
    Logger::info("Prompt token sequence length: {}", prompt_tokens.size());
    
    // 准备输入tensor
    torch::Tensor input_ids = torch::zeros({1, (int32_t)prompt_tokens.size()}, 
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        input_ids[0][i] = prompt_tokens[i];
    }
    
    // 生成文本（自回归生成）
    std::vector<int64_t> generated_tokens = prompt_tokens;
    generated_tokens.reserve(prompt_tokens.size() + max_new_tokens);
    
    Logger::info("Starting generation...");
    Logger::info("Generation parameters: temperature={}, top_k={}", temperature, top_k);
    
    // 记录推理开始时间
    auto inference_start = std::chrono::high_resolution_clock::now();
    
    torch::NoGradGuard no_grad;
    
    for (int i = 0; i < max_new_tokens; ++i) {
        // 记录当前token生成开始时间
        auto token_start = std::chrono::high_resolution_clock::now();
        
        // 当前序列长度
        int current_len = input_ids.size(1);
        
        // 如果序列太长，只保留最近的max_seq_length个token（滑动窗口）
        if (current_len > cfg_.max_seq_length) {
            int start_idx = current_len - cfg_.max_seq_length;
            input_ids = input_ids.slice(1, start_idx, current_len);
            current_len = cfg_.max_seq_length;
        }
        
        // 前向传播：使用整个当前序列
        auto logits = model_->forward(input_ids);
        
        // 获取最后一个位置的logits (vocab_size)
        auto next_token_logits = logits[0][current_len - 1];
        
        // 应用 Top-K 采样（如果 top_k > 0）
        if (top_k > 0 && top_k < next_token_logits.size(0)) {
            // 获取 top_k 个最大值的索引和值
            auto topk_result = torch::topk(next_token_logits, top_k);
            auto topk_values = std::get<0>(topk_result);  // top_k 个最大值
            auto topk_indices = std::get<1>(topk_result); // top_k 个最大值的索引
            
            // 创建一个新的 logits tensor，只保留 top_k 个值，其他设为负无穷
            auto filtered_logits = torch::full_like(next_token_logits, 
                -std::numeric_limits<float>::infinity());
            
            // 将 top_k 个值放回原位置
            for (int j = 0; j < top_k; ++j) {
                int64_t idx = topk_indices[j].item<int64_t>();
                filtered_logits[idx] = topk_values[j];
            }
            
            next_token_logits = filtered_logits;
        }
        
        // 使用温度采样（temperature sampling）增加随机性
        auto scaled_logits = next_token_logits / temperature;
        auto probs = torch::softmax(scaled_logits, 0);
        
        // 采样下一个token
        auto next_token = torch::multinomial(probs, 1).item<int64_t>();
        
        // 添加到生成序列
        generated_tokens.push_back(next_token);
        
        // 将新token添加到输入序列（用于下一次迭代）
        auto new_token_tensor = torch::tensor({{next_token}}, 
            torch::TensorOptions().dtype(torch::kLong).device(device_));
        input_ids = torch::cat({input_ids, new_token_tensor}, 1);
        
        // 记录当前token生成结束时间
        auto token_end = std::chrono::high_resolution_clock::now();
        auto token_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            token_end - token_start
        );
        
        // 计算总体推理时间
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            token_end - inference_start
        );
        
        // 计算速度（tokens/秒）
        double tokens_per_second = 0.0;
        if (total_duration.count() > 0) {
            tokens_per_second = ((i + 1) * 1000.0) / total_duration.count();
        }
        
        // 每10个token打印一次进度
        if ((i + 1) % 10 == 0) {
            std::ostringstream speed_str;
            speed_str << std::fixed << std::setprecision(2) << tokens_per_second;
            Logger::info("Generated {}/{} tokens | Speed: {} tokens/sec | Current token time: {} ms", 
                (i + 1), max_new_tokens, speed_str.str(), token_duration.count());
        }
    }
    
    // 计算总推理时间
    auto inference_end = std::chrono::high_resolution_clock::now();
    auto total_inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        inference_end - inference_start
    );
    result.total_time_sec = total_inference_time.count() / 1000.0;
    result.total_time_ms = total_inference_time.count();
    
    // 计算平均速度
    if (result.total_time_sec > 0) {
        result.avg_tokens_per_second = max_new_tokens / result.total_time_sec;
    }
    
    // 将token序列转换为文本
    result.generated_tokens = generated_tokens;
    result.generated_text = tokensToText(generated_tokens);
    
    return result;
}

