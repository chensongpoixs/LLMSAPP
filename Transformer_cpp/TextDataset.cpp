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
 * 文本数据集实现
 */

#include "TextDataset.h"
#include "Logger.h"
#include <algorithm>
#include <cctype>
#include <limits>

TextDataset::TextDataset(const std::string& filepath, int seq_len, 
                         std::shared_ptr<Tiktoken> encoder)
    : filepath_(filepath)
    , seq_len_(seq_len)
    , current_pos_(0)
    , encoder_(encoder) {
    
    // 如果没有提供编码器，使用简单编码器
    if (!encoder_) {
        encoder_ = tiktoken::create_simple_encoding();
        Logger::info("Using simple encoding (character-level)");
    } else {
        Logger::info("Using tiktoken encoding: {}", encoder_->getName());
    }
    
    vocab_size_ = static_cast<int>(encoder_->getVocabSize());
    Logger::info("Vocab size: {}", vocab_size_);
}

bool TextDataset::load() {
    std::ifstream file(filepath_);
    if (!file.is_open()) {
        Logger::error("Failed to open file: {}", filepath_);
        return false;
    }
    
    // 读取整个文件
    std::ostringstream buffer;
    buffer << file.rdbuf();
    text_ = buffer.str();
    file.close();
    
    if (text_.empty()) {
        Logger::error("File is empty: {}", filepath_);
        return false;
    }
    
    Logger::info("Successfully loaded file: {}", filepath_);
    Logger::info("File size: {} characters", text_.size());
    
    // Convert to token sequence
    tokenize();
    
    Logger::info("Token sequence length: {}", tokens_.size());
    Logger::info("Dataset size (number of samples): {}", size());
    
    return true;
}

void TextDataset::tokenize() {
    tokens_.clear();
    
    // 使用 tiktoken 编码器进行 tokenization
    std::vector<uint32_t> encoded_tokens = encoder_->encode(text_);
    
    // 转换为 int64_t 类型
    tokens_.reserve(encoded_tokens.size());
    for (uint32_t token : encoded_tokens) {
        tokens_.push_back(static_cast<int64_t>(token));
    }
    
    Logger::info("Tokenized text: {} characters -> {} tokens", 
                 text_.size(), tokens_.size());
}

size_t TextDataset::size() const {
    if (tokens_.size() < seq_len_ + 1) {
        return 0;
    }
    // 每个位置都可以作为一个样本的起点
    return tokens_.size() - seq_len_;
}

std::vector<int64_t> TextDataset::textToTokens(const std::string& text) const {
    std::vector<uint32_t> encoded = encoder_->encode(text);
    std::vector<int64_t> tokens;
    tokens.reserve(encoded.size());
    for (uint32_t token : encoded) {
        tokens.push_back(static_cast<int64_t>(token));
    }
    return tokens;
}

std::string TextDataset::tokensToText(const std::vector<int64_t>& tokens) const {
    std::vector<uint32_t> token_ids;
    token_ids.reserve(tokens.size());
    for (int64_t token : tokens) {
        if (token >= 0 && token <= std::numeric_limits<uint32_t>::max()) {
            token_ids.push_back(static_cast<uint32_t>(token));
        }
    }
    return encoder_->decode(token_ids);
}

std::pair<torch::Tensor, torch::Tensor> TextDataset::getBatch(int batch_size, torch::Device device) {
    if (tokens_.size() < seq_len_ + 1) {
        Logger::error("Token sequence too short to generate batch");
        return std::make_pair(
            torch::zeros({batch_size, seq_len_}, torch::TensorOptions().dtype(torch::kLong).device(device)),
            torch::zeros({batch_size, seq_len_}, torch::TensorOptions().dtype(torch::kLong).device(device))
        );
    }
    
    // 准备批次数据
    std::vector<int64_t> input_data;
    std::vector<int64_t> target_data;
    input_data.reserve(batch_size * seq_len_);
    target_data.reserve(batch_size * seq_len_);
    
    // 生成batch_size个样本
    for (int i = 0; i < batch_size; ++i) {
        // 随机选择起始位置（避免总是从开头开始）
        size_t start_pos = current_pos_;
        
        // 如果当前位置超出范围，从头开始
        if (start_pos + seq_len_ >= tokens_.size()) {
            start_pos = 0;
        }
        
        // 提取输入序列（从start_pos到start_pos+seq_len-1）
        for (int j = 0; j < seq_len_; ++j) {
            input_data.push_back(tokens_[start_pos + j]);
        }
        
        // 提取目标序列（从start_pos+1到start_pos+seq_len），用于预测下一个token
        for (int j = 0; j < seq_len_; ++j) {
            target_data.push_back(tokens_[start_pos + j + 1]);
        }
        
        // 更新当前位置（滑动窗口，每次移动seq_len）
        current_pos_ = (start_pos + seq_len_) % (tokens_.size() - seq_len_);
    }
    
    // 创建tensor（使用clone确保数据被复制）
    auto input_tensor = torch::from_blob(
        input_data.data(),
        {batch_size, seq_len_},
        torch::TensorOptions().dtype(torch::kLong)
    ).clone().to(device);
    
    auto target_tensor = torch::from_blob(
        target_data.data(),
        {batch_size, seq_len_},
        torch::TensorOptions().dtype(torch::kLong)
    ).clone().to(device);
    
    return std::make_pair(input_tensor, target_tensor);
}

