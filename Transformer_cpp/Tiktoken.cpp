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
 * Tiktoken Implementation
 * 
 * BPE tokenizer 的 C++ 实现
 */

#include "Tiktoken.h"
#include "TiktokenGPT2.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cctype>
#include <regex>
#include <limits>
#include <set>
#include <functional>

Tiktoken::Tiktoken(const std::string& name,
                   const std::string& pat_str,
                   const std::map<std::vector<uint8_t>, uint32_t>& mergeable_ranks,
                   const std::map<std::string, uint32_t>& special_tokens)
    : name_(name), pat_str_(pat_str), mergeable_ranks_(mergeable_ranks), special_tokens_(special_tokens) {
    
    // 构建反向映射
    for (const auto& pair : special_tokens_) {
        special_tokens_reverse_[pair.second] = pair.first;
    }
    
    // 计算词汇表大小
    vocab_size_ = 0;
    for (const auto& pair : mergeable_ranks_) {
        vocab_size_ = std::max(vocab_size_, static_cast<size_t>(pair.second + 1));
    }
    for (const auto& pair : special_tokens_) {
        vocab_size_ = std::max(vocab_size_, static_cast<size_t>(pair.second + 1));
    }
}

std::vector<uint8_t> Tiktoken::stringToBytes(const std::string& text) const {
    std::vector<uint8_t> bytes;
    bytes.reserve(text.size());
    for (char c : text) {
        bytes.push_back(static_cast<uint8_t>(c));
    }
    return bytes;
}

std::string Tiktoken::bytesToString(const std::vector<uint8_t>& bytes) const {
    std::string text;
    text.reserve(bytes.size());
    for (uint8_t b : bytes) {
        text += static_cast<char>(b);
    }
    return text;
}

std::vector<std::string> Tiktoken::splitText(const std::string& text) const {
    // 简化版本：按空格和标点分割
    // 实际 tiktoken 使用正则表达式，这里简化处理
    std::vector<std::string> parts;
    std::string current;
    
    for (char c : text) {
        if (std::isspace(c) || std::ispunct(c)) {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
            parts.push_back(std::string(1, c));
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        parts.push_back(current);
    }
    
    return parts;
}

std::vector<uint32_t> Tiktoken::applyBPE(const std::vector<uint8_t>& bytes) const {
    if (bytes.empty()) {
        return {};
    }
    
    // 构建快速查找表：将字节对映射到 rank
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> rank_map;
    for (const auto& pair : mergeable_ranks_) {
        if (pair.first.size() == 2) {
            uint32_t token0 = static_cast<uint32_t>(pair.first[0]);
            uint32_t token1 = static_cast<uint32_t>(pair.first[1]);
            rank_map[{token0, token1}] = pair.second;
        }
    }
    
    // 初始化为字节值
    std::vector<uint32_t> tokens;
    tokens.reserve(bytes.size());
    for (uint8_t b : bytes) {
        tokens.push_back(static_cast<uint32_t>(b));
    }
    
    // 迭代应用 BPE 合并
    while (true) {
        auto [pos, new_token] = findBestMerge(tokens, rank_map);
        if (pos == std::numeric_limits<size_t>::max()) {
            break;  // 没有更多合并
        }
        
        // 执行合并
        tokens[pos] = new_token;
        tokens.erase(tokens.begin() + pos + 1);
    }
    
    return tokens;
}

std::pair<size_t, uint32_t> Tiktoken::findBestMerge(
    const std::vector<uint32_t>& tokens,
    const std::map<std::pair<uint32_t, uint32_t>, uint32_t>& rank_map) const {
    
    if (tokens.size() < 2) {
        return {std::numeric_limits<size_t>::max(), 0};
    }
    
    uint32_t best_rank = std::numeric_limits<uint32_t>::max();
    size_t best_pos = std::numeric_limits<size_t>::max();
    
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        std::pair<uint32_t, uint32_t> pair = {tokens[i], tokens[i + 1]};
        auto it = rank_map.find(pair);
        if (it != rank_map.end() && it->second < best_rank) {
            best_rank = it->second;
            best_pos = i;
        }
    }
    
    if (best_pos != std::numeric_limits<size_t>::max()) {
        return {best_pos, rank_map.at({tokens[best_pos], tokens[best_pos + 1]})};
    }
    
    return {std::numeric_limits<size_t>::max(), 0};
}

std::vector<uint32_t> Tiktoken::encode(const std::string& text,
                                       const std::vector<std::string>& allowed_special,
                                       const std::vector<std::string>& disallowed_special) const {
    std::vector<uint32_t> result;
    
    // 检查特殊 token
    std::set<std::string> allowed_set(allowed_special.begin(), allowed_special.end());
    std::set<std::string> disallowed_set(disallowed_special.begin(), disallowed_special.end());
    
    // 简化处理：先检查特殊 token
    std::string remaining = text;
    size_t pos = 0;
    
    while (pos < remaining.size()) {
        // 查找特殊 token
        bool found_special = false;
        size_t best_match_pos = std::numeric_limits<size_t>::max();
        std::string best_match_token;
        uint32_t best_match_id = 0;
        
        for (const auto& pair : special_tokens_) {
            size_t found = remaining.find(pair.first, pos);
            if (found != std::string::npos && found < best_match_pos) {
                // 检查是否允许
                if (!disallowed_set.empty() && disallowed_set.find(pair.first) != disallowed_set.end()) {
                    continue;  // 不允许的特殊 token
                }
                if (!allowed_set.empty() && allowed_set.find(pair.first) == allowed_set.end()) {
                    continue;  // 不在允许列表中
                }
                
                best_match_pos = found;
                best_match_token = pair.first;
                best_match_id = pair.second;
                found_special = true;
            }
        }
        
        if (found_special && best_match_pos == pos) {
            // 找到特殊 token，添加它
            result.push_back(best_match_id);
            pos += best_match_token.size();
        } else {
            // 处理普通文本
            size_t end_pos = found_special ? best_match_pos : remaining.size();
            std::string segment = remaining.substr(pos, end_pos - pos);
            
            // 转换为字节并应用 BPE
            auto bytes = stringToBytes(segment);
            auto tokens = applyBPE(bytes);
            result.insert(result.end(), tokens.begin(), tokens.end());
            
            pos = end_pos;
        }
    }
    
    return result;
}

std::string Tiktoken::decode(const std::vector<uint32_t>& tokens) const {
    std::vector<uint8_t> bytes;
    
    // 构建反向映射：token_id -> 字节序列
    std::map<uint32_t, std::vector<uint8_t>> token_to_bytes;
    
    // 首先添加单字节映射（0-255）
    for (int i = 0; i < 256; ++i) {
        token_to_bytes[static_cast<uint32_t>(i)] = {static_cast<uint8_t>(i)};
    }
    
    // 添加合并规则的映射
    for (const auto& pair : mergeable_ranks_) {
        token_to_bytes[pair.second] = pair.first;
    }
    
    // 递归展开函数
    std::function<void(uint32_t, std::vector<uint8_t>&)> expandToken = 
        [&](uint32_t token_id, std::vector<uint8_t>& result) {
            // 检查是否为特殊 token
            auto special_it = special_tokens_reverse_.find(token_id);
            if (special_it != special_tokens_reverse_.end()) {
                // 特殊 token，直接添加字符串的字节
                std::string special_str = special_it->second;
                for (char c : special_str) {
                    result.push_back(static_cast<uint8_t>(c));
                }
                return;
            }
            
            // 查找 token 对应的字节序列
            auto it = token_to_bytes.find(token_id);
            if (it != token_to_bytes.end()) {
                const auto& byte_seq = it->second;
                // 如果字节序列长度大于1，可能是合并的 token，需要递归展开
                if (byte_seq.size() == 1) {
                    // 单字节，直接添加
                    result.push_back(byte_seq[0]);
                } else {
                    // 多字节，可能是合并的，但这里简化处理，直接添加
                    // 实际 BPE 中，合并的 token 应该已经是最小单位
                    result.insert(result.end(), byte_seq.begin(), byte_seq.end());
                }
            } else {
                // 如果找不到，假设是单字节 token（回退）
                if (token_id < 256) {
                    result.push_back(static_cast<uint8_t>(token_id));
                }
            }
        };
    
    // 解码所有 tokens
    for (uint32_t token : tokens) {
        expandToken(token, bytes);
    }
    
    return bytesToString(bytes);
}

bool Tiktoken::isSpecialToken(uint32_t token_id) const {
    return special_tokens_reverse_.find(token_id) != special_tokens_reverse_.end();
}

uint32_t Tiktoken::getSpecialTokenId(const std::string& special_token) const {
    auto it = special_tokens_.find(special_token);
    if (it != special_tokens_.end()) {
        return it->second;
    }
    return std::numeric_limits<uint32_t>::max();
}

// 工厂函数实现
namespace tiktoken {
    
    std::shared_ptr<Tiktoken> create_simple_encoding() {
        // 创建简单的字符级编码器
        std::map<std::vector<uint8_t>, uint32_t> mergeable_ranks;
        
        // 为每个字节（0-255）分配 token ID
        for (int i = 0; i < 256; ++i) {
            mergeable_ranks[{static_cast<uint8_t>(i)}] = static_cast<uint32_t>(i);
        }
        
        std::map<std::string, uint32_t> special_tokens;
        
        return std::make_shared<Tiktoken>("simple", "", mergeable_ranks, special_tokens);
    }
    
    std::shared_ptr<Tiktoken> create_gpt2_encoding() {
        auto config = tiktoken_gpt2::get_gpt2_config();
        return std::make_shared<Tiktoken>(
            config.name,
            config.pat_str,
            config.mergeable_ranks,
            config.special_tokens
        );
    }
    
    std::shared_ptr<Tiktoken> get_encoding(const std::string& encoding_name) {
        if (encoding_name == "gpt2") {
            return create_gpt2_encoding();
        }
        
        if (encoding_name == "simple" || encoding_name.empty()) {
            return create_simple_encoding();
        }
        
        // 可以添加更多编码，如 cl100k_base, o200k_base 等
        // 需要从文件或配置加载 mergeable_ranks
        
        return create_simple_encoding();
    }
    
    std::shared_ptr<Tiktoken> encoding_for_model(const std::string& model_name) {
        // 根据模型名称返回对应的编码器
        if (model_name.find("gpt-2") != std::string::npos || 
            model_name == "gpt2") {
            return create_gpt2_encoding();
        }
        
        if (model_name.find("gpt-4") != std::string::npos || 
            model_name.find("gpt-3.5") != std::string::npos) {
            // 应该返回 cl100k_base，这里简化
            return get_encoding("simple");
        }
        
        return get_encoding("simple");
    }
    
    std::shared_ptr<Tiktoken> load_encoding_from_file(const std::string& config_file,
                                                      const std::string& encoding_name) {
        // 尝试加载 GPT-2 配置
        auto config = tiktoken_gpt2::load_gpt2_config_from_files(config_file);
        
        if (config.mergeable_ranks.empty()) {
            return nullptr;  // 加载失败
        }
        
        return std::make_shared<Tiktoken>(
            encoding_name.empty() ? config.name : encoding_name,
            config.pat_str,
            config.mergeable_ranks,
            config.special_tokens
        );
    }
}

