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
#include <limits>
#include <set>
#include <unordered_set>
#include <functional>

#ifdef TIKTOKEN_USE_RE2
#include <re2/re2.h>
#include <re2/stringpiece.h>
#else
#include <regex>
#endif

Tiktoken::Tiktoken(const std::string& name,
                   const std::string& pat_str,
                   const std::map<std::vector<uint8_t>, uint32_t>& mergeable_ranks,
                   const std::map<std::string, uint32_t>& special_tokens)
    : name_(name), pat_str_(pat_str), mergeable_ranks_(mergeable_ranks), special_tokens_(special_tokens),
      rank_map_initialized_(false)
#ifdef TIKTOKEN_USE_RE2
      , regex_pattern_(nullptr), regex_compiled_(false)
#endif
{
    
    // 参数验证
    if (name.empty()) {
        throw TiktokenException("Encoder name cannot be empty");
    }
    
    // 构建反向映射
    for (const auto& pair : special_tokens_) {
        if (special_tokens_reverse_.find(pair.second) != special_tokens_reverse_.end()) {
            throw TiktokenException("Duplicate special token ID: " + std::to_string(pair.second));
        }
        special_tokens_reverse_[pair.second] = pair.first;
    }
    
    // 计算词汇表大小
    vocab_size_ = 0;
    for (const auto& pair : mergeable_ranks_) {
        if (pair.first.empty()) {
            throw TiktokenException("Empty byte sequence in mergeable_ranks");
        }
        vocab_size_ = std::max(vocab_size_, static_cast<size_t>(pair.second + 1));
    }
    for (const auto& pair : special_tokens_) {
        vocab_size_ = std::max(vocab_size_, static_cast<size_t>(pair.second + 1));
    }
    
    if (vocab_size_ == 0) {
        throw TiktokenException("Vocab size cannot be zero");
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
    std::vector<std::string> parts;
    
    // 如果 pat_str_ 为空，使用简化分割
    if (pat_str_.empty()) {
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
    
#ifdef TIKTOKEN_USE_RE2
    // 使用 RE2 进行正则表达式匹配（支持完整的 Unicode）
    try {
        // 编译正则表达式（只编译一次，缓存结果）
        if (!regex_compiled_ || !regex_pattern_) {
            regex_pattern_ = std::make_unique<RE2>(pat_str_);
            if (!regex_pattern_->ok()) {
                throw TiktokenException("Invalid regex pattern: " + regex_pattern_->error());
            }
            regex_compiled_ = true;
        }
        
        // 使用 RE2 进行匹配
        // RE2 支持完整的 Unicode 属性类（\p{L}, \p{N} 等）
        re2::StringPiece input(text);
        re2::StringPiece match;
        
        // 使用 RE2::FindAndConsume 找到所有匹配的片段
        // FindAndConsume 会消耗匹配的部分，并返回匹配的内容
        // 注意：FindAndConsume 的第二个参数是捕获组，我们需要使用整个模式
        std::string match_str;
        while (RE2::FindAndConsume(&input, *regex_pattern_, &match_str)) {
            if (!match_str.empty()) {
                parts.push_back(match_str);
            }
        }
        
        // 处理剩余部分（如果有未匹配的文本）
        if (!input.empty()) {
            parts.push_back(std::string(input.data(), input.length()));
        }
        
    } catch (const TiktokenException& e) {
        // RE2 编译错误，回退到简化分割
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
    }
#else
    // 回退到 std::regex（不支持完整 Unicode）
    try {
        // 构建正则表达式
        // 将 \p{L} 替换为 [a-zA-Z\u0080-\uFFFF]（字母和 Unicode 字母）
        // 将 \p{N} 替换为 [0-9\u0660-\u0669\u06F0-\u06F9]（数字和 Unicode 数字）
        std::string regex_pattern = pat_str_;
        
        // 替换 Unicode 类别（简化处理）
        if (regex_pattern.find("\\p{L}") != std::string::npos) {
            size_t pos = 0;
            while ((pos = regex_pattern.find("\\p{L}", pos)) != std::string::npos) {
                regex_pattern.replace(pos, 5, "[a-zA-Z\\x80-\\xFF]");
                pos += 15;
            }
        }
        if (regex_pattern.find("\\p{N}") != std::string::npos) {
            size_t pos = 0;
            while ((pos = regex_pattern.find("\\p{N}", pos)) != std::string::npos) {
                regex_pattern.replace(pos, 5, "[0-9\\x80-\\xFF]");
                pos += 15;
            }
        }
        
        std::regex pattern(regex_pattern, std::regex_constants::ECMAScript | std::regex_constants::optimize);
        std::sregex_iterator iter(text.begin(), text.end(), pattern);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            std::smatch match = *iter;
            if (!match.str().empty()) {
                parts.push_back(match.str());
            }
        }
    } catch (const std::regex_error& e) {
        // 正则表达式错误，回退到简化分割
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
    }
#endif
    
    return parts;
}

void Tiktoken::buildRankMap(std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t>& rank_map) const {
    // 如果已经初始化，直接返回缓存的 rank_map
    if (rank_map_initialized_ && !rank_map_cache_.empty()) {
        rank_map = rank_map_cache_;
        return;
    }
    
    rank_map.clear();
    
    // 首先构建 token_id -> 字节序列的映射
    std::unordered_map<uint32_t, std::vector<uint8_t>> token_to_bytes;
    
    // 添加单字节 token (0-255)
    for (int i = 0; i < 256; ++i) {
        token_to_bytes[static_cast<uint32_t>(i)] = {static_cast<uint8_t>(i)};
    }
    
    // 添加所有 mergeable_ranks 中的 token
    for (const auto& pair : mergeable_ranks_) {
        token_to_bytes[pair.second] = pair.first;
        
        // 如果是双字节 token，添加到 rank_map
        if (pair.first.size() == 2) {
            uint32_t token0 = static_cast<uint32_t>(pair.first[0]);
            uint32_t token1 = static_cast<uint32_t>(pair.first[1]);
            rank_map[{token0, token1}] = pair.second;
        }
    }
    
    // 构建多字节 token 对的 rank_map（支持递归合并）
    // 遍历所有可能的 token 对组合
    for (const auto& pair1 : mergeable_ranks_) {
        uint32_t token1_id = pair1.second;
        const auto& bytes1 = pair1.first;
        
        if (bytes1.empty()) continue;
        
        uint8_t last_byte = bytes1.back();
        
        // 查找所有以 bytes1 最后一个字节开头的 token
        for (const auto& pair2 : mergeable_ranks_) {
            const auto& bytes2 = pair2.first;
            if (bytes2.empty()) continue;
            
            // 如果 pair2 的第一个字节等于 pair1 的最后一个字节，可以合并
            if (bytes2.front() == last_byte) {
                uint32_t token2_id = pair2.second;
                
                // 构建合并后的字节序列
                std::vector<uint8_t> merged_bytes = bytes1;
                merged_bytes.insert(merged_bytes.end(), bytes2.begin() + 1, bytes2.end());
                
                // 查找是否有这个合并规则的 rank
                auto it = mergeable_ranks_.find(merged_bytes);
                if (it != mergeable_ranks_.end()) {
                    rank_map[{token1_id, token2_id}] = it->second;
                }
            }
        }
        
        // 也检查单字节 token 对
        for (int i = 0; i < 256; ++i) {
            if (static_cast<uint8_t>(i) == last_byte) {
                std::vector<uint8_t> merged_bytes = bytes1;
                merged_bytes.push_back(static_cast<uint8_t>(i));
                
                auto it = mergeable_ranks_.find(merged_bytes);
                if (it != mergeable_ranks_.end()) {
                    rank_map[{token1_id, static_cast<uint32_t>(i)}] = it->second;
                }
            }
        }
    }
    
    // 缓存 rank_map
    rank_map_cache_ = rank_map;
    rank_map_initialized_ = true;
}

std::vector<uint32_t> Tiktoken::applyBPE(const std::vector<uint8_t>& bytes) const {
    if (bytes.empty()) {
        return {};
    }
    
    // 构建完整的 rank_map（包括多字节 token 对）
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> rank_map;
    buildRankMap(rank_map);
    
    // 初始化为字节值
    std::vector<uint32_t> tokens;
    tokens.reserve(bytes.size());
    for (uint8_t b : bytes) {
        tokens.push_back(static_cast<uint32_t>(b));
    }
    
    // 迭代应用 BPE 合并（支持多轮合并）
    size_t max_iterations = tokens.size() * 2;  // 防止无限循环
    size_t iteration = 0;
    
    while (iteration < max_iterations) {
        auto [pos, new_token] = findBestMerge(tokens, rank_map);
        if (pos == std::numeric_limits<size_t>::max()) {
            break;  // 没有更多合并
        }
        
        // 执行合并
        tokens[pos] = new_token;
        tokens.erase(tokens.begin() + pos + 1);
        iteration++;
    }
    
    if (iteration >= max_iterations) {
        // 警告：达到最大迭代次数，可能存在问题
        // 但不抛出异常，返回当前结果
    }
    
    return tokens;
}

std::pair<size_t, uint32_t> Tiktoken::findBestMerge(
    const std::vector<uint32_t>& tokens,
    const std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t>& rank_map) const {
    
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
        auto it = rank_map.find({tokens[best_pos], tokens[best_pos + 1]});
        if (it != rank_map.end()) {
            return {best_pos, it->second};
        }
    }
    
    return {std::numeric_limits<size_t>::max(), 0};
}

std::vector<uint32_t> Tiktoken::encode(const std::string& text,
                                       const std::vector<std::string>& allowed_special,
                                       const std::vector<std::string>& disallowed_special) const {
    if (text.empty()) {
        return {};
    }
    
    std::vector<uint32_t> result;
    result.reserve(text.size() / 2);  // 预分配空间
    
    // 检查特殊 token
    std::unordered_set<std::string> allowed_set(allowed_special.begin(), allowed_special.end());
    std::unordered_set<std::string> disallowed_set(disallowed_special.begin(), disallowed_special.end());
    
    // 如果 disallowed_special 为空且 allowed_special 也为空，默认不允许特殊 token
    bool allow_all_special = !allowed_set.empty() && allowed_set.count("all") > 0;
    
    // 使用正则表达式分割文本（如果 pat_str_ 不为空）
    std::vector<std::string> segments;
    if (!pat_str_.empty()) {
        segments = splitText(text);
    } else {
        // 如果没有正则表达式模式，直接处理整个文本
        segments.push_back(text);
    }
    
    // 处理每个分割的片段
    for (const std::string& segment : segments) {
        if (segment.empty()) {
            continue;
        }
        
        // 检查是否包含特殊 token
        bool found_special = false;
        size_t best_match_pos = std::numeric_limits<size_t>::max();
        std::string best_match_token;
        uint32_t best_match_id = 0;
        
        // 查找特殊 token（按长度从长到短排序，优先匹配长的）
        // 使用 vector 存储并按长度排序
        std::vector<std::pair<std::string, uint32_t>> sorted_special;
        for (const auto& pair : special_tokens_) {
            sorted_special.push_back(pair);
        }
        std::sort(sorted_special.begin(), sorted_special.end(),
                  [](const auto& a, const auto& b) {
                      return a.first.length() > b.first.length();
                  });
        
        for (const auto& pair : sorted_special) {
            size_t found = segment.find(pair.first);
            if (found != std::string::npos && found < best_match_pos) {
                // 检查是否允许
                if (!disallowed_set.empty() && disallowed_set.find(pair.first) != disallowed_set.end()) {
                    continue;  // 不允许的特殊 token
                }
                if (!allow_all_special && !allowed_set.empty() && allowed_set.find(pair.first) == allowed_set.end()) {
                    continue;  // 不在允许列表中
                }
                
                best_match_pos = found;
                best_match_token = pair.first;
                best_match_id = pair.second;
                found_special = true;
            }
        }
        
        if (found_special) {
            // 处理特殊 token 前后的文本
            if (best_match_pos > 0) {
                // 处理特殊 token 之前的文本
                std::string before = segment.substr(0, best_match_pos);
                auto bytes = stringToBytes(before);
                auto tokens = applyBPE(bytes);
                result.insert(result.end(), tokens.begin(), tokens.end());
            }
            
            // 添加特殊 token
            result.push_back(best_match_id);
            
            // 处理特殊 token 之后的文本
            if (best_match_pos + best_match_token.size() < segment.size()) {
                std::string after = segment.substr(best_match_pos + best_match_token.size());
                auto bytes = stringToBytes(after);
                auto tokens = applyBPE(bytes);
                result.insert(result.end(), tokens.begin(), tokens.end());
            }
        } else {
            // 没有特殊 token，直接处理整个片段
            auto bytes = stringToBytes(segment);
            auto tokens = applyBPE(bytes);
            result.insert(result.end(), tokens.begin(), tokens.end());
        }
    }
    
    return result;
}
//    
//    while (pos < remaining.size()) {
//        // 查找特殊 token
//        bool found_special = false;
//        size_t best_match_pos = std::numeric_limits<size_t>::max();
//        std::string best_match_token;
//        uint32_t best_match_id = 0;
//        
//        for (const auto& pair : special_tokens_) {
//            size_t found = remaining.find(pair.first, pos);
//            if (found != std::string::npos && found < best_match_pos) {
//                // 检查是否允许
//                if (!disallowed_set.empty() && disallowed_set.find(pair.first) != disallowed_set.end()) {
//                    continue;  // 不允许的特殊 token
//                }
//                if (!allowed_set.empty() && allowed_set.find(pair.first) == allowed_set.end()) {
//                    continue;  // 不在允许列表中
//                }
//                
//                best_match_pos = found;
//                best_match_token = pair.first;
//                best_match_id = pair.second;
//                found_special = true;
//            }
//        }
//        
//        if (found_special && best_match_pos == pos) {
//            // 找到特殊 token，添加它
//            result.push_back(best_match_id);
//            pos += best_match_token.size();
//        } else {
//            // 处理普通文本
//            size_t end_pos = found_special ? best_match_pos : remaining.size();
//            std::string segment = remaining.substr(pos, end_pos - pos);
//            
//            // 转换为字节并应用 BPE
//            auto bytes = stringToBytes(segment);
//            auto tokens = applyBPE(bytes);
//            result.insert(result.end(), tokens.begin(), tokens.end());
//            
//            pos = end_pos;
//        }
//    }
//    
//    return result;
//}

std::string Tiktoken::decode(const std::vector<uint32_t>& tokens) const {
    if (tokens.empty()) {
        return "";
    }
    
    std::vector<uint8_t> bytes;
    bytes.reserve(tokens.size() * 2);  // 预分配空间
    
    // 使用缓存的 token_to_bytes 映射
    if (token_to_bytes_cache_.empty()) {
        // 首先添加单字节映射（0-255）
        for (int i = 0; i < 256; ++i) {
            token_to_bytes_cache_[static_cast<uint32_t>(i)] = {static_cast<uint8_t>(i)};
        }
        
        // 添加所有合并规则的映射
        for (const auto& pair : mergeable_ranks_) {
            token_to_bytes_cache_[pair.second] = pair.first;
        }
    }
    
    const auto& token_to_bytes = token_to_bytes_cache_;
    
    // 递归展开函数（正确处理多字节 token）
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
                
                // 如果字节序列长度大于1，需要检查是否是合并的 token
                // 在 BPE 中，合并的 token 可能包含其他 token_id
                // 但这里我们直接使用字节序列，因为 mergeable_ranks 已经包含了完整的映射
                result.insert(result.end(), byte_seq.begin(), byte_seq.end());
            } else {
                // 如果找不到，回退处理
                if (token_id < 256) {
                    // 单字节 token
                    result.push_back(static_cast<uint8_t>(token_id));
                } else {
                    // 未知 token，尝试作为 UTF-8 字节处理
                    // 这是一个错误情况，但为了健壮性，我们跳过它
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

