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
 * GPT-2 Tokenizer Configuration Implementation
 * 
 * GPT-2 tokenizer 配置的实现
 */

#include "TiktokenGPT2.h"
#include "Tiktoken.h"  // 包含 TiktokenException
#include "json.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>

using json = nlohmann::json;

namespace tiktoken_gpt2 {

GPT2Config get_gpt2_config() {
    GPT2Config config;
    config.name = "gpt2";
    
    // GPT-2 的正则表达式模式
    // 匹配：'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    // 简化版本：匹配单词、数字、标点和空白
    config.pat_str = R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+)";
    
    // GPT-2 的特殊 token
    // <|endoftext|> 是 GPT-2 的唯一特殊 token，token ID 是 50256
    config.special_tokens["<|endoftext|>"] = 50256;
    
    // 初始化 mergeable_ranks
    // 注意：完整的 GPT-2 合并规则有 50000+ 条，这里提供一个基础框架
    // 实际使用时应该从文件加载完整的合并规则
    
    // 首先添加单字节 token (0-255)
    uint32_t token_id = 0;
    for (int i = 0; i < 256; ++i) {
        config.mergeable_ranks[{static_cast<uint8_t>(i)}] = token_id++;
    }
    
    // 然后添加 BPE 合并规则
    // 这里提供一个示例，实际应该从文件加载
    // GPT-2 的合并规则从 token ID 256 开始，到 50255 结束（共 50000 个合并规则）
    // 特殊 token <|endoftext|> 是 50256
    
    // 示例：添加一些常见的合并规则
    // 实际实现中，这些应该从 merges.txt 文件加载
    // 格式：每行是 "token1 token2"，按频率排序
    
    config.vocab_size = 50257;  // GPT-2 词汇表大小
    
    return config;
}

std::map<std::vector<uint8_t>, uint32_t> load_bpe_merges(const std::string& merges_file) {
    std::map<std::vector<uint8_t>, uint32_t> mergeable_ranks;
    
    // 首先添加单字节 token (0-255)
    for (int i = 0; i < 256; ++i) {
        mergeable_ranks[{static_cast<uint8_t>(i)}] = static_cast<uint32_t>(i);
    }
    
    std::ifstream file(merges_file);
    if (!file.is_open()) {
        throw TiktokenException("Failed to open merges file: " + merges_file);
    }
    
    std::string line;
    uint32_t token_id = 256;  // BPE 合并规则从 256 开始
    
    // 读取合并规则
    // GPT-2 merges.txt 格式：每行是 "token1 token2"
    // token 可能是：
    // 1. 单字节字符（如 "a", "b"）
    // 2. 多字节字符（如 "Ġhello" 中的 "Ġ" 表示空格）
    // 3. 特殊字符（如 "\n", "\t"）
    // 4. 转义字符（如 "\\n", "\\t"）
    
    // 辅助函数：处理转义字符
    auto unescapeToken = [](const std::string& token) -> std::string {
        std::string result;
        result.reserve(token.size());
        for (size_t i = 0; i < token.size(); ++i) {
            if (token[i] == '\\' && i + 1 < token.size()) {
                switch (token[i + 1]) {
                    case 'n': result += '\n'; i++; break;
                    case 't': result += '\t'; i++; break;
                    case 'r': result += '\r'; i++; break;
                    case '\\': result += '\\'; i++; break;
                    case '\'': result += '\''; i++; break;
                    case '"': result += '"'; i++; break;
                    default: result += token[i]; break;
                }
            } else {
                result += token[i];
            }
        }
        return result;
    };
    
    while (std::getline(file, line)) {
        // 跳过空行和注释
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // 移除行尾的换行符和空白
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty()) {
            continue;
        }
        
        // 查找第一个空格，分割两个 token
        size_t space_pos = line.find(' ');
        if (space_pos == std::string::npos) {
            continue;  // 无效行
        }
        
        std::string token1_str = line.substr(0, space_pos);
        std::string token2_str = line.substr(space_pos + 1);
        
        // 处理转义字符
        token1_str = unescapeToken(token1_str);
        token2_str = unescapeToken(token2_str);
        
        // 将 token 字符串转换为字节向量（UTF-8 编码）
        // GPT-2 使用字节级 BPE，所以直接按字节处理
        std::vector<uint8_t> pair_bytes;
        
        // token1 的字节（UTF-8 编码）
        for (size_t i = 0; i < token1_str.length(); ++i) {
            // 对于多字节 UTF-8 字符，直接取字节值
            pair_bytes.push_back(static_cast<uint8_t>(token1_str[i]));
        }
        
        // token2 的字节（UTF-8 编码）
        for (size_t i = 0; i < token2_str.length(); ++i) {
            pair_bytes.push_back(static_cast<uint8_t>(token2_str[i]));
        }
        
        // 添加到合并规则
        if (!pair_bytes.empty()) {
            mergeable_ranks[pair_bytes] = token_id++;
            
            // 限制最大 token ID（GPT-2 的词汇表大小是 50257）
            if (token_id >= 50256) {
                break;  // 达到最大 token ID
            }
        }
    }
    
    file.close();
    return mergeable_ranks;
}

std::map<std::string, uint32_t> load_encoder_json(const std::string& encoder_file) {
    std::map<std::string, uint32_t> encoder;
    
    std::ifstream file(encoder_file);
    if (!file.is_open()) {
        return encoder;  // 文件不存在，返回空映射
    }
    
    try {
        // 使用 nlohmann/json 解析 JSON 文件
        json j;
        file >> j;
        file.close();
        
        // 遍历 JSON 对象，提取 token -> id 映射
        if (j.is_object()) {
            for (auto& [key, value] : j.items()) {
                if (value.is_number_unsigned() || value.is_number_integer()) {
                    encoder[key] = static_cast<uint32_t>(value.get<uint32_t>());
                }
            }
        }
    } catch (const json::exception& e) {
        // JSON 解析失败，返回空映射
        // 可以记录错误日志
        return encoder;
    }
    
    return encoder;
}

GPT2Config load_gpt2_config_from_files(const std::string& vocab_file,
                                        const std::string& encoder_file) {
    GPT2Config config = get_gpt2_config();
    
    // 加载 BPE merges
    if (!vocab_file.empty()) {
        auto loaded_merges = load_bpe_merges(vocab_file);
        if (!loaded_merges.empty()) {
            config.mergeable_ranks = loaded_merges;
        }
    }
    
    // 加载编码器 JSON（如果提供）
    // encoder.json 包含 token -> id 的映射，可以用于验证
    if (!encoder_file.empty()) {
        auto encoder = load_encoder_json(encoder_file);
        // encoder.json 主要用于验证，实际的编码使用 mergeable_ranks
        // 但我们可以用它来补充特殊 token 信息
        // 注意：GPT-2 的 encoder.json 中，<|endoftext|> 的 ID 是 50256
    }
    
    // 更新词汇表大小
    if (!config.mergeable_ranks.empty()) {
        size_t max_id = 0;
        for (const auto& pair : config.mergeable_ranks) {
            max_id = std::max(max_id, static_cast<size_t>(pair.second));
        }
        for (const auto& pair : config.special_tokens) {
            max_id = std::max(max_id, static_cast<size_t>(pair.second));
        }
        config.vocab_size = max_id + 1;
    } else {
        // 如果没有加载 merges，使用默认大小
        config.vocab_size = 50257;
    }
    
    return config;
}

} // namespace tiktoken_gpt2

