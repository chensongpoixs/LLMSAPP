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
 * Tiktoken C++ Implementation
 * 
 * 参考 OpenAI tiktoken (https://github.com/openai/tiktoken) 实现的 C++ 版本
 * 
 * 功能：
 * - BPE (Byte Pair Encoding) tokenizer
 * - 文本编码为 token IDs
 * - token IDs 解码为文本
 * - 支持自定义编码规则
 * - 高性能实现
 * 
 * BPE 原理：
 * 1. 从字符级别开始
 * 2. 迭代合并最常见的字符对
 * 3. 构建合并规则表（mergeable_ranks）
 * 4. 使用规则进行编码和解码
 */

#ifndef TIKTOKEN_H
#define TIKTOKEN_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <cstdint>

/**
 * Tiktoken 编码器类
 * 
 * 实现 BPE tokenizer，支持文本编码和解码
 */
class Tiktoken {
public:
    /**
     * 构造函数
     * 
     * @param name: 编码器名称（如 "cl100k_base", "o200k_base"）
     * @param pat_str: 正则表达式模式字符串（用于分割文本）
     * @param mergeable_ranks: BPE 合并规则表，键为字节对，值为 token ID
     * @param special_tokens: 特殊 token 映射，键为特殊 token 字符串，值为 token ID
     */
    Tiktoken(const std::string& name,
             const std::string& pat_str,
             const std::map<std::vector<uint8_t>, uint32_t>& mergeable_ranks,
             const std::map<std::string, uint32_t>& special_tokens = {});
    
    /**
     * 析构函数
     */
    ~Tiktoken() = default;
    
    /**
     * 编码文本为 token IDs
     * 
     * @param text: 输入文本
     * @param allowed_special: 允许的特殊 token 集合（空表示不允许）
     * @param disallowed_special: 不允许的特殊 token 集合（空表示允许所有）
     * @return token ID 向量
     */
    std::vector<uint32_t> encode(const std::string& text,
                                 const std::vector<std::string>& allowed_special = {},
                                 const std::vector<std::string>& disallowed_special = {}) const;
    
    /**
     * 解码 token IDs 为文本
     * 
     * @param tokens: token ID 向量
     * @return 解码后的文本
     */
    std::string decode(const std::vector<uint32_t>& tokens) const;
    
    /**
     * 获取编码器名称
     */
    std::string getName() const { return name_; }
    
    /**
     * 获取词汇表大小（最大 token ID + 1）
     */
    size_t getVocabSize() const { return vocab_size_; }
    
    /**
     * 检查 token ID 是否为特殊 token
     */
    bool isSpecialToken(uint32_t token_id) const;
    
    /**
     * 获取特殊 token 的 ID
     */
    uint32_t getSpecialTokenId(const std::string& special_token) const;

private:
    std::string name_;
    std::string pat_str_;
    std::map<std::vector<uint8_t>, uint32_t> mergeable_ranks_;
    std::map<std::string, uint32_t> special_tokens_;
    std::map<uint32_t, std::string> special_tokens_reverse_;  // 反向映射
    size_t vocab_size_;
    
    /**
     * 将字符串转换为字节向量
     */
    std::vector<uint8_t> stringToBytes(const std::string& text) const;
    
    /**
     * 将字节向量转换为字符串
     */
    std::string bytesToString(const std::vector<uint8_t>& bytes) const;
    
    /**
     * 使用正则表达式分割文本（简化版本，使用空格和标点分割）
     */
    std::vector<std::string> splitText(const std::string& text) const;
    
    /**
     * 对字节序列应用 BPE 合并规则
     */
    std::vector<uint32_t> applyBPE(const std::vector<uint8_t>& bytes) const;
    
    /**
     * 查找可以合并的字节对
     */
    std::pair<size_t, uint32_t> findBestMerge(const std::vector<uint32_t>& tokens,
                                             const std::map<std::pair<uint32_t, uint32_t>, uint32_t>& rank_map) const;
};

/**
 * Tiktoken 工厂函数
 * 
 * 创建预定义的编码器
 */
namespace tiktoken {
    /**
     * 获取编码器
     * 
     * @param encoding_name: 编码器名称
     * @return 编码器指针
     */
    std::shared_ptr<Tiktoken> get_encoding(const std::string& encoding_name);
    
    /**
     * 为特定模型获取编码器
     * 
     * @param model_name: 模型名称（如 "gpt-4", "gpt-3.5-turbo"）
     * @return 编码器指针
     */
    std::shared_ptr<Tiktoken> encoding_for_model(const std::string& model_name);
    
    /**
     * 创建简单的字符级编码器（用于测试和演示）
     * 
     * @return 编码器指针
     */
    std::shared_ptr<Tiktoken> create_simple_encoding();
    
    /**
     * 创建 GPT-2 编码器
     * 
     * @return 编码器指针
     */
    std::shared_ptr<Tiktoken> create_gpt2_encoding();
    
    /**
     * 从文件加载编码器配置
     * 
     * @param config_file: 配置文件路径（JSON 格式）
     * @param encoding_name: 编码器名称
     * @return 编码器指针，失败返回 nullptr
     */
    std::shared_ptr<Tiktoken> load_encoding_from_file(const std::string& config_file,
                                                       const std::string& encoding_name);
}

#endif // TIKTOKEN_H

