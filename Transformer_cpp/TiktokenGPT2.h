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
 * GPT-2 Tokenizer Configuration
 * 
 * GPT-2 tokenizer 的配置和辅助函数
 * 用于加载和初始化 GPT-2 的 BPE 合并规则
 */

#ifndef TIKTOKEN_GPT2_H
#define TIKTOKEN_GPT2_H

#include "Tiktoken.h"
#include <string>
#include <map>
#include <vector>

namespace tiktoken_gpt2 {
    /**
     * GPT-2 编码器配置
     */
    struct GPT2Config {
        std::string name;
        std::string pat_str;  // 正则表达式模式
        std::map<std::vector<uint8_t>, uint32_t> mergeable_ranks;
        std::map<std::string, uint32_t> special_tokens;
        size_t vocab_size;
    };
    
    /**
     * 获取 GPT-2 的默认配置
     * 
     * @return GPT-2 配置
     */
    GPT2Config get_gpt2_config();
    
    /**
     * 从文件加载 GPT-2 配置
     * 
     * @param vocab_file: 词汇表文件路径（BPE merges 文件）
     * @param encoder_file: 编码器文件路径（JSON 格式）
     * @return GPT-2 配置，失败返回空配置
     */
    GPT2Config load_gpt2_config_from_files(const std::string& vocab_file,
                                           const std::string& encoder_file = "");
    
    /**
     * 加载 BPE merges 文件
     * 
     * @param merges_file: merges 文件路径
     * @return mergeable_ranks 映射
     */
    std::map<std::vector<uint8_t>, uint32_t> load_bpe_merges(const std::string& merges_file);
    
    /**
     * 加载编码器 JSON 文件
     * 
     * @param encoder_file: 编码器文件路径
     * @return 编码器映射（token -> id）
     */
    std::map<std::string, uint32_t> load_encoder_json(const std::string& encoder_file);
}

#endif // TIKTOKEN_GPT2_H

