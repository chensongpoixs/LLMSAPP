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
 * Generator Class
 * 
 * Transformer模型文本生成器
 * 封装推理和文本生成功能
 */

#ifndef GENERATOR_H
#define GENERATOR_H

#include "GPTModel.h"
#include "ModelConfig.h"
#include "Tiktoken.h"
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

/**
 * 生成结果结构体
 */
struct GenerationResult {
    std::string prompt;
    std::string generated_text;
    std::vector<int64_t> generated_tokens;
    double total_time_sec;
    int64_t total_time_ms;
    double avg_tokens_per_second;
};

/**
 * 文本生成器类
 */
class Generator {
public:
    /**
     * 构造函数
     * 
     * @param model: GPT模型
     * @param device: 推理设备（CPU或GPU）
     * @param cfg: 模型配置
     * @param encoder: tiktoken 编码器（如果为 nullptr，则使用简单编码器）
     */
    Generator(std::shared_ptr<GPTModel> model,
              torch::Device device,
              const ModelConfig& cfg,
              std::shared_ptr<Tiktoken> encoder = nullptr);
    
    /**
     * 析构函数
     */
    ~Generator() = default;
    
    /**
     * 生成文本
     * 
     * @param prompt: 提示词
     * @param max_new_tokens: 最大生成token数量
     * @param temperature: 温度参数（控制随机性，默认0.8）
     * @param top_k: Top-K采样参数（只从概率最高的k个token中采样，0表示不使用top_k，默认0）
     * @return 生成结果
     */
    GenerationResult generate(const std::string& prompt,
                             int max_new_tokens = 100,
                             double temperature = 0.8,
                             int top_k = 25);
    
    /**
     * 加载模型
     * 
     * @param model_path: 模型文件路径
     * @return 是否加载成功
     */
    bool loadModel(const std::string& model_path);
    
    /**
     * 获取模型
     * 
     * @return 模型指针
     */
    std::shared_ptr<GPTModel> getModel() const { return model_; }
    
    /**
     * 获取编码器
     * 
     * @return 编码器指针
     */
    std::shared_ptr<Tiktoken> getEncoder() const { return encoder_; }

private:
    std::shared_ptr<GPTModel> model_;
    torch::Device device_;
    ModelConfig cfg_;
    std::shared_ptr<Tiktoken> encoder_;  // tiktoken 编码器
    
    /**
     * 将文本转换为token IDs（使用 tiktoken）
     * 
     * @param text: 输入文本
     * @return token ID序列
     */
    std::vector<int64_t> textToTokens(const std::string& text);
    
    /**
     * 将token IDs转换为文本（使用 tiktoken）
     * 
     * @param tokens: token ID序列
     * @return 文本字符串
     */
    std::string tokensToText(const std::vector<int64_t>& tokens);
};

#endif // GENERATOR_H

