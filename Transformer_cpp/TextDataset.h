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
 * 文本数据集类 (TextDataset)
 * 
 * 用于读取文本文件并转换为可用于训练的token序列。
 * 
 * 功能：
 * - 读取文本文件
 * - 字符级tokenization（将字符映射为token IDs）
 * - 生成训练批次（输入序列和目标序列）
 * - 支持滑动窗口生成多个训练样本
 */

#ifndef TEXT_DATASET_H
#define TEXT_DATASET_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <torch/torch.h>

/**
 * 文本数据集类
 */
class TextDataset {
public:
    /**
     * 构造函数
     * @param filepath: 文本文件路径
     * @param seq_len: 序列长度（每个样本的长度）
     * @param vocab_size: 词汇表大小（最大token ID）
     */
    TextDataset(const std::string& filepath, int seq_len, int vocab_size = 256);
    
    /**
     * 加载文本文件
     * @return 是否成功加载
     */
    bool load();
    
    /**
     * 获取数据集大小（样本数量）
     */
    size_t size() const;
    
    /**
     * 获取一个批次的数据
     * @param batch_size: 批次大小
     * @param device: 设备（CPU或GPU）
     * @return pair<input_ids, target_ids>
     */
    std::pair<torch::Tensor, torch::Tensor> getBatch(int batch_size, torch::Device device);
    
    /**
     * 获取词汇表大小
     */
    int getVocabSize() const { return vocab_size_; }
    
    /**
     * 将字符转换为token ID
     */
    int charToToken(char c) const;
    
    /**
     * 将token ID转换为字符
     */
    char tokenToChar(int token_id) const;

private:
    std::string filepath_;           // 文件路径
    int seq_len_;                    // 序列长度
    int vocab_size_;                 // 词汇表大小
    std::string text_;                // 加载的文本内容
    std::vector<int> tokens_;        // token序列
    size_t current_pos_;              // 当前读取位置
    
    /**
     * 将文本转换为token序列
     */
    void tokenize();
};

#endif // TEXT_DATASET_H

