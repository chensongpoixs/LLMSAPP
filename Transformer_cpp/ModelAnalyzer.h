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
 * Model Analyzer Class
 * 
 * 模型分析器
 * 用于读取、分析和显示模型的所有参数，方便理解模型结构
 */

#ifndef MODEL_ANALYZER_H
#define MODEL_ANALYZER_H

#include "GPTModel.h"
#include "ModelConfig.h"
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <map>

/**
 * 参数统计信息
 */
struct ParameterStats {
    std::string name;
    std::vector<int64_t> shape;
    size_t numel;
    double mean;
    double std;
    double min;
    double max;
    size_t memory_bytes;
};

/**
 * 模型分析器类
 */
class ModelAnalyzer {
public:
    /**
     * 构造函数
     * 
     * @param model: GPT模型
     */
    ModelAnalyzer(std::shared_ptr<GPTModel> model);
    
    /**
     * 析构函数
     */
    ~ModelAnalyzer() = default;
    
    /**
     * 分析模型
     * 收集所有参数信息
     */
    void analyze();
    
    /**
     * 打印模型结构
     */
    void printModelStructure() const;
    
    /**
     * 打印所有参数信息
     */
    void printAllParameters() const;
    
    /**
     * 打印参数统计信息
     */
    void printParameterStatistics() const;
    
    /**
     * 打印参数摘要
     */
    void printParameterSummary() const;
    
    /**
     * 打印参数说明（每个参数的作用和含义）
     */
    void printParameterDescriptions() const;
    
    /**
     * 打印模型架构可视化图表
     */
    void printModelArchitectureDiagram() const;
    
    /**
     * 打印参数关系图
     */
    void printParameterRelationshipDiagram() const;
    
    /**
     * 打印参数详细注释和可视化图表
     */
    void printParameterDetailedAnnotations() const;
    
    /**
     * 打印参数维度关系图
     */
    void printParameterDimensionDiagram() const;
    
    /**
     * 打印参数计算流程图
     */
    void printParameterComputationFlow() const;
    
    /**
     * 保存模型信息到文件
     * 
     * @param output_file: 输出文件路径
     */
    void saveModelInfo(const std::string& output_file) const;
    
    /**
     * 获取参数统计信息
     * 
     * @return 参数统计信息列表
     */
    const std::vector<ParameterStats>& getParameterStats() const { return param_stats_; }
    
    /**
     * 获取总参数量
     * 
     * @return 总参数量
     */
    size_t getTotalParameters() const { return total_params_; }
    
    /**
     * 获取总内存占用（字节）
     * 
     * @return 总内存占用
     */
    size_t getTotalMemoryBytes() const { return total_memory_bytes_; }

private:
    std::shared_ptr<GPTModel> model_;
    std::vector<ParameterStats> param_stats_;
    size_t total_params_;
    size_t total_memory_bytes_;
    
    /**
     * 计算参数统计信息
     * 
     * @param param: 参数张量
     * @param name: 参数名称
     * @return 统计信息
     */
    ParameterStats computeStats(const torch::Tensor& param, const std::string& name) const;
    
    /**
     * 格式化形状字符串
     * 
     * @param shape: 形状向量
     * @return 形状字符串
     */
    std::string formatShape(const std::vector<int64_t>& shape) const;
    
    /**
     * 格式化内存大小
     * 
     * @param bytes: 字节数
     * @return 格式化后的字符串
     */
    std::string formatMemorySize(size_t bytes) const;
    
    /**
     * 获取参数说明
     * 
     * @param param_name: 参数名称
     * @return 参数说明字符串
     */
    std::string getParameterDescription(const std::string& param_name) const;
    
    /**
     * 初始化参数说明映射
     */
    void initializeParameterDescriptions();
    
    std::map<std::string, std::string> param_descriptions_;
};

#endif // MODEL_ANALYZER_H

