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
 * Trainer Class
 * 
 * Transformer模型训练器
 * 封装完整的训练流程，包括数据加载、训练循环、模型保存等
 */

#ifndef TRAINER_H
#define TRAINER_H

#include "GPTModel.h"
#include "ModelConfig.h"
#include "TextDataset.h"
#include "TrainingUtils.h"
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

/**
 * Epoch信息结构体
 */
struct EpochInfo {
    int epoch;
    float avg_loss;
    float min_loss;
    float max_loss;
    double duration_sec;
    int64_t samples;
    double samples_per_sec;
};

/**
 * 训练器类
 */
class Trainer {
public:
    /**
     * 构造函数
     * 
     * @param model: GPT模型
     * @param dataset: 训练数据集
     * @param device: 训练设备（CPU或GPU）
     * @param cfg: 模型配置
     */
    Trainer(std::shared_ptr<GPTModel> model,
            TextDataset& dataset,
            torch::Device device,
            const ModelConfig& cfg);
    
    /**
     * 析构函数
     */
    ~Trainer() = default;
    
    /**
     * 开始训练
     * 
     * @param num_epochs: 训练轮数
     * @param batch_size: 批次大小
     * @param learning_rate: 学习率
     * @param weight_decay: 权重衰减
     * @param clip_grad_norm: 梯度裁剪的最大范数
     * @param data_file: 数据文件路径
     * @return 是否训练成功
     */
    bool train(int num_epochs,
               int batch_size,
               double learning_rate = 3e-4,
               double weight_decay = 0.1,
               double clip_grad_norm = 1.0,
               const std::string& data_file = "the-verdict.txt");
    
    /**
     * 获取训练历史
     * 
     * @return epoch历史信息
     */
    const std::vector<EpochInfo>& getEpochHistory() const { return epoch_history_; }
    
    /**
     * 获取最佳损失
     * 
     * @return 最佳损失值
     */
    float getBestLoss() const { return best_loss_; }
    
    /**
     * 获取实验目录路径
     * 
     * @return 实验目录路径
     */
    std::string getExpDir() const { return exp_dir_; }

private:
    std::shared_ptr<GPTModel> model_;
    TextDataset* dataset_;
    torch::Device device_;
    ModelConfig cfg_;
    
    // 训练状态
    std::vector<EpochInfo> epoch_history_;
    float best_loss_;
    float previous_loss_;
    
    // 实验目录
    std::string exp_dir_;
    std::string last_checkpoint_path_;
    std::string best_checkpoint_path_;
    
    // 优化器和损失函数
    std::unique_ptr<torch::optim::Adam> optimizer_;
    std::unique_ptr<torch::nn::CrossEntropyLoss> criterion_;
    
    /**
     * 初始化训练环境
     */
    void initializeTraining(double learning_rate, double weight_decay, 
                           int batch_size, int num_epochs);
    
    /**
     * 训练一个epoch
     * 
     * @param epoch: 当前epoch编号
     * @param num_epochs: 总epoch数
     * @param batch_size: 批次大小
     * @param num_batches_per_epoch: 每个epoch的批次数量
     * @param clip_grad_norm: 梯度裁剪的最大范数
     */
    void trainEpoch(int epoch, int num_epochs, int batch_size, 
                   int num_batches_per_epoch, double clip_grad_norm);
    
    /**
     * 打印训练历史表格
     */
    void printTrainingHistory() const;
    
    /**
     * 打印损失趋势图
     */
    void printLossTrendChart() const;
    
    /**
     * 执行验证
     * 
     * @param batch_size: 批次大小
     * @return 验证损失
     */
    float validate(int batch_size);
    
    /**
     * 执行推理测试
     * 
     * @param prompt: 提示词
     * @param max_new_tokens: 最大生成token数量
     */
    void inferenceTest(const std::string& prompt = "Hello World!!!", 
                      int max_new_tokens = 100);
};

#endif // TRAINER_H

