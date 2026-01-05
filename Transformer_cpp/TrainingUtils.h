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
 * Training Utilities
 * 
 * 训练相关的工具函数集合
 * 包含训练、验证、设备解析、模型保存等功能
 */

#ifndef TRAINING_UTILS_H
#define TRAINING_UTILS_H

#include "GPTModel.h"
#include "ModelConfig.h"
#include <torch/torch.h>
#include <string>
#include <memory>

/**
 * 训练一个批次的数据
 * 
 * @param model: GPT 模型
 * @param input_ids: 输入 token IDs (batch_size, seq_len)
 * @param target_ids: 目标 token IDs (batch_size, seq_len)，用于计算损失
 * @param criterion: 损失函数（交叉熵损失）
 * @param optimizer: 优化器（Adam）
 * @param clip_grad_norm: 梯度裁剪的最大范数
 * @return 损失值
 */
float train_batch(
    std::shared_ptr<GPTModel> model,
    torch::Tensor input_ids,
    torch::Tensor target_ids,
    torch::nn::CrossEntropyLoss& criterion,
    torch::optim::Adam& optimizer,
    double clip_grad_norm = 1.0);

/**
 * 验证一个批次的数据（不更新参数）
 * 
 * @param model: GPT 模型
 * @param input_ids: 输入 token IDs
 * @param target_ids: 目标 token IDs
 * @param criterion: 损失函数
 * @return 损失值
 */
float validate_batch(
    std::shared_ptr<GPTModel> model,
    torch::Tensor input_ids,
    torch::Tensor target_ids,
    torch::nn::CrossEntropyLoss& criterion);

/**
 * 解析命令行参数，获取设备类型
 * 
 * @param argc: 参数数量
 * @param argv: 参数数组
 * @return 设备类型（torch::Device）
 */
torch::Device parse_device(int argc, char* argv[]);

/**
 * 获取runs/train目录下最大编号的实验文件夹
 * YOLOv风格：exp, exp2, exp3, ...
 * 
 * @param train_dir: runs/train目录路径
 * @return 最大编号，如果没有找到则返回0（表示使用exp）
 */
int get_max_exp_number(const std::string& train_dir);

/**
 * 获取实验目录名称（YOLOv风格）
 * 
 * @param exp_num: 实验编号
 * @return 目录名称（exp, exp2, exp3, ...）
 */
std::string get_exp_dir_name(int exp_num);

/**
 * 保存模型到指定路径
 * 
 * @param model: 要保存的模型
 * @param save_path: 保存路径
 * @return 是否保存成功
 */
bool save_model_checkpoint(std::shared_ptr<GPTModel> model, const std::string& save_path);

/**
 * 保存训练配置到YAML文件（YOLOv风格）
 * 
 * @param exp_dir: 实验目录
 * @param cfg: 模型配置
 * @param learning_rate: 学习率
 * @param weight_decay: 权重衰减
 * @param batch_size: 批次大小
 * @param num_epochs: 训练轮数
 */
void save_training_config(const std::string& exp_dir, const ModelConfig& cfg, 
                         double learning_rate, double weight_decay, 
                         int batch_size, int num_epochs);

/**
 * 查找runs/train目录下最新的模型文件
 * 优先查找best.pth，如果没有则查找last.pth
 * 
 * @param train_dir: runs/train目录路径（默认为"runs/train"）
 * @return 最新模型文件的完整路径，如果未找到则返回空字符串
 */
std::string find_latest_model(const std::string& train_dir = "runs/train");

#endif // TRAINING_UTILS_H

