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
 * Trainer Implementation
 * 
 * 训练器实现
 */

#include "Trainer.h"
#include "Logger.h"
#include "Generator.h"
#include "TrainingUtils.h"
#include <torch/torch.h>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>  // CUDA Runtime API
Trainer::Trainer(std::shared_ptr<GPTModel> model,
                 TextDataset& dataset,
                 torch::Device device,
                 const ModelConfig& cfg)
    : model_(model), dataset_(&dataset), device_(device), cfg_(cfg),
      best_loss_(std::numeric_limits<float>::max()),
      previous_loss_(std::numeric_limits<float>::max()) {
}

void Trainer::initializeTraining(double learning_rate, double weight_decay, 
                                int batch_size, int num_epochs) {
    // 创建损失函数
    criterion_ = std::make_unique<torch::nn::CrossEntropyLoss>(
        torch::nn::CrossEntropyLossOptions().ignore_index(-1)
    );
    //criterion_->to(device_);
    
    // 创建优化器
    optimizer_ = std::make_unique<torch::optim::Adam>(
        model_->parameters(),
        torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay)
    );
    
    // 设置模型保存路径（YOLOv风格：runs/train/exp, runs/train/exp2, ...）
    std::string base_dir = "runs/train";
    int current_exp_num = get_max_exp_number(base_dir);
    current_exp_num += 1;  // 找到最大编号后+1
    
    std::string exp_dir_name = get_exp_dir_name(current_exp_num);
    exp_dir_ = base_dir + "/" + exp_dir_name;
    std::string weights_dir = exp_dir_ + "/weights";
    last_checkpoint_path_ = weights_dir + "/last.pth";
    best_checkpoint_path_ = weights_dir + "/best.pth";
    
    // 创建实验目录结构
    namespace fs = std::filesystem;
    try {
        fs::create_directories(weights_dir);
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Logger::info("Experiment Directory (YOLOv Style)");
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Logger::info("  ├─ Experiment: {}", exp_dir_name);
        Logger::info("  ├─ Experiment Directory: {}", exp_dir_);
        Logger::info("  ├─ Weights Directory: {}", weights_dir);
        Logger::info("  ├─ Last Checkpoint: {}", last_checkpoint_path_);
        Logger::info("  └─ Best Checkpoint: {}", best_checkpoint_path_);
        
        // 保存训练配置
        save_training_config(exp_dir_, cfg_, learning_rate, weight_decay, batch_size, num_epochs);
    } catch (const std::exception& e) {
        Logger::error("Failed to create experiment directory: {}", e.what());
    }
}

bool Trainer::train(int num_epochs, int batch_size, double learning_rate,
                   double weight_decay, double clip_grad_norm, const std::string& data_file) {
    // 初始化训练环境
    initializeTraining(learning_rate, weight_decay, batch_size, num_epochs);
    
    // 计算每个epoch的批次数量
    int num_batches_per_epoch = static_cast<int>(dataset_->size()) / batch_size;
    if (num_batches_per_epoch == 0) {
        num_batches_per_epoch = 1;
    }
    
    // YOLOv5 风格的训练开始信息
    Logger::info("");
    Logger::info("Starting training for {} epochs...", num_epochs);
    Logger::info("");
    Logger::info("Epoch   GPU_mem   loss       lr        Instances   Size");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // 记录总体训练时间
    auto training_start = std::chrono::high_resolution_clock::now();
    
    // 训练循环
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        trainEpoch(epoch, num_epochs, batch_size, num_batches_per_epoch, clip_grad_norm);
    }
    
    // 计算总训练时间
    auto training_end = std::chrono::high_resolution_clock::now();
    auto total_training_time = std::chrono::duration_cast<std::chrono::seconds>(
        training_end - training_start
    );
    
    // 打印训练总结
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Training Completed!");
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("  ├─ Total Epochs: {}", num_epochs);
    std::ostringstream best_loss_str;
    best_loss_str << std::fixed << std::setprecision(6) << best_loss_;
    Logger::info("  ├─ Best Loss: {}", best_loss_str.str());
    Logger::info("  ├─ Total Training Time: {} seconds", total_training_time.count());
    std::ostringstream avg_time_str;
    avg_time_str << std::fixed << std::setprecision(2) << (total_training_time.count() / static_cast<double>(num_epochs)) << " seconds";
    Logger::info("  └─ Average per Epoch: {}", avg_time_str.str());
    
    // 打印训练历史
    printTrainingHistory();
    
    // 执行验证
    validate(batch_size);
    
    // 执行推理测试
    inferenceTest();
    
    return true;
}

void Trainer::trainEpoch(int epoch, int num_epochs, int batch_size, 
                        int num_batches_per_epoch, double clip_grad_norm) {
    // 记录每个 epoch 的总损失
    float epoch_loss = 0.0f;
    float min_batch_loss = std::numeric_limits<float>::max();
    float max_batch_loss = std::numeric_limits<float>::min();
    
    // 用于计算移动平均损失
    std::vector<float> recent_losses;
    const int loss_window = 10;
    
    auto epoch_start = std::chrono::high_resolution_clock::now();
    
    // 获取学习率
    double current_lr = 0.0;
    for (auto& group : optimizer_->param_groups()) {
        current_lr = group.options().get_lr();
        break;
    }
    
    // 获取 GPU 内存使用（如果使用 GPU）
    // 使用 CUDA Runtime API 直接获取内存信息
    std::string gpu_mem_str = "N/A";
    if (device_.is_cuda()) {
        try {
            // 获取设备索引
            int device_index = device_.index();
            
            // 检查设备索引是否有效
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err != cudaSuccess || device_index < 0 || device_index >= device_count) {
                // 设备索引无效
                std::ostringstream gpu_ss;
                gpu_ss << "GPU:" << device_index;
                gpu_mem_str = gpu_ss.str();
            } else {
                // 保存当前 CUDA 设备
                int original_device = 0;
                err = cudaGetDevice(&original_device);
                bool need_restore = (err == cudaSuccess);
                
                // 切换到目标设备
                err = cudaSetDevice(device_index);
                if (err != cudaSuccess) {
                    // 如果切换失败，显示设备标识
                    std::ostringstream gpu_ss;
                    gpu_ss << "GPU:" << device_index;
                    gpu_mem_str = gpu_ss.str();
                } else {
                    // 获取内存信息
                    size_t free_mem = 0;
                    size_t total_mem = 0;
                    
                    err = cudaMemGetInfo(&free_mem, &total_mem);
                    if (err == cudaSuccess) {
                        // 计算已使用内存 = 总内存 - 可用内存
                        size_t used_mem = total_mem - free_mem;
                        
                        // 转换为 GB
                        double mem_gb = used_mem / (1024.0 * 1024.0 * 1024.0);
                        
                        std::ostringstream gpu_mem_ss;
                        gpu_mem_ss << std::fixed << std::setprecision(1) << mem_gb << "G";
                        gpu_mem_str = gpu_mem_ss.str();
                    } else {
                        // 如果获取内存信息失败，显示设备标识
                        std::ostringstream gpu_ss;
                        gpu_ss << "GPU:" << device_index;
                        gpu_mem_str = gpu_ss.str();
                    }
                    
                    // 恢复原始设备（如果之前成功获取了）
                    if (need_restore) {
                        cudaError_t restore_err = cudaSetDevice(original_device);
                        // 忽略恢复失败的错误，因为我们已经尝试了
                        (void)restore_err;  // 避免未使用变量警告
                    }
                }
            }
        } catch (const std::exception& e) {
            // 如果发生异常，显示设备标识
            std::ostringstream gpu_ss;
            gpu_ss << "GPU:" << device_.index();
            gpu_mem_str = gpu_ss.str();
        } catch (...) {
            // 捕获所有其他异常
            std::ostringstream gpu_ss;
            gpu_ss << "GPU:" << device_.index();
            gpu_mem_str = gpu_ss.str();
        }
    }
    
    // YOLOv5 风格的 epoch 头部（在第一个 batch 前打印）
    std::ostringstream epoch_header;
    epoch_header << std::setw(5) << (epoch + 1) << "/" << num_epochs
                 << std::setw(9) << gpu_mem_str
                 << std::setw(11) << "..."  // loss 会在进度条中显示
                 << std::setw(11) << std::scientific << std::setprecision(2) << current_lr
                 << std::setw(13) << (batch_size * num_batches_per_epoch)
                 << std::setw(9) << batch_size << ":";
    
    for (int batch_idx = 0; batch_idx < num_batches_per_epoch; ++batch_idx) {
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        // 从数据集中获取一个批次的数据
        auto [input_ids, target_ids] = dataset_->getBatch(batch_size, device_);
        
        // 训练一个批次
        float loss = train_batch(
            model_, 
            input_ids, 
            target_ids, 
            *criterion_, 
            *optimizer_, 
            clip_grad_norm
        );
        
        epoch_loss += loss;
        min_batch_loss = std::min(min_batch_loss, loss);
        max_batch_loss = std::max(max_batch_loss, loss);
        
        // 更新移动平均损失
        recent_losses.push_back(loss);
        if (recent_losses.size() > loss_window) {
            recent_losses.erase(recent_losses.begin());
        }
        float avg_loss = 0.0f;
        for (float l : recent_losses) {
            avg_loss += l;
        }
        avg_loss /= recent_losses.size();
        
        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            batch_end - batch_start
        );
        
        // 计算训练速度（样本/秒）
        int64_t samples_per_second = (batch_size * 1000) / std::max(1L, static_cast<long>(batch_duration.count()));
        
        // 计算进度百分比
        float progress = ((batch_idx + 1) * 100.0f) / num_batches_per_epoch;
        
        // 计算梯度范数
        double grad_norm = 0.0;
        for (const auto& param : model_->parameters()) {
            if (param.grad().defined()) {
                grad_norm += param.grad().norm().item<double>() * param.grad().norm().item<double>();
            }
        }
        grad_norm = std::sqrt(grad_norm);
        
        // YOLOv5 风格的进度条和指标显示
        // 绘制进度条
        const int bar_width = 50;
        int filled = static_cast<int>((progress / 100.0f) * bar_width);
        std::string progress_bar = "";
        for (int i = 0; i < bar_width; ++i) {
            if (i < filled) {
                progress_bar += "█";
            } else {
                progress_bar += " ";
            }
        }
        
        // 计算已用时间和预计剩余时间
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            batch_end - epoch_start
        ).count();
        int remaining = (batch_idx + 1 > 0) ? 
            static_cast<int>((elapsed * (num_batches_per_epoch - batch_idx - 1)) / (batch_idx + 1)) : 0;
        
        // 计算迭代速度（it/s）
        double it_per_sec = (elapsed > 0) ? (batch_idx + 1) / static_cast<double>(elapsed) : 0.0;
        
        // YOLOv5 风格的输出格式
        std::ostringstream batch_info;
        // 所有 batch 都包含 epoch 头部以保持对齐
        batch_info << epoch_header.str() << " " << std::fixed << std::setprecision(0) << progress << "%|"
                   << progress_bar << "| "
                   << std::setw(4) << (batch_idx + 1) << "/" << num_batches_per_epoch;
        
        batch_info << " ["
                   << std::setfill('0') << std::setw(2) << (elapsed / 60) << ":"
                   << std::setfill('0') << std::setw(2) << (elapsed % 60) << "<"
                   << std::setfill('0') << std::setw(2) << (remaining / 60) << ":"
                   << std::setfill('0') << std::setw(2) << (remaining % 60)
                   << ", " << std::fixed << std::setprecision(2) << it_per_sec << "it/s"
                   << ", loss=" << std::setprecision(4) << loss;
        
        // 使用 \r 覆盖同一行（YOLOv5 风格）
        if (batch_idx < num_batches_per_epoch - 1) {
            // 使用 std::cout 直接输出，以便使用 \r
            std::cout << "\r" << batch_info.str() << std::flush;
        } else {
            // 最后一个 batch，打印完整信息并换行
            std::cout << "\r" << batch_info.str() << std::endl;
        }
    }
    
    auto epoch_end = std::chrono::high_resolution_clock::now();
    auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        epoch_end - epoch_start
    );
    auto epoch_duration_sec = epoch_duration.count() / 1000.0;
    
    // 计算平均损失
    float avg_loss = epoch_loss / num_batches_per_epoch;
    
    // 更新最佳损失
    bool is_best = false;
    if (avg_loss < best_loss_) {
        best_loss_ = avg_loss;
        is_best = true;
    }
    
    // 计算总样本数
    int64_t total_samples = batch_size * num_batches_per_epoch;
    double samples_per_second = total_samples / std::max(0.001, epoch_duration_sec);
    
    // YOLOv5 风格的 Epoch 总结（更新表头行）
    std::ostringstream epoch_summary;
    epoch_summary << std::setw(5) << (epoch + 1) << "/" << num_epochs
                  << std::setw(9) << gpu_mem_str
                  << std::fixed << std::setprecision(4) << std::setw(11) << avg_loss
                  << std::scientific << std::setprecision(2) << std::setw(11) << current_lr
                  << std::fixed << std::setw(13) << total_samples
                  << std::setw(9) << batch_size;
    Logger::info("{}", epoch_summary.str());
    
    // 保存last.pth（每个epoch都保存最新的模型）
    if (save_model_checkpoint(model_, last_checkpoint_path_)) {
        Logger::info("Saving last checkpoint to: {}", last_checkpoint_path_);
    } else {
        Logger::warning("Failed to save last checkpoint");
    }
    
    // 如果当前损失比上一次小，保存best.pth
    if (avg_loss < previous_loss_) {
        std::ostringstream prev_loss_ss, curr_loss_ss;
        prev_loss_ss << std::fixed << std::setprecision(6) << previous_loss_;
        curr_loss_ss << std::fixed << std::setprecision(6) << avg_loss;
        
        Logger::info("Loss improved: {} -> {} (saving best checkpoint)", 
            prev_loss_ss.str(), curr_loss_ss.str());
        
        if (save_model_checkpoint(model_, best_checkpoint_path_)) {
            Logger::info("Best checkpoint saved to: {}", best_checkpoint_path_);
        } else {
            Logger::warning("Failed to save best checkpoint");
        }
    } else {
        std::ostringstream prev_loss_ss, curr_loss_ss;
        prev_loss_ss << std::fixed << std::setprecision(6) << previous_loss_;
        curr_loss_ss << std::fixed << std::setprecision(6) << avg_loss;
        
        Logger::info("Loss did not improve: {} -> {} (best checkpoint not updated)", 
            prev_loss_ss.str(), curr_loss_ss.str());
    }
    
    // 更新上一次损失值
    previous_loss_ = avg_loss;
    
    // 保存当前epoch的信息
    EpochInfo epoch_info;
    epoch_info.epoch = epoch + 1;
    epoch_info.avg_loss = avg_loss;
    epoch_info.min_loss = min_batch_loss;
    epoch_info.max_loss = max_batch_loss;
    epoch_info.duration_sec = epoch_duration_sec;
    epoch_info.samples = total_samples;
    epoch_info.samples_per_sec = samples_per_second;
    epoch_history_.push_back(epoch_info);
}

void Trainer::printTrainingHistory() const {
    Logger::info("");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("Training History - All Epochs Summary");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("");
    Logger::info("┌───────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
    Logger::info("│ Epoch │  Avg Loss    │   Min Loss   │   Max Loss   │   Time (s)   │   Samples    │  Speed (s/s) │");
    Logger::info("├───────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");
    
    for (const auto& info : epoch_history_) {
        std::ostringstream avg_loss_ss, min_loss_ss, max_loss_ss, time_ss, samples_ss, speed_ss, row_ss;
        avg_loss_ss << std::fixed << std::setprecision(6) << info.avg_loss;
        min_loss_ss << std::fixed << std::setprecision(6) << info.min_loss;
        max_loss_ss << std::fixed << std::setprecision(6) << info.max_loss;
        time_ss << std::fixed << std::setprecision(2) << info.duration_sec;
        samples_ss << info.samples;
        speed_ss << std::fixed << std::setprecision(1) << info.samples_per_sec;
        
        // Format the row with proper alignment
        row_ss << "│ " << std::setw(5) << info.epoch << " │ "
               << std::setw(12) << avg_loss_ss.str() << " │ "
               << std::setw(12) << min_loss_ss.str() << " │ "
               << std::setw(12) << max_loss_ss.str() << " │ "
               << std::setw(12) << time_ss.str() << " │ "
               << std::setw(12) << samples_ss.str() << " │ "
               << std::setw(12) << speed_ss.str() << " │";
        
        Logger::info("{}", row_ss.str());
    }
    
    Logger::info("└───────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
    Logger::info("");
    
    // 打印损失趋势分析
    if (epoch_history_.size() > 1) {
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Logger::info("Loss Trend Analysis");
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        float first_loss = epoch_history_[0].avg_loss;
        float last_loss = epoch_history_.back().avg_loss;
        float loss_reduction = first_loss - last_loss;
        float loss_reduction_percent = (loss_reduction / first_loss) * 100.0f;
        
        std::ostringstream first_loss_ss, last_loss_ss, reduction_ss, percent_ss;
        first_loss_ss << std::fixed << std::setprecision(6) << first_loss;
        last_loss_ss << std::fixed << std::setprecision(6) << last_loss;
        reduction_ss << std::fixed << std::setprecision(6) << loss_reduction;
        percent_ss << std::fixed << std::setprecision(2) << loss_reduction_percent;
        
        Logger::info("  ├─ First Epoch Loss:  {}", first_loss_ss.str());
        Logger::info("  ├─ Last Epoch Loss:   {}", last_loss_ss.str());
        Logger::info("  ├─ Total Reduction:  {} ({})", reduction_ss.str(), percent_ss.str() + "%");
        
        // 计算平均损失变化率
        float total_change = 0.0f;
        int change_count = 0;
        for (size_t i = 1; i < epoch_history_.size(); ++i) {
            float change = epoch_history_[i-1].avg_loss - epoch_history_[i].avg_loss;
            total_change += change;
            change_count++;
        }
        float avg_change = change_count > 0 ? total_change / change_count : 0.0f;
        std::ostringstream avg_change_ss;
        avg_change_ss << std::fixed << std::setprecision(6) << avg_change;
        Logger::info("  └─ Average Loss Change per Epoch: {}", avg_change_ss.str());
        Logger::info("");
        
        // 绘制损失趋势图
        printLossTrendChart();
    }
}

void Trainer::printLossTrendChart() const {
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("Loss Trend Chart");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("");
    
    // 图表参数
    const int chart_height = 20;
    const int chart_width = std::min(80, static_cast<int>(epoch_history_.size() * 2));
    
    // 找到损失的最小值和最大值
    float min_loss = std::numeric_limits<float>::max();
    float max_loss = std::numeric_limits<float>::min();
    for (const auto& info : epoch_history_) {
        min_loss = std::min(min_loss, info.avg_loss);
        max_loss = std::max(max_loss, info.avg_loss);
    }
    
    // 添加边距
    float loss_range = max_loss - min_loss;
    float margin = loss_range * 0.1f;
    min_loss -= margin;
    max_loss += margin;
    loss_range = max_loss - min_loss;
    
    // 创建图表网格
    std::vector<std::vector<char>> chart(chart_height, std::vector<char>(chart_width, ' '));
    
    // 绘制Y轴
    for (int i = 0; i < chart_height; ++i) {
        chart[i][0] = '│';
    }
    
    // 绘制X轴
    for (int j = 0; j < chart_width; ++j) {
        chart[chart_height - 1][j] = '─';
    }
    chart[chart_height - 1][0] = '└';
    
    // 绘制损失点并连接
    std::vector<int> y_positions;
    for (size_t idx = 0; idx < epoch_history_.size(); ++idx) {
        float loss = epoch_history_[idx].avg_loss;
        int y_pos = chart_height - 1 - static_cast<int>(((loss - min_loss) / loss_range) * (chart_height - 1));
        y_pos = std::max(0, std::min(chart_height - 1, y_pos));
        y_positions.push_back(y_pos);
        
        int x_pos = static_cast<int>((static_cast<float>(idx) / (epoch_history_.size() - 1)) * (chart_width - 1));
        x_pos = std::max(1, std::min(chart_width - 1, x_pos));
        
        if (y_pos >= 0 && y_pos < chart_height && x_pos > 0 && x_pos < chart_width) {
            chart[y_pos][x_pos] = '*';
        }
    }
    
    // 连接相邻的点
    for (size_t idx = 1; idx < y_positions.size(); ++idx) {
        int prev_y = y_positions[idx - 1];
        int curr_y = y_positions[idx];
        int prev_x = static_cast<int>((static_cast<float>(idx - 1) / (epoch_history_.size() - 1)) * (chart_width - 1));
        int curr_x = static_cast<int>((static_cast<float>(idx) / (epoch_history_.size() - 1)) * (chart_width - 1));
        prev_x = std::max(1, std::min(chart_width - 1, prev_x));
        curr_x = std::max(1, std::min(chart_width - 1, curr_x));
        
        // 使用Bresenham算法绘制直线
        int dx = std::abs(curr_x - prev_x);
        int dy = std::abs(curr_y - prev_y);
        int sx = prev_x < curr_x ? 1 : -1;
        int sy = prev_y < curr_y ? 1 : -1;
        int err = dx - dy;
        
        int x = prev_x;
        int y = prev_y;
        while (true) {
            if (y >= 0 && y < chart_height && x > 0 && x < chart_width) {
                if (chart[y][x] == ' ') {
                    chart[y][x] = '-';
                }
            }
            
            if (x == curr_x && y == curr_y) break;
            
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x += sx;
            }
            if (e2 < dx) {
                err += dx;
                y += sy;
            }
        }
    }
    
    // 打印图表
    std::ostringstream y_label_ss;
    y_label_ss << std::fixed << std::setprecision(4) << max_loss;
    Logger::info("{} ┌─ Loss", y_label_ss.str());
    
    for (int i = 0; i < chart_height - 1; ++i) {
        std::ostringstream line_ss;
        if (i % 5 == 0 || i == 0) {
            float loss_value = max_loss - (static_cast<float>(i) / (chart_height - 1)) * loss_range;
            line_ss << std::fixed << std::setprecision(4) << std::setw(8) << loss_value << " │";
        } else {
            line_ss << "         │";
        }
        
        for (int j = 1; j < chart_width; ++j) {
            line_ss << chart[i][j];
        }
        
        Logger::info("{}", line_ss.str());
    }
    
    // X轴
    std::ostringstream x_axis_ss;
    x_axis_ss << "         └";
    for (int j = 1; j < chart_width; ++j) {
        x_axis_ss << chart[chart_height - 1][j];
    }
    Logger::info("{}", x_axis_ss.str());
    
    // X轴标签
    std::ostringstream x_label_ss;
    x_label_ss << "         ";
    for (size_t idx = 0; idx < epoch_history_.size(); ++idx) {
        int x_pos = static_cast<int>((static_cast<float>(idx) / (epoch_history_.size() - 1)) * (chart_width - 1));
        x_pos = std::max(1, std::min(chart_width - 1, x_pos));
        
        if (idx == 0 || idx == epoch_history_.size() - 1 || 
            (idx % std::max(1, static_cast<int>(epoch_history_.size() / 10)) == 0)) {
            std::ostringstream epoch_ss;
            epoch_ss << epoch_history_[idx].epoch;
            int label_len = static_cast<int>(epoch_ss.str().length());
            int start_pos = std::max(0, x_pos - label_len / 2);
            
            if (start_pos + label_len <= chart_width) {
                while (static_cast<int>(x_label_ss.str().length()) < start_pos + 1) {
                    x_label_ss << " ";
                }
                x_label_ss << epoch_ss.str();
            }
        }
    }
    Logger::info("{}", x_label_ss.str());
    
    // X轴标题
    std::ostringstream x_title_ss;
    x_title_ss << "         ";
    int title_start = (chart_width - 5) / 2;
    for (int i = 0; i < title_start; ++i) {
        x_title_ss << " ";
    }
    x_title_ss << "Epoch";
    Logger::info("{}", x_title_ss.str());
    Logger::info("");
}

float Trainer::validate(int batch_size) {
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("Running validation...");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    auto val_start = std::chrono::high_resolution_clock::now();
    
    // 从数据集中获取验证数据
    auto [val_input_ids, val_target_ids] = dataset_->getBatch(batch_size, device_);
    
    // 验证
    float val_loss = validate_batch(
        model_, 
        val_input_ids, 
        val_target_ids, 
        *criterion_
    );
    
    auto val_end = std::chrono::high_resolution_clock::now();
    auto val_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        val_end - val_start
    );
    
    std::ostringstream val_loss_str;
    val_loss_str << std::fixed << std::setprecision(6) << val_loss;
    Logger::info("  ├─ Validation Loss: {}", val_loss_str.str());
    Logger::info("  └─ Validation Time: {} ms", val_duration.count());
    
    return val_loss;
}

void Trainer::inferenceTest(const std::string& prompt, int max_new_tokens) {
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("Starting inference test...");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // 使用Generator进行推理
    Generator generator(model_, device_, cfg_);
    GenerationResult result = generator.generate(prompt, max_new_tokens);
    
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("Inference Results:");
    Logger::info("Original Prompt: \"{}\"", result.prompt);
    Logger::info("Generated Text: \"{}\"", result.generated_text);
    Logger::info("Number of Generated Tokens: {}", result.generated_tokens.size());
    std::ostringstream avg_speed_str;
    avg_speed_str << std::fixed << std::setprecision(2) << result.avg_tokens_per_second;
    Logger::info("Total Inference Time: {:.2f} seconds ({} ms)", result.total_time_sec, result.total_time_ms);
    Logger::info("Average Generation Speed: {} tokens/sec", avg_speed_str.str());
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

