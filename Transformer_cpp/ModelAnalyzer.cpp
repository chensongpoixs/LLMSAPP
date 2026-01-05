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
 * Model Analyzer Implementation
 * 
 * 模型分析器实现
 */

#include "ModelAnalyzer.h"
#include "Logger.h"
#include <torch/torch.h>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>

ModelAnalyzer::ModelAnalyzer(std::shared_ptr<GPTModel> model)
    : model_(model), total_params_(0), total_memory_bytes_(0) {
    initializeParameterDescriptions();
}

void ModelAnalyzer::analyze() {
    param_stats_.clear();
    total_params_ = 0;
    total_memory_bytes_ = 0;
    
    // 遍历所有命名参数
    for (const auto& pair : model_->named_parameters()) {
        const std::string& name = pair.key();
        const torch::Tensor& param = pair.value();
        
        ParameterStats stats = computeStats(param, name);
        param_stats_.push_back(stats);
        
        total_params_ += stats.numel;
        total_memory_bytes_ += stats.memory_bytes;
    }
}

ParameterStats ModelAnalyzer::computeStats(const torch::Tensor& param, const std::string& name) const {
    ParameterStats stats;
    stats.name = name;
    
    // 获取形状
    auto sizes = param.sizes();
    stats.shape.assign(sizes.begin(), sizes.end());
    stats.numel = param.numel();
    stats.memory_bytes = stats.numel * param.element_size();
    
    // 计算统计信息
    if (stats.numel > 0) {
        auto param_cpu = param.cpu().to(torch::kFloat32);
        auto param_flat = param_cpu.flatten();
        
        stats.mean = param_flat.mean().item<double>();
        stats.std = param_flat.std().item<double>();
        stats.min = param_flat.min().item<double>();
        stats.max = param_flat.max().item<double>();
    } else {
        stats.mean = 0.0;
        stats.std = 0.0;
        stats.min = 0.0;
        stats.max = 0.0;
    }
    
    return stats;
}

std::string ModelAnalyzer::formatShape(const std::vector<int64_t>& shape) const {
    if (shape.empty()) {
        return "[]";
    }
    
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i < shape.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

std::string ModelAnalyzer::formatMemorySize(size_t bytes) const {
    std::ostringstream oss;
    if (bytes < 1024) {
        oss << bytes << " B";
    } else if (bytes < 1024 * 1024) {
        oss << std::fixed << std::setprecision(2) << (bytes / 1024.0) << " KB";
    } else if (bytes < 1024 * 1024 * 1024) {
        oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0)) << " MB";
    } else {
        oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GB";
    }
    return oss.str();
}

void ModelAnalyzer::printModelStructure() const {
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Model Structure");
    Logger::info("═══════════════════════════════════════════════════════════════");
    
    // 打印模型层次结构
    Logger::info("Model Architecture: GPT (Decoder-Only Transformer)");
    Logger::info("");
    
    // 按模块分组显示参数
    std::map<std::string, std::vector<const ParameterStats*>> module_groups;
    
    for (const auto& stats : param_stats_) {
        // 提取模块名称（参数名的第一部分）
        size_t dot_pos = stats.name.find('.');
        std::string module_name = (dot_pos != std::string::npos) ? 
            stats.name.substr(0, dot_pos) : "root";
        module_groups[module_name].push_back(&stats);
    }
    
    for (const auto& group : module_groups) {
        Logger::info("Module: {}", group.first);
        for (const auto* stats : group.second) {
            Logger::info("  └─ {}: {}", stats->name, formatShape(stats->shape));
        }
        Logger::info("");
    }
}

void ModelAnalyzer::printAllParameters() const {
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("All Model Parameters");
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("");
    
    Logger::info("┌─────────────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
    Logger::info("│ Parameter Name                      │ Shape        │ Elements     │ Mean         │ Std          │ Min          │ Max          │");
    Logger::info("├─────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤");
    
    for (const auto& stats : param_stats_) {
        std::ostringstream row_ss;
        row_ss << "│ " << std::setw(35) << std::left << stats.name.substr(0, 35) << " │ "
               << std::setw(12) << formatShape(stats.shape) << " │ "
               << std::setw(12) << stats.numel << " │ "
               << std::fixed << std::setprecision(6) << std::setw(12) << stats.mean << " │ "
               << std::setw(12) << stats.std << " │ "
               << std::setw(12) << stats.min << " │ "
               << std::setw(12) << stats.max << " │";
        
        Logger::info("{}", row_ss.str());
    }
    
    Logger::info("└─────────────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘");
    Logger::info("");
}

void ModelAnalyzer::printParameterStatistics() const {
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Parameter Statistics (Detailed)");
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("");
    
    for (const auto& stats : param_stats_) {
        Logger::info("Parameter: {}", stats.name);
        Logger::info("  ├─ Shape: {}", formatShape(stats.shape));
        Logger::info("  ├─ Elements: {}", stats.numel);
        Logger::info("  ├─ Memory: {}", formatMemorySize(stats.memory_bytes));
        
        std::ostringstream mean_ss, std_ss, min_ss, max_ss;
        mean_ss << std::fixed << std::setprecision(6) << stats.mean;
        std_ss << std::fixed << std::setprecision(6) << stats.std;
        min_ss << std::fixed << std::setprecision(6) << stats.min;
        max_ss << std::fixed << std::setprecision(6) << stats.max;
        
        Logger::info("  ├─ Mean: {}", mean_ss.str());
        Logger::info("  ├─ Std:  {}", std_ss.str());
        Logger::info("  ├─ Min:  {}", min_ss.str());
        Logger::info("  └─ Max:  {}", max_ss.str());
        Logger::info("");
    }
}

void ModelAnalyzer::printParameterSummary() const {
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Model Parameter Summary");
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("");
    
    Logger::info("  ├─ Total Parameters: {}", total_params_);
    Logger::info("  ├─ Total Memory: {}", formatMemorySize(total_memory_bytes_));
    Logger::info("  ├─ Number of Parameter Tensors: {}", param_stats_.size());
    
    // 按类型统计
    size_t trainable_params = 0;
    size_t trainable_memory = 0;
    
    for (const auto& param : model_->parameters()) {
        if (param.requires_grad()) {
            trainable_params += param.numel();
            trainable_memory += param.numel() * param.element_size();
        }
    }
    
    Logger::info("  ├─ Trainable Parameters: {}", trainable_params);
    Logger::info("  ├─ Trainable Memory: {}", formatMemorySize(trainable_memory));
    Logger::info("  └─ Non-trainable Parameters: {}", total_params_ - trainable_params);
    Logger::info("");
    
    // 按模块统计
    Logger::info("Parameter Distribution by Module:");
    std::map<std::string, size_t> module_params;
    std::map<std::string, size_t> module_memory;
    
    for (const auto& stats : param_stats_) {
        size_t dot_pos = stats.name.find('.');
        std::string module_name = (dot_pos != std::string::npos) ? 
            stats.name.substr(0, dot_pos) : "root";
        module_params[module_name] += stats.numel;
        module_memory[module_name] += stats.memory_bytes;
    }
    
    for (const auto& pair : module_params) {
        double percentage = (pair.second * 100.0) / total_params_;
        std::ostringstream pct_ss;
        pct_ss << std::fixed << std::setprecision(2) << percentage;
        Logger::info("  ├─ {}: {} ({}%) - {}", 
            pair.first, pair.second, pct_ss.str(), formatMemorySize(module_memory[pair.first]));
    }
    Logger::info("");
}

void ModelAnalyzer::saveModelInfo(const std::string& output_file) const {
    std::ofstream file(output_file);
    if (!file.is_open()) {
        Logger::error("Failed to open file for writing: {}", output_file);
        return;
    }
    
    file << "========================================\n";
    file << "Model Information\n";
    file << "========================================\n\n";
    
    file << "Total Parameters: " << total_params_ << "\n";
    file << "Total Memory: " << formatMemorySize(total_memory_bytes_) << "\n";
    file << "Number of Parameter Tensors: " << param_stats_.size() << "\n\n";
    
    file << "========================================\n";
    file << "All Parameters\n";
    file << "========================================\n\n";
    
    for (const auto& stats : param_stats_) {
        file << "Parameter: " << stats.name << "\n";
        file << "  Shape: " << formatShape(stats.shape) << "\n";
        file << "  Elements: " << stats.numel << "\n";
        file << "  Memory: " << formatMemorySize(stats.memory_bytes) << "\n";
        file << "  Mean: " << std::fixed << std::setprecision(6) << stats.mean << "\n";
        file << "  Std:  " << stats.std << "\n";
        file << "  Min:  " << stats.min << "\n";
        file << "  Max:  " << stats.max << "\n";
        file << "\n";
    }
    
    file.close();
    Logger::info("Model information saved to: {}", output_file);
}

void ModelAnalyzer::initializeParameterDescriptions() {
    // Token Embedding
    param_descriptions_["token_embedding.weight"] = 
        "Token Embedding Weight: Maps token IDs to dense vectors. Shape: [vocab_size, embedding_dim]. "
        "Each row represents the embedding vector for a specific token in the vocabulary. "
        "Formula: E_token = Embedding[token_id], where E_token is a vector of size embedding_dim. "
        "This is a lookup table that converts discrete token IDs into continuous vector representations. "
        "The embedding vectors are learned during training and capture semantic information about tokens.";
    
    // Position Embedding
    param_descriptions_["position_embedding.weight"] = 
        "Position Embedding Weight: Encodes position information for each sequence position. "
        "Shape: [max_seq_length, embedding_dim]. Each row represents the embedding for a specific position. "
        "Formula: E_pos = PositionEmbedding[pos], where pos is the position index (0 to seq_len-1). "
        "These embeddings are added to token embeddings to provide positional information to the model. "
        "Since Transformers have no inherent notion of sequence order, position embeddings are crucial.";
    
    // Transformer Blocks - Multi-Head Attention
    param_descriptions_["transformer_blocks.*.mha.q_proj.weight"] = 
        "Query Projection Weight: Linear transformation for Query vectors in Multi-Head Attention. "
        "Shape: [embedding_dim, embedding_dim]. Projects input to query space. "
        "Formula: Q = X * W_q, where X is input [batch, seq_len, embedding_dim] and W_q is this weight matrix. "
        "For each attention head h, Q_h = X * W_q_h, where W_q_h is a slice of this matrix. "
        "The Query vectors are used to compute attention scores: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V.";
    
    param_descriptions_["transformer_blocks.*.mha.q_proj.bias"] = 
        "Query Projection Bias: Bias term for Query projection (if enabled). "
        "Formula: Q = X * W_q + b_q, where b_q is this bias vector. "
        "Shape: [embedding_dim]. Added element-wise to the output of the linear transformation.";
    
    param_descriptions_["transformer_blocks.*.mha.k_proj.weight"] = 
        "Key Projection Weight: Linear transformation for Key vectors in Multi-Head Attention. "
        "Shape: [embedding_dim, embedding_dim]. Projects input to key space. "
        "Formula: K = X * W_k, where X is input and W_k is this weight matrix. "
        "Key vectors are used to compute attention scores with Query vectors. "
        "The dot product QK^T measures the similarity between queries and keys.";
    
    param_descriptions_["transformer_blocks.*.mha.k_proj.bias"] = 
        "Key Projection Bias: Bias term for Key projection (if enabled). "
        "Formula: K = X * W_k + b_k, where b_k is this bias vector. "
        "Shape: [embedding_dim]. Added element-wise to the output of the linear transformation.";
    
    param_descriptions_["transformer_blocks.*.mha.v_proj.weight"] = 
        "Value Projection Weight: Linear transformation for Value vectors in Multi-Head Attention. "
        "Shape: [embedding_dim, embedding_dim]. Projects input to value space. "
        "Formula: V = X * W_v, where X is input and W_v is this weight matrix. "
        "Value vectors contain the actual information to be aggregated based on attention scores. "
        "The weighted sum of values gives the attention output: Output = Attention(Q, K, V).";
    
    param_descriptions_["transformer_blocks.*.mha.v_proj.bias"] = 
        "Value Projection Bias: Bias term for Value projection (if enabled). "
        "Formula: V = X * W_v + b_v, where b_v is this bias vector. "
        "Shape: [embedding_dim]. Added element-wise to the output of the linear transformation.";
    
    param_descriptions_["transformer_blocks.*.mha.out_proj.weight"] = 
        "Output Projection Weight: Linear transformation that combines multi-head attention outputs. "
        "Shape: [embedding_dim, embedding_dim]. Concatenates and projects attention heads back to embedding_dim. "
        "Formula: Output = Concat(head_1, ..., head_h) * W_o, where head_i is the output of attention head i. "
        "After computing attention for each head independently, the outputs are concatenated and projected. "
        "This allows the model to attend to information from different representation subspaces.";
    
    param_descriptions_["transformer_blocks.*.mha.out_proj.bias"] = 
        "Output Projection Bias: Bias term for output projection. "
        "Formula: Output = Concat(head_1, ..., head_h) * W_o + b_o, where b_o is this bias vector. "
        "Shape: [embedding_dim]. Added element-wise to the output of the linear transformation.";
    
    // Transformer Blocks - Feed Forward Network
    param_descriptions_["transformer_blocks.*.ffn.fc1.weight"] = 
        "Feed Forward Network First Linear Layer Weight: First linear transformation in FFN. "
        "Shape: [ffn_dim, embedding_dim]. Typically ffn_dim = 4 * embedding_dim (e.g., 3072 for embedding_dim=768).";
    
    param_descriptions_["transformer_blocks.*.ffn.fc1.bias"] = 
        "Feed Forward Network First Linear Layer Bias: Bias term for first FFN layer.";
    
    param_descriptions_["transformer_blocks.*.ffn.fc2.weight"] = 
        "Feed Forward Network Second Linear Layer Weight: Second linear transformation in FFN. "
        "Shape: [embedding_dim, ffn_dim]. Projects back to embedding_dim.";
    
    param_descriptions_["transformer_blocks.*.ffn.fc2.bias"] = 
        "Feed Forward Network Second Linear Layer Bias: Bias term for second FFN layer.";
    
    // Transformer Blocks - Layer Normalization
    param_descriptions_["transformer_blocks.*.ln1.weight"] = 
        "First Layer Normalization Weight: Learnable scale parameter for first layer norm in Transformer block. "
        "Shape: [embedding_dim]. Applied before Multi-Head Attention (Pre-LayerNorm architecture).";
    
    param_descriptions_["transformer_blocks.*.ln1.bias"] = 
        "First Layer Normalization Bias: Learnable shift parameter for first layer norm.";
    
    param_descriptions_["transformer_blocks.*.ln2.weight"] = 
        "Second Layer Normalization Weight: Learnable scale parameter for second layer norm in Transformer block. "
        "Shape: [embedding_dim]. Applied before Feed Forward Network.";
    
    param_descriptions_["transformer_blocks.*.ln2.bias"] = 
        "Second Layer Normalization Bias: Learnable shift parameter for second layer norm.";
    
    // Final Layer Normalization
    param_descriptions_["layer_norm.weight"] = 
        "Final Layer Normalization Weight: Learnable scale parameter for final layer norm. "
        "Shape: [embedding_dim]. Applied after all Transformer blocks, before output head.";
    
    param_descriptions_["layer_norm.bias"] = 
        "Final Layer Normalization Bias: Learnable shift parameter for final layer norm.";
    
    // Output Head
    param_descriptions_["out_head.weight"] = 
        "Output Head Weight: Final linear projection to vocabulary size. "
        "Shape: [vocab_size, embedding_dim]. Maps hidden states to logits over vocabulary.";
    
    param_descriptions_["out_head.bias"] = 
        "Output Head Bias: Bias term for output head. Shape: [vocab_size].";
}

std::string ModelAnalyzer::getParameterDescription(const std::string& param_name) const {
    // 直接匹配
    auto it = param_descriptions_.find(param_name);
    if (it != param_descriptions_.end()) {
        return it->second;
    }
    
    // 模式匹配（使用通配符 *）
    for (const auto& pair : param_descriptions_) {
        std::string pattern = pair.first;
        std::string name = param_name;
        
        // 检查模式中是否包含通配符
        size_t star_pos = pattern.find("*");
        if (star_pos != std::string::npos) {
            // 提取模式的前缀和后缀
            std::string prefix = pattern.substr(0, star_pos);
            std::string suffix = pattern.substr(star_pos + 1);
            
            // 检查参数名是否以前缀开头
            if (name.find(prefix) == 0) {
                // 检查后缀是否在参数名中（在前缀之后）
                size_t suffix_pos = name.find(suffix, prefix.length());
                if (suffix_pos != std::string::npos) {
                    // 验证前缀和后缀之间的部分是有效的（数字或点）
                    std::string middle = name.substr(prefix.length(), suffix_pos - prefix.length());
                    bool valid = true;
                    for (char c : middle) {
                        if (c != '.' && (c < '0' || c > '9')) {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) {
                        return pair.second;
                    }
                }
            }
        }
    }
    
    // 根据参数名推断说明
    if (param_name.find("weight") != std::string::npos) {
        return "Weight parameter: Learnable weights for linear transformation or embedding.";
    } else if (param_name.find("bias") != std::string::npos) {
        return "Bias parameter: Learnable bias term added to linear transformation output.";
    }
    
    return "Parameter: No description available.";
}

void ModelAnalyzer::printParameterDescriptions() const {
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Parameter Descriptions");
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("");
    
    // 按模块分组
    std::map<std::string, std::vector<const ParameterStats*>> module_groups;
    
    for (const auto& stats : param_stats_) {
        size_t dot_pos = stats.name.find('.');
        std::string module_name = (dot_pos != std::string::npos) ? 
            stats.name.substr(0, dot_pos) : "root";
        module_groups[module_name].push_back(&stats);
    }
    
    for (const auto& group : module_groups) {
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Logger::info("Module: {}", group.first);
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        for (const auto* stats : group.second) {
            Logger::info("");
            Logger::info("Parameter: {}", stats->name);
            Logger::info("  Shape: {}", formatShape(stats->shape));
            Logger::info("  Description:");
            
            std::string desc = getParameterDescription(stats->name);
            // 将长描述分行显示
            size_t pos = 0;
            size_t line_width = 80;
            while (pos < desc.length()) {
                size_t end_pos = pos + line_width;
                if (end_pos >= desc.length()) {
                    Logger::info("    {}", desc.substr(pos));
                    break;
                }
                
                // 尝试在空格处断开
                size_t break_pos = desc.rfind(' ', end_pos);
                if (break_pos == std::string::npos || break_pos < pos) {
                    break_pos = end_pos;
                }
                
                Logger::info("    {}", desc.substr(pos, break_pos - pos));
                pos = break_pos + 1;
            }
        }
        Logger::info("");
    }
}

void ModelAnalyzer::printModelArchitectureDiagram() const {
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Model Architecture Diagram");
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("");
    
    Logger::info("GPT (Decoder-Only Transformer) Architecture:");
    Logger::info("");
    Logger::info("┌─────────────────────────────────────────────────────────────────┐");
    Logger::info("│ Input: Token IDs [batch_size, seq_len]                        │");
    Logger::info("└─────────────────────────────────────────────────────────────────┘");
    Logger::info("                              │");
    Logger::info("                              ▼");
    Logger::info("┌─────────────────────────────────────────────────────────────────┐");
    Logger::info("│ 1. Token Embedding                                             │");
    Logger::info("│    token_embedding.weight [vocab_size, embedding_dim]           │");
    Logger::info("│    Output: [batch_size, seq_len, embedding_dim]                │");
    Logger::info("└─────────────────────────────────────────────────────────────────┘");
    Logger::info("                              │");
    Logger::info("                              ▼");
    Logger::info("┌─────────────────────────────────────────────────────────────────┐");
    Logger::info("│ 2. Position Embedding                                          │");
    Logger::info("│    position_embedding.weight [max_seq_len, embedding_dim]      │");
    Logger::info("│    Output: [batch_size, seq_len, embedding_dim]                │");
    Logger::info("└─────────────────────────────────────────────────────────────────┘");
    Logger::info("                              │");
    Logger::info("                              ▼");
    Logger::info("┌─────────────────────────────────────────────────────────────────┐");
    Logger::info("│ 3. Embedding Fusion: token_emb + pos_emb                      │");
    Logger::info("│    + Dropout                                                   │");
    Logger::info("└─────────────────────────────────────────────────────────────────┘");
    Logger::info("                              │");
    Logger::info("                              ▼");
    
    // 获取层数（从参数名推断）
    int n_layers = 0;
    for (const auto& stats : param_stats_) {
        if (stats.name.find("transformer_blocks.") == 0) {
            size_t start = strlen("transformer_blocks.");
            size_t dot = stats.name.find('.', start);
            if (dot != std::string::npos) {
                std::string layer_str = stats.name.substr(start, dot - start);
                try {
                    int layer = std::stoi(layer_str);
                    n_layers = std::max(n_layers, layer + 1);
                } catch (...) {
                    // Ignore
                }
            }
        }
    }
    
    if (n_layers == 0) n_layers = 12; // Default
    
    Logger::info("┌─────────────────────────────────────────────────────────────────┐");
    Logger::info("│ 4. Transformer Blocks ({} layers)                              │", n_layers);
    Logger::info("└─────────────────────────────────────────────────────────────────┘");
    Logger::info("                              │");
    
    for (int i = 0; i < n_layers && i < 3; ++i) {
        Logger::info("                    ┌─────────────┐");
        Logger::info("                    │ Block {}     │", i);
        Logger::info("                    └─────────────┘");
        Logger::info("                              │");
        Logger::info("                    ┌─────────────────────────────┐");
        Logger::info("                    │ 4.{}.1 Layer Norm 1          │", i + 1);
        Logger::info("                    │   ln1.weight [embed_dim]    │");
        Logger::info("                    └─────────────────────────────┘");
        Logger::info("                              │");
        Logger::info("                    ┌─────────────────────────────┐");
        Logger::info("                    │ 4.{}.2 Multi-Head Attention  │", i + 1);
        Logger::info("                    │   - q_proj [d, d]           │");
        Logger::info("                    │   - k_proj [d, d]           │");
        Logger::info("                    │   - v_proj [d, d]           │");
        Logger::info("                    │   - out_proj [d, d]          │");
        Logger::info("                    │   + Residual Connection     │");
        Logger::info("                    └─────────────────────────────┘");
        Logger::info("                              │");
        Logger::info("                    ┌─────────────────────────────┐");
        Logger::info("                    │ 4.{}.3 Layer Norm 2          │", i + 1);
        Logger::info("                    │   ln2.weight [embed_dim]    │");
        Logger::info("                    └─────────────────────────────┘");
        Logger::info("                              │");
        Logger::info("                    ┌─────────────────────────────┐");
        Logger::info("                    │ 4.{}.4 Feed Forward Network  │", i + 1);
        Logger::info("                    │   - fc1 [ffn_dim, d]        │");
        Logger::info("                    │   - GELU Activation          │");
        Logger::info("                    │   - fc2 [d, ffn_dim]        │");
        Logger::info("                    │   + Residual Connection     │");
        Logger::info("                    └─────────────────────────────┘");
        if (i < n_layers - 1 && i < 2) {
            Logger::info("                              │");
            Logger::info("                              ▼");
        }
    }
    
    if (n_layers > 3) {
        Logger::info("                              │");
        Logger::info("                              ▼");
        Logger::info("                    ... ({} more blocks) ...", n_layers - 3);
        Logger::info("                              │");
    }
    
    Logger::info("                              ▼");
    Logger::info("┌─────────────────────────────────────────────────────────────────┐");
    Logger::info("│ 5. Final Layer Normalization                                    │");
    Logger::info("│    layer_norm.weight [embedding_dim]                           │");
    Logger::info("└─────────────────────────────────────────────────────────────────┘");
    Logger::info("                              │");
    Logger::info("                              ▼");
    Logger::info("┌─────────────────────────────────────────────────────────────────┐");
    Logger::info("│ 6. Output Head                                                 │");
    Logger::info("│    out_head.weight [vocab_size, embedding_dim]                 │");
    Logger::info("│    Output: [batch_size, seq_len, vocab_size] (Logits)           │");
    Logger::info("└─────────────────────────────────────────────────────────────────┘");
    Logger::info("");
}

void ModelAnalyzer::printParameterRelationshipDiagram() const {
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Parameter Relationship Diagram");
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("");
    
    Logger::info("Parameter Flow and Relationships:");
    Logger::info("");
    Logger::info("┌─────────────────────────────────────────────────────────────┐");
    Logger::info("│ Embedding Layer                                             │");
    Logger::info("├─────────────────────────────────────────────────────────────┤");
    Logger::info("│                                                               │");
    Logger::info("│  token_embedding.weight                                      │");
    Logger::info("│         │                                                     │");
    Logger::info("│         ├──► Maps token IDs to vectors                       │");
    Logger::info("│         │                                                     │");
    Logger::info("│  position_embedding.weight                                   │");
    Logger::info("│         │                                                     │");
    Logger::info("│         ├──► Adds position information                        │");
    Logger::info("│         │                                                     │");
    Logger::info("│         └──► Combined: token_emb + pos_emb                   │");
    Logger::info("└─────────────────────────────────────────────────────────────┘");
    Logger::info("                              │");
    Logger::info("                              ▼");
    Logger::info("┌─────────────────────────────────────────────────────────────┐");
    Logger::info("│ Transformer Block (Repeated N times)                         │");
    Logger::info("├─────────────────────────────────────────────────────────────┤");
    Logger::info("│                                                               │");
    Logger::info("│  ┌─────────────────────────────────────────────┐            │");
    Logger::info("│  │ Multi-Head Attention (MHA)                  │            │");
    Logger::info("│  │                                             │            │");
    Logger::info("│  │  ln1.weight ──► Normalize input             │            │");
    Logger::info("│  │       │                                      │            │");
    Logger::info("│  │       ├──► q_proj.weight ──► Query vectors  │            │");
    Logger::info("│  │       ├──► k_proj.weight ──► Key vectors    │            │");
    Logger::info("│  │       ├──► v_proj.weight ──► Value vectors │            │");
    Logger::info("│  │       │                                      │            │");
    Logger::info("│  │       └──► out_proj.weight ──► Combine heads│            │");
    Logger::info("│  │                                             │            │");
    Logger::info("│  │       └──► + Residual (input)               │            │");
    Logger::info("│  └─────────────────────────────────────────────┘            │");
    Logger::info("│                              │                               │");
    Logger::info("│                              ▼                               │");
    Logger::info("│  ┌─────────────────────────────────────────────┐            │");
    Logger::info("│  │ Feed Forward Network (FFN)                  │            │");
    Logger::info("│  │                                             │            │");
    Logger::info("│  │  ln2.weight ──► Normalize input             │            │");
    Logger::info("│  │       │                                      │            │");
    Logger::info("│  │       ├──► fc1.weight ──► Expand dimension  │            │");
    Logger::info("│  │       │   (typically 4x)                     │            │");
    Logger::info("│  │       ├──► GELU Activation                  │            │");
    Logger::info("│  │       ├──► fc2.weight ──► Project back      │            │");
    Logger::info("│  │       │                                      │            │");
    Logger::info("│  │       └──► + Residual (MHA output)         │            │");
    Logger::info("│  └─────────────────────────────────────────────┘            │");
    Logger::info("└─────────────────────────────────────────────────────────────┘");
    Logger::info("                              │");
    Logger::info("                              ▼");
    Logger::info("┌─────────────────────────────────────────────────────────────┐");
    Logger::info("│ Output Layer                                                │");
    Logger::info("├─────────────────────────────────────────────────────────────┤");
    Logger::info("│                                                               │");
    Logger::info("│  layer_norm.weight ──► Final normalization                 │");
    Logger::info("│         │                                                     │");
    Logger::info("│         └──► out_head.weight ──► Vocabulary logits         │");
    Logger::info("│                                                               │");
    Logger::info("└─────────────────────────────────────────────────────────────┘");
    Logger::info("");
    
    Logger::info("Key Relationships:");
    Logger::info("  • All embedding_dim dimensions must match across layers");
    Logger::info("  • MHA projects: embedding_dim → embedding_dim (per head)");
    Logger::info("  • FFN expands: embedding_dim → ffn_dim → embedding_dim");
    Logger::info("  • Residual connections require matching dimensions");
    Logger::info("  • Output head maps: embedding_dim → vocab_size");
    Logger::info("");
}

void ModelAnalyzer::printParameterDetailedAnnotations() const {
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Parameter Detailed Annotations with Formulas");
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("");
    
    // 按模块分组显示详细注释
    std::map<std::string, std::vector<const ParameterStats*>> module_groups;
    
    for (const auto& stats : param_stats_) {
        size_t dot_pos = stats.name.find('.');
        std::string module_name = (dot_pos != std::string::npos) ? 
            stats.name.substr(0, dot_pos) : "root";
        module_groups[module_name].push_back(&stats);
    }
    
    for (const auto& group : module_groups) {
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Logger::info("Module: {}", group.first);
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        for (const auto* stats : group.second) {
            Logger::info("");
            Logger::info("┌─────────────────────────────────────────────────────────────┐");
            Logger::info("│ Parameter: {}", stats->name);
            Logger::info("├─────────────────────────────────────────────────────────────┤");
            Logger::info("│ Shape: {}", formatShape(stats->shape));
            Logger::info("│ Elements: {} | Memory: {}", stats->numel, formatMemorySize(stats->memory_bytes));
            Logger::info("│ Statistics: Mean={:.6f}, Std={:.6f}, Min={:.6f}, Max={:.6f}", 
                stats->mean, stats->std, stats->min, stats->max);
            Logger::info("├─────────────────────────────────────────────────────────────┤");
            
            std::string desc = getParameterDescription(stats->name);
            Logger::info("│ Description:                                              │");
            
            // 将描述分行显示
            size_t pos = 0;
            size_t line_width = 75;
            while (pos < desc.length()) {
                size_t end_pos = pos + line_width;
                if (end_pos >= desc.length()) {
                    Logger::info("│   {}", desc.substr(pos));
                    break;
                }
                
                size_t break_pos = desc.rfind(' ', end_pos);
                if (break_pos == std::string::npos || break_pos < pos) {
                    break_pos = end_pos;
                }
                
                Logger::info("│   {}", desc.substr(pos, break_pos - pos));
                pos = break_pos + 1;
            }
            Logger::info("└─────────────────────────────────────────────────────────────┘");
        }
        Logger::info("");
    }
}

void ModelAnalyzer::printParameterDimensionDiagram() const {
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Parameter Dimension Flow Diagram");
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("");
    
    Logger::info("Dimension Flow Through Model:");
    Logger::info("");
    Logger::info("Input: [batch_size, seq_len] (Token IDs)");
    Logger::info("  │");
    Logger::info("  ├─► Token Embedding");
    Logger::info("  │   token_embedding.weight: [vocab_size, embedding_dim]");
    Logger::info("  │   Output: [batch_size, seq_len, embedding_dim]");
    Logger::info("  │");
    Logger::info("  ├─► Position Embedding");
    Logger::info("  │   position_embedding.weight: [max_seq_len, embedding_dim]");
    Logger::info("  │   Output: [batch_size, seq_len, embedding_dim]");
    Logger::info("  │");
    Logger::info("  └─► Combined: [batch_size, seq_len, embedding_dim]");
    Logger::info("      │");
    Logger::info("      ▼");
    Logger::info("  ┌───────────────────────────────────────────────────────────┐");
    Logger::info("  │ Transformer Block (Repeated N times)                     │");
    Logger::info("  ├───────────────────────────────────────────────────────────┤");
    Logger::info("  │ Input: [batch_size, seq_len, embedding_dim]              │");
    Logger::info("  │   │                                                       │");
    Logger::info("  │   ├─► Layer Norm 1                                       │");
    Logger::info("  │   │   ln1.weight: [embedding_dim]                        │");
    Logger::info("  │   │   Output: [batch_size, seq_len, embedding_dim]      │");
    Logger::info("  │   │                                                       │");
    Logger::info("  │   ├─► Multi-Head Attention                                │");
    Logger::info("  │   │   ├─ q_proj.weight: [embedding_dim, embedding_dim]  │");
    Logger::info("  │   │   ├─ k_proj.weight: [embedding_dim, embedding_dim]  │");
    Logger::info("  │   │   ├─ v_proj.weight: [embedding_dim, embedding_dim]   │");
    Logger::info("  │   │   └─ out_proj.weight: [embedding_dim, embedding_dim] │");
    Logger::info("  │   │   Output: [batch_size, seq_len, embedding_dim]      │");
    Logger::info("  │   │   + Residual: [batch_size, seq_len, embedding_dim]  │");
    Logger::info("  │   │                                                       │");
    Logger::info("  │   ├─► Layer Norm 2                                       │");
    Logger::info("  │   │   ln2.weight: [embedding_dim]                        │");
    Logger::info("  │   │   Output: [batch_size, seq_len, embedding_dim]      │");
    Logger::info("  │   │                                                       │");
    Logger::info("  │   └─► Feed Forward Network                               │");
    Logger::info("  │       ├─ fc1.weight: [ffn_dim, embedding_dim]            │");
    Logger::info("  │       │   Output: [batch_size, seq_len, ffn_dim]        │");
    Logger::info("  │       ├─ GELU Activation                                  │");
    Logger::info("  │       ├─ fc2.weight: [embedding_dim, ffn_dim]            │");
    Logger::info("  │       └─ Output: [batch_size, seq_len, embedding_dim]   │");
    Logger::info("  │           + Residual: [batch_size, seq_len, embedding_dim]│");
    Logger::info("  │                                                           │");
    Logger::info("  └─► Output: [batch_size, seq_len, embedding_dim]          │");
    Logger::info("      │");
    Logger::info("      ▼");
    Logger::info("  Final Layer Norm");
    Logger::info("  layer_norm.weight: [embedding_dim]");
    Logger::info("  Output: [batch_size, seq_len, embedding_dim]");
    Logger::info("  │");
    Logger::info("  ▼");
    Logger::info("  Output Head");
    Logger::info("  out_head.weight: [vocab_size, embedding_dim]");
    Logger::info("  Output: [batch_size, seq_len, vocab_size] (Logits)");
    Logger::info("");
}

void ModelAnalyzer::printParameterComputationFlow() const {
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Parameter Computation Flow Diagram");
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("");
    
    Logger::info("Forward Pass Computation Flow:");
    Logger::info("");
    Logger::info("Step 1: Token Embedding");
    Logger::info("  Input: token_ids [batch, seq_len]");
    Logger::info("  Operation: E_token = token_embedding.weight[token_ids]");
    Logger::info("  Output: E_token [batch, seq_len, embedding_dim]");
    Logger::info("");
    Logger::info("Step 2: Position Embedding");
    Logger::info("  Input: position indices [0, 1, ..., seq_len-1]");
    Logger::info("  Operation: E_pos = position_embedding.weight[positions]");
    Logger::info("  Output: E_pos [seq_len, embedding_dim]");
    Logger::info("");
    Logger::info("Step 3: Embedding Fusion");
    Logger::info("  Operation: X = E_token + E_pos");
    Logger::info("  Output: X [batch, seq_len, embedding_dim]");
    Logger::info("");
    Logger::info("Step 4: Transformer Block (for each block):");
    Logger::info("  ┌─────────────────────────────────────────────────────────┐");
    Logger::info("  │ 4.1: Layer Normalization 1                              │");
    Logger::info("  │   X_norm = LayerNorm(X, ln1.weight, ln1.bias)          │");
    Logger::info("  │                                                         │");
    Logger::info("  │ 4.2: Multi-Head Attention                              │");
    Logger::info("  │   Q = X_norm * q_proj.weight + q_proj.bias             │");
    Logger::info("  │   K = X_norm * k_proj.weight + k_proj.bias             │");
    Logger::info("  │   V = X_norm * v_proj.weight + v_proj.bias             │");
    Logger::info("  │   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V   │");
    Logger::info("  │   MHA_out = Attention(Q, K, V) * out_proj.weight      │");
    Logger::info("  │   X = X + MHA_out  (Residual Connection)               │");
    Logger::info("  │                                                         │");
    Logger::info("  │ 4.3: Layer Normalization 2                              │");
    Logger::info("  │   X_norm2 = LayerNorm(X, ln2.weight, ln2.bias)         │");
    Logger::info("  │                                                         │");
    Logger::info("  │ 4.4: Feed Forward Network                               │");
    Logger::info("  │   FFN_hidden = GELU(X_norm2 * fc1.weight + fc1.bias)   │");
    Logger::info("  │   FFN_out = FFN_hidden * fc2.weight + fc2.bias         │");
    Logger::info("  │   X = X + FFN_out  (Residual Connection)                │");
    Logger::info("  └─────────────────────────────────────────────────────────┘");
    Logger::info("");
    Logger::info("Step 5: Final Layer Normalization");
    Logger::info("  Operation: X_final = LayerNorm(X, layer_norm.weight, layer_norm.bias)");
    Logger::info("  Output: X_final [batch, seq_len, embedding_dim]");
    Logger::info("");
    Logger::info("Step 6: Output Head");
    Logger::info("  Operation: logits = X_final * out_head.weight^T + out_head.bias");
    Logger::info("  Output: logits [batch, seq_len, vocab_size]");
    Logger::info("");
    Logger::info("Key Formulas:");
    Logger::info("  • LayerNorm(x, γ, β) = γ * (x - μ) / (σ + ε) + β");
    Logger::info("    where μ = mean(x), σ = std(x), ε = 1e-5");
    Logger::info("  • Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V");
    Logger::info("    where d_k is the dimension of keys (embedding_dim / n_heads)");
    Logger::info("  • GELU(x) = x * Φ(x), where Φ is the CDF of standard normal");
    Logger::info("");
}

