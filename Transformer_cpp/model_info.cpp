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
 * Model Information Tool
 * 
 * 模型信息工具
 * 用于读取和分析保存的模型，显示所有参数信息，方便理解模型结构
 * 
 * 功能：
 * 1. 加载保存的模型文件
 * 2. 显示模型结构
 * 3. 显示所有参数的详细信息（形状、统计信息等）
 * 4. 保存模型信息到文件
 */

#include "GPTModel.h"
#include "ModelConfig.h"
#include "Logger.h"
#include "ModelAnalyzer.h"
#include "TrainingUtils.h"
#include <torch/torch.h>
#include <memory>
#include <string>
#include <iomanip>

int main(int argc, char* argv[]) {
    // ========================================================================
    // 0. Configure logger
    // ========================================================================
    Logger::getInstance().setLogLevel(LogLevel::INFO);
    Logger::getInstance().setShowTimestamp(true);
    Logger::getInstance().setShowLevel(true);
    
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Transformer Model Information Tool");
    Logger::info("═══════════════════════════════════════════════════════════════");
    
    // ========================================================================
    // 1. Parse command line arguments
    // ========================================================================
    std::string model_path = "transformer_model_full.pth";
    std::string output_file = "model_info.txt";
    bool save_to_file = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" || arg == "-m") {
            if (i + 1 < argc) {
                model_path = argv[++i];
            }
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                output_file = argv[++i];
                save_to_file = true;
            }
        } else if (arg == "--help" || arg == "-h") {
            Logger::info("Usage: {} [options]", argv[0]);
            Logger::info("Options:");
            Logger::info("  --model, -m <path>    Model file path (default: transformer_model_full.pth)");
            Logger::info("  --output, -o <path>    Output file path for model information");
            Logger::info("  --help, -h             Show this help message");
            return 0;
        }
    }
    
    Logger::info("Model file: {}", model_path);
    if (save_to_file) {
        Logger::info("Output file: {}", output_file);
    }
    
    // ========================================================================
    // 2. Create model configuration
    // ========================================================================
    ModelConfig cfg;
    cfg.vocab_size = 50257;        // Vocab size (GPT-2 standard)
    cfg.max_seq_length = 1024;     // Max sequence length
    cfg.embedding_dim = 768;       // Embedding dimension (GPT-2 standard)
    cfg.n_heads = 12;              // Number of attention heads
    cfg.n_layers = 12;             // Number of Transformer layers
    cfg.drop_rate = 0.1;           // Dropout rate
    cfg.qkv_bias = false;          // No bias
    
    Logger::info("Model Configuration:");
    Logger::info("  - Vocab Size: {}", cfg.vocab_size);
    Logger::info("  - Max Sequence Length: {}", cfg.max_seq_length);
    Logger::info("  - Embedding Dimension: {}", cfg.embedding_dim);
    Logger::info("  - Number of Attention Heads: {}", cfg.n_heads);
    Logger::info("  - Number of Transformer Layers: {}", cfg.n_layers);
    Logger::info("  - Dropout Rate: {}", cfg.drop_rate);
    
    // ========================================================================
    // 3. Load model
    // ========================================================================
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("Loading model...");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    std::shared_ptr<GPTModel> model = std::make_shared<GPTModel>(cfg);
    
    try {
        torch::load(model, model_path);
        Logger::info("✓ Successfully loaded model from: {}", model_path);
    } catch (const std::exception& e) {
        Logger::error("✗ Failed to load model from {}: {}", model_path, e.what());
        Logger::info("Using untrained model for analysis");
    }
    
    // ========================================================================
    // 4. Analyze model
    // ========================================================================
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Logger::info("Analyzing model...");
    Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    ModelAnalyzer analyzer(model);
    analyzer.analyze();
    
    // ========================================================================
    // 5. Print model information
    // ========================================================================
    // 打印模型架构图
    analyzer.printModelArchitectureDiagram();
    
    // 打印参数关系图
    analyzer.printParameterRelationshipDiagram();
    
    // 打印参数维度关系图
    analyzer.printParameterDimensionDiagram();
    
    // 打印参数计算流程图
    analyzer.printParameterComputationFlow();
    
    // 打印模型结构
    analyzer.printModelStructure();
    
    // 打印参数详细注释（包含公式）
    analyzer.printParameterDetailedAnnotations();
    
    // 打印参数说明
    analyzer.printParameterDescriptions();
    
    // 打印参数摘要
    analyzer.printParameterSummary();
    
    // 打印所有参数
    analyzer.printAllParameters();
    
    // 打印详细统计信息
    analyzer.printParameterStatistics();
    
    // ========================================================================
    // 6. Save model information to file (if requested)
    // ========================================================================
    if (save_to_file) {
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Logger::info("Saving model information to file...");
        Logger::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        analyzer.saveModelInfo(output_file);
    }
    
    Logger::info("═══════════════════════════════════════════════════════════════");
    Logger::info("Model analysis completed!");
    Logger::info("═══════════════════════════════════════════════════════════════");
    
    return 0;
}

