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
 * Tiktoken Usage Example
 * 
 * 演示如何使用 C++ 版本的 tiktoken 库进行文本编码和解码
 * 
 * 参考: https://github.com/openai/tiktoken
 */

#include "Tiktoken.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>

int main(int argc, char* argv[]) {
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "Tiktoken C++ Example" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::endl;
    
    // 测试简单编码器
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "Test 1: Simple Encoding" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    
    auto enc = tiktoken::create_simple_encoding();
    std::cout << "Created encoding: " << enc->getName() << std::endl;
    std::cout << "Vocab size: " << enc->getVocabSize() << std::endl;
    std::cout << std::endl;
    
    // 测试文本
    std::string test_text = "hello world";
    std::cout << "Test text: \"" << test_text << "\"" << std::endl;
    
    // 编码
    std::vector<uint32_t> tokens = enc->encode(test_text);
    std::cout << "Encoded tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    std::cout << "Number of tokens: " << tokens.size() << std::endl;
    std::cout << std::endl;
    
    // 解码
    std::string decoded_text = enc->decode(tokens);
    std::cout << "Decoded text: \"" << decoded_text << "\"" << std::endl;
    
    // 验证往返
    if (decoded_text == test_text) {
        std::cout << "✓ Round-trip test passed!" << std::endl;
    } else {
        std::cout << "✗ Round-trip test failed!" << std::endl;
    }
    std::cout << std::endl;
    
    // 测试更多文本
    std::vector<std::string> test_texts = {
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "你好，世界！",
        "🚀 AI is amazing!"
    };
    
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "Testing Multiple Texts" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    
    for (const auto& text : test_texts) {
        std::cout << std::endl;
        std::cout << "Text: \"" << text << "\"" << std::endl;
        auto encoded = enc->encode(text);
        std::cout << "Tokens: " << encoded.size() << " tokens" << std::endl;
        std::cout << "Token IDs: [";
        for (size_t i = 0; i < std::min(encoded.size(), size_t(10)); ++i) {
            std::cout << encoded[i];
            if (i < std::min(encoded.size(), size_t(10)) - 1) {
                std::cout << ", ";
            }
        }
        if (encoded.size() > 10) {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;
        
        auto decoded = enc->decode(encoded);
        std::cout << "Decoded: \"" << decoded << "\"" << std::endl;
        std::cout << "Round-trip: " << (decoded == text ? "✓ Pass" : "✗ Fail") << std::endl;
    }
    
    std::cout << std::endl;
    // 测试 GPT-2 编码器
    std::cout << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "Test 2: GPT-2 Encoding" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    
    auto gpt2_enc = tiktoken::create_gpt2_encoding();
    std::cout << "Created GPT-2 encoding: " << gpt2_enc->getName() << std::endl;
    std::cout << "Vocab size: " << gpt2_enc->getVocabSize() << std::endl;
    std::cout << std::endl;
    
    // 测试 GPT-2 编码
    std::vector<std::string> gpt2_test_texts = {
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "GPT-2 is a language model."
    };
    
    for (const auto& text : gpt2_test_texts) {
        std::cout << "Text: \"" << text << "\"" << std::endl;
        auto encoded = gpt2_enc->encode(text);
        std::cout << "Tokens: " << encoded.size() << " tokens" << std::endl;
        std::cout << "Token IDs: [";
        for (size_t i = 0; i < std::min(encoded.size(), size_t(10)); ++i) {
            std::cout << encoded[i];
            if (i < std::min(encoded.size(), size_t(10)) - 1) {
                std::cout << ", ";
            }
        }
        if (encoded.size() > 10) {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;
        
        auto decoded = gpt2_enc->decode(encoded);
        std::cout << "Decoded: \"" << decoded << "\"" << std::endl;
        std::cout << "Round-trip: " << (decoded == text ? "✓ Pass" : "✗ Fail") << std::endl;
        std::cout << std::endl;
    }
    
    // 测试特殊 token
    std::cout << "Testing special token <|endoftext|>:" << std::endl;
    uint32_t eot_id = gpt2_enc->getSpecialTokenId("<|endoftext|>");
    if (eot_id != std::numeric_limits<uint32_t>::max()) {
        std::cout << "  Special token ID: " << eot_id << std::endl;
        std::vector<uint32_t> eot_tokens = {eot_id};
        std::string eot_decoded = gpt2_enc->decode(eot_tokens);
        std::cout << "  Decoded: \"" << eot_decoded << "\"" << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "Example completed!" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    
    return 0;
}

