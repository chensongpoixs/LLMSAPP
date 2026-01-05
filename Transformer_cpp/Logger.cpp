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
 * 日志模块实现
 */

#include "Logger.h"
#include <algorithm>

// ANSI 颜色代码
#ifdef _WIN32
    // Windows 控制台颜色（需要启用虚拟终端处理）
    #define COLOR_RESET   "\033[0m"
    #define COLOR_DEBUG   "\033[36m"  // 青色
    #define COLOR_INFO    "\033[32m"  // 绿色
    #define COLOR_WARNING "\033[33m"  // 黄色
    #define COLOR_ERROR   "\033[31m"  // 红色
#else
    // Linux/Mac 颜色代码
    #define COLOR_RESET   "\033[0m"
    #define COLOR_DEBUG   "\033[36m"
    #define COLOR_INFO    "\033[32m"
    #define COLOR_WARNING "\033[33m"
    #define COLOR_ERROR   "\033[31m"
#endif

Logger::Logger() 
    : min_level_(LogLevel::DEBUG)
    , show_timestamp_(true)
    , show_level_(true)
    , file_output_enabled_(false)
    , use_colors_(true) {
}

Logger::~Logger() {
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::setLogLevel(LogLevel level) {
    min_level_ = level;
}

void Logger::setFileOutput(bool enabled, const std::string& filename) {
    file_output_enabled_ = enabled;
    
    if (enabled) {
        if (log_file_.is_open()) {
            log_file_.close();
        }
        log_file_.open(filename, std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "警告: 无法打开日志文件: " << filename << std::endl;
            file_output_enabled_ = false;
        }
    } else {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }
}

void Logger::setShowTimestamp(bool enabled) {
    show_timestamp_ = enabled;
}

void Logger::setShowLevel(bool enabled) {
    show_level_ = enabled;
}

std::string Logger::getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string Logger::getLevelString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO ";
        case LogLevel::WARNING: return "WARN ";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

std::string Logger::getLevelColor(LogLevel level) {
    if (!use_colors_) {
        return "";
    }
    
    switch (level) {
        case LogLevel::DEBUG:   return COLOR_DEBUG;
        case LogLevel::INFO:    return COLOR_INFO;
        case LogLevel::WARNING: return COLOR_WARNING;
        case LogLevel::ERROR:   return COLOR_ERROR;
        default:                return COLOR_RESET;
    }
}

void Logger::log(LogLevel level, const std::string& message) {
    // 检查日志级别
    if (level < min_level_) {
        return;
    }
    
    std::ostringstream log_line;
    
    // 构建日志行
    if (show_timestamp_) {
        log_line << "[" << getTimestamp() << "] ";
    }
    
    if (show_level_) {
        log_line << "[" << getLevelString(level) << "] ";
    }
    
    log_line << message;
    
    std::string log_message = log_line.str();
    
    // 输出到控制台（带颜色）
    if (use_colors_) {
        std::cout << getLevelColor(level) << log_message << COLOR_RESET << std::endl;
    } else {
        std::cout << log_message << std::endl;
    }
    
    // 输出到文件（不带颜色）
    if (file_output_enabled_ && log_file_.is_open()) {
        log_file_ << log_message << std::endl;
        log_file_.flush();
    }
}

