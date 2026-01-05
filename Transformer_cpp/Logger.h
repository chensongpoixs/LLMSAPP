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
 * 日志模块 (Logger)
 * 
 * 提供统一的日志记录功能，用于训练和推理过程中的信息输出。
 * 
 * 功能特性：
 * - 多日志级别：DEBUG, INFO, WARNING, ERROR
 * - 时间戳：自动添加时间戳
 * - 格式化输出：支持格式化字符串
 * - 控制台输出：默认输出到控制台
 * - 文件输出：可选，支持输出到文件
 * - 日志级别过滤：只输出指定级别及以上的日志
 * 
 * 使用示例：
 *   Logger::info("模型创建完成，参数量: {}", total_params);
 *   Logger::warning("CUDA 不可用，使用 CPU");
 *   Logger::error("训练失败: {}", error_msg);
 *   Logger::debug("批次 {} 处理完成", batch_idx);
 */

#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <memory>

/**
 * 日志级别枚举
 */
enum class LogLevel {
    DEBUG = 0,  // 调试信息
    INFO = 1,   // 一般信息
    WARNING = 2, // 警告信息
    ERROR = 3    // 错误信息
};

/**
 * 日志类
 * 
 * 单例模式，提供全局日志记录功能
 */
class Logger {
public:
    /**
     * 获取日志实例（单例）
     */
    static Logger& getInstance();
    
    /**
     * 设置日志级别
     * @param level: 最低日志级别，低于此级别的日志将被过滤
     */
    void setLogLevel(LogLevel level);
    
    /**
     * 设置是否输出到文件
     * @param enabled: 是否启用文件输出
     * @param filename: 日志文件名（可选，默认 "transformer.log"）
     */
    void setFileOutput(bool enabled, const std::string& filename = "transformer.log");
    
    /**
     * 设置是否显示时间戳
     * @param enabled: 是否显示时间戳
     */
    void setShowTimestamp(bool enabled);
    
    /**
     * 设置是否显示日志级别
     * @param enabled: 是否显示日志级别
     */
    void setShowLevel(bool enabled);
    
    /**
     * 记录日志（内部方法）
     * @param level: 日志级别
     * @param message: 日志消息
     */
    void log(LogLevel level, const std::string& message);
    
    // 静态便捷方法 - 格式化版本
    template<typename... Args>
    static void debug(const std::string& format, Args... args) {
        Logger& logger = getInstance();
        if (LogLevel::DEBUG >= logger.min_level_) {
            std::ostringstream oss;
            logger.formatStringImpl(oss, format, args...);
            logger.log(LogLevel::DEBUG, oss.str());
        }
    }
    
    template<typename... Args>
    static void info(const std::string& format, Args... args) {
        Logger& logger = getInstance();
        if (LogLevel::INFO >= logger.min_level_) {
            std::ostringstream oss;
            logger.formatStringImpl(oss, format, args...);
            logger.log(LogLevel::INFO, oss.str());
        }
    }
    
    template<typename... Args>
    static void warning(const std::string& format, Args... args) {
        Logger& logger = getInstance();
        if (LogLevel::WARNING >= logger.min_level_) {
            std::ostringstream oss;
            logger.formatStringImpl(oss, format, args...);
            logger.log(LogLevel::WARNING, oss.str());
        }
    }
    
    template<typename... Args>
    static void error(const std::string& format, Args... args) {
        Logger& logger = getInstance();
        if (LogLevel::ERROR >= logger.min_level_) {
            std::ostringstream oss;
            logger.formatStringImpl(oss, format, args...);
            logger.log(LogLevel::ERROR, oss.str());
        }
    }
    
    // 无参数版本
    static void debug(const std::string& message) {
        getInstance().log(LogLevel::DEBUG, message);
    }
    
    static void info(const std::string& message) {
        getInstance().log(LogLevel::INFO, message);
    }
    
    static void warning(const std::string& message) {
        getInstance().log(LogLevel::WARNING, message);
    }
    
    static void error(const std::string& message) {
        getInstance().log(LogLevel::ERROR, message);
    }

private:
    Logger();
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    /**
     * 格式化字符串实现（简单的 {} 替换）
     */
    template<typename T>
    void formatStringImpl(std::ostringstream& oss, const std::string& format, T value) {
        size_t pos = format.find("{}");
        if (pos != std::string::npos) {
            oss << format.substr(0, pos) << value << format.substr(pos + 2);
        } else {
            oss << format;
        }
    }
    
    template<typename T, typename... Args>
    void formatStringImpl(std::ostringstream& oss, const std::string& format, T value, Args... args) {
        size_t pos = format.find("{}");
        if (pos != std::string::npos) {
            oss << format.substr(0, pos) << value;
            formatStringImpl(oss, format.substr(pos + 2), args...);
        } else {
            oss << format;
        }
    }
    
    /**
     * 获取当前时间戳字符串
     */
    std::string getTimestamp();
    
    /**
     * 获取日志级别字符串
     */
    std::string getLevelString(LogLevel level);
    
    /**
     * 获取日志级别颜色代码（控制台）
     */
    std::string getLevelColor(LogLevel level);
    
    LogLevel min_level_;           // 最低日志级别
    bool show_timestamp_;          // 是否显示时间戳
    bool show_level_;              // 是否显示日志级别
    bool file_output_enabled_;     // 是否启用文件输出
    std::ofstream log_file_;       // 日志文件流
    bool use_colors_;              // 是否使用颜色（控制台）
};

#endif // LOGGER_H

