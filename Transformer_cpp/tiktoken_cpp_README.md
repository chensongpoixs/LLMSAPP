# Tiktoken C++ 独立静态库模块

## 概述

`tiktoken_cpp` 是一个完全独立的 C++ 静态库模块，实现了 BPE (Byte Pair Encoding) tokenizer，参考 [OpenAI tiktoken](https://github.com/openai/tiktoken)。

## 特性

- ✅ **完全独立**：不依赖任何外部库（包括 libtorch）
- ✅ **可单独使用**：可以独立编译和链接
- ✅ **模块化设计**：清晰的接口和实现分离
- ✅ **支持 GPT-2**：内置 GPT-2 tokenizer 支持
- ✅ **可扩展**：支持自定义编码器和从文件加载配置

## 文件结构

```
tiktoken_cpp/
├── Tiktoken.h          # 核心编码器类头文件
├── Tiktoken.cpp        # 核心编码器实现
├── TiktokenGPT2.h      # GPT-2 配置头文件
└── TiktokenGPT2.cpp    # GPT-2 配置实现
```

## 编译

### 作为独立模块编译

```cmake
# 在 CMakeLists.txt 中
add_library(tiktoken_cpp STATIC 
    Tiktoken.h
    Tiktoken.cpp
    TiktokenGPT2.h
    TiktokenGPT2.cpp
)

target_include_directories(tiktoken_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET tiktoken_cpp PROPERTY CXX_STANDARD 17)
```

### 链接到其他项目

```cmake
# 链接 tiktoken_cpp 库
target_link_libraries(your_target PRIVATE tiktoken_cpp)
```

## 使用示例

### 基本使用

```cpp
#include "Tiktoken.h"

// 创建编码器
auto enc = tiktoken::create_simple_encoding();

// 编码文本
std::string text = "hello world";
std::vector<uint32_t> tokens = enc->encode(text);

// 解码 tokens
std::string decoded = enc->decode(tokens);
```

### GPT-2 编码器

```cpp
#include "Tiktoken.h"

// 创建 GPT-2 编码器
auto gpt2_enc = tiktoken::create_gpt2_encoding();

// 编码文本
std::vector<uint32_t> tokens = gpt2_enc->encode("Hello, world!");

// 解码
std::string text = gpt2_enc->decode(tokens);
```

### 从文件加载

```cpp
// 从 merges.txt 文件加载 GPT-2 配置
auto enc = tiktoken::load_encoding_from_file("merges.txt", "gpt2");
```

## API 参考

### Tiktoken 类

#### 构造函数
```cpp
Tiktoken(const std::string& name,
         const std::string& pat_str,
         const std::map<std::vector<uint8_t>, uint32_t>& mergeable_ranks,
         const std::map<std::string, uint32_t>& special_tokens = {});
```

#### 主要方法
- `std::vector<uint32_t> encode(const std::string& text, ...)` - 编码文本
- `std::string decode(const std::vector<uint32_t>& tokens)` - 解码 tokens
- `std::string getName() const` - 获取编码器名称
- `size_t getVocabSize() const` - 获取词汇表大小

### 工厂函数

```cpp
namespace tiktoken {
    // 创建简单编码器
    std::shared_ptr<Tiktoken> create_simple_encoding();
    
    // 创建 GPT-2 编码器
    std::shared_ptr<Tiktoken> create_gpt2_encoding();
    
    // 获取编码器（按名称）
    std::shared_ptr<Tiktoken> get_encoding(const std::string& encoding_name);
    
    // 为模型获取编码器
    std::shared_ptr<Tiktoken> encoding_for_model(const std::string& model_name);
    
    // 从文件加载编码器
    std::shared_ptr<Tiktoken> load_encoding_from_file(
        const std::string& config_file,
        const std::string& encoding_name);
}
```

## 依赖关系

```
tiktoken_cpp (独立模块)
    │
    ├─► 无外部依赖
    │
    └─► 可被其他模块链接
        ├─► transformer_utils (可选)
        └─► 其他项目 (可选)
```

## 编译要求

- C++17 或更高版本
- CMake 3.18 或更高版本
- 标准 C++ 库（无其他依赖）

## 许可证

与主项目相同的许可证。

