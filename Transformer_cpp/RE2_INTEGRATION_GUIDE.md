# RE2 集成指南

## 📋 概述

本项目已集成 Google RE2 正则表达式库，以提供完整的 Unicode 支持。RE2 支持完整的 Unicode 属性类（如 `\p{L}`, `\p{N}` 等），这正是 GPT-2 tokenizer 所需的。

## ✅ 已完成的集成

### 1. **CMake 配置**
- ✅ 自动查找系统安装的 RE2
- ✅ 如果未找到，使用 `FetchContent` 自动下载和编译
- ✅ 自动链接 RE2 库到 `transformer_utils`

### 2. **代码集成**
- ✅ 在 `Tiktoken.h` 中添加了条件编译支持
- ✅ 在 `Tiktoken.cpp` 中实现了 RE2 版本的 `splitText()` 函数
- ✅ 保留了 `std::regex` 作为回退选项

### 3. **功能特性**
- ✅ 支持完整的 Unicode 属性类（`\p{L}`, `\p{N}`, `\p{S}` 等）
- ✅ 正则表达式编译缓存（只编译一次，多次使用）
- ✅ 自动错误处理和回退机制

## 🔧 使用方法

### 编译项目

RE2 会自动集成，无需手动操作：

```bash
cd Transformer_cpp
mkdir build && cd build
cmake ..
cmake --build .
```

如果系统未安装 RE2，CMake 会自动下载并编译。

### 代码使用

使用方式与之前完全相同，无需修改代码：

```cpp
#include "Tiktoken.h"

// 创建编码器
auto enc = tiktoken::create_gpt2_encoding();

// 编码文本（现在支持完整的 Unicode）
std::vector<uint32_t> tokens = enc->encode("Hello, 世界！🚀");

// 解码 tokens
std::string text = enc->decode(tokens);
```

## 📊 改进效果

### Unicode 支持对比

| 特性 | std::regex | RE2 |
|------|-----------|-----|
| `\p{L}` (字母) | ❌ 不支持 | ✅ 完整支持 |
| `\p{N}` (数字) | ❌ 不支持 | ✅ 完整支持 |
| `\p{S}` (符号) | ❌ 不支持 | ✅ 完整支持 |
| Unicode 字符 | ⚠️ 部分支持 | ✅ 完整支持 |
| 性能 | ⚠️ 中等 | ✅ 高性能 |

### 性能提升

- **正则表达式编译**：只编译一次，缓存结果
- **匹配速度**：RE2 使用 DFA，性能优于 `std::regex`
- **内存使用**：RE2 的内存使用更高效

## 🔍 技术细节

### 条件编译

代码使用条件编译，支持两种模式：

```cpp
#ifdef TIKTOKEN_USE_RE2
    // 使用 RE2（完整 Unicode 支持）
    RE2::FindAndConsume(&input, *regex_pattern_, &match);
#else
    // 回退到 std::regex（简化 Unicode 支持）
    std::regex pattern(regex_pattern);
#endif
```

### 正则表达式缓存

RE2 正则表达式对象被缓存，避免重复编译：

```cpp
if (!regex_compiled_ || !regex_pattern_) {
    regex_pattern_ = std::make_unique<RE2>(pat_str_);
    regex_compiled_ = true;
}
```

## 🐛 故障排除

### 问题 1: RE2 编译失败

**症状**：CMake 配置时出错

**解决方案**：
1. 确保 CMake 版本 >= 3.18
2. 确保有网络连接（用于下载 RE2）
3. 或者手动安装 RE2：
   ```bash
   # Linux
   sudo apt-get install libre2-dev
   
   # macOS
   brew install re2
   ```

### 问题 2: 链接错误

**症状**：链接时找不到 RE2 库

**解决方案**：
1. 检查 CMake 是否正确找到了 RE2
2. 确保 `target_link_libraries` 包含 `re2::re2`

### 问题 3: 运行时错误

**症状**：运行时正则表达式匹配失败

**解决方案**：
1. 检查正则表达式模式是否正确
2. 查看错误消息（RE2 提供详细的错误信息）
3. 代码会自动回退到简化分割

## 📝 配置选项

### 禁用 RE2（使用 std::regex）

如果不想使用 RE2，可以修改 `CMakeLists.txt`：

```cmake
# 注释掉或删除以下行
# add_definitions(-DTIKTOKEN_USE_RE2)
# target_compile_definitions(transformer_utils PUBLIC TIKTOKEN_USE_RE2)
```

代码会自动回退到 `std::regex`。

### 使用系统安装的 RE2

如果系统已安装 RE2，CMake 会自动使用：

```bash
# Linux
sudo apt-get install libre2-dev

# macOS  
brew install re2
```

## 🎯 下一步

1. **测试 Unicode 支持**：使用各种 Unicode 字符测试编码/解码
2. **性能测试**：对比 RE2 和 std::regex 的性能
3. **错误处理**：完善错误处理和日志记录

## 📚 参考资源

- [RE2 GitHub](https://github.com/google/re2)
- [RE2 文档](https://github.com/google/re2/wiki)
- [Unicode 属性类](https://www.regular-expressions.info/unicode.html)

---

**最后更新**：2026-01-01  
**版本**：1.0

