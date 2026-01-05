# RE2 CMake 配置选项说明

## 📋 概述

RE2 支持多个 CMake 配置选项，允许您根据需求定制构建过程。本文档列出了所有可用的配置选项及其说明。

## 🔧 可用的 CMake 配置选项

### 1. **RE2_BUILD_TESTING**

**类型**: `BOOL`  
**默认值**: `OFF`  
**说明**: 是否构建 RE2 的测试套件

**使用示例**:
```bash
cmake -DRE2_BUILD_TESTING=ON ..
```

**推荐设置**: `OFF`（除非您需要运行 RE2 的测试）

---

### 2. **RE2_BUILD_BENCHMARK**

**类型**: `BOOL`  
**默认值**: `OFF`  
**说明**: 是否构建 RE2 的基准测试

**使用示例**:
```bash
cmake -DRE2_BUILD_BENCHMARK=ON ..
```

**推荐设置**: `OFF`（除非您需要运行性能基准测试）

---

### 3. **RE2_USE_ICU**

**类型**: `BOOL`  
**默认值**: `OFF`  
**说明**: 是否使用 ICU 库以支持完整的 Unicode 属性

**使用示例**:
```bash
cmake -DRE2_USE_ICU=ON ..
```

**注意**:
- RE2 本身已经支持基本的 Unicode 属性（`\p{L}`, `\p{N}` 等）
- 启用 ICU 可以提供更完整的 Unicode 支持，但需要安装 ICU 库
- 对于 GPT-2 tokenizer，基本 Unicode 支持已足够

**推荐设置**: `OFF`（除非需要完整的 ICU Unicode 支持）

---

### 4. **BUILD_SHARED_LIBS**

**类型**: `BOOL`  
**默认值**: `OFF`  
**说明**: 是否构建动态链接库（`.dll`/`.so`）而不是静态库（`.lib`/`.a`）

**使用示例**:
```bash
cmake -DBUILD_SHARED_LIBS=ON ..
```

**推荐设置**: `OFF`（静态库更便于分发，无需额外的 DLL）

---

### 5. **RE2_BUILD_FRAMEWORK**

**类型**: `BOOL`  
**默认值**: `OFF`  
**说明**: 在 Apple 平台（macOS/iOS）上将 RE2 构建为框架（`.framework`）

**使用示例**:
```bash
cmake -DRE2_BUILD_FRAMEWORK=ON ..
```

**推荐设置**: `OFF`（除非在 Apple 平台需要框架格式）

---

## 📝 在 CMakeLists.txt 中的配置

当前项目的 `CMakeLists.txt` 已经配置了这些选项的默认值：

```cmake
# 设置 RE2 的默认编译选项（可在命令行覆盖）
option(RE2_BUILD_TESTING "Build RE2 tests" OFF)
option(RE2_BUILD_BENCHMARK "Build RE2 benchmarks" OFF)
option(RE2_USE_ICU "Use ICU library for full Unicode support" OFF)
option(RE2_BUILD_FRAMEWORK "Build RE2 as framework on Apple platforms" OFF)
```

## 🎯 使用示例

### 基本配置（推荐）

```bash
cd Transformer_cpp
mkdir build && cd build
cmake ..
cmake --build .
```

这将使用默认配置（关闭测试和基准测试，使用静态库）。

### 启用 ICU 支持

如果需要完整的 Unicode 支持（需要先安装 ICU 库）：

```bash
# Linux
sudo apt-get install libicu-dev

# macOS
brew install icu4c

# 然后配置
cmake -DRE2_USE_ICU=ON ..
cmake --build .
```

### 构建测试和基准测试

如果需要运行 RE2 的测试和基准测试：

```bash
cmake -DRE2_BUILD_TESTING=ON -DRE2_BUILD_BENCHMARK=ON ..
cmake --build .
```

### 构建动态库

如果需要构建动态链接库：

```bash
cmake -DBUILD_SHARED_LIBS=ON ..
cmake --build .
```

## 🔍 查看当前配置

在 CMake 配置阶段，会显示 RE2 的配置信息：

```
-- RE2: Downloaded and built from source
-- RE2_BUILD_TESTING: OFF
-- RE2_BUILD_BENCHMARK: OFF
-- RE2_USE_ICU: OFF
```

## ⚙️ 高级配置

### 使用 CMake GUI

1. 打开 CMake GUI
2. 设置源代码目录和构建目录
3. 点击 "Configure"
4. 在变量列表中查找 `RE2_*` 开头的选项
5. 修改所需选项
6. 点击 "Generate"

### 在 CMakeLists.txt 中硬编码

如果需要永久设置某些选项，可以在 `CMakeLists.txt` 中修改：

```cmake
# 永久启用 ICU 支持
set(RE2_USE_ICU ON CACHE BOOL "Use ICU for full Unicode" FORCE)
```

## 📊 配置选项对比

| 选项 | 默认值 | 推荐值 | 影响 |
|------|--------|--------|------|
| `RE2_BUILD_TESTING` | OFF | OFF | 编译时间、二进制大小 |
| `RE2_BUILD_BENCHMARK` | OFF | OFF | 编译时间、二进制大小 |
| `RE2_USE_ICU` | OFF | OFF | Unicode 支持完整性、依赖项 |
| `BUILD_SHARED_LIBS` | OFF | OFF | 库类型、分发方式 |
| `RE2_BUILD_FRAMEWORK` | OFF | OFF | 平台特定格式 |

## 🐛 故障排除

### 问题 1: ICU 未找到

**症状**: 配置时提示找不到 ICU

**解决方案**:
```bash
# 安装 ICU 库
sudo apt-get install libicu-dev  # Linux
brew install icu4c              # macOS

# 或禁用 ICU
cmake -DRE2_USE_ICU=OFF ..
```

### 问题 2: 编译时间过长

**症状**: RE2 编译时间很长

**解决方案**:
```bash
# 禁用测试和基准测试
cmake -DRE2_BUILD_TESTING=OFF -DRE2_BUILD_BENCHMARK=OFF ..
```

### 问题 3: 链接错误

**症状**: 链接时找不到 RE2 库

**解决方案**:
- 检查 `target_link_libraries` 是否包含 `re2::re2`
- 确保 RE2 已正确构建
- 检查库文件路径

## 📚 参考资源

- [RE2 GitHub](https://github.com/google/re2)
- [RE2 CMakeLists.txt](https://github.com/google/re2/blob/main/CMakeLists.txt)
- [CMake Options 文档](https://cmake.org/cmake/help/latest/manual/cmake.1.html#options)

---

**最后更新**：2026-01-01  
**版本**：1.0

