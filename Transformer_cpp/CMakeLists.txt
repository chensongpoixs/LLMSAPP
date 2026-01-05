cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Transformer)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)


# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 查找 libtorch
# 注意: 需要先下载 libtorch 并设置 CMAKE_PREFIX_PATH
# 例如: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ============================================================================
# nlohmann/json 库（用于 JSON 解析）
# ============================================================================
# 使用本地 json.hpp 文件（位于 tiktoken 目录下）
message(STATUS "nlohmann/json: Using local json.hpp from tiktoken directory")

# ============================================================================
# 正则表达式库选择（RE2 或 std::regex）
# ============================================================================
# 用户可以选择使用 RE2 库（完整 Unicode 支持）或 std::regex（标准库，有限 Unicode 支持）
# 使用方式: cmake -DUSE_RE2=ON ..  (使用 RE2，默认)
#          cmake -DUSE_RE2=OFF .. (使用 std::regex)

option(USE_RE2 "Use RE2 library for full Unicode regex support (recommended)" ON)

if(USE_RE2)
    message(STATUS "========================================")
    message(STATUS "Using RE2 library for regex support")
    message(STATUS "========================================")
    
    # ============================================================================
    # RE2 正则表达式库（用于完整的 Unicode 支持）
    # ============================================================================
    # RE2 CMake 配置选项（可在命令行通过 -D 参数覆盖）
    # 例如: cmake -DRE2_BUILD_TESTING=ON -DRE2_USE_ICU=ON ..
    
    # RE2 配置选项说明：
    # - RE2_BUILD_TESTING: 是否构建 RE2 测试（默认 OFF，推荐关闭以加快编译）
    # - RE2_BUILD_BENCHMARK: 是否构建 RE2 基准测试（默认 OFF）
    # - RE2_USE_ICU: 是否使用 ICU 库以支持完整 Unicode（默认 OFF，RE2 本身已支持基本 Unicode）
    # - BUILD_SHARED_LIBS: 是否构建动态库（默认 OFF，构建静态库）
    # - RE2_BUILD_FRAMEWORK: 在 Apple 平台构建为框架（默认 OFF）
    
    # 设置 RE2 的默认编译选项（可在命令行覆盖）
    option(RE2_BUILD_TESTING "Build RE2 tests" OFF)
    option(RE2_BUILD_BENCHMARK "Build RE2 benchmarks" OFF)
    option(RE2_USE_ICU "Use ICU library for full Unicode support" OFF)
    option(RE2_BUILD_FRAMEWORK "Build RE2 as framework on Apple platforms" OFF)
    
    # 如果使用 FetchContent，需要在 FetchContent_MakeAvailable 之前设置选项
    # 尝试查找系统安装的 RE2
    find_package(re2 QUIET)
    
    if(NOT re2_FOUND)
        # 如果未找到，尝试使用 FetchContent 下载
        include(FetchContent)
        
        # 设置 FetchContent 选项（在 Declare 之前）
        set(FETCHCONTENT_QUIET OFF)  # 显示下载进度
        
        FetchContent_Declare(
            re2
            GIT_REPOSITORY https://github.com/google/re2.git
            GIT_TAG main  # 使用 main 分支（或指定稳定版本标签）
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
        )
        
        # 在 MakeAvailable 之前设置 RE2 的 CMake 选项
        # 这些选项会被传递给 RE2 的 CMakeLists.txt
        set(RE2_BUILD_TESTING ${RE2_BUILD_TESTING} CACHE BOOL "Disable RE2 testing" FORCE)
        set(RE2_BUILD_BENCHMARK ${RE2_BUILD_BENCHMARK} CACHE BOOL "Disable RE2 benchmarks" FORCE)
        set(RE2_USE_ICU ${RE2_USE_ICU} CACHE BOOL "Use ICU for full Unicode" FORCE)
        set(RE2_BUILD_FRAMEWORK ${RE2_BUILD_FRAMEWORK} CACHE BOOL "Build as framework" FORCE)
        
        # 确保构建静态库（推荐）
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
        
        FetchContent_MakeAvailable(re2)
        message(STATUS "RE2: Downloaded and built from source")
        message(STATUS "RE2_BUILD_TESTING: ${RE2_BUILD_TESTING}")
        message(STATUS "RE2_BUILD_BENCHMARK: ${RE2_BUILD_BENCHMARK}")
        message(STATUS "RE2_USE_ICU: ${RE2_USE_ICU}")
    else()
        message(STATUS "RE2: Found system installation")
        message(STATUS "RE2 version: ${re2_VERSION}")
    endif()
    
    # 定义使用 RE2 的宏
    add_definitions(-DTIKTOKEN_USE_RE2)
    target_compile_definitions(transformer_utils PUBLIC TIKTOKEN_USE_RE2)
    
    # 链接 RE2 库
    set(RE2_LINK_LIBRARY re2::re2)
    
else()
    message(STATUS "========================================")
    message(STATUS "Using std::regex for regex support")
    message(STATUS "Note: std::regex has limited Unicode support")
    message(STATUS "========================================")
    
    # 不定义 TIKTOKEN_USE_RE2，代码将使用 std::regex
    # 不需要链接任何额外的库
    set(RE2_LINK_LIBRARY "")
endif()

# ============================================================================
# Transformer 核心模型实现（静态库）
# ============================================================================
set(TRANSFORMER_CORE_SOURCES
    ModelConfig.h
    NewGELU.h
    NewGELU.cpp
    FeedForwardNetwork.h
    FeedForwardNetwork.cpp
    MultiHeadAttention.h
    MultiHeadAttention.cpp
    TransformerBlock.h
    TransformerBlock.cpp
    GPTModel.h
    GPTModel.cpp
)

# 创建静态库
add_library(transformer_core STATIC ${TRANSFORMER_CORE_SOURCES})

# 设置静态库的包含目录
target_include_directories(transformer_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 链接 libtorch 到静态库
target_link_libraries(transformer_core PUBLIC "${TORCH_LIBRARIES}")

# 设置编译选项
set_property(TARGET transformer_core PROPERTY CXX_STANDARD 17)

 

# ============================================================================
# Transformer 工具类（静态库）
# ============================================================================
set(TRANSFORMER_UTILS_SOURCES
    Logger.h
    Logger.cpp
    TextDataset.h
    TextDataset.cpp
    TrainingUtils.h
    TrainingUtils.cpp
    Trainer.h
    Trainer.cpp
    Generator.h
    Generator.cpp
    ModelAnalyzer.h
    ModelAnalyzer.cpp
      Tiktoken.h
    Tiktoken.cpp
    TiktokenGPT2.h
    TiktokenGPT2.cpp
    json.hpp 
)

# 创建工具类静态库
add_library(transformer_utils STATIC ${TRANSFORMER_UTILS_SOURCES})

# 工具库依赖于核心库和 tiktoken 库
# 如果使用 RE2，则链接 RE2 库；否则只使用 std::regex（标准库，无需链接）
if(USE_RE2 AND RE2_LINK_LIBRARY)
    target_link_libraries(transformer_utils PUBLIC transformer_core ${RE2_LINK_LIBRARY})
    message(STATUS "transformer_utils: Linked with RE2")
else()
    target_link_libraries(transformer_utils PUBLIC transformer_core)
    message(STATUS "transformer_utils: Using std::regex (no additional library needed)")
endif()

# 设置编译选项
set_property(TARGET transformer_utils PROPERTY CXX_STANDARD 17)

# ============================================================================
# 可执行文件
# ============================================================================

# 推理程序（文本生成）
add_executable(Transformer_inference
    inference.cpp
)

# 训练程序（包含反向传播）
add_executable(Transformer_train
    train.cpp
)

# 模型信息工具（分析和显示模型参数）
add_executable(Transformer_model_info
    model_info.cpp
)

# Tiktoken 示例程序
add_executable(Transformer_tiktoken_example
    tiktoken_example.cpp
)

# 链接库到可执行文件
target_link_libraries(Transformer_inference PRIVATE transformer_utils)
target_link_libraries(Transformer_train PRIVATE transformer_utils)
target_link_libraries(Transformer_model_info PRIVATE transformer_utils)
target_link_libraries(Transformer_tiktoken_example PRIVATE transformer_utils)

# 设置编译选项
set_property(TARGET Transformer_inference PROPERTY CXX_STANDARD 17)
set_property(TARGET Transformer_train PROPERTY CXX_STANDARD 17)
set_property(TARGET Transformer_model_info PROPERTY CXX_STANDARD 17)
set_property(TARGET Transformer_tiktoken_example PROPERTY CXX_STANDARD 17)

# 在 Windows 上禁用某些警告
if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(transformer_core PUBLIC _CRT_SECURE_NO_WARNINGS) 
    target_compile_definitions(transformer_utils PUBLIC _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(Transformer_inference PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(Transformer_train PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(Transformer_model_info PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# 打印配置信息
message(STATUS "========================================")
message(STATUS "Build Configuration Summary")
message(STATUS "========================================")
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Regex library: ${USE_RE2}")
if(USE_RE2)
    message(STATUS "  - Using RE2 (full Unicode support)")
else()
    message(STATUS "  - Using std::regex (limited Unicode support)")
endif()
message(STATUS "========================================")
