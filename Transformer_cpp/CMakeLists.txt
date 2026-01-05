cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Transformer)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
target_link_libraries(transformer_utils PUBLIC transformer_core  )

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
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
