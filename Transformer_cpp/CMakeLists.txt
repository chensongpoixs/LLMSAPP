cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Transformer_demo)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
# 查找 libtorch
# 注意: 需要先下载 libtorch 并设置 CMAKE_PREFIX_PATH
# 例如: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



# 共享的源文件（模型实现）
file(GLOB model_source
    NewGELU.cpp
    NewGELU.h 
    FeedForwardNetwork.h
    FeedForwardNetwork.cpp
    MultiHeadAttention.h
    MultiHeadAttention.cpp
    TransformerBlock.h
    TransformerBlock.cpp
    GPTModel.h
    GPTModel.cpp
    ModelConfig.h
    Logger.h
    Logger.cpp
    TextDataset.h
    TextDataset.cpp
)

# 示例程序（推理）
add_executable(${PROJECT_NAME} 
    example.cpp
    ${model_source}
)

# 训练程序（包含反向传播）
add_executable(Transformer_train
    train.cpp
    ${model_source}
)

# 链接 libtorch
target_link_libraries(${PROJECT_NAME}  "${TORCH_LIBRARIES}")
target_link_libraries(Transformer_train  "${TORCH_LIBRARIES}")


set(AllFile 
    example.cpp
    train.cpp
    ${model_source}
)
foreach(fileItem ${AllFile})       
    # Get the directory of the source file
    get_filename_component(PARENT_DIR "${fileItem}" DIRECTORY)
	
    # Remove common directory prefix to make the group
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" GROUP "${PARENT_DIR}")
    # Make sure we are using windows slashes
    string(REPLACE "/" "\\" GROUP "${GROUP}")
	#message(status   "---> fileItem = ${fileItem}     GROUP ---->  ${GROUP}")
    # Group into "Source Files" and "Header Files"
    set(GROUP "${GROUP}")
    source_group("${GROUP}" FILES "${fileItem}")
endforeach()

# 设置编译选项
set_property(TARGET ${PROJECT_NAME}  PROPERTY CXX_STANDARD 17)
set_property(TARGET Transformer_train  PROPERTY CXX_STANDARD 17)

# 在 Windows 上禁用某些警告
if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# 打印配置信息
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")

