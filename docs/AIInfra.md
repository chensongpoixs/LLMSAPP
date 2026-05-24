
```

第一层：面试必备（2周内搞定）
├── Transformer推理全流程：prefill / decode / KV cache / 显存估算
├── 主流推理框架概念：vLLM / TensorRT-LLM / SGLang 各解决了什么问题
├── 量化基础：INT8/INT4/FP8，对称 vs 非对称量化，PER-CHANNEL vs PER-TOKEN
└── GPU显存组成：模型参数 / KV Cache / 激活值各占多少，如何估算

第二层：与面试官过招的核心（1个月）
├── CUDA编程实践：至少写一个简单的GEMM kernel
├── TensorRT部署一个开源模型（Qwen/LLaMA），记录性能数据
├── FlashAttention原理：为什么比分块计算快？（IO感知）
└── Continuous Batching / PagedAttention 原理

第三层：成为抢手货（持续）
├── 分布式推理：Tensor Parallelism / Pipeline Parallelism
├── GPU拓扑与通信：NVLink / NVSwitch / RDMA
├── 国产卡适配经验（华为昇腾等）
└── Speculative Decoding / Lookahead Decoding 等前沿加速技术
```

### AI推理部署工程师 · 核心能力栈

#### 🧱 第一层：面试必备（2周内速成）

这部分是高频考点，目标是清晰阐述原理并完成快速估算。

*   **Transformer推理全流程**
    *   **核心阶段**：区分 **Prefill（预填充）**与 **Decode（解码）** 阶段的计算特性。Prefill是计算密集型（Compute Bound），并行处理输入序列；Decode是内存密集型（Memory Bound），逐Token生成。
    *   **KV Cache**：理解其原理（缓存历史Key/Value，避免重复计算）及显存占用公式：$2 \times B \times L \times H \times D \times \text{精度字节}$。
*   **主流推理框架概念**
    *   **vLLM**：核心是解决**显存碎片化和KV Cache预留过大**问题，通过PagedAttention实现零浪费。
    *   **TensorRT-LLM**：NVIDIA官方方案，极致**算子融合与图优化**，插件丰富，紧密结合TensorRT生态。
    *   **SGLang**：强调**结构化生成与编程语言前端**，通过RadixAttention实现前缀缓存，调度效率高。
*   **模型量化基础**
    *   **精度选择**：理解 **INT8 / INT4 / FP8** 各自的数值范围和精度损失权衡。
    *   **量化范式**：区分**对称 vs 非对称量化**（看零点是否为0）；区分**Per-Tensor / Per-Channel / Per-Token** 量化粒度。
*   **GPU显存组成与估算**
    *   **三大显存占用**：模型参数（静态）、**KV Cache（动态，随Batch和Seqlen线性增长）**、激活值（临时）。
    *   **快速估算**：能对一个7B模型在特定精度、批处理大小下的显存瓶颈做口算评估。

---

#### ⚔️ 第二层：与面试官过招的核心（1个月攻关）

这部分是定薪关键，需要展示动手能力和对核心优化的深度理解。

*   **CUDA编程实践**
    *   **入门实战**：能手写一个简单的 **GEMM Kernel**（矩阵乘法），包含Grid/Block划分和共享内存优化思想。
    *   **优化方向**：线程协同、合并访存、Bank Conflict规避。
*   **TensorRT模型部署**
    *   **完整流程**：从ONNX导出（含Qwen/LLaMA）到TensorRT构建（Build），再到C++/Python运行时推理。
    *   **性能基准**：记录 **Latency（延迟）** 与 **Throughput（吞吐）** 数据，分析相对于原生PyTorch的加速比。
*   **FlashAttention原理**
    *   **核心洞见**：掌握**IO感知（IO-Aware）** 设计。通过Tiling分块和Online Softmax，在SRAM中完成全流程，避免将大型注意力矩阵写回HBM，显著降低显存读写量。
    *   **回答关键**：不能说“因为快所以快”，要说明是**硬件黑盒（SRAM/HBM带宽差）与算法重构**。
*   **Continuous Batching 与 PagedAttention**
    *   **Continuous Batching（连续批处理）**：不同于静态Batch，支持在每次Decode迭代中动态增删序列，榨干GPU利用率。
    *   **PagedAttention（分页注意力）**：借鉴操作系统虚拟内存思想，将KV Cache切分为Block存储，按需非连续映射，彻底解决显存碎片化。

---

#### 🚀 第三层：成为抢手货（长期深耕）

这部分体现技术视野的广度、深度和稀缺性。

*   **分布式推理**
    *   **Tensor Parallelism（张量并行）**：把单层权重切分到多卡，配合All-Reduce通信，降低单卡显存。
    *   **Pipeline Parallelism（流水线并行）**：按层切分到多卡，通过微批次流水线掩盖Bubble（空泡）时间。
*   **GPU拓扑与通信**
    *   **协议栈**：掌握单机多卡 **NVLink/NVSwitch** 与多机 **RDMA (InfiniBand/RoCE)** 的原理、带宽和延迟差异。
    *   **影响**：能分析不同并行策略下的通信瓶颈。
*   **国产卡适配经验**
    *   **平台迁移**：将模型迁移至**华为昇腾**等国产平台，使用CANN算子库进行算子适配与精度调优。
    *   **价值**：具备信创背景下的技术方案落地能力。
*   **投机采样**
    *   **原理**：使用小模型（Draft Model）快速生成候选Token序列，由大模型（Target Model）并行验证并修正，在不降低精度的前提下显著加速Decode过程。

---

### 📝 简历项目描述示例（可参考）

**项目名称：基于TensorRT-LLM的Qwen大模型高性能推理服务构建**
*   **技术栈**：TensorRT-LLM, CUDA C++, vLLM, C++ Triton Server, NVLink
*   **项目描述**：负责将开源Qwen-7B模型部署至A100集群，实现低延迟、高吞吐的在线推理API。
*   **核心职责与成果**：
    *   **模型优化**：完成模型导出与TensorRT引擎构建，利用FP8量化与图融合，模型推理延迟降低 **3倍** 以上。
    *   **服务调度**：基于Continuous Batching与PagedAttention机制优化显存分配，单卡并发处理能力提升 **5倍**，显存碎片率降至接近0%。
    *   **定制化开发**：编写CUDA Kernel实现自定义Post-Processing算子，减少Host-Device数据搬移开销。
    *   **性能调优**：通过分析GPU带宽利用率与SM占用率，优化Tensor Parallelism配置，实现双卡NVLink通信下近线性的加速比。


