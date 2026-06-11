# vllm使用实战


## vLLM 具有哪些特点
• 最先进的服务吞吐量；
• PagedAttention 可以有效的管理注意力的键和值；
• 动态批处理请求；
• 优化好的 CUDA 内核；
• 与流行的 HuggingFace 模型无缝集成；
• 高吞吐量服务与各种解码算法，包括并行采样、beam search 等等；
• 张量并行以支持分布式推理；
• 流输出；
• 兼容 OpenAI 的 API 服务。

## vllm参数详解
  vllm -version # 查看vllm版本
  --model # 模型路径
  --tensor-parallel-size # 张量并行大小（TP），作用是将单层内的权重切分到多个 GPU 上并行计算，提高单步推理速度，默认值为 1
  --pipeline-parallel-size # 流水线并行大小（PP），作用是将模型的不同层切分到多个 GPU 上，形成流水线执行，适合超大模型跨多卡部署，默认值为 1
  --max-model-len # 最大模型长度，作用是限制模型的最大输入长度，默认值为2048
  --quantization # 量化类型，作用是将模型参数量化为低精度，减少模型大小和内存占用
  --dtype # 数据类型，作用是指定模型的计算数据类型，如half、float16、bfloat16、float32等，默认值为float16
  --gpu-memory-utilization # GPU内存利用率，作用是指定模型在GPU上的内存占用比例，默认值为0.9，建议根据实际情况调整
  --max-num-seqs # 最大序列数，作用是指定模型在一次推理中最多可以处理的序列数，默认值为128，建议根据实际情况调整
  --enforce-eager # 强制 eager 模式，作用是强制使用 eager 模式，而不是使用 lazy 模式，默认值为False，建议根据实际情况调整
  --max-num-batched-tokens # 最大批量令牌数，作用是指定模型在一次推理中最多可以处理的批量令牌数，默认值为2048，建议根据实际情况调整
  --host # 主机地址，作用是指定模型的主机地址，默认值为0.0.0.0，建议根据实际情况调整
  --port # 端口号，作用是指定模型的端口号，默认值为8000，建议根据实际情况调整
  --max-num-requests # 最大请求数，作用是指定模型在一次推理中最多可以处理的请求数，默认值为128，建议根据实际情况调整
  --enable-prefix-caching # 启用前缀缓存，作用是自动缓存并复用请求间共享前缀（如 system prompt）的 KV 缓存，避免重复计算，显著降低首 token 延迟并提升吞吐量，适合高并发生产场景
  --speculative-config # 投机解码配置，让草稿模型提前预测多个 token，再由主模型并行验证，不损失精度同时大幅提升推理速度与吞吐量。以下是五种主流种类及示例：

  1. Draft Model（草稿模型）—— 最常见方式
     原理：使用一个同 tokenizer 的小模型作为草稿模型，提前生成候选 token，主模型一次性验证。
     示例：
     --speculative-config '{"method": "draft_model", "model": "Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 5}'
     参数说明：
       - method: "draft_model"
       - model: 草稿模型路径（必须与主模型 tokenizer 一致）
       - num_speculative_tokens: 每次投机生成的 token 数（建议 3~8）

  2. N-gram（N 元语法匹配）—— 无需额外模型
     原理：从 prompt 或已生成文本中匹配 N-gram 作为候选 token 序列，零额外显存开销。
     示例：
     --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'
     参数说明：
       - method: "ngram"
       - num_speculative_tokens: 投机 token 数
       - prompt_lookup_max: 最大 N-gram 匹配长度

  3. EAGLE（特征级投机解码）—— 精度更高
     原理：在模型顶层 feature 上训练一个轻量预测头，直接预测下一层 feature，再利用原模型 head 解码 token。
     示例：
     --speculative-config '{"method": "eagle", "model": "yuhuili/EAGLE-Qwen2.5-7B-Instruct", "num_speculative_tokens": 4}'
     参数说明：
       - method: "eagle"
       - model: EAGLE 专用模型路径
       - num_speculative_tokens: 投机 token 数

  4. Medusa（多头并行预测）—— 一次预测多个 token
     原理：在模型顶层添加多个独立预测头，各自预测未来不同位置的 token，一次前向推理即可产出多个候选 token。
     示例：
     --speculative-config '{"method": "medusa", "model": "FasterDecoding/medusa-vicuna-7b-v1.3", "num_speculative_tokens": 4}'
     参数说明：
       - method: "medusa"
       - model: Medusa 权重路径
       - num_speculative_tokens: 投机 token 数

  5. MTP（多 Token 预测）—— 模型原生支持，无需额外模型
     原理：模型自身在训练时就具备多 token 预测头（如 DeepSeek-V3），单次前向直接产出多个未来 token，无需草稿模型，精度无损。
     示例：
     --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
     参数说明：
       - method: "mtp"
       - num_speculative_tokens: 投机 token 数（取决于模型 MTP 头数量，DeepSeek-V3 建议 1~2）

  通用建议：Draft Model 最成熟通用；N-gram 适合 prompt 前缀重复率高的场景（如代码补全）；EAGLE/Medusa 精度更高但需特定模型权重；MTP 精度无损、零额外延迟，但仅限 DeepSeek-V3 等原生支持的模型。

## 并行策略：张量并行 vs 流水线并行

### 核心区别

| 维度 | 张量并行（TP） | 流水线并行（PP） |
| --- | --- | --- |
| 切分粒度 | 同一层内的矩阵/注意力头 | 不同 Transformer 层 |
| 通信模式 | 层内 AllReduce，通信频繁 | 层间点对点传递激活值，通信较少 |
| 主要收益 | 降低单卡显存、加速单层计算 | 突破单卡装不下整层的限制，支持更大模型 |
| 典型瓶颈 | 多卡间带宽要求高 | 流水线气泡（bubble）导致部分 GPU 空闲 |
| 适用场景 | 7B~70B 模型，GPU 间 NVLink 互联 | 70B+ 超大模型，或 GPU 数量 > 单卡可承载层数 |

### 流水线并行原理

将模型的 N 个 Transformer 层按顺序切分到 P 个 GPU 上，每个 GPU 负责一段连续层。推理时，不同 micro-batch 在不同 GPU 上同时处于不同层，形成「流水线」：

```
GPU0: [Layer 0-7]   →  seq1 → seq2 → seq3
GPU1: [Layer 8-15]  →        seq1 → seq2 → seq3
GPU2: [Layer 16-23] →               seq1 → seq2 → seq3
GPU3: [Layer 24-31] →                      seq1 → seq2 → seq3
```

- **Prefill 阶段**：prompt 较长时，各 GPU 依次处理，存在启动/排空气泡。
- **Decode 阶段**：每次只生成 1 个 token，流水线利用率较低，通常与 TP 组合使用效果更好。

### 使用实例

**场景 1：4 卡部署 70B 模型（TP=2 + PP=2，共 4 GPU）**

单层权重过大，2 卡 TP 仍装不下全部层，用 PP 将 80 层拆成两段：

```bash
vllm serve /path/to/Llama-3-70B-Instruct \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9
```

- GPU 0~1：负责 Layer 0~39（TP 切分）
- GPU 2~3：负责 Layer 40~79（TP 切分）

**场景 2：8 卡部署 DeepSeek-V3 671B（TP=8，纯张量并行）**

GPU 间 NVLink 带宽充足、模型支持良好时，优先纯 TP：

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --max-model-len 32768 \
  --dtype bfloat16
```

**场景 3：16 卡部署超大 MoE 模型（TP=4 + PP=4）**

MoE 模型单层显存极大，TP 4 卡仍不够，再叠 PP 4 段：

```bash
vllm serve /path/to/DeepSeek-V3-671B \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 4 \
  --max-model-len 16384 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.95
```

**场景 4：Autodl 单机 4×4090 跑 32B 量化模型（纯 TP，不用 PP）**

32B INT4 量化后单卡约 20GB，4 卡 TP 足够，PP 反而增加延迟：

```bash
vllm serve /root/autodl-tmp/models/tclf90/deepseek-r1-distill-qwen-32b-gptq-int4 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --max-model-len 8192 \
  --quantization gptq \
  --dtype half
```

### 选型建议

- **GPU 数量 ≤ 模型所需 TP 数**：只用 TP，延迟最低。
- **单层权重 > 单卡显存**：TP + PP 组合，满足 `TP × PP = 总 GPU 数`。
- **Decode 延迟敏感**（在线对话）：尽量少用 PP，优先量化 + TP。
- **吞吐量优先**（离线批处理）：可适当增大 PP，换取更大 batch。

## KV Cache 原理

### 为什么需要 KV Cache

Transformer 自回归生成时，每产生一个新 token，都要对**所有历史 token** 做 Attention 计算。若不缓存，每步都要重新计算全部层的 Key（K）和 Value（V），复杂度为 O(n²)。

**KV Cache** 的核心思路：每生成一个 token 后，将其各层计算出的 K、V 向量**缓存到 GPU 显存**，下一步只计算新 token 的 Q、K、V，再用新 Q 与**历史 K、V** 做 Attention。

```
Step 1 (Prefill):  "你好" → 计算 K₁,V₁, K₂,V₂，全部写入 Cache
Step 2 (Decode):   生成 "，" → 只算 K₃,V₃，Attention(Q₃, [K₁,K₂,K₃], [V₁,V₂,V₃])
Step 3 (Decode):   生成 "我" → 只算 K₄,V₄，Attention(Q₄, [K₁..K₄], [V₁..V₄])
```

### 有 KV Cache vs 无 KV Cache

以生成 `"你好，我是 AI"`（共 6 个 token）为例，对比每步的计算量：

**无 KV Cache（每步全量重算）**

每生成 1 个新 token，都要把当前全部 token 重新过一遍所有 Transformer 层，历史 token 的 K、V 每步重复计算：

```
Step 1: 输入 "你"           → 计算 1 个 token 的 K,V
Step 2: 输入 "你好"         → 重新计算 2 个 token 的 K,V（"你" 被算了第 2 次）
Step 3: 输入 "你好，"       → 重新计算 3 个 token 的 K,V（"你""好" 各被算了第 3、2 次）
...
Step 6: 输入完整 6 token    → 重新计算 6 个 token 的 K,V

总计算量 = 1 + 2 + 3 + 4 + 5 + 6 = 21 次 token·层 的前向
         ≈ O(n²)，n 为序列长度
```

**有 KV Cache（增量计算）**

Prefill 阶段一次性算完 prompt，Decode 阶段每步只算 1 个新 token：

```
Step 1 (Prefill):  "你好"   → 计算 K₁,V₁, K₂,V₂，写入 Cache
Step 2 (Decode):   生成 "，" → 只算 K₃,V₃，从 Cache 读取 K₁,K₂,V₁,V₂
Step 3 (Decode):   生成 "我" → 只算 K₄,V₄，从 Cache 读取 K₁..K₃,V₁..V₃
Step 4 (Decode):   生成 "是" → 只算 K₅,V₅
Step 5 (Decode):   生成 " AI"→ 只算 K₆,V₆

总计算量 = 2（Prefill）+ 4（Decode 每步 1 个）= 6 次 token·层 的前向
         ≈ O(n)，n 为序列长度
```

| 对比维度 | 无 KV Cache | 有 KV Cache |
| --- | --- | --- |
| **每步计算** | 重算全部历史 token 的 K、V | 只算新 token 的 K、V，历史从 Cache 读取 |
| **时间复杂度** | O(n²)（n = 当前序列长） | O(n) |
| **显存占用** | 低（不存历史 K、V） | 高（需为每个 token 存 K、V） |
| **生成速度** | 极慢，序列越长越慢 | 快，Decode 每步耗时基本恒定 |
| **适用场景** | 几乎不用于推理；训练时反向传播需要全量计算 | 所有生产级推理框架的默认方案（vLLM、TGI 等） |

**直观对比（Llama-3-8B，生成 512 token）**

| 指标 | 无 KV Cache | 有 KV Cache |
| --- | --- | --- |
| 总前向次数 | 1+2+…+512 ≈ **13 万次** token·层 | 512 次 token·层 |
| 相对速度 | 基准 1× | 约 **250×** 更快 |
| 额外显存 | 0 | 约 **256 MB** / 请求（512 token） |

**代价与权衡**

- **有 Cache 的代价**：用显存换速度。并发越高、序列越长，KV Cache 占用的显存越大（见下方估算公式）。
- **无 Cache 几乎不可用于推理**：仅极短序列（如 n < 8）的 demo 或可忽略；生产环境必须开启 KV Cache。
- **vLLM 的优化方向**：在「必须有 KV Cache」的前提下，用 PageAttention 减少 Cache 的显存浪费，用 Continuous Batching 提高 Cache 的复用率。

### 显存占用估算

单个请求的 KV Cache 显存 ≈ `2 × num_layers × hidden_size × seq_len × dtype_bytes`

以 Llama-3-8B（32 层，hidden=4096，FP16）为例，seq_len=4096 时：

```
2 × 32 × 4096 × 4096 × 2 bytes ≈ 2 GB / 请求
```

并发 128 个请求就需要约 **256 GB** 纯 KV Cache 显存，这也是高并发推理的显存瓶颈所在。

### Prefill vs Decode 对 KV Cache 的不同需求

| 阶段 | 特点 | KV Cache 行为 |
| --- | --- | --- |
| Prefill | 一次性处理整个 prompt | 批量写入 Cache，计算密集 |
| Decode | 每次只生成 1 token | 增量追加 Cache，访存密集 |

vLLM 的 PagedAttention 和 Continuous Batching 正是针对这两个阶段分别优化吞吐与显存利用率（详见后文 PageAttention、Continuous Batching 章节）。

## PageAttention 原理与优点

### 传统 KV Cache 的问题

早期推理框架为每个请求**预分配一段连续显存**（按 `max_seq_len` 上限），存在两大问题：

1. **内部碎片**：实际序列长 512，却按 4096 分配，浪费 ~87% 空间。
2. **外部碎片**：请求长短不一，释放后产生大量不连续空闲块，无法复用。

### PageAttention 核心思想

借鉴操作系统**虚拟内存分页**：将 KV Cache 切分为固定大小的 **Block**（如每 block 存 16 个 token 的 K/V），通过 **Block Table** 映射到物理显存块，逻辑上连续、物理上可不连续。

```
请求 A Block Table: [物理块3] → [物理块7] → [物理块1]   (seq_len=40)
请求 B Block Table: [物理块5] → [物理块2]               (seq_len=25)
```

### 主要优点

| 优点 | 说明 |
| --- | --- |
| **显存利用率接近 100%** | 按需分配 block，无 max_len 预分配浪费，同等显存可服务更多并发 |
| **消除内存碎片** | 固定大小 block 统一管理，释放后可立即被新请求复用 |
| **高效 Continuous Batching** | 请求完成后 block 立刻回收，新请求插入同 batch，无需等整批结束 |
| **原生支持 Prefix Caching** | 共享前缀（如相同 system prompt）的 block 可引用计数共享，避免重复计算 |
| **Copy-on-Write 共享** | 多请求共享前缀 block，仅在新 token 分叉时复制，极大节省显存 |
| **灵活抢占与迁移** | block 粒度小，支持请求优先级调度和显存不足时的优雅降级 |

### 与 `--enable-prefix-caching` 的关系

开启 `--enable-prefix-caching` 后，vLLM 在 PageAttention 的 block 层面对**相同 token 序列前缀**做哈希去重，多个请求命中同一前缀时直接复用 KV block，配合 `--max-num-seqs` 可显著提升 agent 场景（大量相同 system prompt）的吞吐量。

## Continuous Batching 原理

Continuous Batching（连续批处理，也称 **Iteration-level Batching** / 迭代级批处理）是 vLLM 的调度策略：**不按「整批请求」调度，而按「每一步 decode」调度**，让 GPU 始终尽量满负荷运行。

### 传统 Static Batching 的问题

传统框架将多个请求打包成固定 batch，等**整批全部生成完毕**才释放 GPU：

```
Batch = [请求A(100 token), 请求B(10 token), 请求C(50 token)]

t=0~10:   A、B、C 同时运行，GPU 满负荷
t=10:     B 完成 → slot 空出，但 A、C 还在跑，GPU 浪费
t=10~50:  只剩 A、C，利用率下降
t=50~100: 只剩 A，GPU 大量空闲
```

短请求会拖长整批等待时间，GPU 利用率随时间持续下降。

### Continuous Batching 如何工作

每个 iteration（每生成 1 个 token）重新组 batch：

- 某请求**生成完毕** → 立刻移出 batch，释放 KV Cache block
- **新请求随时插入**，无需等上一批全部结束

```
Iteration 1~9:  [A, B, C]          → 3 个请求同时 decode
Iteration 10:   B 完成 → [A, C, D]  ← D 新请求立刻补上 B 的空位
Iteration 11:   [A, C, D, E]       ← E 也插入
...
```

### Static vs Continuous 对比

| 对比维度 | Static Batching | Continuous Batching |
| --- | --- | --- |
| 调度粒度 | 按「整批请求」 | 按「每步 decode」 |
| 请求完成 | 等 batch 内最长请求结束 | 完成即退出，slot 立刻复用 |
| GPU 利用率 | 随请求陆续完成而下降 | 维持高位，新请求持续补位 |
| 适合场景 | 请求长度相近的离线批处理 | 高并发在线服务（长短请求混合） |
| 吞吐提升 | 基准 1× | 通常 **2~10×**（取决于长度差异） |

### 与 KV Cache、PageAttention 的配合

| 组件 | 作用 |
| --- | --- |
| **KV Cache** | 每步只算 1 个新 token，使逐步调度成为可能 |
| **PageAttention** | block 级管理 KV Cache，请求结束 block 立刻回收 |
| **Continuous Batching** | 调度层：决定每步哪些请求进 batch、何时插入/移除 |

三者配合：PageAttention 解决「显存怎么高效分配/回收」，Continuous Batching 解决「请求怎么动态进出 batch」。

### 相关 vLLM 参数

- `--max-num-seqs`：Continuous Batching 同时处理的最大并发序列数，越大吞吐越高，KV Cache 显存占用也越大
- `--max-num-batched-tokens`：单步推理最多处理的 token 总数（Prefill + Decode 合计），控制单步计算量上限

## 使用实例
  pip install vllm # 安装vllm
  pip install modelscope # 安装modelscope
  modelscope download --model tclf90/deepseek-r1-distill-qwen-32b-gptq-int4 # 下载模型
  ![](./images/down_models.png)

  vllm serve /root/autodl-tmp/models/tclf90/deepseek-r1-distill-qwen-32b-gptq-int4 --tensor-parallel-size 1 --max-mode
l-len 32768 --enforce-eager --quantization gptq --dtype half


  vllm serve /root/autodl-tmp/models/tclf90/deepseek-r1-distill-qwen-32b-gptq-int4 --tensor-parallel-size 1 --max-mode
l-len 4096 --quantization gptq --dtype half --gpu-memory-utilization 0.8 --max-num-seqs 8 --enforce-eager

  vllm serve /root/autodl-tmp/models/tclf90/deepseek-r1-distill-qwen-32b-gptq-int4  --tensor-parallel-size 1 \
  --max-model-len 1024 \
  --quantization gptq \
  --dtype half \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 2 \
  --enforce-eager
  

## SGLang 部署模型示例
pip install -U sglang

python -m sglang.launch_server \
--model-path deepseek-ai/DeepSeek-V4-Flash \
--tp 2 \
--context-length 262144 \
--quant fp8 \
--enable-cache-report \
--host 0.0.0.0 --port 30000

## VLLM 与 SGLang 区别
   SGLang 的 RadixAttention + prefix caching 对 agent 共享 prompt ⼯作负载⽐ vLLM 更友好。如果项⽬⾥ agent 调⽤密集，优先
考虑 SGLang；如果是混合⼯作负载或要 MTP（Multi-Token Prediction 多 token 预测）speculative decoding（推测解码，⼀次
预测多个 token 加速⽣成），优先 vLLM。
