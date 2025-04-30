# tinyllama
## LLM-Micro

An efficient inference engine for On-Device LLM.

### 1、源码编译

#### 1.1 Linux Arm64服务器编译

建议环境：鲲鹏920服务器（Arm64）

```bash
# 下载代码
git clone git@gitee.com:mindspore/llm-micro.git

cd llm-micro/
cmake -S . -B build_linux -DCMAKE_BUILD_TYPE=Release # 执行cmake构建
cmake --build build_linux/ -j20 # 执行make编译,-j20指采用20线程编译，可按需修改。
```

当前编译出可执行文件：

- ./build_linux/test/ut/kernel/ut-kernel：算子库测试用例
- ./build_linux/test/ut/runtime/ut-llama：llama模型测试用例
- ./build_linux/test/ut/runtime/ut-offloader：Flash offload模块测试用例
- ./build_linux/test/ut/runtime/ut-parallel：线程池并行推理测试用例
- ./build_linux/test/ut/runtime/ut-rtutils：运行时工具函数（runtime utilization）测试用例

#### 1.2 Android手机交叉编译

> 注意：Android NDK发布件只支持x86_64架构，不支持Arm64，鲲鹏服务器无法交叉编译！

下载Android NDK [r26](https://developer.android.com/ndk/downloads?hl=zh-cn)

```bash
# 在开发机下载ndk
wget https://dl.google.com/android/repository/android-ndk-r26d-linux.zip

# 解压zip包
unzip android-ndk-r26d-linux.zip
```

编译：

```
export ANDROID_NDK=/path/to/tool/android-ndk-r26d

bash script/build_android.sh
```

执行：将ut用来通过adb拷贝到Android手机上执行，执行方式与服务器类型。

### 2、运行UT & PerfTest

llm-micro采用TDD开发模式，使用多种用例确保功能和性能符合预期。

#### 2.1 ut-kernel

ut-kernel采用python numpy生成标杆数据，用于比对算子计算结果，依赖pytest和numpy包。

```bash
# 推荐使用conda创建python虚拟环境，避免污染个人开发环境
conda create -n llm-micro-env python=3.10

# 激活新建的conda环境
conda activate llm-micro-env

pip install pytest numpy
```

生成标杆数据：

```bash
pytest test/ut/kernel/
```

执行ut-kernel

```
./build_linux/test/ut/kernel/ut-kernel
```

#### 2.2 ut-offloader

ut-offloader是Flash Offload模块用例。

执行ut-offloader:

```
./build_linux/test/ut/runtime/ut-offloader
```

#### 2.3 ut-parallel

ut-parallel是线程池并行计算用例，用于测试并行加速逻辑的正确性。

执行ut-parallel：

```
./build_linux/test/ut/runtime/ut-parallel
```

#### 2.4 ut-llama

ut-llama是llama推理用例，包含：

- LlamaTest.Load：测试模型权重加载（基于Tinyllama 1.1B）；
- LlamaTest.Chat：测试稠密模型推理（基于Tinyllama 1.1B）；
- LlamaTest.SparseChat：测试动态稀疏模型推理（基于Relullama 7B）；

- **稠密模型推理**：[TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

下载权重：（此处用简化的git-lfs下载，对于小模型比较方便；大模型推荐[huggingface-cli](https://huggingface.co/docs/huggingface_hub/guides/cli)下载）

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

下载完毕，使用[llama.cpp](https://github.com/ggerganov/llama.cpp)的Python转换脚本，生成gguf权重文件

```bash
git clone https://github.com/ggerganov/llama.cpp.git

cd llama.cpp

# 安装依赖包
pip install -r requirements.txt

# 执行tinyllama转换，执行结束在TinyLlama-1.1B-Chat-v1.0/生成ggml-model-f16.gguf
python convert.py /path/to/TinyLlama-1.1B-Chat-v1.0/ # 改成你环境的Tinyllama路径！
```

将文件链接到llm-micro指定目录（ut-llama只会加载ut/data/下的gguf）：

```bash
cd llm-micro
ln -s /path/to/ggml-model-f16.gguf test/data/tiny-llama-1.1b-f16.gguf # 改成你自己的gguf路径！
```

执行用例：

```
./build_linux/test/ut/runtime/ut-llama --gtest_filter=LlamaTest.Chat
```

- **稀疏模型推理**：Relu-Llama-7B

下载模型权重（注意，PowerIner已提供[gguf文件](https://huggingface.co/PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF)，此处直接下载，无需转换）：

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF # 此命令耗时数小时！
```

将文件链接到llm-micro指定目录（ut-llama只会加载ut/data/下的gguf）：

```bash
cd llm-micro
ln -s /path/to/ReluLLaMA-7B-PowerInfer-GGUF/llama-7b-relu.powerinfer.gguf test/data/llama-7b-relu.powerinfer.gguf # 改成你自己的gguf路径！
```

执行用例：

```bash
# 不带Flash Offload，仅动态稀疏加速
./build_linux/test/ut/runtime/ut-llama --gtest_filter=LlamaTest.Chat

# 动态稀疏加速 + Flash Offload
./build_linux/test/ut/runtime/ut-llama --gtest_filter=LlamaTest.SparseOffloaderChat
```

#### 2.5 perf_test

使用google benchmark框架，执行matmul算子在多种输入尺寸下的性能测试：

```bash
./build_linux/test/perf_test/bench_matmul
```

### 3、框架设计

llm-micro架构示意图：

![image-20240926222537835](doc/images//llm-micro-arch.png)

- 模型组件侧：参考Transformers库设计，采用解耦的Model/Tokenizers/Samplers三种核心部件，构成llm推理pipeline；
- 推理运行时层：提供内存、权重offload、线程任务等管理功能；
- 算子接口层：提供统一的算子API，屏蔽硬件差异，降低后续多硬件协同推理的开发成本；
- 算子库：包括CPU、GPU、NPU算子，提供高性能算子实现。

### 4、Todo List

#### 4.1 模型量化

- [ ] Q4量化推理：支持4bit量化，甚至更低比特
- [ ] convert脚本：当前复用llama.cpp转换脚本，需重新实现一套，包括gguf模型格式
- [ ] [T-MAC](https://github.com/microsoft/T-MAC)：使用查表替代浮点乘法计算

#### 4.2 模型泛化

- [ ] 支持qwen
- [ ] 支持mistral（MoE）
- [ ] 支持mamba

#### 4.3 稀疏加速

- [x] PowerInfer动态稀疏
- [ ] [MInference-1.0：利用attn三种基础稀疏模式的动态组合来加速

#### 4.4 投机推理

- [ ] [LookAhead](https://arxiv.org/abs/2312.12728)：基于Trie-tree查询候选词
- [ ] [LookAhead Decoding](https://github.com/hao-ai-lab/LookaheadDecoding)：基于雅克比矩阵并行解码
- [ ] 稀疏和投机技术的叠加

#### 4.5 KVCache压缩

- [x] SnapKV：已在探索中
- [ ] [MiniCache](https://arxiv.org/abs/2405.14366)
- [ ] MLA：改造Attention降低内存

#### 4.6 权重Offload/Swap

- [x] [llm in a flash](https://arxiv.org/abs/2312.11514)：已实现原型，待性能优化，如：使用线程池异步并行加载权重；
