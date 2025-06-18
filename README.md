下面是将你提供的官方 CUDA Samples 项目的 README 内容翻译成中文后的版本：

---

# CUDA 示例代码集

本示例项目为 CUDA 开发者提供了众多示例，展示了 CUDA Toolkit 的各种功能。本版本支持 [CUDA Toolkit 12.9](https://developer.nvidia.com/cuda-downloads)。

---

## 发布说明

本节描述该 CUDA Samples 项目在 GitHub 上的发布说明。

### 更新日志

详见：[Revision History](./CHANGELOG.md)

---

## 开始使用

### 前置条件

请前往 [CUDA Toolkit 下载页面](https://developer.nvidia.com/cuda-downloads)，下载并安装适合你系统平台的版本。详细的系统要求和安装指导可参考：

* Linux 安装指南
* Windows 安装指南

---

## 获取 CUDA 示例代码

有两种方式：

1. **通过 git 克隆：**

   ```bash
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```

2. **下载 ZIP 压缩包：**
   在 GitHub 页面点击 “Download ZIP”，然后解压即可使用全部示例。

---

## 构建 CUDA 示例

### 构建方式

示例使用 CMake 管理，支持 Linux、Windows 以及交叉编译到 Tegra 设备。

#### Linux 上编译：

确保已安装 CMake（3.20+），如未安装可使用系统包管理器：

```bash
sudo apt install cmake
```

然后在示例根目录：

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

构建完成后，在相应目录中运行示例程序。

#### Windows 上编译：

需 Visual Studio 2019（16.5 以上）支持 CMake 项目导入，或命令行方式：

```bat
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
```

然后打开生成的 `CUDA_Samples.sln`，编译并在输出目录运行想要的示例。

---

### 启用 GPU 上调试

示例默认不开启 gpu 调试，以保持性能。如需使用 `cuda-gdb` 调试，编译时需加上 `-G`，并在 CMake 中设置：

```bash
cmake -DENABLE_CUDA_DEBUG=True ..
```

---

### 平台特定示例

部分示例依赖特定平台，如 Tegra，可通过 CMake 参数 `-DBUILD_TEGRA=True` 来启用。也支持交叉编译。

---

### 交叉编译到 Tegra

确保已安装 Tegra 的交叉编译工具链，使用：

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/toolchain-aarch64-linux.cmake \
  -DTARGET_FS=/path/to/target/system
make -j$(nproc)
```

编译完后将可执行文件复制到 Tegra 设备运行。

---

### 为汽车 Linux 平台构建

指定 CUDA include 和 lib 路径，如：

```bash
cmake \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_LIBRARY_PATH=/usr/local/cuda/orin/lib64/ \
  -DCMAKE_INCLUDE_PATH=/usr/local/cuda/orin/include \
  -DBUILD_TEGRA=True \
  ..
```

---

### QNX 平台说明

目前 QNX 的交叉编译支持尚未完全验证，可查看早期 12.8 版本以获取方法。

---

## 批量运行所有示例作为测试

虽然示例集不是完整测试套件，但可用于快速检查系统环境。仓库内提供 `run_tests.py` 脚本，支持如下参数：

| 参数           | 说明                  |
| ------------ | ------------------- |
| `--dir`      | 指定要搜索可执行文件的目录       |
| `--config`   | 指定测试配置 JSON 文件      |
| `--output`   | 指定保存 stdout 日志的输出目录 |
| `--parallel` | 并行运行示例的数量           |

示例用法：

```bash
cd build
make -j$(nproc)

cd ..
python3 run_tests.py \
  --output ./test \
  --dir ./build/Samples \
  --config test_args.json
```

若全部执行成功，将输出类似：

```
Test Summary:
Ran 199 test runs for 180 executables.
All test runs passed!
```

若有失败项，则会列出失败的示例及返回码。

---

## 示例代码列表

* **0. Introduction**：基础入门示例，演示关键概念和 CUDA 运行时 API。
* **1. Utilities**：工具示例，如查询设备信息、测带宽等。
* **2. Concepts and Techniques**：展示常用 CUDA 技术。
* **3. CUDA Features**：高级特性演示，如 Cooperative Groups、Dynamic Parallelism、CUDA Graphs 等。
* **4. CUDA Libraries**：展示如何使用 cuBLAS、cuFFT、cuSPARSE 等 CUDA 库。
* **5. Domain Specific**：领域特定示例（图形、金融、图像处理等）。
* **6. Performance**：性能优化相关示例。
* **7. libNVVM**：使用 NVVM IR 及运行时编译相关示例。

---

## 依赖说明

部分示例依赖第三方库，如 FreeImage、MPI、OpenGL、Vulkan、GLFW、OpenMP 等。具体依赖会在每个子示例的 README 中说明，缺失时该示例会自动跳过。

### 常见第三方依赖：

* **FreeImage**：Linux 可以通过包管理器安装；Windows 将 DLL 放入 `Common/FreeImage/Dist/x64` 中，并在 CMake 中设置 `FreeImage` 路径。
* **MPI**：Linux 可用系统包安装；Windows 可安装 MS-MPI SDK。
* **图形 API（DirectX/OpenGL/Vulkan）**：需安装相应 SDK。
* **GLFW**：Windows 下载预编译版本并在 CMake 中指定路径。
* **OpenMP**：Linux 通常自带，若使用 clang 则需额外安装 `libomp`。

---

## CUDA 特性支持

部分示例需要以下功能，需对应 GPU 或平台支持：

* **CUDA 回调 (CUFFT callback)**
* **动态并行（CDP）**
* **Cooperative Groups**（多 block、多设备）
* **cuBLAS/cuFFT/cuRAND/cuSPARSE/cuSOLVER 等库**
* **CUDA IPC（跨进程通信）**
* **NPP/NVGRAPH/NVJPEG 等库**
* **NVRTC（运行时编译）**
* **Stream 优先级**
* **统一虚拟内存 UVM**
* **FP16（半精度浮点）**
* **C++11 支持**
* **CMake**

---

## 贡献指南

欢迎反馈 issue 或建议，当前暂不接受外部 PR。代码风格遵循 Google C++ Style Guide。

* CUDA 编程指南
* NVIDIA 加速计算博客

---

希望这份中文翻译对你阅读与使用 CUDA 示例项目有帮助！如需进一步辅助（如某个示例的运行说明、示例代码解析等），随时告诉我 😊
