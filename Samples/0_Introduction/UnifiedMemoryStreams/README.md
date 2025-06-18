以下是你提供的 **UnifiedMemoryStreams - Unified Memory Streams** 示例的中文翻译：

---

# UnifiedMemoryStreams - 统一内存流（Unified Memory Streams）

## 示例简介

本示例展示了在单个 GPU 上，结合使用 OpenMP 和 CUDA 流（streams）操作统一内存（Unified Memory）的方式。

## 核心概念

* CUDA 系统集成（CUDA Systems Integration）
* OpenMP 并行编程
* cuBLAS 库调用
* 多线程处理（Multithreading）
* 统一内存（Unified Memory）
* CUDA 流和事件（CUDA Streams and Events）

## 支持的 SM 架构

（未具体列出，但通常支持 compute capability ≥ 6.0 的架构）

## 支持的操作系统

* Linux
* Windows

## 支持的 CPU 架构

* x86\_64（常见于 PC）
* armv7l（常见于嵌入式设备，如 Jetson）

## 涉及的 CUDA API

### [CUDA 运行时 API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

使用了以下函数：

* `cudaStreamDestroy`：销毁 CUDA 流
* `cudaFree`：释放 CUDA 内存
* `cudaMallocManaged`：分配统一内存
* `cudaStreamAttachMemAsync`：将内存异步附加到流
* `cudaSetDevice`：设置当前设备
* `cudaDeviceSynchronize`：同步当前设备
* `cudaStreamSynchronize`：同步指定流
* `cudaStreamCreate`：创建 CUDA 流
* `cudaGetDeviceProperties`：获取设备属性

## 构建与运行所需依赖

* [OpenMP](../../../README.md#openmp)：用于主机侧的多线程并行处理
* [UVM - Unified Virtual Memory](../../../README.md#uvm)：统一虚拟内存支持
* [cuBLAS](../../../README.md#cublas)：NVIDIA 提供的高性能 BLAS 库

## 前置条件

请下载安装与你系统平台对应的 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)。

确保上述依赖（OpenMP、UVM、cuBLAS）已正确安装。

## 参考资料（了解更多）

（此处通常会链接到 NVIDIA 的官方文档、博客或 CUDA 编程指南等）

---

如需进一步解释这个示例的原理、代码结构，或如何在你的 GPU 上运行它，也可以告诉我，我可以逐步说明。
