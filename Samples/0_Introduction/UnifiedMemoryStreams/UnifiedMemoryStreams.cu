/\* 版权所有 (c) 2022，NVIDIA CORPORATION。保留所有权利。
\*

* 允许以源代码和二进制形式使用和再发行，是否修改均可，但需满足以下条件：
* * 源代码再发行必须保留上述版权声明、本条件列表以及以下免责声明。
* * 二进制形式再发行必须在提供的文档和/或其他材料中重现上述版权声明、本条件列表及以下免责声明。
* * 未经 NVIDIA CORPORATION 事先书面许可，不得使用 NVIDIA CORPORATION 名称或其贡献者的名称为本软件衍生产品进行推广或背书。
*
* 本软件由版权持有人按“原样”提供，不附带任何明示或暗示的保证，包括但不限于对适销性和特定用途适用性的暗示保证。
* 在任何情况下，无论是合同责任、严格责任还是侵权责任（包括疏忽或其他原因）下，版权持有人或贡献者均不对因使用本软件而导致的任何直接、间接、偶发、特殊、
* 示范性或后果性损害（包括但不限于采购替代商品或服务、使用损失、数据丢失或利润损失，或业务中断）承担责任，
* 即使已被告知发生此类损害的可能性。
  \*/



/*
 * 本示例实现了一个使用线程和 CUDA 流执行任务的简单消费者模型，
 * 所有数据存储在统一内存（Unified Memory）中，任务可由 CPU（主机）或 GPU（设备）处理
 */

// 系统头文件
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <vector>
#ifdef USE_PTHREADS
#include <pthread.h>  // 可选支持 POSIX 线程
#else
#include <omp.h>      // 默认使用 OpenMP 多线程
#endif
#include <stdlib.h>

// cuBLAS 线性代数库
#include <cublas_v2.h>

// CUDA 辅助函数（错误检查等）
#include <helper_cuda.h>

// 兼容 Windows 上缺失的 drand48() 函数
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
void   srand48(long seed) { srand((unsigned int)seed); }
double drand48() { return double(rand()) / RAND_MAX; }
#endif

const char *sSDKname = "UnifiedMemoryStreams";

// 定义一个简单的任务结构体（包含矩阵、向量、结果）
template <typename T>
struct Task {
    unsigned int size, id;
    T *data;     // 方阵数据
    T *result;   // 结果向量
    T *vector;   // 输入向量

    // 构造函数初始化指针为空
    Task() : size(0), id(0), data(NULL), result(NULL), vector(NULL) {}

    // 初始化构造函数：为矩阵和向量分配统一内存
    Task(unsigned int s) : size(s), id(0), data(NULL), result(NULL) {
        checkCudaErrors(cudaMallocManaged(&data, sizeof(T) * size * size));
        checkCudaErrors(cudaMallocManaged(&result, sizeof(T) * size));
        checkCudaErrors(cudaMallocManaged(&vector, sizeof(T) * size));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // 析构函数：释放内存
    ~Task() {
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaFree(data));
        checkCudaErrors(cudaFree(result));
        checkCudaErrors(cudaFree(vector));
    }

    // 为任务重新分配大小，初始化数据
    void allocate(const unsigned int s, const unsigned int unique_id) {
        id = unique_id;
        size = s;
        checkCudaErrors(cudaMallocManaged(&data, sizeof(T) * size * size));
        checkCudaErrors(cudaMallocManaged(&result, sizeof(T) * size));
        checkCudaErrors(cudaMallocManaged(&vector, sizeof(T) * size));
        checkCudaErrors(cudaDeviceSynchronize());

        // 初始化矩阵和向量随机值
        for (unsigned int i = 0; i < size * size; i++) {
            data[i] = drand48();
        }
        for (unsigned int i = 0; i < size; i++) {
            result[i] = 0.;
            vector[i] = drand48();
        }
    }
};

#ifdef USE_PTHREADS
// POSIX 线程使用的结构体参数
struct threadData_t {
    int             tid;
    Task<double>   *TaskListPtr;
    cudaStream_t   *streams;
    cublasHandle_t *handles;
    int             taskSize;
};

typedef struct threadData_t threadData;
#endif

// 主机端 GEMV 运算（行优先）
template <typename T>
void gemv(int m, int n, T alpha, T *A, T *x, T beta, T *result) {
    for (int i = 0; i < n; i++) {
        result[i] *= beta;
        for (int j = 0; j < n; j++) {
            result[i] += A[i * n + j] * x[j];
        }
    }
}

#ifdef USE_PTHREADS
// 多线程任务执行函数（线程入口）
void *execute(void *inpArgs) {
    threadData *dataPtr = (threadData *)inpArgs;
    int tid = dataPtr->tid;
    cudaStream_t *stream = dataPtr->streams;
    cublasHandle_t *handle = dataPtr->handles;

    for (int i = 0; i < dataPtr->taskSize; i++) {
        Task<double> &t = dataPtr->TaskListPtr[i];
        if (t.size < 100) {
            printf("任务[%d], 线程[%d] 在主机执行 (%d)\n", t.id, tid, t.size);
            checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.data, 0, cudaMemAttachHost));
            checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.vector, 0, cudaMemAttachHost));
            checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.result, 0, cudaMemAttachHost));
            checkCudaErrors(cudaStreamSynchronize(stream[0]));
            gemv(t.size, t.size, 1.0, t.data, t.vector, 0.0, t.result);
        } else {
            printf("任务[%d], 线程[%d] 在设备执行 (%d)\n", t.id, tid, t.size);
            double one = 1.0, zero = 0.0;
            checkCudaErrors(cublasSetStream(handle[tid + 1], stream[tid + 1]));
            checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.data, 0, cudaMemAttachSingle));
            checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.vector, 0, cudaMemAttachSingle));
            checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.result, 0, cudaMemAttachSingle));
            checkCudaErrors(cublasDgemv(handle[tid + 1], CUBLAS_OP_N, t.size, t.size, &one,
                                        t.data, t.size, t.vector, 1, &zero, t.result, 1));
        }
    }

    pthread_exit(NULL);
}
#else
// OpenMP 执行函数
template <typename T>
void execute(Task<T> &t, cublasHandle_t *handle, cudaStream_t *stream, int tid) {
    if (t.size < 100) {
        printf("任务[%d], 线程[%d] 在主机执行 (%d)\n", t.id, tid, t.size);
        checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.data, 0, cudaMemAttachHost));
        checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.vector, 0, cudaMemAttachHost));
        checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.result, 0, cudaMemAttachHost));
        checkCudaErrors(cudaStreamSynchronize(stream[0]));
        gemv(t.size, t.size, 1.0, t.data, t.vector, 0.0, t.result);
    } else {
        printf("任务[%d], 线程[%d] 在设备执行 (%d)\n", t.id, tid, t.size);
        double one = 1.0, zero = 0.0;
        checkCudaErrors(cublasSetStream(handle[tid + 1], stream[tid + 1]));
        checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.data, 0, cudaMemAttachSingle));
        checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.vector, 0, cudaMemAttachSingle));
        checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.result, 0, cudaMemAttachSingle));
        checkCudaErrors(cublasDgemv(handle[tid + 1], CUBLAS_OP_N, t.size, t.size, &one,
                                    t.data, t.size, t.vector, 1, &zero, t.result, 1));
    }
}
#endif

// 初始化一批随机大小的任务
template <typename T>
void initialise_tasks(std::vector<Task<T>> &TaskList) {
    for (unsigned int i = 0; i < TaskList.size(); i++) {
        int size = std::max((int)(drand48() * 1000.0), 64);
        TaskList[i].allocate(size, i);
    }
}

int main(int argc, char **argv) {
    cudaDeviceProp device_prop;
    int dev_id = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));

    // 检查是否支持统一内存
    if (!device_prop.managedMemory) {
        fprintf(stderr, "当前设备不支持 Unified Memory\n");
        exit(EXIT_WAIVED);
    }

    // 检查设备是否允许计算
    if (device_prop.computeMode == cudaComputeModeProhibited) {
        fprintf(stderr, "设备处于禁止模式，无法运行\n");
        exit(EXIT_WAIVED);
    }

    // 初始化随机种子
    int seed = (int)time(NULL);
    srand48(seed);

    const int nthreads = 4;

    // 创建 CUDA 流和 cuBLAS handle
    cudaStream_t *streams = new cudaStream_t[nthreads + 1];
    cublasHandle_t *handles = new cublasHandle_t[nthreads + 1];

    for (int i = 0; i < nthreads + 1; i++) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
        checkCudaErrors(cublasCreate(&handles[i]));
    }

    // 初始化任务
    unsigned int N = 40;
    std::vector<Task<double>> TaskList(N);
    initialise_tasks(TaskList);

    printf("开始在主机 / 设备上执行任务...\n");

#ifdef USE_PTHREADS
    // POSIX 多线程调度
    pthread_t threads[nthreads];
    threadData *InputToThreads = new threadData[nthreads];
    for (int i = 0; i < nthreads; i++) {
        checkCudaErrors(cudaSetDevice(dev_id));
        InputToThreads[i].tid = i;
        InputToThreads[i].streams = streams;
        InputToThreads[i].handles = handles;
        if ((TaskList.size() / nthreads) == 0) {
            InputToThreads[i].taskSize = (TaskList.size() / nthreads);
            InputToThreads[i].TaskListPtr = &TaskList[i * (TaskList.size() / nthreads)];
        } else {
            if (i == nthreads - 1) {
                InputToThreads[i].taskSize = (TaskList.size() / nthreads) + (TaskList.size() % nthreads);
                InputToThreads[i].TaskListPtr =
                    &TaskList[i * (TaskList.size() / nthreads) + (TaskList.size() % nthreads)];
            } else {
                InputToThreads[i].taskSize = (TaskList.size() / nthreads);
                InputToThreads[i].TaskListPtr = &TaskList[i * (TaskList.size() / nthreads)];
            }
        }
        pthread_create(&threads[i], NULL, &execute, &InputToThreads[i]);
    }
    for (int i = 0; i < nthreads; i++) {
        pthread_join(threads[i], NULL);
    }
#else
    // 使用 OpenMP 并行执行任务
    omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < TaskList.size(); i++) {
        checkCudaErrors(cudaSetDevice(dev_id));
        int tid = omp_get_thread_num();
        execute(TaskList[i], handles, streams, tid);
    }
#endif

    cudaDeviceSynchronize();

    // 清理资源
    for (int i = 0; i < nthreads + 1; i++) {
        cudaStreamDestroy(streams[i]);
        cublasDestroy(handles[i]);
    }

    std::vector<Task<double>>().swap(TaskList);

    printf("全部任务完成！\n");
    exit(EXIT_SUCCESS);
}
