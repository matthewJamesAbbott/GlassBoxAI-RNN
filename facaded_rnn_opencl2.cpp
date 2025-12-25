//
// Facade RNN - OpenCL Version (Stage 1: Types, Macros, Utility)
// Ported from CUDA for GlassBoxAI-RNN
// Matthew Abbott 2025
//
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <random>
#include <memory>
#include <iomanip>
#include <random> // For host-side random


using namespace std;

// OpenCL error checking macro
#define CL_CHECK(err) \
    do { \
        if (err != CL_SUCCESS) { \
            cerr << "OpenCL error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << (int)err << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Type definitions
typedef std::vector<float> FArray;
typedef std::vector<FArray> TFArray2D;
typedef std::vector<TFArray2D> TFArray3D;

// Enum definitions (match rnn_opencl.cpp)
enum ActivationType { atSigmoid = 0, atTanh = 1, atReLU = 2, atLinear = 3 };
enum LossType { ltMSE = 0, ltCrossEntropy = 1 };
enum CellType { ctSimpleRNN = 0, ctLSTM = 1, ctGRU = 2 };
enum GateType { gtForget, gtInput, gtOutput, gtCellCandidate, gtUpdate, gtReset, gtHiddenCandidate };

// Utility functions
float ClipValue(float v, float maxVal) {
    if (v > maxVal) return maxVal;
    if (v < -maxVal) return -maxVal;
    return v;
}

float RandomInit(float scale) {
    return ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
}

// --------- OpenCL Kernels as string ---------
const char* kernelSource = R"CLC(
// Device-side utils
float d_sigmoid(float x) {
    float clamped = fmax(-500.0f, fmin(500.0f, x));
    return 1.0f / (1.0f + exp(-clamped));
}
float d_tanh_act(float x) { return tanh(x); }
float d_relu(float x) { return x > 0.0f ? x : 0.0f; }
float d_clip_value(float v, float maxVal) {
    if (v > maxVal) return maxVal;
    if (v < -maxVal) return -maxVal;
    return v;
}
float d_apply_activation(float x, int actType) {
    switch(actType) {
        case 0: return d_sigmoid(x);
        case 1: return d_tanh_act(x);
        case 2: return d_relu(x);
        case 3: return x;
        default: return x;
    }
}
float d_activation_derivative(float y, int actType) {
    switch(actType) {
        case 0: return y * (1.0f - y);
        case 1: return 1.0f - y * y;
        case 2: return y > 0.0f ? 1.0f : 0.0f;
        case 3: return 1.0f;
        default: return 1.0f;
    }
}

// Set to zero
__kernel void k_zero(__global float* arr, int n) {
    int i = get_global_id(0);
    if (i < n) arr[i] = 0.0f;
}

// Fill with value
__kernel void k_fill(__global float* arr, float val, int n) {
    int i = get_global_id(0);
    if (i < n) arr[i] = val;
}

// Vector addition
__kernel void k_vec_add(__global float* C, __global const float* A, __global const float* B, int n) {
    int i = get_global_id(0);
    if (i < n) C[i] = A[i] + B[i];
}

// Vector scale
__kernel void k_vec_scale(__global float* A, float scale, int n) {
    int i = get_global_id(0);
    if (i < n) A[i] *= scale;
}

// Matrix-vector multiply: y = W * x + b, then activation
__kernel void k_matvec_bias_act(__global float* y, __global const float* W,
                                __global const float* x, __global const float* b,
                                int rows, int cols, int actType) {
    int row = get_global_id(0);
    if (row < rows) {
        float sum = b[row];
        for (int j = 0; j < cols; j++) sum += W[row * cols + j] * x[j];
        y[row] = d_apply_activation(sum, actType);
    }
}
// Matrix-vector multiply: y = W * x + b
__kernel void k_matvec_bias(__global float* y, __global const float* W,
                            __global const float* x, __global const float* b,
                            int rows, int cols) {
    int row = get_global_id(0);
    if (row < rows) {
        float sum = b[row];
        for (int j = 0; j < cols; j++) sum += W[row * cols + j] * x[j];
        y[row] = sum;
    }
}

// Concatenate two vectors: out = [a, b]
__kernel void k_concat(__global float* out, __global const float* a, __global const float* b, int sizeA, int sizeB) {
    int i = get_global_id(0);
    if (i < sizeA) out[i] = a[i];
    else if (i < sizeA+sizeB) out[i] = b[i-sizeA];
}
)CLC";

// ======================= OpenCL Buffer Management Classes =======================
class CLArray {
public:
    cl_mem d_ptr;
    int size;
    cl_context context;
    cl_command_queue queue;

    CLArray(cl_context ctx, cl_command_queue q) : d_ptr(nullptr), size(0), context(ctx), queue(q) {}

    void allocate(int n) {
        cl_int err;
        if (d_ptr) clReleaseMemObject(d_ptr);
        d_ptr = (n > 0) ? clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), nullptr, &err) : nullptr;
        if (n > 0) CL_CHECK(err);
        size = n;
    }

    void free() {
        if (d_ptr) {
            clReleaseMemObject(d_ptr);
            d_ptr = nullptr;
            size = 0;
        }
    }

    void zero(cl_kernel k_zero_kernel) {
        if (d_ptr && size > 0) {
            size_t globalSize = ((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            clSetKernelArg(k_zero_kernel, 0, sizeof(cl_mem), &d_ptr);
            clSetKernelArg(k_zero_kernel, 1, sizeof(int), &size);
            cl_int err = clEnqueueNDRangeKernel(queue, k_zero_kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
            CL_CHECK(err);
            clFinish(queue);
        }
    }

    void fill(float val, cl_kernel k_fill_kernel) {
        if (d_ptr && size > 0) {
            size_t globalSize = ((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            clSetKernelArg(k_fill_kernel, 0, sizeof(cl_mem), &d_ptr);
            clSetKernelArg(k_fill_kernel, 1, sizeof(float), &val);
            clSetKernelArg(k_fill_kernel, 2, sizeof(int), &size);
            cl_int err = clEnqueueNDRangeKernel(queue, k_fill_kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
            CL_CHECK(err);
            clFinish(queue);
        }
    }

    void copyToDevice(const float* src, int n) {
        allocate(n);
        clEnqueueWriteBuffer(queue, d_ptr, CL_TRUE, 0, n * sizeof(float), src, 0, nullptr, nullptr);
    }

    void copyToDevice(const FArray& src) {
        copyToDevice(src.data(), (int)src.size());
    }

    void copyToHost(float* dst, int n) {
        if (!d_ptr || n <= 0) return;
        clEnqueueReadBuffer(queue, d_ptr, CL_TRUE, 0, n * sizeof(float), dst, 0, nullptr, nullptr);
    }

    void copyToHost(FArray& dst) {
        if (size <= 0) { dst.clear(); return; }
        dst.resize(size);
        clEnqueueReadBuffer(queue, d_ptr, CL_TRUE, 0, size * sizeof(float), dst.data(), 0, nullptr, nullptr);
    }

    ~CLArray() { free(); }
};

class CLMatrix {
public:
    cl_mem d_ptr;
    int rows, cols;
    cl_context context;
    cl_command_queue queue;

    CLMatrix(cl_context ctx, cl_command_queue q) : d_ptr(nullptr), rows(0), cols(0), context(ctx), queue(q) {}

    void allocate(int r, int c) {
        cl_int err;
        int n = r * c;
        if (d_ptr) clReleaseMemObject(d_ptr);
        d_ptr = (n > 0) ? clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), nullptr, &err) : nullptr;
        if (n > 0) CL_CHECK(err);
        rows = r; cols = c;
    }

    void free() {
        if (d_ptr) {
            clReleaseMemObject(d_ptr);
            d_ptr = nullptr;
            rows = cols = 0;
        }
    }

    void zero(cl_kernel k_zero_kernel) {
        if (d_ptr && rows * cols > 0) {
            int n = rows * cols;
            size_t globalSize = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            clSetKernelArg(k_zero_kernel, 0, sizeof(cl_mem), &d_ptr);
            clSetKernelArg(k_zero_kernel, 1, sizeof(int), &n);
            cl_int err = clEnqueueNDRangeKernel(queue, k_zero_kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
            CL_CHECK(err);
            clFinish(queue);
        }
    }

    void fill(float val, cl_kernel k_fill_kernel) {
        if (d_ptr && rows * cols > 0) {
            int n = rows * cols;
            size_t globalSize = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            clSetKernelArg(k_fill_kernel, 0, sizeof(cl_mem), &d_ptr);
            clSetKernelArg(k_fill_kernel, 1, sizeof(float), &val);
            clSetKernelArg(k_fill_kernel, 2, sizeof(int), &n);
            cl_int err = clEnqueueNDRangeKernel(queue, k_fill_kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
            CL_CHECK(err);
            clFinish(queue);
        }
    }

    void copyToDevice(const TFArray2D& src) {
        if (src.empty()) return;
        int r = (int)src.size(), c = (int)src[0].size();
        allocate(r, c);
        std::vector<float> flat(r * c);
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                flat[i * c + j] = src[i][j];
        clEnqueueWriteBuffer(queue, d_ptr, CL_TRUE, 0, flat.size() * sizeof(float), flat.data(), 0, nullptr, nullptr);
    }

    void copyToHost(TFArray2D& dst) {
        if (!d_ptr || rows <= 0 || cols <= 0) { dst.clear(); return; }
        dst.resize(rows);
        std::vector<float> flat(rows * cols);
        clEnqueueReadBuffer(queue, d_ptr, CL_TRUE, 0, flat.size() * sizeof(float), flat.data(), 0, nullptr, nullptr);
        for (int i = 0; i < rows; i++) {
            dst[i].resize(cols);
            for (int j = 0; j < cols; j++)
                dst[i][j] = flat[i * cols + j];
        }
    }

    ~CLMatrix() { free(); }
};
class SimpleRNNCell {
public:
    int inputSize, hiddenSize;
    ActivationType activation;

    CLMatrix* g_Wih;
    CLMatrix* g_Whh;
    CLArray* g_Bh;
    CLMatrix* g_dWih;
    CLMatrix* g_dWhh;
    CLArray* g_dBh;
    CLArray* g_Sum;
    CLArray* g_H;
    CLArray* g_PreH;
    cl_context context;
    cl_command_queue queue;

    SimpleRNNCell(int inSize, int hidSize, ActivationType act,
                  cl_context ctx, cl_command_queue q, cl_kernel k_zero_kernel)
        : inputSize(inSize), hiddenSize(hidSize), activation(act), context(ctx), queue(q)
    {
        // Allocate device buffers
        g_Wih = new CLMatrix(context, queue);
        g_Whh = new CLMatrix(context, queue);
        g_Bh = new CLArray(context, queue);
        g_dWih = new CLMatrix(context, queue);
        g_dWhh = new CLMatrix(context, queue);
        g_dBh = new CLArray(context, queue);
        g_Sum = new CLArray(context, queue);
        g_H = new CLArray(context, queue);
        g_PreH = new CLArray(context, queue);

        // Init weights and biases (host-side, upload to device)
        float scale = sqrtf(2.0f / (inSize + hidSize));
        TFArray2D Wih_init(hidSize, FArray(inSize));
        TFArray2D Whh_init(hidSize, FArray(hidSize));
        FArray Bh_init(hidSize, 0.0f);

                auto initMat = [&](DArray2D& M, int rows, int cols, float sc) {
            std::uniform_real_distribution<> dis(-sc, sc);
            M.resize(rows);
            for (int i = 0; i < rows; ++i) {
                M[i].resize(cols);
                for (int j = 0; j < cols; ++j)
                    M[i][j] = dis(gen);

        for (int i = 0; i < hidSize; i++) RandomInit(Wih_init[i], scale);
        for (int i = 0; i < hidSize; i++) RandomInit(Whh_init[i], scale);

        g_Wih->copyToDevice(Wih_init);
        g_Whh->copyToDevice(Whh_init);
        g_Bh->copyToDevice(Bh_init);

        g_dWih->allocate(hidSize, inSize);   g_dWih->zero(k_zero_kernel);
        g_dWhh->allocate(hidSize, hidSize);  g_dWhh->zero(k_zero_kernel);
        g_dBh->allocate(hidSize);            g_dBh->zero(k_zero_kernel);

        g_Sum->allocate(hidSize);
        g_H->allocate(hidSize);
        g_PreH->allocate(hidSize);
    }

    ~SimpleRNNCell() {
        delete g_Wih; delete g_Whh; delete g_Bh;
        delete g_dWih; delete g_dWhh; delete g_dBh;
        delete g_Sum; delete g_H; delete g_PreH;
    }

    // Device-side forward
    void forward(const FArray& input, const FArray& prevH,
                 FArray& H, FArray& PreH,
                 cl_kernel k_matvec, cl_kernel k_simple_rnn_forward) {
        // Upload inputs
        CLArray g_input(context, queue), g_prevH(context, queue);
        g_input.copyToDevice(input);
        g_prevH.copyToDevice(prevH);

        // Run kernel (TODO)
        // After computing, download to host:
        g_H->copyToHost(H);
        g_PreH->copyToHost(PreH);
    }

    void backward(...) {}
    void applyGradients(float LR, float clipVal) {}
    void resetGradients(cl_kernel k_zero_kernel) {}
};

// ======================= LSTM Cell (OpenCL) =======================
class LSTMCell {
public:
    int inputSize, hiddenSize, concatSize;
    ActivationType activation;
    cl_context context;
    cl_command_queue queue;

    // Device buffers
    CLMatrix *g_Wf, *g_Wi, *g_Wc, *g_Wo;
    CLArray *g_Bf, *g_Bi, *g_Bc, *g_Bo;
    CLMatrix *g_dWf, *g_dWi, *g_dWc, *g_dWo;
    CLArray *g_dBf, *g_dBi, *g_dBc, *g_dBo;
    CLArray *g_SumF, *g_SumI, *g_SumC, *g_SumO;
    CLArray *g_H, *g_C, *g_Fg, *g_Ig, *g_CTilde, *g_Og, *g_TanhC;
    CLArray *g_Concat, *g_PrevH, *g_PrevC;

    LSTMCell(int inSize, int hidSize, ActivationType act, cl_context ctx, cl_command_queue q, cl_kernel k_zero_kernel)
        : inputSize(inSize), hiddenSize(hidSize), activation(act), context(ctx), queue(q)
    {
        concatSize = inputSize + hiddenSize;
        g_Wf = new CLMatrix(context, queue);
        g_Wi = new CLMatrix(context, queue);
        g_Wc = new CLMatrix(context, queue);
        g_Wo = new CLMatrix(context, queue);
        g_Bf = new CLArray(context, queue);
        g_Bi = new CLArray(context, queue);
        g_Bc = new CLArray(context, queue);
        g_Bo = new CLArray(context, queue);
        g_dWf = new CLMatrix(context, queue);
        g_dWi = new CLMatrix(context, queue);
        g_dWc = new CLMatrix(context, queue);
        g_dWo = new CLMatrix(context, queue);
        g_dBf = new CLArray(context, queue);
        g_dBi = new CLArray(context, queue);
        g_dBc = new CLArray(context, queue);
        g_dBo = new CLArray(context, queue);

        g_SumF = new CLArray(context, queue);
        g_SumI = new CLArray(context, queue);
        g_SumC = new CLArray(context, queue);
        g_SumO = new CLArray(context, queue);

        g_H = new CLArray(context, queue); g_C = new CLArray(context, queue);
        g_Fg = new CLArray(context, queue); g_Ig = new CLArray(context, queue);
        g_CTilde = new CLArray(context, queue); g_Og = new CLArray(context, queue); g_TanhC = new CLArray(context, queue);

        g_Concat = new CLArray(context, queue); g_PrevH = new CLArray(context, queue); g_PrevC = new CLArray(context, queue);
    }

    ~LSTMCell() {
        delete g_Wf; delete g_Wi; delete g_Wc; delete g_Wo;
        delete g_Bf; delete g_Bi; delete g_Bc; delete g_Bo;
        delete g_dWf; delete g_dWi; delete g_dWc; delete g_dWo;
        delete g_dBf; delete g_dBi; delete g_dBc; delete g_dBo;
        delete g_SumF; delete g_SumI; delete g_SumC; delete g_SumO;
        delete g_H; delete g_C; delete g_Fg; delete g_Ig; delete g_CTilde;
        delete g_Og; delete g_TanhC; delete g_Concat; delete g_PrevH; delete g_PrevC;
    }

    void forward(...) {}
    void backward(...) {}
    void applyGradients(float LR, float clipVal) {}
    void resetGradients(cl_kernel k_zero_kernel) {}
};


// Helper function for host random initialization
inline void RandomInit(FArray& arr, float scale) {
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-scale, scale);
    for (auto& v : arr) v = dis(gen);
}


// ======================= Output Layer (OpenCL) =======================
class OutputLayer {
public:
    int inputSize, outputSize;
    ActivationType activation;
    cl_context context;
    cl_command_queue queue;

    CLMatrix* g_W;
    CLArray* g_B;
    CLMatrix* g_dW;
    CLArray* g_dB;
    CLArray* g_Pre;
    CLArray* g_Out;

    OutputLayer(int inSize, int outSize, ActivationType act, cl_context ctx, cl_command_queue q, cl_kernel k_zero_kernel)
        : inputSize(inSize), outputSize(outSize), activation(act), context(ctx), queue(q)
    {
        g_W = new CLMatrix(context, queue); g_B = new CLArray(context, queue);
        g_dW = new CLMatrix(context, queue); g_dB = new CLArray(context, queue);
        g_Pre = new CLArray(context, queue); g_Out = new CLArray(context, queue);

        // Init weights and biases (host-side, upload to device)
        float scale = sqrtf(2.0f / inSize);
        TFArray2D W_init(outSize, FArray(inSize));
        FArray B_init(outSize, 0.0f);
        for (int i = 0; i < outSize; i++) RandomInit(W_init[i], scale);

        g_W->copyToDevice(W_init);
        g_B->copyToDevice(B_init);

        g_dW->allocate(outSize, inSize);   g_dW->zero(k_zero_kernel);
        g_dB->allocate(outSize);           g_dB->zero(k_zero_kernel);
        g_Pre->allocate(outSize);
        g_Out->allocate(outSize);
    }

    // Device-side forward: y = activation(W*input + B)
    void forward(const FArray& input, FArray& output, FArray& pre,
                 cl_kernel k_matvec, cl_kernel k_activate) {
        // Upload input
        CLArray g_input(context, queue);
        g_input.copyToDevice(input);

        // Matrix-vector kernel: (optionally split matvec, then activation, or use a fused kernel)
        // [Kernel launches not implemented here—need to match your kernels and calling conventions]
        // After computing, download to host:
        g_Out->copyToHost(output);
        g_Pre->copyToHost(pre);
    }
    // (backward/applyGradients/resetGradients as before)
};

// (Pattern for LSTMCell and GRUCell would follow similarly)

// ======================= LSTM Cell (OpenCL) =======================
class LSTMCell {
public:
    int inputSize, hiddenSize, concatSize;
    ActivationType activation;
    cl_context context;
    cl_command_queue queue;

    // Device buffers
    CLMatrix *g_Wf, *g_Wi, *g_Wc, *g_Wo;
    CLArray *g_Bf, *g_Bi, *g_Bc, *g_Bo;
    CLMatrix *g_dWf, *g_dWi, *g_dWc, *g_dWo;
    CLArray *g_dBf, *g_dBi, *g_dBc, *g_dBo;
    CLArray *g_SumF, *g_SumI, *g_SumC, *g_SumO;
    CLArray *g_H, *g_C, *g_Fg, *g_Ig, *g_CTilde, *g_Og, *g_TanhC;
    CLArray *g_Concat, *g_PrevH, *g_PrevC;

    LSTMCell(int inSize, int hidSize, ActivationType act, cl_context ctx, cl_command_queue q, cl_kernel k_zero_kernel)
    : inputSize(inSize), hiddenSize(hidSize), activation(act), context(ctx), queue(q)
    {
        concatSize = inputSize + hiddenSize;
        float scale = sqrtf(2.0f / concatSize);

        // --- Host-side initial weights/biases ---
        TFArray2D W_init(hiddenSize, FArray(concatSize));
        for (int i = 0; i < hiddenSize; i++) RandomInit(W_init[i], scale);
        TFArray2D Wi_init = W_init, Wc_init = W_init, Wo_init = W_init;
        for (int i = 0; i < hiddenSize; i++) RandomInit(Wi_init[i], scale);
        for (int i = 0; i < hiddenSize; i++) RandomInit(Wc_init[i], scale);
        for (int i = 0; i < hiddenSize; i++) RandomInit(Wo_init[i], scale);

        FArray Bf_init(hiddenSize, 1.0f), Bi_init(hiddenSize, 0.0f),
        Bc_init(hiddenSize, 0.0f), Bo_init(hiddenSize, 0.0f);

        // --- Allocate device buffers & upload initial weights/biases ---
        g_Wf = new CLMatrix(context, queue); g_Wf->copyToDevice(W_init);
        g_Wi = new CLMatrix(context, queue); g_Wi->copyToDevice(Wi_init);
        g_Wc = new CLMatrix(context, queue); g_Wc->copyToDevice(Wc_init);
        g_Wo = new CLMatrix(context, queue); g_Wo->copyToDevice(Wo_init);

        g_Bf = new CLArray(context, queue); g_Bf->copyToDevice(Bf_init);
        g_Bi = new CLArray(context, queue); g_Bi->copyToDevice(Bi_init);
        g_Bc = new CLArray(context, queue); g_Bc->copyToDevice(Bc_init);
        g_Bo = new CLArray(context, queue); g_Bo->copyToDevice(Bo_init);

        // --- Allocate and zero gradients ---
        g_dWf = new CLMatrix(context, queue); g_dWf->allocate(hiddenSize, concatSize); g_dWf->zero(k_zero_kernel);
        g_dWi = new CLMatrix(context, queue); g_dWi->allocate(hiddenSize, concatSize); g_dWi->zero(k_zero_kernel);
        g_dWc = new CLMatrix(context, queue); g_dWc->allocate(hiddenSize, concatSize); g_dWc->zero(k_zero_kernel);
        g_dWo = new CLMatrix(context, queue); g_dWo->allocate(hiddenSize, concatSize); g_dWo->zero(k_zero_kernel);
        g_dBf = new CLArray(context, queue); g_dBf->allocate(hiddenSize); g_dBf->zero(k_zero_kernel);
        g_dBi = new CLArray(context, queue); g_dBi->allocate(hiddenSize); g_dBi->zero(k_zero_kernel);
        g_dBc = new CLArray(context, queue); g_dBc->allocate(hiddenSize); g_dBc->zero(k_zero_kernel);
        g_dBo = new CLArray(context, queue); g_dBo->allocate(hiddenSize); g_dBo->zero(k_zero_kernel);

        // --- Internal buffers for temporaries/gates ---
        g_SumF = new CLArray(context, queue); g_SumF->allocate(hiddenSize);
        g_SumI = new CLArray(context, queue); g_SumI->allocate(hiddenSize);
        g_SumC = new CLArray(context, queue); g_SumC->allocate(hiddenSize);
        g_SumO = new CLArray(context, queue); g_SumO->allocate(hiddenSize);

        g_H = new CLArray(context, queue); g_H->allocate(hiddenSize);
        g_C = new CLArray(context, queue); g_C->allocate(hiddenSize);
        g_Fg = new CLArray(context, queue); g_Fg->allocate(hiddenSize);
        g_Ig = new CLArray(context, queue); g_Ig->allocate(hiddenSize);
        g_CTilde = new CLArray(context, queue); g_CTilde->allocate(hiddenSize);
        g_Og = new CLArray(context, queue); g_Og->allocate(hiddenSize);
        g_TanhC = new CLArray(context, queue); g_TanhC->allocate(hiddenSize);

        g_Concat = new CLArray(context, queue); g_Concat->allocate(concatSize);
        g_PrevH = new CLArray(context, queue); g_PrevH->allocate(hiddenSize);
        g_PrevC = new CLArray(context, queue); g_PrevC->allocate(hiddenSize);
    }

    ~LSTMCell() {
        delete g_Wf; delete g_Wi; delete g_Wc; delete g_Wo;
        delete g_Bf; delete g_Bi; delete g_Bc; delete g_Bo;
        delete g_dWf; delete g_dWi; delete g_dWc; delete g_dWo;
        delete g_dBf; delete g_dBi; delete g_dBc; delete g_dBo;
        delete g_SumF; delete g_SumI; delete g_SumC; delete g_SumO;
        delete g_H; delete g_C; delete g_Fg; delete g_Ig; delete g_CTilde;
        delete g_Og; delete g_TanhC; delete g_Concat; delete g_PrevH; delete g_PrevC;
    }

    // Device-side forward (placeholder: will use k_concat, then k_lstm_forward kernels)
    void forward(const FArray& input, const FArray& prevH, const FArray& prevC,
                 FArray& H, FArray& C, FArray& Fg, FArray& Ig, FArray& CTilde,
                 FArray& Og, FArray& TanhC,
                 cl_kernel k_concat, cl_kernel k_lstm_forward) {
        // Upload input, prevH, prevC as CLArray
        // Launch k_concat, then k_lstm_forward
        // Download outputs to host arrays
                 }

                 // backward/applyGradients/resetGradients analogously, as before
};

// ======================= GRU Cell (OpenCL) =======================
class GRUCell {
public:
    int inputSize, hiddenSize, concatSize;
    ActivationType activation;
    cl_context context;
    cl_command_queue queue;

    // Device buffers
    CLMatrix *g_Wz, *g_Wr, *g_Wh;
    CLArray *g_Bz, *g_Br, *g_Bh;
    CLMatrix *g_dWz, *g_dWr, *g_dWh;
    CLArray *g_dBz, *g_dBr, *g_dBh;
    CLArray *g_SumZ, *g_SumR, *g_SumH;
    CLArray *g_H, *g_Z, *g_R, *g_HTilde;
    CLArray *g_Concat, *g_ConcatR, *g_PrevH;

    GRUCell(int inSize, int hidSize, ActivationType act, cl_context ctx, cl_command_queue q, cl_kernel k_zero_kernel)
    : inputSize(inSize), hiddenSize(hidSize), activation(act), context(ctx), queue(q)
    {
        concatSize = inputSize + hiddenSize;
        float scale = sqrtf(2.0f / concatSize);

        TFArray2D Wz_init(hiddenSize, FArray(concatSize)), Wr_init(hiddenSize, FArray(concatSize)), Wh_init(hiddenSize, FArray(concatSize));
        for (int i = 0; i < hiddenSize; i++) RandomInit(Wz_init[i], scale);
        for (int i = 0; i < hiddenSize; i++) RandomInit(Wr_init[i], scale);
        for (int i = 0; i < hiddenSize; i++) RandomInit(Wh_init[i], scale);
        FArray Bz_init(hiddenSize, 0.0f), Br_init(hiddenSize, 0.0f), Bh_init(hiddenSize, 0.0f);

        g_Wz = new CLMatrix(context, queue); g_Wz->copyToDevice(Wz_init);
        g_Wr = new CLMatrix(context, queue); g_Wr->copyToDevice(Wr_init);
        g_Wh = new CLMatrix(context, queue); g_Wh->copyToDevice(Wh_init);

        g_Bz = new CLArray(context, queue); g_Bz->copyToDevice(Bz_init);
        g_Br = new CLArray(context, queue); g_Br->copyToDevice(Br_init);
        g_Bh = new CLArray(context, queue); g_Bh->copyToDevice(Bh_init);

        g_dWz = new CLMatrix(context, queue); g_dWz->allocate(hiddenSize, concatSize); g_dWz->zero(k_zero_kernel);
        g_dWr = new CLMatrix(context, queue); g_dWr->allocate(hiddenSize, concatSize); g_dWr->zero(k_zero_kernel);
        g_dWh = new CLMatrix(context, queue); g_dWh->allocate(hiddenSize, concatSize); g_dWh->zero(k_zero_kernel);

        g_dBz = new CLArray(context, queue); g_dBz->allocate(hiddenSize); g_dBz->zero(k_zero_kernel);
        g_dBr = new CLArray(context, queue); g_dBr->allocate(hiddenSize); g_dBr->zero(k_zero_kernel);
        g_dBh = new CLArray(context, queue); g_dBh->allocate(hiddenSize); g_dBh->zero(k_zero_kernel);

        // Internal buffers for temporaries/gates
        g_SumZ = new CLArray(context, queue); g_SumZ->allocate(hiddenSize);
        g_SumR = new CLArray(context, queue); g_SumR->allocate(hiddenSize);
        g_SumH = new CLArray(context, queue); g_SumH->allocate(hiddenSize);

        g_H      = new CLArray(context, queue); g_H->allocate(hiddenSize);
        g_Z      = new CLArray(context, queue); g_Z->allocate(hiddenSize);
        g_R      = new CLArray(context, queue); g_R->allocate(hiddenSize);
        g_HTilde = new CLArray(context, queue); g_HTilde->allocate(hiddenSize);

        g_Concat  = new CLArray(context, queue); g_Concat->allocate(concatSize);
        g_ConcatR = new CLArray(context, queue); g_ConcatR->allocate(concatSize);
        g_PrevH   = new CLArray(context, queue); g_PrevH->allocate(hiddenSize);
    }

    ~GRUCell() {
        delete g_Wz; delete g_Wr; delete g_Wh;
        delete g_Bz; delete g_Br; delete g_Bh;
        delete g_dWz; delete g_dWr; delete g_dWh;
        delete g_dBz; delete g_dBr; delete g_dBh;
        delete g_SumZ; delete g_SumR; delete g_SumH;
        delete g_H; delete g_Z; delete g_R; delete g_HTilde;
        delete g_Concat; delete g_ConcatR; delete g_PrevH;
    }

    // Device-side forward (placeholder: will use k_concat, k_gru_gates, k_gru_hidden kernels)
    void forward(const FArray& input, const FArray& prevH,
                 FArray& H, FArray& Z, FArray& R, FArray& HTilde,
                 cl_kernel k_concat, cl_kernel k_gru_gates, cl_kernel k_gru_hidden) {
        // Upload inputs, run required kernels, download to host
                 }

                 // backward/applyGradients/resetGradients analogously, as before
};

// =============== OpenCLRNNFacade Model Wrapper ===============
class OpenCLRNNFacade {
public:
    int inputSize, outputSize;
    std::vector<int> hiddenSizes;
    CellType cellType;
    ActivationType activation, outputActivation;
    LossType lossType;
    float learningRate, gradientClip;
    int bpttSteps, sequenceLen;

    // Cells/layers
    std::vector<std::unique_ptr<SimpleRNNCell>> simpleCells;
    std::vector<std::unique_ptr<LSTMCell>>      lstmCells;
    std::vector<std::unique_ptr<GRUCell>>       gruCells;
    std::unique_ptr<OutputLayer> outputLayer;

    // OpenCL context
    cl_context context;
    cl_command_queue queue;

    // Persistent OpenCL kernel handles
    cl_kernel k_zero_kernel;
    cl_kernel k_fill_kernel;
    cl_kernel k_matvec_bias_kernel;
    cl_kernel k_matvec_bias_act_kernel;
    cl_kernel k_concat_kernel;
    // (add more kernels as needed for forward/backward)

    // Device "state" buffers, temp arrays, etc. as needed (analogous to your CUDA d_H, d_C, etc.)
    // std::vector<CLArray> d_H, d_C, d_PreH, ...

    // Constructor: creates all needed cells and layers
    OpenCLRNNFacade(
        int inSize, int outSize,
        const std::vector<int>& hidSizes,
        CellType cell,
        ActivationType act, ActivationType outAct,
        LossType loss,
        float lr, float gradClip,
        int bptt,
        cl_context ctx, cl_command_queue q,
        cl_kernel zeroK, cl_kernel fillK, cl_kernel matvecK, cl_kernel matvecActK, cl_kernel concatK)
    : inputSize(inSize), outputSize(outSize), hiddenSizes(hidSizes),
    cellType(cell), activation(act), outputActivation(outAct),
    lossType(loss), learningRate(lr), gradientClip(gradClip),
    bpttSteps(bptt), context(ctx), queue(q),
    k_zero_kernel(zeroK), k_fill_kernel(fillK),
    k_matvec_bias_kernel(matvecK), k_matvec_bias_act_kernel(matvecActK), k_concat_kernel(concatK)
    {
        // Build N layers/cells according to hiddenSizes
        for (size_t l = 0; l < hiddenSizes.size(); ++l) {
            int in_dim = (l == 0) ? inputSize : hiddenSizes[l-1];
            int hid_dim = hiddenSizes[l];
            if (cellType == ctSimpleRNN)
                simpleCells.push_back(std::make_unique<SimpleRNNCell>(in_dim, hid_dim, activation, context, queue, k_zero_kernel));
            else if (cellType == ctLSTM)
                lstmCells.push_back(std::make_unique<LSTMCell>(in_dim, hid_dim, activation, context, queue, k_zero_kernel));
            else if (cellType == ctGRU)
                gruCells.push_back(std::make_unique<GRUCell>(in_dim, hid_dim, activation, context, queue, k_zero_kernel));
        }
        // Output layer connects last hidden layer to output
        int lastHidden = hiddenSizes.empty() ? inputSize : hiddenSizes.back();
        outputLayer = std::make_unique<OutputLayer>(lastHidden, outputSize, outputActivation, context, queue, k_zero_kernel);
    }

    // ================ Forward, backward, predict skeletons (to be implemented) ==================

    // Forward pass for a sequence (single or batch)
    void forwardSequence(const std::vector<FArray>& inputs, std::vector<FArray>& outputs) {
        // Example logic:
        // 1. For each timestep:
        // 2.   Pass input through each layer (chain across time if recurrent)
        // 3.   Collect output from outputLayer
        // 4.   Store for output
    }

    // (Add methods for training, BPTT, loss, save/load, etc.)

    // Destructor will auto-clean (assuming kernels/context handled elsewhere)
    ~OpenCLRNNFacade() = default;
};


// ======================= GRU Cell (OpenCL) =======================
class GRUCell {
public:
    int inputSize, hiddenSize, concatSize;
    ActivationType activation;
    cl_context context;
    cl_command_queue queue;

    // Device buffers
    CLMatrix *g_Wz, *g_Wr, *g_Wh;
    CLArray *g_Bz, *g_Br, *g_Bh;
    CLMatrix *g_dWz, *g_dWr, *g_dWh;
    CLArray *g_dBz, *g_dBr, *g_dBh;
    CLArray *g_SumZ, *g_SumR, *g_SumH;
    CLArray *g_H, *g_Z, *g_R, *g_HTilde;
    CLArray *g_Concat, *g_ConcatR, *g_PrevH;

    GRUCell(int inSize, int hidSize, ActivationType act, cl_context ctx, cl_command_queue q, cl_kernel k_zero_kernel)
        : inputSize(inSize), hiddenSize(hidSize), activation(act), context(ctx), queue(q)
    {
        concatSize = inputSize + hiddenSize;
        g_Wz = new CLMatrix(context, queue); g_Wr = new CLMatrix(context, queue); g_Wh = new CLMatrix(context, queue);
        g_Bz = new CLArray(context, queue); g_Br = new CLArray(context, queue); g_Bh = new CLArray(context, queue);
        g_dWz = new CLMatrix(context, queue); g_dWr = new CLMatrix(context, queue); g_dWh = new CLMatrix(context, queue);
        g_dBz = new CLArray(context, queue); g_dBr = new CLArray(context, queue); g_dBh = new CLArray(context, queue);
        g_SumZ = new CLArray(context, queue); g_SumR = new CLArray(context, queue); g_SumH = new CLArray(context, queue);

        g_H = new CLArray(context, queue); g_Z = new CLArray(context, queue);
        g_R = new CLArray(context, queue); g_HTilde = new CLArray(context, queue);

        g_Concat = new CLArray(context, queue); g_ConcatR = new CLArray(context, queue); g_PrevH = new CLArray(context, queue);
    }

    ~GRUCell() {
        delete g_Wz; delete g_Wr; delete g_Wh;
        delete g_Bz; delete g_Br; delete g_Bh;
        delete g_dWz; delete g_dWr; delete g_dWh;
        delete g_dBz; delete g_dBr; delete g_dBh;
        delete g_SumZ; delete g_SumR; delete g_SumH;
        delete g_H; delete g_Z; delete g_R; delete g_HTilde;
        delete g_Concat; delete g_ConcatR; delete g_PrevH;
    }

    void forward(...) {}
    void backward(...) {}
    void applyGradients(float LR, float clipVal) {}
    void resetGradients(cl_kernel k_zero_kernel) {}
};

// ======================= Output Layer (OpenCL) =======================
class OutputLayer {
public:
    int inputSize, outputSize;
    ActivationType activation;
    cl_context context;
    cl_command_queue queue;

    CLMatrix* g_W;
    CLArray* g_B;
    CLMatrix* g_dW;
    CLArray* g_dB;
    CLArray* g_Pre;
    CLArray* g_Out;

    OutputLayer(int inSize, int outSize, ActivationType act, cl_context ctx, cl_command_queue q, cl_kernel k_zero_kernel)
        : inputSize(inSize), outputSize(outSize), activation(act), context(ctx), queue(q)
    {
        g_W = new CLMatrix(context, queue); g_B = new CLArray(context, queue);
        g_dW = new CLMatrix(context, queue); g_dB = new CLArray(context, queue);
        g_Pre = new CLArray(context, queue); g_Out = new CLArray(context, queue);
        // More initialization as needed
    }

    ~OutputLayer() {
        delete g_W; delete g_B; delete g_dW; delete g_dB; delete g_Pre; delete g_Out;
    }

    void forward(const FArray& input, FArray& output, FArray& pre, cl_kernel k_matvec, cl_kernel k_activate) {}
    void backward(...) {}
    void applyGradients(float LR, float clipVal) {}
    void resetGradients(cl_kernel k_zero_kernel) {}
};

// =================== CLI Argument Parsing and Main Command Preparation ====================
#include <cctype> // For isdigit

// Helper to split comma-separated string into a vector of ints
std::vector<int> ParseIntList(const std::string& str) {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        int n = atoi(token.c_str());
        if (n > 0) result.push_back(n);
    }
    return result;
}

// Top-level main() will use these variables:
struct CLIOptions {
    CommandType command = cmdNone;
    int inputSize = 0, outputSize = 0, epochs = 100, bpttSteps = 1;
    std::vector<int> hiddenSizes;
    std::string modelFile, saveFile, dataFile;
    float lr = 0.01f, clipVal = 1.0f;
    CellType cellType = ctSimpleRNN;
    LossType lossType = ltMSE;
    ActivationType hiddenAct = atTanh, outputAct = atSigmoid;
    bool normalize = false;
};

// Argument parsing routine:
CLIOptions ParseCLI(int argc, char** argv) {
    CLIOptions opt;
    if (argc < 2) { opt.command = cmdHelp; return opt; }

    std::string cmdStr = argv[1];
    if (cmdStr == "create")      opt.command = cmdCreate;
    else if (cmdStr == "train")  opt.command = cmdTrain;
    else if (cmdStr == "predict")opt.command = cmdPredict;
    else if (cmdStr == "info")   opt.command = cmdInfo;
    else if (cmdStr == "help" || cmdStr == "--help") opt.command = cmdHelp;
    else { std::cout << "Unknown command: " << argv[1] << "\n"; opt.command = cmdHelp; }

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        size_t eq = arg.find('=');

        if (eq == std::string::npos) {
            if (arg == "--normalize") { opt.normalize = true; continue; }
            std::cout << "Invalid argument: " << arg << "\n";
            continue;
        }

        std::string key = arg.substr(0, eq);
        std::string valueStr = arg.substr(eq + 1);
        if      (key == "--input")   opt.inputSize = atoi(valueStr.c_str());
        else if (key == "--output")  opt.outputSize = atoi(valueStr.c_str());
        else if (key == "--hidden")  opt.hiddenSizes = ParseIntList(valueStr);
        else if (key == "--type")    opt.cellType = ParseCellType(valueStr);
        else if (key == "--loss")    opt.lossType = ParseLossType(valueStr);
        else if (key == "--save")    opt.saveFile = valueStr;
        else if (key == "--model")   opt.modelFile = valueStr;
        else if (key == "--data")    opt.dataFile = valueStr;
        else if (key == "--epochs")  opt.epochs = atoi(valueStr.c_str());
        else if (key == "--lr")      opt.lr = atof(valueStr.c_str());
        else if (key == "--clip")    opt.clipVal = atof(valueStr.c_str());
    }
    return opt;
}
