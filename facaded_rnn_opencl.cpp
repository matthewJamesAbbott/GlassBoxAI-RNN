//
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

using namespace std;

// OpenCL error check macro, similar to CL_CHECK in rnn_opencl.cpp
#define CL_CHECK(err) \
    do { \
        if (err != CL_SUCCESS) { \
            std::cerr << "OpenCL error at " << __FILE__ << ":" << __LINE__ << " - " << (int)err << std::endl; \
            exit(1); \
        } \
    } while(0)

#define BLOCK_SIZE 256

// Enums matching CUDA
enum ActivationType { atSigmoid=0, atTanh=1, atReLU=2, atLinear=3 };
enum LossType { ltMSE=0, ltCrossEntropy=1 };
enum CellType { ctSimpleRNN=0, ctLSTM=1, ctGRU=2 };
enum GateType { gtForget=0, gtInput=1, gtOutput=2, gtCellCandidate=3, gtUpdate=4, gtReset=5, gtHiddenCandidate=6 };

// Typedefs
typedef std::vector<float> DArray;
typedef std::vector<DArray> DArray2D;
typedef std::vector<DArray2D> DArray3D;

struct HistogramBin {
    float RangeMin, RangeMax;
    int Count;
    float Percentage;
};

struct GateSaturationStats {
    int NearZeroCount, NearOneCount, TotalCount;
    float NearZeroPct, NearOnePct;
};

struct GradientScaleStats {
    int Timestep;
    float MeanAbsGrad, MaxAbsGrad, MinAbsGrad;
};

struct LayerNormStats {
    float Mean, Variance, Gamma, Beta;
};

struct OptimizerStateRecord {
    float Momentum, Velocity, Beta1Power, Beta2Power;
};

static random_device rd;
static mt19937 gen(rd());

// Analytics utility functions (host-side, portable from CUDA)
// E.g. statistics on gradients, gates, activations – use these in train/test loops
float compute_mean_abs(const std::vector<float>& arr) {
    float sum = 0.0;
    for (float v : arr) sum += std::abs(v);
    return sum / arr.size();
}
float compute_max_abs(const std::vector<float>& arr) {
    float maxv = 0.0;
    for (float v : arr) maxv = std::max(maxv, std::abs(v));
    return maxv;
}
float compute_min_abs(const std::vector<float>& arr) {
    float minv = arr.empty() ? 0.0 : std::abs(arr[0]);
    for (float v : arr) minv = std::min(minv, std::abs(v));
    return minv;
}


// Double-precision OpenCL kernels mirroring CUDA RNN kernels
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

const char* kernelSource = R"CLC(
// --- Device-side activation functions ---
float d_sigmoid(float x) {
    float clamped = fmax(-500.0, fmin(500.0, x));
    return 1.0 / (1.0 + exp(-clamped));
}
float d_tanh_act(float x) { return tanh(x); }
float d_relu(float x) { return x > 0.0 ? x : 0.0; }

float d_apply_activation(float x, int actType) {
    switch (actType) {
        case 0: return d_sigmoid(x);
        case 1: return d_tanh_act(x);
        case 2: return d_relu(x);
        case 3: return x;
        default: return x;
    }
}
float d_activation_derivative(float y, int actType) {
    switch (actType) {
        case 0: return y * (1.0 - y);
        case 1: return 1.0 - y * y;
        case 2: return y > 0.0 ? 1.0 : 0.0;
        case 3: return 1.0;
        default: return 1.0;
    }
}
float d_clip_value(float v, float maxVal) {
    if (v > maxVal) return maxVal;
    if (v < -maxVal) return -maxVal;
    return v;
}
// ------- Basic kernels -------

// Vector addition: C = A + B
__kernel void k_vec_add(__global float* C, __global const float* A, __global const float* B, int n) {
    int idx = get_global_id(0);
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// Vector scale: A = A * scale
__kernel void k_vec_scale(__global float* A, float scale, int n) {
    int idx = get_global_id(0);
    if (idx < n)
        A[idx] *= scale;
}

// Zero array
__kernel void k_zero_array(__global float* A, int n) {
    int idx = get_global_id(0);
    if (idx < n)
        A[idx] = 0.0;
}

// Fill array with value
__kernel void k_fill_array(__global float* A, float val, int n) {
    int idx = get_global_id(0);
    if (idx < n)
        A[idx] = val;
}

// Matrix-vector multiply: y = W * x + b, then apply activation
// W is [rows x cols], x [cols], y [rows], b [rows]
__kernel void k_matvec_bias_act(__global float* y, __global const float* W, __global const float* x,
                                __global const float* b, int rows, int cols, int actType) {
    int row = get_global_id(0);
    if (row < rows) {
        float sum = b[row];
        for (int j = 0; j < cols; j++) {
            sum += W[row * cols + j] * x[j];
        }
        y[row] = d_apply_activation(sum, actType);
    }
}

// Matrix-vector multiply, no activation: y = W * x + b
__kernel void k_matvec_bias(__global float* y, __global const float* W, __global const float* x,
                            __global const float* b, int rows, int cols) {
    int row = get_global_id(0);
    if (row < rows) {
        float sum = b[row];
        for (int j = 0; j < cols; j++)
            sum += W[row * cols + j] * x[j];
        y[row] = sum;
    }
}

// Concatenate two vectors: out = [a, b]
__kernel void k_concat(__global float* out, __global const float* a, __global const float* b, int sizeA, int sizeB) {
    int idx = get_global_id(0);
    if (idx < sizeA)
        out[idx] = a[idx];
    else if (idx < sizeA + sizeB)
        out[idx] = b[idx - sizeA];
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// --- SimpleRNN cell forward (per timestep) ---
__kernel void k_simple_rnn_forward(__global float* H, __global float* PreH,
    __global const float* Wih, __global const float* Whh, __global const float* Bh,
    __global const float* Input, __global const float* PrevH,
    int inputSize, int hiddenSize, int actType) {

    int i = get_global_id(0);
    if (i < hiddenSize) {
        float sum = Bh[i];
        for (int j = 0; j < inputSize; j++)
            sum += Wih[i * inputSize + j] * Input[j];
        for (int j = 0; j < hiddenSize; j++)
            sum += Whh[i * hiddenSize + j] * PrevH[j];
        PreH[i] = sum;
        H[i] = d_apply_activation(sum, actType);
    }
}

// --- SimpleRNN backward: dHRaw, weight/bias grads, dInput, dPrevH ---

__kernel void k_simple_rnn_backward_dHRaw(__global float* dHRaw, __global const float* dH,
                                          __global const float* H, int hiddenSize,
                                          int actType, float clipVal) {
    int i = get_global_id(0);
    if (i < hiddenSize) {
        float deriv;
        switch (actType) {
            case 0: deriv = H[i] * (1.0 - H[i]); break;        // sigmoid
            case 1: deriv = 1.0 - H[i]*H[i]; break;            // tanh
            case 2: deriv = H[i] > 0.0 ? 1.0 : 0.0; break;     // ReLU
            default: deriv = 1.0;
        }
        float raw = dH[i] * deriv;
        dHRaw[i] = (raw >  clipVal) ?  clipVal :
                   (raw < -clipVal) ? -clipVal : raw;
    }
}

__kernel void k_accumulate_dW(__global float* dW, __global const float* dHRaw, __global const float* X,
                              int rows, int cols) {
    int idx = get_global_id(0);
    int total = rows * cols;
    if (idx < total) {
        int i = idx / cols;
        int j = idx % cols;
        // WARNING: OpenCL 1.x does NOT guarantee atomic add on float
        // Host-side reduction recommended for real BPTT!
        dW[idx] += dHRaw[i] * X[j];
    }
}

__kernel void k_accumulate_dB(__global float* dB, __global const float* dHRaw, int size) {
    int i = get_global_id(0);
    if (i < size)
        dB[i] += dHRaw[i];
}

__kernel void k_compute_dInput(__global float* dInput, __global const float* W, __global const float* dHRaw,
                               int inputSize, int hiddenSize) {
    int j = get_global_id(0);
    if (j < inputSize) {
        float sum = 0.0;
        for (int i = 0; i < hiddenSize; ++i)
            sum += W[i * inputSize + j] * dHRaw[i];
        dInput[j] = sum;
    }
}

__kernel void k_compute_dPrevH(__global float* dPrevH, __global const float* Whh, __global const float* dHRaw,
                               int hiddenSize) {
    int j = get_global_id(0);
    if (j < hiddenSize) {
        float sum = 0.0;
        for (int i = 0; i < hiddenSize; ++i)
            sum += Whh[i * hiddenSize + j] * dHRaw[i];
        dPrevH[j] = sum;
    }
}

// --- LSTM cell forward (per timestep) ---
__kernel void k_lstm_forward(__global float* H, __global float* C, __global float* FG, __global float* IG,
    __global float* CTilde, __global float* OG, __global float* TanhC,
    __global const float* Wf, __global const float* Wi, __global const float* Wc, __global const float* Wo,
    __global const float* Bf, __global const float* Bi, __global const float* Bc, __global const float* Bo,
    __global const float* Concat, __global const float* PrevC,
    int concatSize, int hiddenSize) {

    int k = get_global_id(0);
    if (k < hiddenSize) {
        float sumF = Bf[k], sumI = Bi[k], sumC = Bc[k], sumO = Bo[k];
        for (int j = 0; j < concatSize; j++) {
            float cj = Concat[j];
            sumF += Wf[k * concatSize + j] * cj;
            sumI += Wi[k * concatSize + j] * cj;
            sumC += Wc[k * concatSize + j] * cj;
            sumO += Wo[k * concatSize + j] * cj;
        }
        FG[k]     = d_sigmoid(sumF);
        IG[k]     = d_sigmoid(sumI);
        CTilde[k] = d_tanh_act(sumC);
        OG[k]     = d_sigmoid(sumO);
        C[k]      = FG[k] * PrevC[k] + IG[k] * CTilde[k];
        TanhC[k]  = tanh(C[k]);
        H[k]      = OG[k] * TanhC[k];
    }
}

// --- GRU cell forward part 1: Z/R gates ---
__kernel void k_gru_forward_gates(__global float* Z, __global float* R,
    __global const float* Wz, __global const float* Wr,
    __global const float* Bz, __global const float* Br,
    __global const float* Concat,
    int concatSize, int hiddenSize) {

    int k = get_global_id(0);
    if (k < hiddenSize) {
        float sumZ = Bz[k], sumR = Br[k];
        for (int j = 0; j < concatSize; j++) {
            sumZ += Wz[k * concatSize + j] * Concat[j];
            sumR += Wr[k * concatSize + j] * Concat[j];
        }
        Z[k] = d_sigmoid(sumZ);
        R[k] = d_sigmoid(sumR);
    }
}

// --- GRU cell forward part 2: HTilde/H ---
__kernel void k_gru_forward_hidden(__global float* H, __global float* HTilde,
    __global const float* Wh, __global const float* Bh,
    __global const float* Input, __global const float* PrevH,
    __global const float* Z, __global const float* R,
    int inputSize, int hiddenSize) {

    int k = get_global_id(0);
    if (k < hiddenSize) {
        int concatSize = inputSize + hiddenSize;
        float sumH = Bh[k];
        // Input part
        for (int j = 0; j < inputSize; j++)
            sumH += Wh[k * concatSize + j] * Input[j];
        // Reset-gated hidden part
        for (int j = 0; j < hiddenSize; j++)
            sumH += Wh[k * concatSize + inputSize + j] * (R[j] * PrevH[j]);

        HTilde[k] = d_tanh_act(sumH);
        H[k] = (1.0 - Z[k]) * PrevH[k] + Z[k] * HTilde[k];
    }
}

// --- Output layer forward: y = W*x+B, then activation
__kernel void k_output_forward(__global float* Output, __global float* Pre,
    __global const float* W, __global const float* B,
    __global const float* Input,
    int inputSize, int outputSize, int actType) {

    int i = get_global_id(0);
    if (i < outputSize) {
        float sum = B[i];
        for (int j = 0; j < inputSize; j++)
            sum += W[i * inputSize + j] * Input[j];
        Pre[i] = sum;
        Output[i] = d_apply_activation(sum, actType);
    }
}


// ===================== LSTM Backward Kernels =====================

// Compute LSTM gate derivatives (for BPTT)
__kernel void k_lstm_backward_gates(
    __global float* dOG, __global float* dCTotal, __global float* dFG,
    __global float* dIG, __global float* dCTilde,
    __global const float* dH, __global const float* dC,
    __global const float* FG, __global const float* IG,
    __global const float* CTilde, __global const float* OG,
    __global const float* TanhC, __global const float* PrevC,
    int hiddenSize, float clipVal)
{
    int k = get_global_id(0);
    if (k < hiddenSize) {
        float og = OG[k], tanhc = TanhC[k];
        float fg = FG[k], ig = IG[k], ct = CTilde[k], pc = PrevC[k];

        float raw_og = dH[k] * tanhc * og * (1.0 - og);
        dOG[k] = (raw_og >  clipVal) ?  clipVal : ((raw_og < -clipVal) ? -clipVal : raw_og);

        float dcTotal = dH[k] * og * (1.0 - tanhc * tanhc) + dC[k];
        dcTotal = (dcTotal > clipVal) ? clipVal : ((dcTotal < -clipVal) ? -clipVal : dcTotal);
        dCTotal[k] = dcTotal;

        float raw_fg = dcTotal * pc * fg * (1.0 - fg);
        dFG[k] = (raw_fg >  clipVal) ?  clipVal : ((raw_fg < -clipVal) ? -clipVal : raw_fg);

        float raw_ig = dcTotal * ct * ig * (1.0 - ig);
        dIG[k] = (raw_ig >  clipVal) ?  clipVal : ((raw_ig < -clipVal) ? -clipVal : raw_ig);

        float raw_cTilde = dcTotal * ig * (1.0 - ct * ct);
        dCTilde[k] = (raw_cTilde >  clipVal) ?  clipVal : ((raw_cTilde < -clipVal) ? -clipVal : raw_cTilde);
    }
}

__kernel void k_lstm_backward_dPrevC(__global float* dPrevC, __global const float* dCTotal,
                                     __global const float* FG, int hiddenSize) {
    int k = get_global_id(0);
    if (k < hiddenSize)
        dPrevC[k] = dCTotal[k] * FG[k];}

    // Accumulate dW, dB (host-side reduction recommended in production)
__kernel void k_lstm_backward_weights(
   __global float* dWf, __global float* dWi, __global float* dWc, __global float* dWo,
   __global float* dBf, __global float* dBi, __global float* dBc, __global float* dBo,
   __global const float* dFG, __global const float* dIG,
   __global const float* dCTilde, __global const float* dOG,
   __global const float* Concat,
   int hiddenSize, int concatSize)
   {
      int idx = get_global_id(0);
      int total = hiddenSize * concatSize;
      if (idx < total) {
         int k = idx / concatSize;
         int j = idx % concatSize;
         float cj = Concat[j];
         dWf[idx] += dFG[k] * cj;
         dWi[idx] += dIG[k] * cj;
         dWc[idx] += dCTilde[k] * cj;
         dWo[idx] += dOG[k] * cj;
      }

     // For bias, just let first hiddenSize threads do this
     if (idx < hiddenSize) {
        dBf[idx] += dFG[idx];
        dBi[idx] += dIG[idx];
        dBc[idx] += dCTilde[idx];
        dBo[idx] += dOG[idx];
     }
}

// dInput and dPrevH computation
__kernel void k_lstm_backward_dInput_dPrevH(
   __global float* dInput, __global float* dPrevH,
   __global const float* Wf, __global const float* Wi,
   __global const float* Wc, __global const float* Wo,
   __global const float* dFG, __global const float* dIG,
   __global const float* dCTilde, __global const float* dOG,
   int inputSize, int hiddenSize, int concatSize)
   {
      int j = get_global_id(0);
      if (j < concatSize) {
         float sum = 0.0;
         for (int k = 0; k < hiddenSize; k++) {
            int idx = k * concatSize + j;
            sum += Wf[idx] * dFG[k] +
            Wi[idx] * dIG[k] +
            Wc[idx] * dCTilde[k] +
            Wo[idx] * dOG[k];
         }
         if (j < inputSize)
            dInput[j] = sum;
         else
            dPrevH[j - inputSize] = sum;
      }
   }
   )CLC";

/*
 * Device buffer handle and helpers (for float precision OpenCL, facade style)
 * Analogous to DeviceArray (CUDA) and CLArray (rnn_opencl.cpp, but float)
 */

class CLArray {
public:
    cl_mem d_ptr;
    int size;
    cl_context context;
    cl_command_queue queue;

    CLArray(cl_context ctx, cl_command_queue q)
        : d_ptr(nullptr), size(0), context(ctx), queue(q) {}

    void allocate(int n) {
        cl_int err;
        if (d_ptr) clReleaseMemObject(d_ptr);
        d_ptr = nullptr;
        if (n > 0) {
            d_ptr = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), NULL, &err);
            CL_CHECK(err);
            size = n;
        }
    }

    void free() {
        if (d_ptr) {
            clReleaseMemObject(d_ptr);
            d_ptr = nullptr;
        }
        size = 0;
    }

    void zero(cl_kernel k_zero_kernel) {
        if (d_ptr && size > 0) {
            size_t globalSize = ((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            clSetKernelArg(k_zero_kernel, 0, sizeof(cl_mem), &d_ptr);
            clSetKernelArg(k_zero_kernel, 1, sizeof(int), &size);
            cl_int err = clEnqueueNDRangeKernel(queue, k_zero_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
            CL_CHECK(err);
            clFinish(queue);
        }
    }

    void copyToDevice(const DArray& src) {
        allocate((int)src.size());
        cl_int err = clEnqueueWriteBuffer(queue, d_ptr, CL_TRUE, 0, src.size() * sizeof(float), src.data(), 0, NULL, NULL);
        // CL_CHECK(err);
    }

    void copyToHost(DArray& dst) {
        dst.resize(size);
        cl_int err = clEnqueueReadBuffer(queue, d_ptr, CL_TRUE, 0, size * sizeof(float), dst.data(), 0, NULL, NULL);
        CL_CHECK(err);
    }

    void copyToDeviceRaw(const float* src, int n) {
        allocate(n);
        cl_int err = clEnqueueWriteBuffer(queue, d_ptr, CL_TRUE, 0, n * sizeof(float), src, 0, NULL, NULL);
        CL_CHECK(err);
    }

    // fill entire array with value (calls kernel)
    void fill(cl_kernel k_fill, float val) {
        if (!d_ptr || size <= 0) return;
        size_t globalSize = ((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        clSetKernelArg(k_fill, 0, sizeof(cl_mem), &d_ptr);
        clSetKernelArg(k_fill, 1, sizeof(float), &val);
        clSetKernelArg(k_fill, 2, sizeof(int), &size);
        cl_int err = clEnqueueNDRangeKernel(queue, k_fill, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
        CL_CHECK(err);
        clFinish(queue);
    }

    ~CLArray() { free(); }
};

class SimpleRNNCell {
public:
    int inputSize, hiddenSize;
    ActivationType activation;

    // Host-side weights (float, for precision)
    DArray2D Wih, Whh;
    DArray Bh, dBh;
    DArray2D dWih, dWhh;

    // Device-side buffers
    std::unique_ptr<CLArray> g_Wih, g_Whh, g_Bh;
    std::unique_ptr<CLArray> g_dWih, g_dWhh, g_dBh;
    std::unique_ptr<CLArray> g_H, g_PreH, g_Input, g_PrevH;

    cl_context context;
    cl_command_queue queue;

    // SimpleRNNCell initialization
    SimpleRNNCell(int inputSize_, int hiddenSize_, ActivationType activation_,
                  cl_context ctx, cl_command_queue q,
                  cl_kernel k_zero_kernel)
        : inputSize(inputSize_), hiddenSize(hiddenSize_), activation(activation_),
          context(ctx), queue(q)
    {
        float scale = std::sqrt(2.0 / (inputSize + hiddenSize));
        // Initialize and zero host weights (float)
        auto initMat = [&](DArray2D& M, int rows, int cols, float sc) {
            std::uniform_real_distribution<> dis(-sc, sc);
            M.resize(rows);
            for (int i = 0; i < rows; ++i) {
                M[i].resize(cols);
                for (int j = 0; j < cols; ++j)
                    M[i][j] = dis(gen);
            }
        };
        auto zeroMat = [](DArray2D& M, int rows, int cols) {
            M.resize(rows); for (int i = 0; i < rows; ++i) M[i].assign(cols, 0.0);
        };
        auto zeroArr = [](DArray& A, int sz) { A.assign(sz, 0.0); };

        initMat(Wih, hiddenSize, inputSize, scale);
        initMat(Whh, hiddenSize, hiddenSize, scale);
        zeroArr(Bh, hiddenSize);

        zeroMat(dWih, hiddenSize, inputSize);
        zeroMat(dWhh, hiddenSize, hiddenSize);
        zeroArr(dBh, hiddenSize);

        g_Wih  = std::make_unique<CLArray>(context, queue);
        g_Whh  = std::make_unique<CLArray>(context, queue);
        g_Bh   = std::make_unique<CLArray>(context, queue);
        g_dWih = std::make_unique<CLArray>(context, queue);
        g_dWhh = std::make_unique<CLArray>(context, queue);
        g_dBh  = std::make_unique<CLArray>(context, queue);

        // allocate and upload weight buffers
        int idx;
        std::vector<float> w_tmp(hiddenSize * inputSize);  // flatten for OpenCL
        for (int i=0, idx=0; i < hiddenSize; ++i)
            for (int j=0; j < inputSize; ++j)
                w_tmp[idx++] = Wih[i][j];
        g_Wih->allocate(hiddenSize * inputSize);
        g_Wih->copyToDevice(w_tmp);

        w_tmp.resize(hiddenSize * hiddenSize); idx = 0;
        for (int i=0; i < hiddenSize; ++i)
            for (int j=0; j < hiddenSize; ++j)
                w_tmp[idx++] = Whh[i][j];
        g_Whh->allocate(hiddenSize * hiddenSize);
        g_Whh->copyToDevice(w_tmp);

        g_Bh->copyToDevice(Bh);

        g_dWih->allocate(hiddenSize * inputSize); g_dWih->zero(k_zero_kernel);
        g_dWhh->allocate(hiddenSize * hiddenSize); g_dWhh->zero(k_zero_kernel);
        g_dBh->allocate(hiddenSize); g_dBh->zero(k_zero_kernel);

        // temporaries
        g_H    = std::make_unique<CLArray>(context, queue); g_H->allocate(hiddenSize);
        g_PreH = std::make_unique<CLArray>(context, queue); g_PreH->allocate(hiddenSize);
        g_Input= std::make_unique<CLArray>(context, queue);
        g_PrevH= std::make_unique<CLArray>(context, queue);
    }

    void ForwardOpenCL(const DArray& input, const DArray& prevH,
                       DArray& H, DArray& PreH,
                       cl_kernel k_forward) {
        // upload input/prevH
        g_Input->copyToDevice(input);
        g_PrevH->copyToDevice(prevH);

        // Set kernel arguments and enqueue
        int i = 0;
        clSetKernelArg(k_forward, i++, sizeof(cl_mem), &g_H->d_ptr);
        clSetKernelArg(k_forward, i++, sizeof(cl_mem), &g_PreH->d_ptr);
        clSetKernelArg(k_forward, i++, sizeof(cl_mem), &g_Wih->d_ptr);
        clSetKernelArg(k_forward, i++, sizeof(cl_mem), &g_Whh->d_ptr);
        clSetKernelArg(k_forward, i++, sizeof(cl_mem), &g_Bh->d_ptr);
        clSetKernelArg(k_forward, i++, sizeof(cl_mem), &g_Input->d_ptr);
        clSetKernelArg(k_forward, i++, sizeof(cl_mem), &g_PrevH->d_ptr);
        clSetKernelArg(k_forward, i++, sizeof(int), &inputSize);
        clSetKernelArg(k_forward, i++, sizeof(int), &hiddenSize);
        int actT = (int)activation;
        clSetKernelArg(k_forward, i++, sizeof(int), &actT);

        size_t global = ((hiddenSize + BLOCK_SIZE - 1)/BLOCK_SIZE)*BLOCK_SIZE;
        cl_int err = clEnqueueNDRangeKernel(queue, k_forward, 1, NULL, &global, NULL, 0, NULL, NULL);
        CL_CHECK(err);
        clFinish(queue);

        // Download output & preactivation from device
        g_H->copyToHost(H);
        g_PreH->copyToHost(PreH);
    }

    int GetHiddenSize() const { return hiddenSize; }

    // Add other methods for backward pass and weights application as needed (adapted from CUDA & OpenCL base)
};

class LSTMCell {
public:
    int inputSize, hiddenSize, concatSize;
    ActivationType activation;
    cl_context context;
    cl_command_queue queue;

    // Host-side weights & gradients
    DArray2D Wf, Wi, Wc, Wo;
    DArray Bf, Bi, Bc, Bo;
    DArray2D dWf, dWi, dWc, dWo;
    DArray dBf, dBi, dBc, dBo;

    // Device buffers (all weights, temporaries, intermediates)
    std::unique_ptr<CLArray> g_Wf, g_Wi, g_Wc, g_Wo;
    std::unique_ptr<CLArray> g_Bf, g_Bi, g_Bc, g_Bo;
    std::unique_ptr<CLArray> g_SumF, g_SumI, g_SumC, g_SumO;
    std::unique_ptr<CLArray> g_H, g_C, g_FG, g_IG, g_CTilde, g_OG, g_TanhC;
    std::unique_ptr<CLArray> g_Concat, g_PrevH, g_PrevC;
    // Optionally: gradients

    LSTMCell(int inputSize_, int hiddenSize_, ActivationType activation_,
             cl_context ctx, cl_command_queue q, cl_kernel k_zero_kernel)
        : inputSize(inputSize_), hiddenSize(hiddenSize_), activation(activation_),
          context(ctx), queue(q) {

        concatSize = inputSize + hiddenSize;
        float scale = sqrt(2.0 / concatSize);

        auto initMat = [&](DArray2D& M, int rows, int cols, float sc) {
            std::uniform_real_distribution<> dis(-sc, sc);
            M.resize(rows);
            for (int i=0; i < rows; ++i) {
                M[i].resize(cols);
                for (int j=0; j < cols; ++j)
                    M[i][j] = dis(gen);
            }
        };
        auto zeroMat = [](DArray2D& M, int rows, int cols) {
            M.resize(rows); for (int i=0; i < rows; ++i) M[i].assign(cols, 0.0);
        };
        auto zeroArr = [](DArray& A, int sz, float val=0) { A.assign(sz, val); };

        initMat(Wf, hiddenSize, concatSize, scale);
        initMat(Wi, hiddenSize, concatSize, scale);
        initMat(Wc, hiddenSize, concatSize, scale);
        initMat(Wo, hiddenSize, concatSize, scale);
        zeroArr(Bf, hiddenSize, 1.0); // LSTM forget bias initialized to 1.0
        zeroArr(Bi, hiddenSize, 0.0);
        zeroArr(Bc, hiddenSize, 0.0);
        zeroArr(Bo, hiddenSize, 0.0);

        // Gradients
        zeroMat(dWf, hiddenSize, concatSize);
        zeroMat(dWi, hiddenSize, concatSize);
        zeroMat(dWc, hiddenSize, concatSize);
        zeroMat(dWo, hiddenSize, concatSize);
        zeroArr(dBf, hiddenSize);
        zeroArr(dBi, hiddenSize);
        zeroArr(dBc, hiddenSize);
        zeroArr(dBo, hiddenSize);

        // Flatten helpers
        auto flatten = [](const DArray2D& M, int rows, int cols) {
            std::vector<float> v(rows * cols);
            for (int i=0, idx=0; i < rows; ++i)
                for (int j=0; j < cols; ++j)
                    v[idx++] = M[i][j];
            return v;
        };

        g_Wf = std::make_unique<CLArray>(context, queue); g_Wf->allocate(hiddenSize*concatSize); g_Wf->copyToDevice(flatten(Wf, hiddenSize, concatSize));
        g_Wi = std::make_unique<CLArray>(context, queue); g_Wi->allocate(hiddenSize*concatSize); g_Wi->copyToDevice(flatten(Wi, hiddenSize, concatSize));
        g_Wc = std::make_unique<CLArray>(context, queue); g_Wc->allocate(hiddenSize*concatSize); g_Wc->copyToDevice(flatten(Wc, hiddenSize, concatSize));
        g_Wo = std::make_unique<CLArray>(context, queue); g_Wo->allocate(hiddenSize*concatSize); g_Wo->copyToDevice(flatten(Wo, hiddenSize, concatSize));

        g_Bf = std::make_unique<CLArray>(context, queue); g_Bf->allocate(hiddenSize); g_Bf->copyToDevice(Bf);
        g_Bi = std::make_unique<CLArray>(context, queue); g_Bi->allocate(hiddenSize); g_Bi->copyToDevice(Bi);
        g_Bc = std::make_unique<CLArray>(context, queue); g_Bc->allocate(hiddenSize); g_Bc->copyToDevice(Bc);
        g_Bo = std::make_unique<CLArray>(context, queue); g_Bo->allocate(hiddenSize); g_Bo->copyToDevice(Bo);

        // Temporaries (1 per sequence/batch, here 1)
        g_SumF  = std::make_unique<CLArray>(context, queue); g_SumF->allocate(hiddenSize);
        g_SumI  = std::make_unique<CLArray>(context, queue); g_SumI->allocate(hiddenSize);
        g_SumC  = std::make_unique<CLArray>(context, queue); g_SumC->allocate(hiddenSize);
        g_SumO  = std::make_unique<CLArray>(context, queue); g_SumO->allocate(hiddenSize);
        g_H     = std::make_unique<CLArray>(context, queue); g_H->allocate(hiddenSize);
        g_C     = std::make_unique<CLArray>(context, queue); g_C->allocate(hiddenSize);
        g_FG    = std::make_unique<CLArray>(context, queue); g_FG->allocate(hiddenSize);
        g_IG    = std::make_unique<CLArray>(context, queue); g_IG->allocate(hiddenSize);
        g_CTilde= std::make_unique<CLArray>(context, queue); g_CTilde->allocate(hiddenSize);
        g_OG    = std::make_unique<CLArray>(context, queue); g_OG->allocate(hiddenSize);
        g_TanhC = std::make_unique<CLArray>(context, queue); g_TanhC->allocate(hiddenSize);

        g_Concat = std::make_unique<CLArray>(context, queue); g_Concat->allocate(concatSize);
        g_PrevH  = std::make_unique<CLArray>(context, queue); g_PrevH->allocate(hiddenSize);
        g_PrevC  = std::make_unique<CLArray>(context, queue); g_PrevC->allocate(hiddenSize);
    }

    // LSTMCell forward (OpenCL)
    void ForwardOpenCL(const DArray& input, const DArray& prevH, const DArray& prevC,
                       DArray& H, DArray& C,
                       DArray& FG, DArray& IG, DArray& CTilde, DArray& OG, DArray& TanhC,
                       cl_kernel k_concat, cl_kernel k_lstm_forward) {

        // 1. Concat(input, prevH) → g_Concat
        g_Concat->allocate(concatSize);
        std::vector<float> concat(concatSize);
        for(int i=0;i<inputSize;++i) concat[i]=input[i];
        for(int i=0;i<hiddenSize;++i) concat[inputSize+i]=prevH[i];
        g_Concat->copyToDevice(concat);

        // 2. Upload previous cell state
        g_PrevC->copyToDevice(prevC);

        // 3. Enqueue LSTM forward kernel
        int idx = 0;
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_H->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_C->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_FG->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_IG->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_CTilde->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_OG->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_TanhC->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_Wf->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_Wi->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_Wc->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_Wo->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_Bf->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_Bi->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_Bc->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_Bo->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_Concat->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(cl_mem), &g_PrevC->d_ptr);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(int), &concatSize);
        clSetKernelArg(k_lstm_forward, idx++, sizeof(int), &hiddenSize);

        size_t global = ((hiddenSize + BLOCK_SIZE - 1)/BLOCK_SIZE)*BLOCK_SIZE;
        cl_int err = clEnqueueNDRangeKernel(queue, k_lstm_forward, 1, NULL, &global, NULL, 0, NULL, NULL);
        CL_CHECK(err);
        clFinish(queue);

        // 4. Copy outputs to host
        g_H->copyToHost(H);
        g_C->copyToHost(C);
        g_FG->copyToHost(FG);
        g_IG->copyToHost(IG);
        g_CTilde->copyToHost(CTilde);
        g_OG->copyToHost(OG);
        g_TanhC->copyToHost(TanhC);
    }

    int GetHiddenSize() const { return hiddenSize; }

    // Add similar logic for backward/gradient/apply ops as needed.
};


class GRUCell {
public:
    int inputSize, hiddenSize, concatSize;
    ActivationType activation;
    cl_context context;
    cl_command_queue queue;

    // Host-side weights & grads
    DArray2D Wz, Wr, Wh;
    DArray Bz, Br, Bh;
    DArray2D dWz, dWr, dWh;
    DArray dBz, dBr, dBh;

    // Device buffers
    std::unique_ptr<CLArray> g_Wz, g_Wr, g_Wh;
    std::unique_ptr<CLArray> g_Bz, g_Br, g_Bh;
    std::unique_ptr<CLArray> g_H, g_Z, g_R, g_HTilde;
    std::unique_ptr<CLArray> g_Concat, g_Input, g_PrevH;

    GRUCell(int inputSize_, int hiddenSize_, ActivationType activation_,
            cl_context ctx, cl_command_queue q, cl_kernel k_zero_kernel)
        : inputSize(inputSize_), hiddenSize(hiddenSize_), activation(activation_),
          context(ctx), queue(q)
    {
        concatSize = inputSize + hiddenSize;
        float scale = sqrt(2.0 / concatSize);
        // Random/init helpers
        auto initMat = [&](DArray2D& M, int rows, int cols, float sc){
            std::uniform_real_distribution<> dis(-sc, sc);
            M.resize(rows); for (int i=0; i<rows; ++i) {
                M[i].resize(cols); for(int j=0;j<cols;++j) M[i][j]=dis(gen);
            }
        };
        auto zeroMat = [](DArray2D& M, int rows, int cols) {
            M.resize(rows); for(int i=0;i<rows;++i) M[i].assign(cols,0.0);
        };
        auto zeroArr = [](DArray& A, int sz, float v=0.0) { A.assign(sz,v); };

        initMat(Wz, hiddenSize, concatSize, scale);
        initMat(Wr, hiddenSize, concatSize, scale);
        initMat(Wh, hiddenSize, concatSize, scale);
        zeroArr(Bz, hiddenSize, 0.0);
        zeroArr(Br, hiddenSize, 0.0);
        zeroArr(Bh, hiddenSize, 0.0);

        zeroMat(dWz, hiddenSize, concatSize);
        zeroMat(dWr, hiddenSize, concatSize);
        zeroMat(dWh, hiddenSize, concatSize);
        zeroArr(dBz, hiddenSize);
        zeroArr(dBr, hiddenSize);
        zeroArr(dBh, hiddenSize);

        auto flatten = [](const DArray2D& M, int rows, int cols) {
            std::vector<float> v(rows*cols);
            for (int i=0, idx=0; i<rows; ++i)
                for(int j=0;j<cols;++j)
                    v[idx++] = M[i][j];
            return v;
        };

        g_Wz = std::make_unique<CLArray>(context, queue); g_Wz->allocate(hiddenSize*concatSize); g_Wz->copyToDevice(flatten(Wz,hiddenSize,concatSize));
        g_Wr = std::make_unique<CLArray>(context, queue); g_Wr->allocate(hiddenSize*concatSize); g_Wr->copyToDevice(flatten(Wr,hiddenSize,concatSize));
        g_Wh = std::make_unique<CLArray>(context, queue); g_Wh->allocate(hiddenSize*concatSize); g_Wh->copyToDevice(flatten(Wh,hiddenSize,concatSize));
        g_Bz = std::make_unique<CLArray>(context, queue); g_Bz->allocate(hiddenSize); g_Bz->copyToDevice(Bz);
        g_Br = std::make_unique<CLArray>(context, queue); g_Br->allocate(hiddenSize); g_Br->copyToDevice(Br);
        g_Bh = std::make_unique<CLArray>(context, queue); g_Bh->allocate(hiddenSize); g_Bh->copyToDevice(Bh);

        // Temporaries
        g_H      = std::make_unique<CLArray>(context, queue); g_H->allocate(hiddenSize);
        g_Z      = std::make_unique<CLArray>(context, queue); g_Z->allocate(hiddenSize);
        g_R      = std::make_unique<CLArray>(context, queue); g_R->allocate(hiddenSize);
        g_HTilde = std::make_unique<CLArray>(context, queue); g_HTilde->allocate(hiddenSize);
        g_Concat = std::make_unique<CLArray>(context, queue); g_Concat->allocate(concatSize);
        g_Input  = std::make_unique<CLArray>(context, queue); g_Input->allocate(inputSize);
        g_PrevH  = std::make_unique<CLArray>(context, queue); g_PrevH->allocate(hiddenSize);
    }

    void ForwardOpenCL(const DArray& input, const DArray& prevH,
                       DArray& H, DArray& Z, DArray& R, DArray& HTilde,
                       cl_kernel k_concat, cl_kernel k_gates, cl_kernel k_hidden) {
        // Upload input and prevH & form concat host-side (no need for concat kernel here)
        std::vector<float> concat(concatSize);
        for(int i=0;i<inputSize;++i) concat[i]=input[i];
        for(int i=0;i<hiddenSize;++i) concat[inputSize+i]=prevH[i];
        g_Concat->copyToDevice(concat);
        g_Input->copyToDevice(input);
        g_PrevH->copyToDevice(prevH);

        // 1. Z/R gates
        int idx=0;
        clSetKernelArg(k_gates, idx++, sizeof(cl_mem), &g_Z->d_ptr);
        clSetKernelArg(k_gates, idx++, sizeof(cl_mem), &g_R->d_ptr);
        clSetKernelArg(k_gates, idx++, sizeof(cl_mem), &g_Wz->d_ptr);
        clSetKernelArg(k_gates, idx++, sizeof(cl_mem), &g_Wr->d_ptr);
        clSetKernelArg(k_gates, idx++, sizeof(cl_mem), &g_Bz->d_ptr);
        clSetKernelArg(k_gates, idx++, sizeof(cl_mem), &g_Br->d_ptr);
        clSetKernelArg(k_gates, idx++, sizeof(cl_mem), &g_Concat->d_ptr);
        clSetKernelArg(k_gates, idx++, sizeof(int), &concatSize);
        clSetKernelArg(k_gates, idx++, sizeof(int), &hiddenSize);

        size_t global = ((hiddenSize + BLOCK_SIZE - 1)/BLOCK_SIZE) * BLOCK_SIZE;
        cl_int err = clEnqueueNDRangeKernel(queue, k_gates, 1, NULL, &global, NULL, 0, NULL, NULL);
        CL_CHECK(err);
        clFinish(queue);

        // 2. Compute h̃ and H
        idx = 0;
        clSetKernelArg(k_hidden, idx++, sizeof(cl_mem), &g_H->d_ptr);
        clSetKernelArg(k_hidden, idx++, sizeof(cl_mem), &g_HTilde->d_ptr);
        clSetKernelArg(k_hidden, idx++, sizeof(cl_mem), &g_Wh->d_ptr);
        clSetKernelArg(k_hidden, idx++, sizeof(cl_mem), &g_Bh->d_ptr);
        clSetKernelArg(k_hidden, idx++, sizeof(cl_mem), &g_Input->d_ptr);
        clSetKernelArg(k_hidden, idx++, sizeof(cl_mem), &g_PrevH->d_ptr);
        clSetKernelArg(k_hidden, idx++, sizeof(cl_mem), &g_Z->d_ptr);
        clSetKernelArg(k_hidden, idx++, sizeof(cl_mem), &g_R->d_ptr);
        clSetKernelArg(k_hidden, idx++, sizeof(int), &inputSize);
        clSetKernelArg(k_hidden, idx++, sizeof(int), &hiddenSize);

        err = clEnqueueNDRangeKernel(queue, k_hidden, 1, NULL, &global, NULL, 0, NULL, NULL);
        CL_CHECK(err);
        clFinish(queue);

        // Download results
        g_H->copyToHost(H);
        g_Z->copyToHost(Z);
        g_R->copyToHost(R);
        g_HTilde->copyToHost(HTilde);
    }

    int GetHiddenSize() const { return hiddenSize; }
    // For backward, grads, parameter update, adapt from CUDA/previous notes.
};

class OutputLayer {
public:
    int inputSize, outputSize;
    ActivationType activation;
    cl_context context;
    cl_command_queue queue;

    DArray2D W; // [outputSize][inputSize]
    DArray B;   // [outputSize]
    DArray2D dW;
    DArray dB;

    // Device side
    std::unique_ptr<CLArray> g_W, g_B, g_dW, g_dB;
    std::unique_ptr<CLArray> g_Pre, g_Out, g_Input;

    OutputLayer(int inputSize_, int outputSize_, ActivationType activation_,
                cl_context ctx, cl_command_queue q, cl_kernel k_zero_kernel)
        : inputSize(inputSize_), outputSize(outputSize_), activation(activation_),
          context(ctx), queue(q)
    {
        float scale = sqrt(2.0 / inputSize);
        std::uniform_real_distribution<> dis(-scale, scale);

        W.resize(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            W[i].resize(inputSize);
            for (int j = 0; j < inputSize; ++j)
                W[i][j] = dis(gen);
        }
        B.assign(outputSize, 0.0);

        dW.resize(outputSize);
        for (int i = 0; i < outputSize; ++i)
            dW[i].assign(inputSize, 0.0);
        dB.assign(outputSize, 0.0);

        // Flatten helpers
        auto flatten = [](const DArray2D& M, int rows, int cols) {
            std::vector<float> v(rows * cols);
            for (int i=0, idx=0;i<rows;++i)
                for (int j=0;j<cols;++j)
                    v[idx++] = M[i][j];
            return v;
        };

        g_W  = std::make_unique<CLArray>(context, queue); g_W->allocate(outputSize*inputSize); g_W->copyToDevice(flatten(W,outputSize,inputSize));
        g_B  = std::make_unique<CLArray>(context, queue); g_B->allocate(outputSize); g_B->copyToDevice(B);
        g_dW = std::make_unique<CLArray>(context, queue); g_dW->allocate(outputSize*inputSize); g_dW->zero(k_zero_kernel);
        g_dB = std::make_unique<CLArray>(context, queue); g_dB->allocate(outputSize); g_dB->zero(k_zero_kernel);

        g_Pre   = std::make_unique<CLArray>(context, queue); g_Pre->allocate(outputSize);
        g_Out   = std::make_unique<CLArray>(context, queue); g_Out->allocate(outputSize);
        g_Input = std::make_unique<CLArray>(context, queue); g_Input->allocate(inputSize);
    }

    void ForwardOpenCL(const DArray& input, DArray& output, DArray& pre,
                       cl_kernel k_output_forward) {
        g_Input->copyToDevice(input);

        int idx = 0;
        clSetKernelArg(k_output_forward, idx++, sizeof(cl_mem), &g_Out->d_ptr);
        clSetKernelArg(k_output_forward, idx++, sizeof(cl_mem), &g_Pre->d_ptr);
        clSetKernelArg(k_output_forward, idx++, sizeof(cl_mem), &g_W->d_ptr);
        clSetKernelArg(k_output_forward, idx++, sizeof(cl_mem), &g_B->d_ptr);
        clSetKernelArg(k_output_forward, idx++, sizeof(cl_mem), &g_Input->d_ptr);
        clSetKernelArg(k_output_forward, idx++, sizeof(int), &inputSize);
        clSetKernelArg(k_output_forward, idx++, sizeof(int), &outputSize);
        int actT = activation;
        clSetKernelArg(k_output_forward, idx++, sizeof(int), &actT);
        size_t global = ((outputSize + BLOCK_SIZE - 1)/BLOCK_SIZE)*BLOCK_SIZE;

        cl_int err = clEnqueueNDRangeKernel(queue, k_output_forward, 1, NULL, &global, NULL, 0, NULL, NULL);
        CL_CHECK(err);
        clFinish(queue);

        g_Out->copyToHost(output);
        g_Pre->copyToHost(pre);
    }

    int GetOutputSize() const { return outputSize; }

    // Backward/gradient logic as relevant; parameter update similar to cell classes.
};

class RNNModelOpenCL {
public:
    int inputSize, hiddenSize, outputSize;
    ActivationType hiddenActivation, outputActivation;
    LossType lossType;
    CellType cellType;
    float learningRate;
    float gradClipValue;
    int bpttSteps;
    cl_context context;
    cl_command_queue queue;

    std::unique_ptr<SimpleRNNCell> simpleCell;
    std::unique_ptr<LSTMCell> lstmCell;
    std::unique_ptr<GRUCell> gruCell;
    std::unique_ptr<OutputLayer> outputLayer;

    // OpenCL kernels (should be loaded and compiled at init!)
    cl_kernel k_simple_rnn_forward;
    cl_kernel k_lstm_forward;
    cl_kernel k_gru_gates;
    cl_kernel k_gru_hidden;
    cl_kernel k_output_forward;
    cl_kernel k_zero;
    cl_kernel k_fill;
    cl_kernel k_concat;

    RNNModelOpenCL(int inputSize_, int hiddenSize_, int outputSize_,
                   ActivationType hiddenAct_, ActivationType outputAct_,
                   LossType lossType_, CellType cellType_, float lr, float clipVal,
                   int bpttSteps_,
                   cl_context ctx, cl_command_queue q,
                   cl_kernel ker_simple_rnn_forward,
                   cl_kernel ker_lstm_forward,
                   cl_kernel ker_gru_gates,
                   cl_kernel ker_gru_hidden,
                   cl_kernel ker_output_forward,
                   cl_kernel ker_concat,
                   cl_kernel ker_zero,
                   cl_kernel ker_fill)
    : inputSize(inputSize_), hiddenSize(hiddenSize_), outputSize(outputSize_),
    hiddenActivation(hiddenAct_), outputActivation(outputAct_),
    lossType(lossType_), cellType(cellType_), learningRate(lr), gradClipValue(clipVal),
    bpttSteps(bpttSteps_), context(ctx), queue(q),
    k_simple_rnn_forward(ker_simple_rnn_forward),
    k_lstm_forward(ker_lstm_forward),
    k_gru_gates(ker_gru_gates),
    k_gru_hidden(ker_gru_hidden),
    k_output_forward(ker_output_forward),
    k_concat(ker_concat),
    k_zero(ker_zero),
    k_fill(ker_fill)
    {
        // Create and wire up cells
        if (cellType == ctSimpleRNN) {
            simpleCell = std::make_unique<SimpleRNNCell>(inputSize, hiddenSize, hiddenActivation, context, queue, k_zero);
        } else if (cellType == ctLSTM) {
            lstmCell = std::make_unique<LSTMCell>(inputSize, hiddenSize, hiddenActivation, context, queue, k_zero);
        } else if (cellType == ctGRU) {
            gruCell = std::make_unique<GRUCell>(inputSize, hiddenSize, hiddenActivation, context, queue, k_zero);
        }
        outputLayer = std::make_unique<OutputLayer>(hiddenSize, outputSize, outputActivation, context, queue, k_zero);
    }

    // Forward pass for a sequence, float precision
    void ForwardSequence(const std::vector<DArray>& inputs, std::vector<DArray>& outputs) {
        outputs.clear();
        DArray h(hiddenSize, 0.0), c(hiddenSize, 0.0), prevH(hiddenSize, 0.0), prevC(hiddenSize, 0.0);

        for (size_t t = 0; t < inputs.size(); ++t) {
            if (cellType == ctSimpleRNN) {
                DArray preH;
                simpleCell->ForwardOpenCL(inputs[t], h, h, preH, k_simple_rnn_forward);
            } else if (cellType == ctLSTM) {
                DArray fg, ig, ctilde, og, tanhc;
                lstmCell->ForwardOpenCL(inputs[t], h, c, h, c, fg, ig, ctilde, og, tanhc, k_concat, k_lstm_forward);
            } else if (cellType == ctGRU) {
                DArray z, r, htilde;
                gruCell->ForwardOpenCL(inputs[t], h, h, z, r, htilde, k_concat, k_gru_gates, k_gru_hidden);
            }
            DArray out, pre;
            outputLayer->ForwardOpenCL(h, out, pre, k_output_forward);
            outputs.push_back(out);
        }
    }

    int GetInputSize() const { return inputSize; }
    int GetHiddenSize() const { return hiddenSize; }
    int GetOutputSize() const { return outputSize; }
    CellType GetCellType() const { return cellType; }

    // Backward, optimizer step, save/load omitted -- similar to CUDA/OpenCL implementation
};

int main(int argc, char** argv) {
    srand((unsigned)time(nullptr));
    if (argc < 2) {
        std::cout << "Usage: <command> [options]\n";
        std::cout << "Commands: create, train, predict, info, help\n";
        return 0;
    }
    // Command parse
    std::string cmdStr = argv[1];
    int inputSize = 0, hiddenSize = 0, outputSize = 0, epochs = 100, bpttSteps = 1;
    std::string modelFile, saveFile, dataFile;
    float lr = 0.01, clipVal = 1.0;
    CellType cellType = ctSimpleRNN;
    LossType lossType = ltMSE;
    ActivationType hiddenAct = atTanh, outputAct = atSigmoid;
    bool normalize = false;

    // Simplified argument scan
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        size_t eq = arg.find('=');
        if (eq == std::string::npos) {
            if (arg == "--normalize") { normalize = true; continue; }
            continue;
        }
        std::string key = arg.substr(0, eq);
        std::string valueStr = arg.substr(eq + 1);
        if (key == "--input") inputSize = atoi(valueStr.c_str());
        else if (key == "--hidden") hiddenSize = atoi(valueStr.c_str());
        else if (key == "--output") outputSize = atoi(valueStr.c_str());
        else if (key == "--type") cellType = (valueStr=="lstm")?ctLSTM:(valueStr=="gru"?ctGRU:ctSimpleRNN);
        else if (key == "--loss") lossType = (valueStr=="ce"||valueStr=="crossentropy")?ltCrossEntropy:ltMSE;
        else if (key == "--save") saveFile = valueStr;
        else if (key == "--model") modelFile = valueStr;
        else if (key == "--data") dataFile = valueStr;
        else if (key == "--epochs") epochs = atoi(valueStr.c_str());
        else if (key == "--lr") lr = atof(valueStr.c_str());
        else if (key == "--clip") clipVal = atof(valueStr.c_str());
    }

    // OpenCL device and context setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    err = clGetPlatformIDs(1, &platform, NULL); CL_CHECK(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) { err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL); }
    CL_CHECK(err);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err); CL_CHECK(err);


    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    CL_CHECK(err);

    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print the build log (always!)
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        build_log[log_size] = 0;
        std::cerr << "\nOpenCL build error (" << err << "):\n" << build_log.data() << std::endl;

        // Optionally log the full kernel source, with line numbers, for easier debugging:
        std::cerr << "\nKernel source was:\n";
        std::istringstream src(kernelSource);
        std::string line;
        int lno = 1;
        while (std::getline(src, line)) {
            std::cerr << std::setw(4) << lno++ << ": " << line << '\n';
        }
        exit(1);
    }

    // Kernel objects
    #define KMAKE(name) clCreateKernel(program, name, &err); CL_CHECK(err)
    cl_kernel k_simple_rnn_forward = KMAKE("k_simple_rnn_forward");
    cl_kernel k_lstm_forward       = KMAKE("k_lstm_forward");
    cl_kernel k_gru_gates          = KMAKE("k_gru_forward_gates");
    cl_kernel k_gru_hidden         = KMAKE("k_gru_forward_hidden");
    cl_kernel k_output_forward     = KMAKE("k_output_forward");
    cl_kernel k_zero               = KMAKE("k_zero_array");
    cl_kernel k_fill               = KMAKE("k_fill_array");
    cl_kernel k_concat             = KMAKE("k_concat");

    // Model construction
    RNNModelOpenCL model(inputSize, hiddenSize, outputSize, hiddenAct, outputAct, lossType,
                         cellType, lr, clipVal, bpttSteps, context, queue,
                         k_simple_rnn_forward, k_lstm_forward, k_gru_gates, k_gru_hidden,
                         k_output_forward, k_concat, k_zero, k_fill);

    // Command dispatcher
    if (cmdStr == "create") {
        if (!inputSize || !hiddenSize || !outputSize || saveFile.empty()) {
            std::cout << "Missing parameters for create\n"; return 1;
        }
        // Save model weights (add serialization)
        std::cout << "Model created. (Saving model logic not shown)\n";
    }
    else if (cmdStr == "predict") {
        // load model & data, run model.ForwardSequence(...)
        std::cout << "Prediction command not fully implemented.\n";
    }
    else if (cmdStr == "train") {
        std::cout << "Train command not fully implemented.\n";
    }
    else if (cmdStr == "info") {
        std::cout << "Model info: input=" << model.GetInputSize()
        << " hidden="  << model.GetHiddenSize()
        << " output="  << model.GetOutputSize()
        << " type="    << model.GetCellType() << "\n";
    }
    else if (cmdStr == "help") {
        std::cout << "Help: usage instructions ...\n";
    }

    // Cleanup
    clReleaseKernel(k_simple_rnn_forward);
    clReleaseKernel(k_lstm_forward);
    clReleaseKernel(k_gru_gates);
    clReleaseKernel(k_gru_hidden);
    clReleaseKernel(k_output_forward);
    clReleaseKernel(k_zero);
    clReleaseKernel(k_fill);
    clReleaseKernel(k_concat);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}

