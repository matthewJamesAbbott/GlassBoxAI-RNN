/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <cstdio>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                 << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    } while(0)

#define BLOCK_SIZE 256

// ========== Double atomicAdd for older GPUs ==========
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ double atomicAddDouble(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

// ========== Type Aliases ==========
using DArray = vector<double>;
using TDArray2D = vector<DArray>;
using TDArray3D = vector<TDArray2D>;
using TIntArray = vector<int>;

// ========== Enums ==========
enum TActivationType { atSigmoid, atTanh, atReLU, atLinear };
enum TLossType { ltMSE, ltCrossEntropy };
enum TCellType { ctSimpleRNN, ctLSTM, ctGRU };
enum TCommand { cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdHelp };

// ========== Data Structures ==========
struct TDataSplit {
    TDArray2D TrainInputs, TrainTargets;
    TDArray2D ValInputs, ValTargets;
};

struct TTimeStepCache {
    DArray Input;
    DArray H, C;
    DArray PreH;
    DArray F, I, CTilde, O, TanhC;
    DArray Z, R, HTilde;
    DArray OutVal, OutPre;
    TDArray2D LayerInputs;
};

// ========== CUDA Kernels ==========

__device__ double d_sigmoid(double x) {
    double clamped = fmax(-500.0, fmin(500.0, x));
    return 1.0 / (1.0 + exp(-clamped));
}

__device__ double d_tanh_act(double x) {
    return tanh(x);
}

__device__ double d_relu(double x) {
    return x > 0 ? x : 0;
}

__device__ double d_activation(double x, int actType) {
    switch (actType) {
        case 0: return d_sigmoid(x);  // atSigmoid
        case 1: return d_tanh_act(x); // atTanh
        case 2: return d_relu(x);     // atReLU
        case 3: return x;             // atLinear
        default: return x;
    }
}

__device__ double d_activation_derivative(double y, int actType) {
    switch (actType) {
        case 0: return y * (1.0 - y);     // atSigmoid
        case 1: return 1.0 - y * y;       // atTanh
        case 2: return y > 0 ? 1.0 : 0.0; // atReLU
        case 3: return 1.0;               // atLinear
        default: return 1.0;
    }
}

__device__ double d_clip(double v, double maxVal) {
    if (v > maxVal) return maxVal;
    else if (v < -maxVal) return -maxVal;
    else return v;
}

// Matrix-vector multiply: y = W * x + b
__global__ void k_matvec_add(double* y, const double* W, const double* x, const double* b,
                              int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        double sum = b[i];
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

// Apply activation element-wise
__global__ void k_activate(double* y, const double* x, int n, int actType) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = d_activation(x[i], actType);
    }
}

// LSTM forward kernel - computes gates and cell/hidden state
__global__ void k_lstm_forward(double* H, double* C, double* Fg, double* Ig,
                                double* CTilde, double* Og, double* TanhC,
                                const double* SumF, const double* SumI,
                                const double* SumC, const double* SumO,
                                const double* PrevC, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        Fg[k] = d_sigmoid(SumF[k]);
        Ig[k] = d_sigmoid(SumI[k]);
        CTilde[k] = tanh(SumC[k]);
        Og[k] = d_sigmoid(SumO[k]);
        C[k] = Fg[k] * PrevC[k] + Ig[k] * CTilde[k];
        TanhC[k] = tanh(C[k]);
        H[k] = Og[k] * TanhC[k];
    }
}

// GRU forward kernel - step 1: compute Z and R gates
__global__ void k_gru_gates(double* Z, double* R, const double* SumZ, const double* SumR, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        Z[k] = d_sigmoid(SumZ[k]);
        R[k] = d_sigmoid(SumR[k]);
    }
}

// GRU forward kernel - step 2: compute HTilde and H
__global__ void k_gru_hidden(double* H, double* HTilde, const double* SumH,
                              const double* Z, const double* PrevH, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        HTilde[k] = tanh(SumH[k]);
        H[k] = (1.0 - Z[k]) * PrevH[k] + Z[k] * HTilde[k];
    }
}

// Simple RNN forward kernel
__global__ void k_simple_rnn_forward(double* H, double* PreH, const double* Sum, int hiddenSize, int actType) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hiddenSize) {
        PreH[i] = Sum[i];
        H[i] = d_activation(Sum[i], actType);
    }
}

// Gradient accumulation kernels
__global__ void k_outer_product_add(double* dW, const double* dY, const double* X, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / cols;
    int j = idx % cols;
    if (i < rows && j < cols) {
        atomicAddDouble(&dW[i * cols + j], dY[i] * X[j]);
    }
}

__global__ void k_add_vector(double* dst, const double* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAddDouble(&dst[i], src[i]);
    }
}

__global__ void k_apply_gradients(double* W, double* dW, double lr, double clipVal, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double g = d_clip(dW[i], clipVal);
        W[i] -= lr * g;
        dW[i] = 0.0;
    }
}

__global__ void k_zero(double* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = 0.0;
    }
}

// ========== GPU Memory Helper ==========
class GPUArray {
public:
    double* d_ptr;
    int size;

    GPUArray() : d_ptr(nullptr), size(0) {}

    void allocate(int n) {
        if (d_ptr && size != n) {
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr = nullptr;
        }
        if (!d_ptr && n > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(double)));
        }
        size = n;
    }

    void free() {
        if (d_ptr) {
            cudaFree(d_ptr);
            d_ptr = nullptr;
            size = 0;
        }
    }

    void copyToDevice(const DArray& src) {
        allocate(src.size());
        CUDA_CHECK(cudaMemcpy(d_ptr, src.data(), src.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    void copyToDevice(const double* src, int n) {
        allocate(n);
        CUDA_CHECK(cudaMemcpy(d_ptr, src, n * sizeof(double), cudaMemcpyHostToDevice));
    }

    void copyToHost(DArray& dst) {
        dst.resize(size);
        CUDA_CHECK(cudaMemcpy(dst.data(), d_ptr, size * sizeof(double), cudaMemcpyDeviceToHost));
    }

    void zero() {
        if (d_ptr && size > 0) {
            int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            k_zero<<<blocks, BLOCK_SIZE>>>(d_ptr, size);
        }
    }

    ~GPUArray() { free(); }
};

class GPUMatrix {
public:
    double* d_ptr;
    int rows, cols;

    GPUMatrix() : d_ptr(nullptr), rows(0), cols(0) {}

    void allocate(int r, int c) {
        int n = r * c;
        if (d_ptr && (rows * cols) != n) {
            CUDA_CHECK(cudaFree(d_ptr));
            d_ptr = nullptr;
        }
        if (!d_ptr && n > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(double)));
        }
        rows = r;
        cols = c;
    }

    void free() {
        if (d_ptr) {
            cudaFree(d_ptr);
            d_ptr = nullptr;
            rows = cols = 0;
        }
    }

    void copyToDevice(const TDArray2D& src) {
        if (src.empty()) return;
        allocate(src.size(), src[0].size());
        vector<double> flat(rows * cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                flat[i * cols + j] = src[i][j];
        CUDA_CHECK(cudaMemcpy(d_ptr, flat.data(), flat.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    void copyToHost(TDArray2D& dst) {
        dst.resize(rows);
        vector<double> flat(rows * cols);
        CUDA_CHECK(cudaMemcpy(flat.data(), d_ptr, flat.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < rows; i++) {
            dst[i].resize(cols);
            for (int j = 0; j < cols; j++)
                dst[i][j] = flat[i * cols + j];
        }
    }

    void zero() {
        if (d_ptr && rows * cols > 0) {
            int n = rows * cols;
            int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            k_zero<<<blocks, BLOCK_SIZE>>>(d_ptr, n);
        }
    }

    ~GPUMatrix() { free(); }
};

// ========== Host-side Utility Functions ==========
double ClipValue(double V, double MaxVal) {
    if (V > MaxVal) return MaxVal;
    else if (V < -MaxVal) return -MaxVal;
    else return V;
}

double RandomWeight(double Scale) {
    return ((double)rand() / RAND_MAX - 0.5) * 2.0 * Scale;
}

void InitMatrix(TDArray2D& M, int Rows, int Cols, double Scale) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++)
            M[i][j] = RandomWeight(Scale);
    }
}

void ZeroMatrix(TDArray2D& M, int Rows, int Cols) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++)
            M[i][j] = 0.0;
    }
}

void ZeroArray(DArray& A, int Size) {
    A.resize(Size);
    for (int i = 0; i < Size; i++)
        A[i] = 0.0;
}

DArray ConcatArrays(const DArray& A, const DArray& B) {
    DArray Result(A.size() + B.size());
    for (size_t i = 0; i < A.size(); i++)
        Result[i] = A[i];
    for (size_t i = 0; i < B.size(); i++)
        Result[A.size() + i] = B[i];
    return Result;
}

// ========== TActivation (host) ==========
class TActivation {
public:
    static double Apply(double X, TActivationType ActType) {
        switch (ActType) {
            case atSigmoid: return 1.0 / (1.0 + exp(-max(-500.0, min(500.0, X))));
            case atTanh: return tanh(X);
            case atReLU: return X > 0 ? X : 0;
            case atLinear: return X;
            default: return X;
        }
    }

    static double Derivative(double Y, TActivationType ActType) {
        switch (ActType) {
            case atSigmoid: return Y * (1.0 - Y);
            case atTanh: return 1.0 - Y * Y;
            case atReLU: return Y > 0 ? 1.0 : 0.0;
            case atLinear: return 1.0;
            default: return 1.0;
        }
    }
};

// ========== TLoss (host) ==========
class TLoss {
public:
    static double Compute(const DArray& Pred, const DArray& Target, TLossType LossType) {
        double Result = 0;
        switch (LossType) {
            case ltMSE:
                for (size_t i = 0; i < Pred.size(); i++)
                    Result += (Pred[i] - Target[i]) * (Pred[i] - Target[i]);
                break;
            case ltCrossEntropy:
                for (size_t i = 0; i < Pred.size(); i++) {
                    double P = max(1e-15, min(1.0 - 1e-15, Pred[i]));
                    Result -= (Target[i] * log(P) + (1 - Target[i]) * log(1 - P));
                }
                break;
        }
        return Result / Pred.size();
    }

    static void Gradient(const DArray& Pred, const DArray& Target, TLossType LossType, DArray& Grad) {
        Grad.resize(Pred.size());
        switch (LossType) {
            case ltMSE:
                for (size_t i = 0; i < Pred.size(); i++)
                    Grad[i] = Pred[i] - Target[i];
                break;
            case ltCrossEntropy:
                for (size_t i = 0; i < Pred.size(); i++) {
                    double P = max(1e-15, min(1.0 - 1e-15, Pred[i]));
                    Grad[i] = (P - Target[i]) / (P * (1 - P) + 1e-15);
                }
                break;
        }
    }
};

// ========== TSimpleRNNCell (GPU) ==========
class TSimpleRNNCell {
public:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;
    TDArray2D Wih, Whh;
    DArray Bh;
    TDArray2D dWih, dWhh;
    DArray dBh;

    // GPU arrays
    GPUMatrix g_Wih, g_Whh, g_dWih, g_dWhh;
    GPUArray g_Bh, g_dBh;
    GPUArray g_Sum, g_H, g_PreH, g_Concat;

    TSimpleRNNCell(int InputSize, int HiddenSize, TActivationType Activation) {
        FInputSize = InputSize;
        FHiddenSize = HiddenSize;
        FActivation = Activation;
        double Scale = sqrt(2.0 / (InputSize + HiddenSize));
        InitMatrix(Wih, HiddenSize, InputSize, Scale);
        InitMatrix(Whh, HiddenSize, HiddenSize, Scale);
        ZeroArray(Bh, HiddenSize);
        ZeroMatrix(dWih, HiddenSize, InputSize);
        ZeroMatrix(dWhh, HiddenSize, HiddenSize);
        ZeroArray(dBh, HiddenSize);

        // Upload to GPU
        g_Wih.copyToDevice(Wih);
        g_Whh.copyToDevice(Whh);
        g_Bh.copyToDevice(Bh);
        g_dWih.allocate(HiddenSize, InputSize);
        g_dWhh.allocate(HiddenSize, HiddenSize);
        g_dBh.allocate(HiddenSize);
        g_dWih.zero();
        g_dWhh.zero();
        g_dBh.zero();

        g_Sum.allocate(HiddenSize);
        g_H.allocate(HiddenSize);
        g_PreH.allocate(HiddenSize);
        g_Concat.allocate(InputSize + HiddenSize);
    }

    void Forward(const DArray& Input, const DArray& PrevH, DArray& H, DArray& PreH) {
        // CPU fallback for simplicity - can be GPU accelerated
        H.resize(FHiddenSize);
        PreH.resize(FHiddenSize);
        for (int i = 0; i < FHiddenSize; i++) {
            double Sum = Bh[i];
            for (int j = 0; j < FInputSize; j++)
                Sum += Wih[i][j] * Input[j];
            for (int j = 0; j < FHiddenSize; j++)
                Sum += Whh[i][j] * PrevH[j];
            PreH[i] = Sum;
            H[i] = TActivation::Apply(Sum, FActivation);
        }
    }

    void Backward(const DArray& dH, const DArray& H, const DArray& PreH,
                  const DArray& PrevH, const DArray& Input, double ClipVal,
                  DArray& dInput, DArray& dPrevH) {
        DArray dHRaw(FHiddenSize);
        dInput.resize(FInputSize);
        dPrevH.resize(FHiddenSize);
        for (int i = 0; i < FInputSize; i++) dInput[i] = 0;
        for (int i = 0; i < FHiddenSize; i++) dPrevH[i] = 0;

        for (int i = 0; i < FHiddenSize; i++)
            dHRaw[i] = ClipValue(dH[i] * TActivation::Derivative(H[i], FActivation), ClipVal);

        for (int i = 0; i < FHiddenSize; i++) {
            for (int j = 0; j < FInputSize; j++) {
                dWih[i][j] += dHRaw[i] * Input[j];
                dInput[j] += Wih[i][j] * dHRaw[i];
            }
            for (int j = 0; j < FHiddenSize; j++) {
                dWhh[i][j] += dHRaw[i] * PrevH[j];
                dPrevH[j] += Whh[i][j] * dHRaw[i];
            }
            dBh[i] += dHRaw[i];
        }
    }

    void ApplyGradients(double LR, double ClipVal) {
        for (int i = 0; i < FHiddenSize; i++) {
            for (int j = 0; j < FInputSize; j++) {
                Wih[i][j] -= LR * ClipValue(dWih[i][j], ClipVal);
                dWih[i][j] = 0;
            }
            for (int j = 0; j < FHiddenSize; j++) {
                Whh[i][j] -= LR * ClipValue(dWhh[i][j], ClipVal);
                dWhh[i][j] = 0;
            }
            Bh[i] -= LR * ClipValue(dBh[i], ClipVal);
            dBh[i] = 0;
        }
        g_Wih.copyToDevice(Wih);
        g_Whh.copyToDevice(Whh);
        g_Bh.copyToDevice(Bh);
    }

    void ResetGradients() {
        ZeroMatrix(dWih, FHiddenSize, FInputSize);
        ZeroMatrix(dWhh, FHiddenSize, FHiddenSize);
        ZeroArray(dBh, FHiddenSize);
    }

    int GetHiddenSize() { return FHiddenSize; }
};

// ========== TLSTMCell (GPU-accelerated) ==========
class TLSTMCell {
public:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;
    TDArray2D Wf, Wi, Wc, Wo;
    DArray Bf, Bi, Bc, Bo;
    TDArray2D dWf, dWi, dWc, dWo;
    DArray dBf, dBi, dBc, dBo;

    // GPU arrays
    GPUMatrix g_Wf, g_Wi, g_Wc, g_Wo;
    GPUArray g_Bf, g_Bi, g_Bc, g_Bo;
    GPUMatrix g_dWf, g_dWi, g_dWc, g_dWo;
    GPUArray g_dBf, g_dBi, g_dBc, g_dBo;
    GPUArray g_SumF, g_SumI, g_SumC, g_SumO;
    GPUArray g_H, g_C, g_Fg, g_Ig, g_CTilde, g_Og, g_TanhC;
    GPUArray g_Concat, g_PrevH, g_PrevC;

    TLSTMCell(int InputSize, int HiddenSize, TActivationType Activation) {
        FInputSize = InputSize;
        FHiddenSize = HiddenSize;
        FActivation = Activation;
        int ConcatSize = InputSize + HiddenSize;
        double Scale = sqrt(2.0 / ConcatSize);

        InitMatrix(Wf, HiddenSize, ConcatSize, Scale);
        InitMatrix(Wi, HiddenSize, ConcatSize, Scale);
        InitMatrix(Wc, HiddenSize, ConcatSize, Scale);
        InitMatrix(Wo, HiddenSize, ConcatSize, Scale);

        Bf.resize(HiddenSize);
        Bi.resize(HiddenSize);
        Bc.resize(HiddenSize);
        Bo.resize(HiddenSize);
        for (int i = 0; i < HiddenSize; i++) {
            Bf[i] = 1.0;
            Bi[i] = 0;
            Bc[i] = 0;
            Bo[i] = 0;
        }

        ZeroMatrix(dWf, HiddenSize, ConcatSize);
        ZeroMatrix(dWi, HiddenSize, ConcatSize);
        ZeroMatrix(dWc, HiddenSize, ConcatSize);
        ZeroMatrix(dWo, HiddenSize, ConcatSize);
        ZeroArray(dBf, HiddenSize);
        ZeroArray(dBi, HiddenSize);
        ZeroArray(dBc, HiddenSize);
        ZeroArray(dBo, HiddenSize);

        // Upload to GPU
        g_Wf.copyToDevice(Wf); g_Wi.copyToDevice(Wi);
        g_Wc.copyToDevice(Wc); g_Wo.copyToDevice(Wo);
        g_Bf.copyToDevice(Bf); g_Bi.copyToDevice(Bi);
        g_Bc.copyToDevice(Bc); g_Bo.copyToDevice(Bo);

        g_dWf.allocate(HiddenSize, ConcatSize); g_dWf.zero();
        g_dWi.allocate(HiddenSize, ConcatSize); g_dWi.zero();
        g_dWc.allocate(HiddenSize, ConcatSize); g_dWc.zero();
        g_dWo.allocate(HiddenSize, ConcatSize); g_dWo.zero();
        g_dBf.allocate(HiddenSize); g_dBf.zero();
        g_dBi.allocate(HiddenSize); g_dBi.zero();
        g_dBc.allocate(HiddenSize); g_dBc.zero();
        g_dBo.allocate(HiddenSize); g_dBo.zero();

        g_SumF.allocate(HiddenSize); g_SumI.allocate(HiddenSize);
        g_SumC.allocate(HiddenSize); g_SumO.allocate(HiddenSize);
        g_H.allocate(HiddenSize); g_C.allocate(HiddenSize);
        g_Fg.allocate(HiddenSize); g_Ig.allocate(HiddenSize);
        g_CTilde.allocate(HiddenSize); g_Og.allocate(HiddenSize);
        g_TanhC.allocate(HiddenSize);
        g_Concat.allocate(ConcatSize);
        g_PrevH.allocate(HiddenSize); g_PrevC.allocate(HiddenSize);
    }

    void Forward(const DArray& Input, const DArray& PrevH, const DArray& PrevC,
                 DArray& H, DArray& C, DArray& Fg, DArray& Ig, DArray& CTilde,
                 DArray& Og, DArray& TanhC) {
        int ConcatSize = FInputSize + FHiddenSize;
        DArray Concat = ConcatArrays(Input, PrevH);

        // Upload concat and prev states
        g_Concat.copyToDevice(Concat);
        g_PrevC.copyToDevice(PrevC);

        int blocks = (FHiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Compute gate sums: SumX = Wx * Concat + Bx
        k_matvec_add<<<blocks, BLOCK_SIZE>>>(g_SumF.d_ptr, g_Wf.d_ptr, g_Concat.d_ptr, g_Bf.d_ptr, FHiddenSize, ConcatSize);
        k_matvec_add<<<blocks, BLOCK_SIZE>>>(g_SumI.d_ptr, g_Wi.d_ptr, g_Concat.d_ptr, g_Bi.d_ptr, FHiddenSize, ConcatSize);
        k_matvec_add<<<blocks, BLOCK_SIZE>>>(g_SumC.d_ptr, g_Wc.d_ptr, g_Concat.d_ptr, g_Bc.d_ptr, FHiddenSize, ConcatSize);
        k_matvec_add<<<blocks, BLOCK_SIZE>>>(g_SumO.d_ptr, g_Wo.d_ptr, g_Concat.d_ptr, g_Bo.d_ptr, FHiddenSize, ConcatSize);

        // Compute gates, cell, and hidden state
        k_lstm_forward<<<blocks, BLOCK_SIZE>>>(g_H.d_ptr, g_C.d_ptr, g_Fg.d_ptr, g_Ig.d_ptr,
                                                g_CTilde.d_ptr, g_Og.d_ptr, g_TanhC.d_ptr,
                                                g_SumF.d_ptr, g_SumI.d_ptr, g_SumC.d_ptr, g_SumO.d_ptr,
                                                g_PrevC.d_ptr, FHiddenSize);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Download results
        g_H.copyToHost(H);
        g_C.copyToHost(C);
        g_Fg.copyToHost(Fg);
        g_Ig.copyToHost(Ig);
        g_CTilde.copyToHost(CTilde);
        g_Og.copyToHost(Og);
        g_TanhC.copyToHost(TanhC);
    }

    void Backward(const DArray& dH, const DArray& dC, const DArray& H, const DArray& C,
                  const DArray& Fg, const DArray& Ig, const DArray& CTilde,
                  const DArray& Og, const DArray& TanhC, const DArray& PrevH,
                  const DArray& PrevC, const DArray& Input, double ClipVal,
                  DArray& dInput, DArray& dPrevH, DArray& dPrevC) {
        // CPU backward for now (can be GPU accelerated)
        DArray Concat = ConcatArrays(Input, PrevH);
        int ConcatSize = Concat.size();
        DArray dOg(FHiddenSize), dCTotal(FHiddenSize), dFg(FHiddenSize);
        DArray dIg(FHiddenSize), dCTilde(FHiddenSize);
        dInput.resize(FInputSize);
        dPrevH.resize(FHiddenSize);
        dPrevC.resize(FHiddenSize);

        for (int k = 0; k < FInputSize; k++) dInput[k] = 0;
        for (int k = 0; k < FHiddenSize; k++) {
            dPrevH[k] = 0;
            dPrevC[k] = 0;
        }

        for (int k = 0; k < FHiddenSize; k++) {
            dOg[k] = ClipValue(dH[k] * TanhC[k] * TActivation::Derivative(Og[k], atSigmoid), ClipVal);
            dCTotal[k] = ClipValue(dH[k] * Og[k] * (1 - TanhC[k] * TanhC[k]) + dC[k], ClipVal);
            dFg[k] = ClipValue(dCTotal[k] * PrevC[k] * TActivation::Derivative(Fg[k], atSigmoid), ClipVal);
            dIg[k] = ClipValue(dCTotal[k] * CTilde[k] * TActivation::Derivative(Ig[k], atSigmoid), ClipVal);
            dCTilde[k] = ClipValue(dCTotal[k] * Ig[k] * TActivation::Derivative(CTilde[k], atTanh), ClipVal);
            dPrevC[k] = dCTotal[k] * Fg[k];
        }

        for (int k = 0; k < FHiddenSize; k++) {
            for (int j = 0; j < ConcatSize; j++) {
                dWf[k][j] += dFg[k] * Concat[j];
                dWi[k][j] += dIg[k] * Concat[j];
                dWc[k][j] += dCTilde[k] * Concat[j];
                dWo[k][j] += dOg[k] * Concat[j];

                if (j < FInputSize)
                    dInput[j] += Wf[k][j] * dFg[k] + Wi[k][j] * dIg[k] +
                                 Wc[k][j] * dCTilde[k] + Wo[k][j] * dOg[k];
                else
                    dPrevH[j - FInputSize] += Wf[k][j] * dFg[k] + Wi[k][j] * dIg[k] +
                                               Wc[k][j] * dCTilde[k] + Wo[k][j] * dOg[k];
            }
            dBf[k] += dFg[k];
            dBi[k] += dIg[k];
            dBc[k] += dCTilde[k];
            dBo[k] += dOg[k];
        }
    }

    void ApplyGradients(double LR, double ClipVal) {
        int ConcatSize = FInputSize + FHiddenSize;
        for (int k = 0; k < FHiddenSize; k++) {
            for (int j = 0; j < ConcatSize; j++) {
                Wf[k][j] -= LR * ClipValue(dWf[k][j], ClipVal);
                Wi[k][j] -= LR * ClipValue(dWi[k][j], ClipVal);
                Wc[k][j] -= LR * ClipValue(dWc[k][j], ClipVal);
                Wo[k][j] -= LR * ClipValue(dWo[k][j], ClipVal);
                dWf[k][j] = 0; dWi[k][j] = 0; dWc[k][j] = 0; dWo[k][j] = 0;
            }
            Bf[k] -= LR * ClipValue(dBf[k], ClipVal);
            Bi[k] -= LR * ClipValue(dBi[k], ClipVal);
            Bc[k] -= LR * ClipValue(dBc[k], ClipVal);
            Bo[k] -= LR * ClipValue(dBo[k], ClipVal);
            dBf[k] = 0; dBi[k] = 0; dBc[k] = 0; dBo[k] = 0;
        }
        // Re-upload weights
        g_Wf.copyToDevice(Wf); g_Wi.copyToDevice(Wi);
        g_Wc.copyToDevice(Wc); g_Wo.copyToDevice(Wo);
        g_Bf.copyToDevice(Bf); g_Bi.copyToDevice(Bi);
        g_Bc.copyToDevice(Bc); g_Bo.copyToDevice(Bo);
    }

    void ResetGradients() {
        int ConcatSize = FInputSize + FHiddenSize;
        ZeroMatrix(dWf, FHiddenSize, ConcatSize);
        ZeroMatrix(dWi, FHiddenSize, ConcatSize);
        ZeroMatrix(dWc, FHiddenSize, ConcatSize);
        ZeroMatrix(dWo, FHiddenSize, ConcatSize);
        ZeroArray(dBf, FHiddenSize);
        ZeroArray(dBi, FHiddenSize);
        ZeroArray(dBc, FHiddenSize);
        ZeroArray(dBo, FHiddenSize);
    }

    int GetHiddenSize() { return FHiddenSize; }
};

// ========== TGRUCell (GPU-accelerated) ==========
class TGRUCell {
public:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;
    TDArray2D Wz, Wr, Wh;
    DArray Bz, Br, Bh;
    TDArray2D dWz, dWr, dWh;
    DArray dBz, dBr, dBh;

    GPUMatrix g_Wz, g_Wr, g_Wh;
    GPUArray g_Bz, g_Br, g_Bh;
    GPUArray g_SumZ, g_SumR, g_SumH;
    GPUArray g_H, g_Z, g_R, g_HTilde;
    GPUArray g_Concat, g_ConcatR, g_PrevH;

    TGRUCell(int InputSize, int HiddenSize, TActivationType Activation) {
        FInputSize = InputSize;
        FHiddenSize = HiddenSize;
        FActivation = Activation;
        int ConcatSize = InputSize + HiddenSize;
        double Scale = sqrt(2.0 / ConcatSize);

        InitMatrix(Wz, HiddenSize, ConcatSize, Scale);
        InitMatrix(Wr, HiddenSize, ConcatSize, Scale);
        InitMatrix(Wh, HiddenSize, ConcatSize, Scale);

        ZeroArray(Bz, HiddenSize);
        ZeroArray(Br, HiddenSize);
        ZeroArray(Bh, HiddenSize);

        ZeroMatrix(dWz, HiddenSize, ConcatSize);
        ZeroMatrix(dWr, HiddenSize, ConcatSize);
        ZeroMatrix(dWh, HiddenSize, ConcatSize);
        ZeroArray(dBz, HiddenSize);
        ZeroArray(dBr, HiddenSize);
        ZeroArray(dBh, HiddenSize);

        g_Wz.copyToDevice(Wz); g_Wr.copyToDevice(Wr); g_Wh.copyToDevice(Wh);
        g_Bz.copyToDevice(Bz); g_Br.copyToDevice(Br); g_Bh.copyToDevice(Bh);
        g_SumZ.allocate(HiddenSize); g_SumR.allocate(HiddenSize); g_SumH.allocate(HiddenSize);
        g_H.allocate(HiddenSize); g_Z.allocate(HiddenSize);
        g_R.allocate(HiddenSize); g_HTilde.allocate(HiddenSize);
        g_Concat.allocate(ConcatSize); g_ConcatR.allocate(ConcatSize);
        g_PrevH.allocate(HiddenSize);
    }

    void Forward(const DArray& Input, const DArray& PrevH,
                 DArray& H, DArray& Z, DArray& R, DArray& HTilde) {
        int ConcatSize = FInputSize + FHiddenSize;
        DArray Concat = ConcatArrays(Input, PrevH);

        g_Concat.copyToDevice(Concat);
        g_PrevH.copyToDevice(PrevH);

        int blocks = (FHiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Compute Z and R gate sums
        k_matvec_add<<<blocks, BLOCK_SIZE>>>(g_SumZ.d_ptr, g_Wz.d_ptr, g_Concat.d_ptr, g_Bz.d_ptr, FHiddenSize, ConcatSize);
        k_matvec_add<<<blocks, BLOCK_SIZE>>>(g_SumR.d_ptr, g_Wr.d_ptr, g_Concat.d_ptr, g_Br.d_ptr, FHiddenSize, ConcatSize);

        // Apply sigmoid to get Z and R
        k_gru_gates<<<blocks, BLOCK_SIZE>>>(g_Z.d_ptr, g_R.d_ptr, g_SumZ.d_ptr, g_SumR.d_ptr, FHiddenSize);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Download R to build ConcatR
        g_R.copyToHost(R);
        DArray ConcatR(ConcatSize);
        for (int k = 0; k < FInputSize; k++)
            ConcatR[k] = Input[k];
        for (int k = 0; k < FHiddenSize; k++)
            ConcatR[FInputSize + k] = R[k] * PrevH[k];
        g_ConcatR.copyToDevice(ConcatR);

        // Compute H candidate sum
        k_matvec_add<<<blocks, BLOCK_SIZE>>>(g_SumH.d_ptr, g_Wh.d_ptr, g_ConcatR.d_ptr, g_Bh.d_ptr, FHiddenSize, ConcatSize);

        // Compute HTilde and H
        k_gru_hidden<<<blocks, BLOCK_SIZE>>>(g_H.d_ptr, g_HTilde.d_ptr, g_SumH.d_ptr, g_Z.d_ptr, g_PrevH.d_ptr, FHiddenSize);

        CUDA_CHECK(cudaDeviceSynchronize());

        g_H.copyToHost(H);
        g_Z.copyToHost(Z);
        g_HTilde.copyToHost(HTilde);
    }

    void Backward(const DArray& dH, const DArray& H, const DArray& Z,
                  const DArray& R, const DArray& HTilde, const DArray& PrevH,
                  const DArray& Input, double ClipVal,
                  DArray& dInput, DArray& dPrevH) {
        // CPU backward
        DArray Concat = ConcatArrays(Input, PrevH);
        int ConcatSize = Concat.size();

        DArray ConcatR(ConcatSize);
        for (int k = 0; k < FInputSize; k++)
            ConcatR[k] = Input[k];
        for (int k = 0; k < FHiddenSize; k++)
            ConcatR[FInputSize + k] = R[k] * PrevH[k];

        DArray dZ(FHiddenSize), dR(FHiddenSize), dHTilde(FHiddenSize);
        dInput.resize(FInputSize);
        dPrevH.resize(FHiddenSize);

        for (int k = 0; k < FInputSize; k++) dInput[k] = 0;
        for (int k = 0; k < FHiddenSize; k++) {
            dPrevH[k] = dH[k] * (1 - Z[k]);
            dR[k] = 0;
        }

        for (int k = 0; k < FHiddenSize; k++) {
            dHTilde[k] = ClipValue(dH[k] * Z[k] * TActivation::Derivative(HTilde[k], atTanh), ClipVal);
            dZ[k] = ClipValue(dH[k] * (HTilde[k] - PrevH[k]) * TActivation::Derivative(Z[k], atSigmoid), ClipVal);
        }

        for (int k = 0; k < FHiddenSize; k++) {
            for (int j = 0; j < ConcatSize; j++) {
                dWh[k][j] += dHTilde[k] * ConcatR[j];
                if (j < FInputSize)
                    dInput[j] += Wh[k][j] * dHTilde[k];
                else {
                    dR[j - FInputSize] += Wh[k][j] * dHTilde[k] * PrevH[j - FInputSize];
                    dPrevH[j - FInputSize] += Wh[k][j] * dHTilde[k] * R[j - FInputSize];
                }
            }
            dBh[k] += dHTilde[k];
        }

        for (int k = 0; k < FHiddenSize; k++)
            dR[k] = ClipValue(dR[k] * TActivation::Derivative(R[k], atSigmoid), ClipVal);

        for (int k = 0; k < FHiddenSize; k++) {
            for (int j = 0; j < ConcatSize; j++) {
                dWz[k][j] += dZ[k] * Concat[j];
                dWr[k][j] += dR[k] * Concat[j];
                if (j < FInputSize)
                    dInput[j] += Wz[k][j] * dZ[k] + Wr[k][j] * dR[k];
                else
                    dPrevH[j - FInputSize] += Wz[k][j] * dZ[k] + Wr[k][j] * dR[k];
            }
            dBz[k] += dZ[k];
            dBr[k] += dR[k];
        }
    }

    void ApplyGradients(double LR, double ClipVal) {
        int ConcatSize = FInputSize + FHiddenSize;
        for (int k = 0; k < FHiddenSize; k++) {
            for (int j = 0; j < ConcatSize; j++) {
                Wz[k][j] -= LR * ClipValue(dWz[k][j], ClipVal);
                Wr[k][j] -= LR * ClipValue(dWr[k][j], ClipVal);
                Wh[k][j] -= LR * ClipValue(dWh[k][j], ClipVal);
                dWz[k][j] = 0; dWr[k][j] = 0; dWh[k][j] = 0;
            }
            Bz[k] -= LR * ClipValue(dBz[k], ClipVal);
            Br[k] -= LR * ClipValue(dBr[k], ClipVal);
            Bh[k] -= LR * ClipValue(dBh[k], ClipVal);
            dBz[k] = 0; dBr[k] = 0; dBh[k] = 0;
        }
        g_Wz.copyToDevice(Wz); g_Wr.copyToDevice(Wr); g_Wh.copyToDevice(Wh);
        g_Bz.copyToDevice(Bz); g_Br.copyToDevice(Br); g_Bh.copyToDevice(Bh);
    }

    void ResetGradients() {
        int ConcatSize = FInputSize + FHiddenSize;
        ZeroMatrix(dWz, FHiddenSize, ConcatSize);
        ZeroMatrix(dWr, FHiddenSize, ConcatSize);
        ZeroMatrix(dWh, FHiddenSize, ConcatSize);
        ZeroArray(dBz, FHiddenSize);
        ZeroArray(dBr, FHiddenSize);
        ZeroArray(dBh, FHiddenSize);
    }

    int GetHiddenSize() { return FHiddenSize; }
};

// ========== TOutputLayer ==========
class TOutputLayer {
public:
    int FInputSize, FOutputSize;
    TActivationType FActivation;
    TDArray2D W;
    DArray B;
    TDArray2D dW;
    DArray dB;

    GPUMatrix g_W;
    GPUArray g_B, g_Pre, g_Out;

    TOutputLayer(int InputSize, int OutputSize, TActivationType Activation) {
        FInputSize = InputSize;
        FOutputSize = OutputSize;
        FActivation = Activation;
        double Scale = sqrt(2.0 / InputSize);
        InitMatrix(W, OutputSize, InputSize, Scale);
        ZeroArray(B, OutputSize);
        ZeroMatrix(dW, OutputSize, InputSize);
        ZeroArray(dB, OutputSize);

        g_W.copyToDevice(W);
        g_B.copyToDevice(B);
        g_Pre.allocate(OutputSize);
        g_Out.allocate(OutputSize);
    }

    void Forward(const DArray& Input, DArray& Output, DArray& Pre) {
        Pre.resize(FOutputSize);
        Output.resize(FOutputSize);
        for (int i = 0; i < FOutputSize; i++) {
            double Sum = B[i];
            for (int j = 0; j < FInputSize; j++)
                Sum += W[i][j] * Input[j];
            Pre[i] = Sum;
        }

        if (FActivation == atLinear) {
            for (int i = 0; i < FOutputSize; i++)
                Output[i] = Pre[i];
        } else {
            for (int i = 0; i < FOutputSize; i++)
                Output[i] = TActivation::Apply(Pre[i], FActivation);
        }
    }

    void Backward(const DArray& dOut, const DArray& Output, const DArray& Pre,
                  const DArray& Input, double ClipVal, DArray& dInput) {
        DArray dPre(FOutputSize);
        dInput.resize(FInputSize);
        for (int j = 0; j < FInputSize; j++) dInput[j] = 0;

        for (int i = 0; i < FOutputSize; i++)
            dPre[i] = ClipValue(dOut[i] * TActivation::Derivative(Output[i], FActivation), ClipVal);

        for (int i = 0; i < FOutputSize; i++) {
            for (int j = 0; j < FInputSize; j++) {
                dW[i][j] += dPre[i] * Input[j];
                dInput[j] += W[i][j] * dPre[i];
            }
            dB[i] += dPre[i];
        }
    }

    void ApplyGradients(double LR, double ClipVal) {
        for (int i = 0; i < FOutputSize; i++) {
            for (int j = 0; j < FInputSize; j++) {
                W[i][j] -= LR * ClipValue(dW[i][j], ClipVal);
                dW[i][j] = 0;
            }
            B[i] -= LR * ClipValue(dB[i], ClipVal);
            dB[i] = 0;
        }
        g_W.copyToDevice(W);
        g_B.copyToDevice(B);
    }

    void ResetGradients() {
        ZeroMatrix(dW, FOutputSize, FInputSize);
        ZeroArray(dB, FOutputSize);
    }
};

// ========== Forward Declarations ==========
string CellTypeToStr(TCellType ct);
string ActivationToStr(TActivationType act);
string LossToStr(TLossType loss);
TCellType ParseCellType(const string& s);
TActivationType ParseActivation(const string& s);
TLossType ParseLoss(const string& s);
static string ExtractJSONValue(const string& json, const string& key);

// ========== TAdvancedRNN ==========
class TAdvancedRNN {
public:
    int FInputSize, FOutputSize;
    vector<int> FHiddenSizes;
    TCellType FCellType;
    TActivationType FActivation;
    TActivationType FOutputActivation;
    TLossType FLossType;
    double FLearningRate;
    double FGradientClip;
    int FBPTTSteps;

    vector<TSimpleRNNCell*> FSimpleCells;
    vector<TLSTMCell*> FLSTMCells;
    vector<TGRUCell*> FGRUCells;
    TOutputLayer* FOutputLayer;
    TDArray3D FStates;

public:
    TAdvancedRNN(int InputSize, const vector<int>& HiddenSizes, int OutputSize,
                 TCellType CellType, TActivationType Activation,
                 TActivationType OutputActivation, TLossType LossType,
                 double LearningRate, double GradientClip, int BPTTSteps) {
        FInputSize = InputSize;
        FOutputSize = OutputSize;
        FHiddenSizes = HiddenSizes;
        FCellType = CellType;
        FActivation = Activation;
        FOutputActivation = OutputActivation;
        FLossType = LossType;
        FLearningRate = LearningRate;
        FGradientClip = GradientClip;
        FBPTTSteps = BPTTSteps;

        int PrevSize = InputSize;
        switch (CellType) {
            case ctSimpleRNN:
                for (size_t i = 0; i < HiddenSizes.size(); i++) {
                    FSimpleCells.push_back(new TSimpleRNNCell(PrevSize, HiddenSizes[i], Activation));
                    PrevSize = HiddenSizes[i];
                }
                break;
            case ctLSTM:
                for (size_t i = 0; i < HiddenSizes.size(); i++) {
                    FLSTMCells.push_back(new TLSTMCell(PrevSize, HiddenSizes[i], Activation));
                    PrevSize = HiddenSizes[i];
                }
                break;
            case ctGRU:
                for (size_t i = 0; i < HiddenSizes.size(); i++) {
                    FGRUCells.push_back(new TGRUCell(PrevSize, HiddenSizes[i], Activation));
                    PrevSize = HiddenSizes[i];
                }
                break;
        }

        FOutputLayer = new TOutputLayer(PrevSize, OutputSize, OutputActivation);
    }

    ~TAdvancedRNN() {
        switch (FCellType) {
            case ctSimpleRNN:
                for (auto cell : FSimpleCells) delete cell;
                break;
            case ctLSTM:
                for (auto cell : FLSTMCells) delete cell;
                break;
            case ctGRU:
                for (auto cell : FGRUCells) delete cell;
                break;
        }
        delete FOutputLayer;
    }

    TDArray3D InitHiddenStates() {
        TDArray3D Result(FHiddenSizes.size());
        for (size_t i = 0; i < FHiddenSizes.size(); i++) {
            Result[i].resize(2);
            ZeroArray(Result[i][0], FHiddenSizes[i]);
            ZeroArray(Result[i][1], FHiddenSizes[i]);
        }
        return Result;
    }

    TDArray2D ForwardSequence(const TDArray2D& Inputs, vector<TTimeStepCache>& Caches,
                               TDArray3D& States) {
        TDArray2D Result(Inputs.size());
        TDArray3D NewStates = InitHiddenStates();

        for (size_t t = 0; t < Inputs.size(); t++) {
            DArray X = Inputs[t];
            Caches[t].Input = X;

            DArray H, C, PreH, Fg, Ig, CTilde, Og, TanhC, Z, R, HTilde;

            for (size_t layer = 0; layer < FHiddenSizes.size(); layer++) {
                switch (FCellType) {
                    case ctSimpleRNN:
                        FSimpleCells[layer]->Forward(X, States[layer][0], H, PreH);
                        NewStates[layer][0] = H;
                        Caches[t].H = H;
                        Caches[t].PreH = PreH;
                        break;
                    case ctLSTM:
                        FLSTMCells[layer]->Forward(X, States[layer][0], States[layer][1],
                                                    H, C, Fg, Ig, CTilde, Og, TanhC);
                        NewStates[layer][0] = H;
                        NewStates[layer][1] = C;
                        Caches[t].H = H;
                        Caches[t].C = C;
                        Caches[t].F = Fg;
                        Caches[t].I = Ig;
                        Caches[t].CTilde = CTilde;
                        Caches[t].O = Og;
                        Caches[t].TanhC = TanhC;
                        break;
                    case ctGRU:
                        FGRUCells[layer]->Forward(X, States[layer][0], H, Z, R, HTilde);
                        NewStates[layer][0] = H;
                        Caches[t].H = H;
                        Caches[t].Z = Z;
                        Caches[t].R = R;
                        Caches[t].HTilde = HTilde;
                        break;
                }
                X = H;
            }

            DArray OutVal, OutPre;
            FOutputLayer->Forward(X, OutVal, OutPre);
            Caches[t].OutVal = OutVal;
            Caches[t].OutPre = OutPre;
            Result[t] = OutVal;

            States = NewStates;
        }
        return Result;
    }

    double BackwardSequence(const TDArray2D& Targets, const vector<TTimeStepCache>& Caches,
                            const TDArray3D& States) {
        int T_len = Targets.size();
        int BPTTLimit = FBPTTSteps > 0 ? FBPTTSteps : T_len;

        double TotalLoss = 0;

        TDArray2D dStatesH(FHiddenSizes.size());
        TDArray2D dStatesC(FHiddenSizes.size());
        for (size_t layer = 0; layer < FHiddenSizes.size(); layer++) {
            ZeroArray(dStatesH[layer], FHiddenSizes[layer]);
            ZeroArray(dStatesC[layer], FHiddenSizes[layer]);
        }

        for (int t = T_len - 1; t >= max(0, T_len - BPTTLimit); t--) {
            TotalLoss += TLoss::Compute(Caches[t].OutVal, Targets[t], FLossType);
            DArray Grad;
            TLoss::Gradient(Caches[t].OutVal, Targets[t], FLossType, Grad);

            DArray dH;
            FOutputLayer->Backward(Grad, Caches[t].OutVal, Caches[t].OutPre, Caches[t].H, FGradientClip, dH);

            for (int layer = FHiddenSizes.size() - 1; layer >= 0; layer--) {
                DArray dOut(FHiddenSizes[layer]);
                for (int k = 0; k < FHiddenSizes[layer]; k++)
                    dOut[k] = dH[k] + dStatesH[layer][k];

                DArray PrevH, PrevC;
                if (t > 0)
                    PrevH = Caches[t-1].H;
                else
                    ZeroArray(PrevH, FHiddenSizes[layer]);

                DArray dInput, dPrevH, dPrevC;
                switch (FCellType) {
                    case ctSimpleRNN:
                        FSimpleCells[layer]->Backward(dOut, Caches[t].H, Caches[t].PreH, PrevH,
                                                       Caches[t].Input, FGradientClip, dInput, dPrevH);
                        dStatesH[layer] = dPrevH;
                        break;
                    case ctLSTM:
                        if (t > 0)
                            PrevC = Caches[t-1].C;
                        else
                            ZeroArray(PrevC, FHiddenSizes[layer]);

                        {
                            DArray dC(FHiddenSizes[layer]);
                            for (int k = 0; k < FHiddenSizes[layer]; k++)
                                dC[k] = dStatesC[layer][k];

                            FLSTMCells[layer]->Backward(dOut, dC, Caches[t].H, Caches[t].C,
                                                         Caches[t].F, Caches[t].I, Caches[t].CTilde,
                                                         Caches[t].O, Caches[t].TanhC,
                                                         PrevH, PrevC, Caches[t].Input,
                                                         FGradientClip, dInput, dPrevH, dPrevC);
                            dStatesH[layer] = dPrevH;
                            dStatesC[layer] = dPrevC;
                        }
                        break;
                    case ctGRU:
                        FGRUCells[layer]->Backward(dOut, Caches[t].H, Caches[t].Z, Caches[t].R,
                                                    Caches[t].HTilde, PrevH, Caches[t].Input,
                                                    FGradientClip, dInput, dPrevH);
                        dStatesH[layer] = dPrevH;
                        break;
                }

                dH = dInput;
            }
        }

        return TotalLoss / T_len;
    }

    double TrainSequence(const TDArray2D& Inputs, const TDArray2D& Targets) {
        ResetGradients();
        vector<TTimeStepCache> Caches(Inputs.size());
        TDArray3D States = InitHiddenStates();
        ForwardSequence(Inputs, Caches, States);
        double loss = BackwardSequence(Targets, Caches, States);
        ApplyGradients();
        return loss;
    }

    TDArray2D Predict(const TDArray2D& Inputs) {
        vector<TTimeStepCache> Caches(Inputs.size());
        TDArray3D States = InitHiddenStates();
        return ForwardSequence(Inputs, Caches, States);
    }

    double ComputeLoss(const TDArray2D& Inputs, const TDArray2D& Targets) {
        TDArray2D Outputs = Predict(Inputs);
        double Result = 0;
        for (size_t t = 0; t < Outputs.size(); t++)
            Result += TLoss::Compute(Outputs[t], Targets[t], FLossType);
        return Result / Outputs.size();
    }

    string Array1DToJSON(const DArray& Arr) {
        string Result = "[";
        for (size_t i = 0; i < Arr.size(); ++i) {
            if (i > 0) Result += ",";
            char buf[32];
            snprintf(buf, sizeof(buf), "%.17g", Arr[i]);
            Result += buf;
        }
        Result += "]";
        return Result;
    }

    string Array2DToJSON(const TDArray2D& Arr) {
        string Result = "[";
        for (size_t i = 0; i < Arr.size(); ++i) {
            if (i > 0) Result += ",";
            Result += Array1DToJSON(Arr[i]);
        }
        Result += "]";
        return Result;
    }

    void SaveModelToJSON(const string& Filename) {
        ofstream file(Filename);
        
        file << "{\n";
        file << "  \"input_size\": " << FInputSize << ",\n";
        file << "  \"output_size\": " << FOutputSize << ",\n";
        file << "  \"hidden_sizes\": [\n";
        
        for (size_t i = 0; i < FHiddenSizes.size(); ++i) {
            if (i > 0) file << ",\n";
            file << "    " << FHiddenSizes[i];
        }
        file << "\n  ],\n";
        
        file << "  \"cell_type\": \"" << CellTypeToStr(FCellType) << "\",\n";
        file << "  \"activation\": \"" << ActivationToStr(FActivation) << "\",\n";
        file << "  \"output_activation\": \"" << ActivationToStr(FOutputActivation) << "\",\n";
        file << "  \"loss_type\": \"" << LossToStr(FLossType) << "\",\n";
        file << fixed << setprecision(17);
        file << "  \"learning_rate\": " << FLearningRate << ",\n";
        file << "  \"gradient_clip\": " << FGradientClip << ",\n";
        file << "  \"bptt_steps\": " << FBPTTSteps << ",\n";
        file << "  \"dropout_rate\": 0,\n";
        
        file << "  \"cells\": [\n";
        
        switch (FCellType) {
            case ctSimpleRNN:
                for (size_t i = 0; i < FSimpleCells.size(); ++i) {
                    if (i > 0) file << ",\n";
                    file << "    {\n";
                    file << "      \"Wih\": " << Array2DToJSON(FSimpleCells[i]->Wih) << ",\n";
                    file << "      \"Whh\": " << Array2DToJSON(FSimpleCells[i]->Whh) << ",\n";
                    file << "      \"bh\": " << Array1DToJSON(FSimpleCells[i]->Bh) << "\n";
                    file << "    }";
                }
                break;
            case ctLSTM:
                for (size_t i = 0; i < FLSTMCells.size(); ++i) {
                    if (i > 0) file << ",\n";
                    file << "    {\n";
                    file << "      \"Wf\": " << Array2DToJSON(FLSTMCells[i]->Wf) << ",\n";
                    file << "      \"Wi\": " << Array2DToJSON(FLSTMCells[i]->Wi) << ",\n";
                    file << "      \"Wc\": " << Array2DToJSON(FLSTMCells[i]->Wc) << ",\n";
                    file << "      \"Wo\": " << Array2DToJSON(FLSTMCells[i]->Wo) << ",\n";
                    file << "      \"Bf\": " << Array1DToJSON(FLSTMCells[i]->Bf) << ",\n";
                    file << "      \"Bi\": " << Array1DToJSON(FLSTMCells[i]->Bi) << ",\n";
                    file << "      \"Bc\": " << Array1DToJSON(FLSTMCells[i]->Bc) << ",\n";
                    file << "      \"Bo\": " << Array1DToJSON(FLSTMCells[i]->Bo) << "\n";
                    file << "    }";
                }
                break;
            case ctGRU:
                for (size_t i = 0; i < FGRUCells.size(); ++i) {
                    if (i > 0) file << ",\n";
                    file << "    {\n";
                    file << "      \"Wz\": " << Array2DToJSON(FGRUCells[i]->Wz) << ",\n";
                    file << "      \"Wr\": " << Array2DToJSON(FGRUCells[i]->Wr) << ",\n";
                    file << "      \"Wh\": " << Array2DToJSON(FGRUCells[i]->Wh) << ",\n";
                    file << "      \"Bz\": " << Array1DToJSON(FGRUCells[i]->Bz) << ",\n";
                    file << "      \"Br\": " << Array1DToJSON(FGRUCells[i]->Br) << ",\n";
                    file << "      \"Bh\": " << Array1DToJSON(FGRUCells[i]->Bh) << "\n";
                    file << "    }";
                }
                break;
        }
        
        file << "\n  ],\n";
        
        file << "  \"output_layer\": {\n";
        file << "    \"W\": " << Array2DToJSON(FOutputLayer->W) << ",\n";
        file << "    \"B\": " << Array1DToJSON(FOutputLayer->B) << "\n";
        file << "  }\n";
        file << "}\n";
        
        file.close();
        cout << "Model saved to JSON: " << Filename << "\n";
    }

    void LoadModelFromJSON(const string& Filename) {
        ifstream file(Filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << Filename << "\n";
            return;
        }
        
        stringstream buffer;
        buffer << file.rdbuf();
        string Content = buffer.str();
        file.close();
        
        string inputStr = ExtractJSONValue(Content, "input_size");
        if (inputStr.empty()) { cerr << "Error: Could not parse input_size from JSON\n"; return; }
        int inputSize = stoi(inputStr);
        
        string outputStr = ExtractJSONValue(Content, "output_size");
        if (outputStr.empty()) { cerr << "Error: Could not parse output_size from JSON\n"; return; }
        int outputSize = stoi(outputStr);
        
        string cellTypeStr = ExtractJSONValue(Content, "cell_type");
        TCellType cellType = ParseCellType(cellTypeStr);
        
        string hiddenStr = ExtractJSONValue(Content, "hidden_sizes");
        
        string activationStr = ExtractJSONValue(Content, "hidden_activation");
        if (activationStr.empty()) activationStr = ExtractJSONValue(Content, "activation");
        if (activationStr.empty()) activationStr = "tanh";
        
        string outputActStr = ExtractJSONValue(Content, "output_activation");
        if (outputActStr.empty()) outputActStr = "linear";
        
        string lossStr = ExtractJSONValue(Content, "loss_type");
        if (lossStr.empty()) lossStr = "mse";
        
        string lrStr = ExtractJSONValue(Content, "learning_rate");
        double learningRate = lrStr.empty() ? 0.01 : stod(lrStr);
        
        string clipStr = ExtractJSONValue(Content, "gradient_clip");
        double gradientClip = clipStr.empty() ? 5.0 : stod(clipStr);
        
        string bpttStr = ExtractJSONValue(Content, "bptt_steps");
        int bpttSteps = bpttStr.empty() ? 0 : stoi(bpttStr);
        
        TIntArray hiddenSizes;
        size_t openBracket = hiddenStr.find('[');
        size_t closeBracket = hiddenStr.rfind(']');
        if (openBracket != string::npos && closeBracket != string::npos) {
            string arrayContent = hiddenStr.substr(openBracket + 1, closeBracket - openBracket - 1);
            stringstream ss(arrayContent);
            string token;
            while (getline(ss, token, ',')) {
                size_t start = token.find_first_not_of(" \t\n\r");
                size_t end = token.find_last_not_of(" \t\n\r");
                if (start != string::npos && end != string::npos) {
                    token = token.substr(start, end - start + 1);
                    if (!token.empty()) {
                        hiddenSizes.push_back(stoi(token));
                    }
                }
            }
        } else if (!hiddenStr.empty()) {
            hiddenSizes.push_back(stoi(hiddenStr));
        }
        
        FInputSize = inputSize;
        FOutputSize = outputSize;
        FHiddenSizes = hiddenSizes;
        FCellType = cellType;
        FActivation = ParseActivation(activationStr);
        FOutputActivation = ParseActivation(outputActStr);
        FLossType = ParseLoss(lossStr);
        FLearningRate = learningRate;
        FGradientClip = gradientClip;
        FBPTTSteps = bpttSteps;
        
        for (auto cell : FSimpleCells) delete cell;
        for (auto cell : FLSTMCells) delete cell;
        for (auto cell : FGRUCells) delete cell;
        FSimpleCells.clear();
        FLSTMCells.clear();
        FGRUCells.clear();
        if (FOutputLayer) delete FOutputLayer;
        
        int PrevSize = inputSize;
        switch (cellType) {
            case ctSimpleRNN:
                for (size_t i = 0; i < hiddenSizes.size(); ++i) {
                    FSimpleCells.push_back(new TSimpleRNNCell(PrevSize, hiddenSizes[i], FActivation));
                    PrevSize = hiddenSizes[i];
                }
                break;
            case ctLSTM:
                for (size_t i = 0; i < hiddenSizes.size(); ++i) {
                    FLSTMCells.push_back(new TLSTMCell(PrevSize, hiddenSizes[i], FActivation));
                    PrevSize = hiddenSizes[i];
                }
                break;
            case ctGRU:
                for (size_t i = 0; i < hiddenSizes.size(); ++i) {
                    FGRUCells.push_back(new TGRUCell(PrevSize, hiddenSizes[i], FActivation));
                    PrevSize = hiddenSizes[i];
                }
                break;
        }
        
        FOutputLayer = new TOutputLayer(PrevSize, outputSize, FOutputActivation);
        
        cout << "Model loaded from JSON: " << Filename << "\n";
    }

    void ResetGradients() {
        switch (FCellType) {
            case ctSimpleRNN:
                for (auto cell : FSimpleCells) cell->ResetGradients();
                break;
            case ctLSTM:
                for (auto cell : FLSTMCells) cell->ResetGradients();
                break;
            case ctGRU:
                for (auto cell : FGRUCells) cell->ResetGradients();
                break;
        }
        FOutputLayer->ResetGradients();
    }

    void ApplyGradients() {
        switch (FCellType) {
            case ctSimpleRNN:
                for (auto cell : FSimpleCells) cell->ApplyGradients(FLearningRate, FGradientClip);
                break;
            case ctLSTM:
                for (auto cell : FLSTMCells) cell->ApplyGradients(FLearningRate, FGradientClip);
                break;
            case ctGRU:
                for (auto cell : FGRUCells) cell->ApplyGradients(FLearningRate, FGradientClip);
                break;
        }
        FOutputLayer->ApplyGradients(FLearningRate, FGradientClip);
    }
};

// ========== Data Utilities ==========
void SplitData(const TDArray2D& Inputs, const TDArray2D& Targets, double ValSplit, TDataSplit& Split) {
    int N = Inputs.size();
    int ValCount = (int)round(N * ValSplit);
    int TrainCount = N - ValCount;

    vector<int> Indices(N);
    for (int i = 0; i < N; i++)
        Indices[i] = i;

    for (int i = N - 1; i >= 1; i--) {
        int j = rand() % (i + 1);
        swap(Indices[i], Indices[j]);
    }

    Split.TrainInputs.resize(TrainCount);
    Split.TrainTargets.resize(TrainCount);
    Split.ValInputs.resize(ValCount);
    Split.ValTargets.resize(ValCount);

    for (int i = 0; i < TrainCount; i++) {
        Split.TrainInputs[i] = Inputs[Indices[i]];
        Split.TrainTargets[i] = Targets[Indices[i]];
    }

    for (int i = 0; i < ValCount; i++) {
        Split.ValInputs[i] = Inputs[Indices[TrainCount + i]];
        Split.ValTargets[i] = Targets[Indices[TrainCount + i]];
    }
}

// ========== Helper Functions ==========
string CellTypeToStr(TCellType ct) {
    switch (ct) {
        case ctSimpleRNN: return "simplernn";
        case ctLSTM: return "lstm";
        case ctGRU: return "gru";
        default: return "simplernn";
    }
}

string ActivationToStr(TActivationType act) {
    switch (act) {
        case atSigmoid: return "sigmoid";
        case atTanh: return "tanh";
        case atReLU: return "relu";
        case atLinear: return "linear";
        default: return "sigmoid";
    }
}

string LossToStr(TLossType loss) {
    switch (loss) {
        case ltMSE: return "mse";
        case ltCrossEntropy: return "crossentropy";
        default: return "mse";
    }
}

TCellType ParseCellType(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "lstm") return ctLSTM;
    if (lower == "gru") return ctGRU;
    return ctSimpleRNN;
}

TActivationType ParseActivation(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "tanh") return atTanh;
    if (lower == "relu") return atReLU;
    if (lower == "linear") return atLinear;
    return atSigmoid;
}

TLossType ParseLoss(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "crossentropy") return ltCrossEntropy;
    return ltMSE;
}

void ParseIntArrayHelper(const string& s, TIntArray& result) {
    result.clear();
    stringstream ss(s);
    string token;
    while (getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        result.push_back(stoi(token));
    }
}

void ParseDoubleArrayHelper(const string& s, DArray& result) {
    result.clear();
    stringstream ss(s);
    string token;
    while (getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        result.push_back(stod(token));
    }
}

void LoadDataFromCSV(const string& Filename, TDArray2D& Inputs, TDArray2D& Targets) {
    Inputs.clear();
    Targets.clear();
    
    ifstream file(Filename);
    string line;
    
    while (getline(file, line)) {
        if (line.empty()) continue;
        
        DArray InputsArr, TargetsArr;
        stringstream ss(line);
        string token;
        
        vector<double> tokens;
        while (getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);
            tokens.push_back(stod(token));
        }
        
        if (tokens.size() >= 2) {
            size_t splitPoint = tokens.size() / 2;
            InputsArr.assign(tokens.begin(), tokens.begin() + splitPoint);
            TargetsArr.assign(tokens.begin() + splitPoint, tokens.end());
            
            Inputs.push_back(InputsArr);
            Targets.push_back(TargetsArr);
        }
    }
    
    file.close();
}

static string ExtractJSONValue(const string& json, const string& key) {
     string searchKey = "\"" + key + "\"";
     size_t keyPos = json.find(searchKey);
     
     if (keyPos == string::npos) return "";
     
     size_t colonPos = json.find(':', keyPos);
     if (colonPos == string::npos) return "";
     
     size_t startPos = colonPos + 1;
     
     while (startPos < json.length() && (json[startPos] == ' ' || json[startPos] == '\t' 
            || json[startPos] == '\n' || json[startPos] == '\r')) {
         ++startPos;
     }
     
     if (startPos < json.length() && json[startPos] == '"') {
         size_t quotePos1 = startPos;
         size_t quotePos2 = json.find('"', quotePos1 + 1);
         if (quotePos2 != string::npos) {
             return json.substr(quotePos1 + 1, quotePos2 - quotePos1 - 1);
         }
         return "";
     }
     
     if (startPos < json.length() && json[startPos] == '[') {
         size_t bracketCount = 1;
         size_t endPos = startPos + 1;
         while (endPos < json.length() && bracketCount > 0) {
             if (json[endPos] == '[') bracketCount++;
             else if (json[endPos] == ']') bracketCount--;
             if (bracketCount > 0) endPos++;
         }
         return json.substr(startPos, endPos - startPos + 1);
     }
     
     size_t endPos = json.find(',', startPos);
     if (endPos == string::npos) endPos = json.find('}', startPos);
     if (endPos == string::npos) endPos = json.find(']', startPos);
     
     string result = json.substr(startPos, endPos - startPos);
     size_t end = result.find_last_not_of(" \t\n\r");
     if (end != string::npos) {
         result = result.substr(0, end + 1);
     }
     return result;
 }

// ========== Utility Functions ==========
void PrintUsage() {
    cout << "RNN (CUDA Accelerated)\n\n";
    cout << "Commands:\n";
    cout << "  create   Create a new RNN model and save to JSON\n";
    cout << "  train    Train an existing model with data from JSON\n";
    cout << "  predict  Make predictions with a trained model from JSON\n";
    cout << "  info     Display model information from JSON\n";
    cout << "  help     Show this help message\n\n";
    cout << "Create Options:\n";
    cout << "  --input=N              Input layer size (required)\n";
    cout << "  --hidden=N,N,...       Hidden layer sizes (required)\n";
    cout << "  --output=N             Output layer size (required)\n";
    cout << "  --save=FILE.json       Save model to JSON file (required)\n";
    cout << "  --cell=TYPE            simplernn|lstm|gru (default: lstm)\n";
    cout << "  --lr=VALUE             Learning rate (default: 0.01)\n";
    cout << "  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: tanh)\n";
    cout << "  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)\n";
    cout << "  --loss=TYPE            mse|crossentropy (default: mse)\n";
    cout << "  --clip=VALUE           Gradient clipping (default: 5.0)\n";
    cout << "  --bptt=N               BPTT steps (default: 0 = full)\n\n";
    cout << "Train Options:\n";
    cout << "  --model=FILE.json      Load model from JSON file (required)\n";
    cout << "  --data=FILE.csv        Training data CSV file (required)\n";
    cout << "  --save=FILE.json       Save trained model to JSON (required)\n";
    cout << "  --epochs=N             Number of training epochs (default: 100)\n";
    cout << "  --batch=N              Batch size (default: 1)\n";
    cout << "  --lr=VALUE             Override learning rate\n";
    cout << "  --seq-len=N            Sequence length (default: auto-detect)\n\n";
    cout << "Predict Options:\n";
    cout << "  --model=FILE.json      Load model from JSON file (required)\n";
    cout << "  --input=v1,v2,...      Input values as CSV (required)\n\n";
    cout << "Info Options:\n";
    cout << "  --model=FILE.json      Load model from JSON file (required)\n\n";
    cout << "Options:\n";
    cout << "  --input       Input size (create) or input values (predict)\n";
    cout << "  --hidden      Hidden layer sizes (comma-separated)\n";
    cout << "  --output      Output size\n";
    cout << "  --cell        Cell type: simplernn, lstm, gru (default: lstm)\n";
    cout << "  --hidden-act  Hidden activation: sigmoid, tanh, relu, linear (default: tanh)\n";
    cout << "  --output-act  Output activation: sigmoid, tanh, relu, linear (default: linear)\n";
    cout << "  --loss        Loss function: mse, crossentropy (default: mse)\n";
    cout << "  --lr          Learning rate (default: 0.01)\n";
    cout << "  --clip        Gradient clipping value (default: 5.0)\n";
    cout << "  --bptt        BPTT steps (default: 0 = full sequence)\n";
    cout << "  --epochs      Training epochs (default: 100)\n";
    cout << "  --batch       Batch size (default: 1)\n";
    cout << "  --model       Model file path\n";
    cout << "  --data        Data file path\n";
    cout << "  --save        Save file path\n";
    cout << "  --verbose     Verbose output\n";
}

// ========== Main Program ==========
int main(int argc, char* argv[]) {
    if (argc < 2) {
        PrintUsage();
        return 1;
    }

    string CmdStr = argv[1];
    TCommand Command = cmdNone;

    if (CmdStr == "create") Command = cmdCreate;
    else if (CmdStr == "train") Command = cmdTrain;
    else if (CmdStr == "predict") Command = cmdPredict;
    else if (CmdStr == "info") Command = cmdInfo;
    else if (CmdStr == "help" || CmdStr == "--help" || CmdStr == "-h") Command = cmdHelp;
    else {
        cerr << "Unknown command: " << CmdStr << "\n";
        PrintUsage();
        return 1;
    }

    if (Command == cmdHelp) {
        PrintUsage();
        return 0;
    }

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "Error: No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Using CUDA device: " << prop.name << "\n";

    int inputSize = 0;
    int outputSize = 0;
    TIntArray hiddenSizes;
    double learningRate = 0.01;
    double gradientClip = 5.0;
    int epochs = 100;
    int batchSize = 1;
    int seqLen = 0;
    int bpttSteps = 0;
    bool verbose = false;
    TActivationType hiddenAct = atTanh;
    TActivationType outputAct = atLinear;
    TCellType cellType = ctLSTM;
    TLossType lossType = ltMSE;
    string modelFile, saveFile, dataFile;
    DArray inputValues;

    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "--verbose") {
            verbose = true;
        } else {
            size_t eqPos = arg.find('=');
            if (eqPos == string::npos) {
                cerr << "Invalid argument: " << arg << "\n";
                continue;
            }

            string key = arg.substr(0, eqPos);
            string value = arg.substr(eqPos + 1);

            if (key == "--input") {
                if (Command == cmdPredict) {
                    ParseDoubleArrayHelper(value, inputValues);
                } else {
                    inputSize = stoi(value);
                }
            } else if (key == "--hidden") {
                ParseIntArrayHelper(value, hiddenSizes);
            } else if (key == "--output") {
                outputSize = stoi(value);
            } else if (key == "--save") {
                saveFile = value;
            } else if (key == "--model") {
                modelFile = value;
            } else if (key == "--data") {
                dataFile = value;
            } else if (key == "--lr") {
                learningRate = stod(value);
            } else if (key == "--cell") {
                cellType = ParseCellType(value);
            } else if (key == "--hidden-act") {
                hiddenAct = ParseActivation(value);
            } else if (key == "--output-act") {
                outputAct = ParseActivation(value);
            } else if (key == "--loss") {
                lossType = ParseLoss(value);
            } else if (key == "--clip") {
                gradientClip = stod(value);
            } else if (key == "--bptt") {
                bpttSteps = stoi(value);
            } else if (key == "--epochs") {
                epochs = stoi(value);
            } else if (key == "--batch") {
                batchSize = stoi(value);
            } else if (key == "--seq-len") {
                seqLen = stoi(value);
            } else {
                cerr << "Unknown option: " << key << "\n";
            }
        }
    }

    if (Command == cmdCreate) {
        if (inputSize <= 0) { cerr << "Error: --input is required\n"; return 1; }
        if (hiddenSizes.empty()) { cerr << "Error: --hidden is required\n"; return 1; }
        if (outputSize <= 0) { cerr << "Error: --output is required\n"; return 1; }
        if (saveFile.empty()) { cerr << "Error: --save is required\n"; return 1; }

        TAdvancedRNN* RNNModel = new TAdvancedRNN(inputSize, hiddenSizes, outputSize, cellType,
                                   hiddenAct, outputAct, lossType, learningRate,
                                   gradientClip, bpttSteps);

        cout << "Created RNN model:\n";
        cout << "  Input size: " << inputSize << "\n";
        cout << "  Hidden sizes: ";
        for (size_t i = 0; i < hiddenSizes.size(); ++i) {
            if (i > 0) cout << ",";
            cout << hiddenSizes[i];
        }
        cout << "\n";
        cout << "  Output size: " << outputSize << "\n";
        cout << "  Cell type: " << CellTypeToStr(cellType) << "\n";
        cout << "  Hidden activation: " << ActivationToStr(hiddenAct) << "\n";
        cout << "  Output activation: " << ActivationToStr(outputAct) << "\n";
        cout << "  Loss function: " << LossToStr(lossType) << "\n";
        cout << fixed << setprecision(6)
                  << "  Learning rate: " << learningRate << "\n";
        cout << fixed << setprecision(2)
                  << "  Gradient clip: " << gradientClip << "\n";
        cout << "  BPTT steps: " << bpttSteps << "\n";

        RNNModel->SaveModelToJSON(saveFile);
        cout << "Model saved to: " << saveFile << "\n";

        delete RNNModel;
    }
    else if (Command == cmdTrain) {
        if (modelFile.empty()) { cerr << "Error: --model is required\n"; return 1; }
        if (dataFile.empty()) { cerr << "Error: --data is required\n"; return 1; }
        if (saveFile.empty()) { cerr << "Error: --save is required\n"; return 1; }

        cout << "Loading model from JSON: " << modelFile << "\n";
        TAdvancedRNN* RNNModel = new TAdvancedRNN(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
        RNNModel->LoadModelFromJSON(modelFile);
        cout << "Model loaded successfully.\n";

        cout << "Loading training data from: " << dataFile << "\n";
        TDArray2D Inputs, Targets;
        LoadDataFromCSV(dataFile, Inputs, Targets);

        if (Inputs.empty()) {
            cerr << "Error: No data loaded from CSV file\n";
            delete RNNModel;
            return 1;
        }

        cout << "Loaded " << Inputs.size() << " timesteps of training data\n";
        cout << "Starting training for " << epochs << " epochs...\n";

        for (int Epoch = 1; Epoch <= epochs; ++Epoch) {
            double TrainLoss = RNNModel->TrainSequence(Inputs, Targets);

            if (!isnan(TrainLoss) && !isinf(TrainLoss)) {
                if (verbose || (Epoch % 10 == 0) || (Epoch == epochs)) {
                    cout << "Epoch " << setw(4) << Epoch << "/"
                              << epochs << " - Loss: "
                              << fixed << setprecision(6) << TrainLoss << "\n";
                }
            }
        }

        cout << "Training completed.\n";
        cout << "Saving trained model to: " << saveFile << "\n";
        RNNModel->SaveModelToJSON(saveFile);

        delete RNNModel;
    }
    else if (Command == cmdPredict) {
        if (modelFile.empty()) { cerr << "Error: --model is required\n"; return 1; }
        if (inputValues.empty()) { cerr << "Error: --input is required\n"; return 1; }

        TAdvancedRNN* RNNModel = new TAdvancedRNN(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
        RNNModel->LoadModelFromJSON(modelFile);
        if (RNNModel == nullptr) { cerr << "Error: Failed to load model\n"; return 1; }

        TDArray2D Inputs(1);
        Inputs[0] = inputValues;

        TDArray2D Predictions = RNNModel->Predict(Inputs);

        cout << "Input: ";
        for (size_t i = 0; i < inputValues.size(); ++i) {
            if (i > 0) cout << ", ";
            cout << fixed << setprecision(4) << inputValues[i];
        }
        cout << "\n";

        if (!Predictions.empty() && !Predictions.back().empty()) {
            cout << "Output: ";
            for (size_t i = 0; i < Predictions.back().size(); ++i) {
                if (i > 0) cout << ", ";
                cout << fixed << setprecision(6) << Predictions.back()[i];
            }
            cout << "\n";

            if (Predictions.back().size() > 1) {
                size_t maxIdx = 0;
                for (size_t i = 1; i < Predictions.back().size(); ++i) {
                    if (Predictions.back()[i] > Predictions.back()[maxIdx]) {
                        maxIdx = i;
                    }
                }
                cout << "Max index: " << maxIdx << "\n";
            }
        }

        delete RNNModel;
    }
    else if (Command == cmdInfo) {
        if (modelFile.empty()) { cerr << "Error: --model is required\n"; return 1; }
        cout << "Loading model from JSON: " << modelFile << "\n";
        TAdvancedRNN* RNNModel = new TAdvancedRNN(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
        RNNModel->LoadModelFromJSON(modelFile);
        
        ifstream file(modelFile);
        stringstream buffer;
        buffer << file.rdbuf();
        string Content = buffer.str();
        file.close();
        
        cout << "Model Information:\n";
        cout << "  Input size: " << ExtractJSONValue(Content, "input_size") << "\n";
        cout << "  Output size: " << ExtractJSONValue(Content, "output_size") << "\n";
        cout << "  Hidden sizes: " << ExtractJSONValue(Content, "hidden_sizes") << "\n";
        cout << "  Cell type: " << ExtractJSONValue(Content, "cell_type") << "\n";
        cout << "  Activation: " << ExtractJSONValue(Content, "activation") << "\n";
        cout << "  Output activation: " << ExtractJSONValue(Content, "output_activation") << "\n";
        cout << "  Loss type: " << ExtractJSONValue(Content, "loss_type") << "\n";
        cout << "  Learning rate: " << ExtractJSONValue(Content, "learning_rate") << "\n";
        cout << "  Gradient clip: " << ExtractJSONValue(Content, "gradient_clip") << "\n";
        cout << "  BPTT steps: " << ExtractJSONValue(Content, "bptt_steps") << "\n";

        delete RNNModel;
    }

    return 0;
}
