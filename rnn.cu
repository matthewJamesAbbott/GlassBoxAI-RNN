//
// Matthew Abbott  2025
// Advanced RNN with BPTT, Gradient Clipping, LSTM/GRU, Batch Processing
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iomanip>
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

// ========== Type Definitions ==========
enum TActivationType { atSigmoid, atTanh, atReLU, atLinear };
enum TLossType { ltMSE, ltCrossEntropy };
enum TCellType { ctSimpleRNN, ctLSTM, ctGRU };

typedef vector<double> DArray;
typedef vector<DArray> TDArray2D;
typedef vector<TDArray2D> TDArray3D;

struct TDataSplit {
    TDArray2D TrainInputs, TrainTargets;
    TDArray2D ValInputs, ValTargets;
};

struct TTimeStepCache {
    DArray Input;
    DArray H, C;
    DArray PreH;
    DArray Fg, Ig, CTilde, Og, TanhC;
    DArray Z, R, HTilde;
    DArray OutPre, OutVal;
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
                        Caches[t].Fg = Fg;
                        Caches[t].Ig = Ig;
                        Caches[t].CTilde = CTilde;
                        Caches[t].Og = Og;
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
                                                         Caches[t].Fg, Caches[t].Ig, Caches[t].CTilde,
                                                         Caches[t].Og, Caches[t].TanhC,
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

    bool SaveModel(const string& FileName) {
        try {
            ofstream File(FileName, ios::binary);
            if (!File.is_open()) {
                cerr << "Error: Cannot open file for writing: " << FileName << endl;
                return false;
            }

            // Write header and configuration
            int NumLayers = FHiddenSizes.size();
            int CellTypeInt = (int)FCellType;
            int ActTypeInt = (int)FActivation;
            int OutActTypeInt = (int)FOutputActivation;
            int LossTypeInt = (int)FLossType;

            File.write((char*)&FInputSize, sizeof(int));
            File.write((char*)&FOutputSize, sizeof(int));
            File.write((char*)&NumLayers, sizeof(int));
            File.write((char*)&CellTypeInt, sizeof(int));
            File.write((char*)&ActTypeInt, sizeof(int));
            File.write((char*)&OutActTypeInt, sizeof(int));
            File.write((char*)&LossTypeInt, sizeof(int));
            File.write((char*)&FLearningRate, sizeof(double));
            File.write((char*)&FGradientClip, sizeof(double));

            // Write hidden sizes
            for (int h : FHiddenSizes)
                File.write((char*)&h, sizeof(int));

            // Write SimpleRNN cells
            for (auto cell : FSimpleCells) {
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < cell->FInputSize; j++) {
                        File.write((char*)&cell->Wih[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < cell->FHiddenSize; j++) {
                        File.write((char*)&cell->Whh[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    File.write((char*)&cell->Bh[i], sizeof(double));
                }
            }

            // Write LSTM cells
            for (auto cell : FLSTMCells) {
                int InputPlusHidden = cell->FInputSize + cell->FHiddenSize;
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.write((char*)&cell->Wf[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.write((char*)&cell->Wi[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.write((char*)&cell->Wc[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.write((char*)&cell->Wo[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    File.write((char*)&cell->Bf[i], sizeof(double));
                    File.write((char*)&cell->Bi[i], sizeof(double));
                    File.write((char*)&cell->Bc[i], sizeof(double));
                    File.write((char*)&cell->Bo[i], sizeof(double));
                }
            }

            // Write GRU cells
            for (auto cell : FGRUCells) {
                int InputPlusHidden = cell->FInputSize + cell->FHiddenSize;
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.write((char*)&cell->Wz[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.write((char*)&cell->Wr[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.write((char*)&cell->Wh[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    File.write((char*)&cell->Bz[i], sizeof(double));
                    File.write((char*)&cell->Br[i], sizeof(double));
                    File.write((char*)&cell->Bh[i], sizeof(double));
                }
            }

            // Write output layer
            for (int i = 0; i < FOutputSize; i++) {
                for (int j = 0; j < FOutputLayer->FInputSize; j++) {
                    File.write((char*)&FOutputLayer->W[i][j], sizeof(double));
                }
            }
            for (int i = 0; i < FOutputSize; i++) {
                File.write((char*)&FOutputLayer->B[i], sizeof(double));
            }

            File.close();
            return true;
        } catch (...) {
            cerr << "Error saving model" << endl;
            return false;
        }
    }

    bool LoadModel(const string& FileName) {
        try {
            ifstream File(FileName, ios::binary);
            if (!File.is_open()) {
                cerr << "Error: File not found: " << FileName << endl;
                return false;
            }

            int NumLayers, CellTypeInt, ActTypeInt, OutActTypeInt, LossTypeInt;
            File.read((char*)&FInputSize, sizeof(int));
            File.read((char*)&FOutputSize, sizeof(int));
            File.read((char*)&NumLayers, sizeof(int));
            File.read((char*)&CellTypeInt, sizeof(int));
            File.read((char*)&ActTypeInt, sizeof(int));
            File.read((char*)&OutActTypeInt, sizeof(int));
            File.read((char*)&LossTypeInt, sizeof(int));
            File.read((char*)&FLearningRate, sizeof(double));
            File.read((char*)&FGradientClip, sizeof(double));

            FCellType = (TCellType)CellTypeInt;
            FActivation = (TActivationType)ActTypeInt;
            FOutputActivation = (TActivationType)OutActTypeInt;
            FLossType = (TLossType)LossTypeInt;

            // Read hidden sizes
            FHiddenSizes.clear();
            for (int i = 0; i < NumLayers; i++) {
                int h;
                File.read((char*)&h, sizeof(int));
                FHiddenSizes.push_back(h);
            }

            // Reinitialize cells based on loaded config
            for (auto cell : FSimpleCells) delete cell;
            for (auto cell : FLSTMCells) delete cell;
            for (auto cell : FGRUCells) delete cell;
            FSimpleCells.clear();
            FLSTMCells.clear();
            FGRUCells.clear();

            if (FCellType == ctSimpleRNN) {
                for (size_t i = 0; i < FHiddenSizes.size(); i++) {
                    int InSize = (i == 0) ? FInputSize : FHiddenSizes[i-1];
                    FSimpleCells.push_back(new TSimpleRNNCell(InSize, FHiddenSizes[i], FActivation));
                }
            } else if (FCellType == ctLSTM) {
                for (size_t i = 0; i < FHiddenSizes.size(); i++) {
                    int InSize = (i == 0) ? FInputSize : FHiddenSizes[i-1];
                    FLSTMCells.push_back(new TLSTMCell(InSize, FHiddenSizes[i], FActivation));
                }
            } else if (FCellType == ctGRU) {
                for (size_t i = 0; i < FHiddenSizes.size(); i++) {
                    int InSize = (i == 0) ? FInputSize : FHiddenSizes[i-1];
                    FGRUCells.push_back(new TGRUCell(InSize, FHiddenSizes[i], FActivation));
                }
            }

            if (FOutputLayer) delete FOutputLayer;
            FOutputLayer = new TOutputLayer(FHiddenSizes.back(), FOutputSize, FOutputActivation);

            // Read SimpleRNN cells
            for (auto cell : FSimpleCells) {
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < cell->FInputSize; j++) {
                        File.read((char*)&cell->Wih[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < cell->FHiddenSize; j++) {
                        File.read((char*)&cell->Whh[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    File.read((char*)&cell->Bh[i], sizeof(double));
                }
            }

            // Read LSTM cells
            for (auto cell : FLSTMCells) {
                int InputPlusHidden = cell->FInputSize + cell->FHiddenSize;
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.read((char*)&cell->Wf[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.read((char*)&cell->Wi[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.read((char*)&cell->Wc[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.read((char*)&cell->Wo[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    File.read((char*)&cell->Bf[i], sizeof(double));
                    File.read((char*)&cell->Bi[i], sizeof(double));
                    File.read((char*)&cell->Bc[i], sizeof(double));
                    File.read((char*)&cell->Bo[i], sizeof(double));
                }
            }

            // Read GRU cells
            for (auto cell : FGRUCells) {
                int InputPlusHidden = cell->FInputSize + cell->FHiddenSize;
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.read((char*)&cell->Wz[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.read((char*)&cell->Wr[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    for (int j = 0; j < InputPlusHidden; j++) {
                        File.read((char*)&cell->Wh[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell->FHiddenSize; i++) {
                    File.read((char*)&cell->Bz[i], sizeof(double));
                    File.read((char*)&cell->Br[i], sizeof(double));
                    File.read((char*)&cell->Bh[i], sizeof(double));
                }
            }

            // Read output layer
            for (int i = 0; i < FOutputSize; i++) {
                for (int j = 0; j < FOutputLayer->FInputSize; j++) {
                    File.read((char*)&FOutputLayer->W[i][j], sizeof(double));
                }
            }
            for (int i = 0; i < FOutputSize; i++) {
                File.read((char*)&FOutputLayer->B[i], sizeof(double));
            }

            File.close();
            FStates = InitHiddenStates();
            return true;
        } catch (...) {
            cerr << "Error loading model" << endl;
            return false;
        }
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

// ========== CLI Helpers ==========
void ShowHelp() {
    cout << "RNN CUDA" << endl;
    cout << "Advanced RNN with BPTT, Gradient Clipping, LSTM/GRU, Batch Processing" << endl;
    cout << endl;
    cout << "Usage: rnn_cuda [OPTIONS]" << endl;
    cout << endl;
    cout << "Training an RNN on sequence data from CSV files (CUDA accelerated)." << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "  -h, --help              Show this help message and exit" << endl;
    cout << "  -i, --input FILE        Input CSV file (required for train/predict)" << endl;
    cout << "  -t, --target FILE       Target CSV file (required for train)" << endl;
    cout << "  -o, --output FILE       Output predictions to FILE" << endl;
    cout << "  -m, --model FILE        Model file to save/load" << endl;
    cout << endl;
    cout << "Model Architecture:" << endl;
    cout << "  --cell TYPE             Cell type: rnn, lstm, gru (default: lstm)" << endl;
    cout << "  --hidden SIZE           Hidden layer size (default: 32)" << endl;
    cout << "  --layers N              Number of hidden layers (default: 1)" << endl;
    cout << "  --activation TYPE       Hidden activation: sigmoid, tanh, relu (default: tanh)" << endl;
    cout << "  --out-activation TYPE   Output activation: sigmoid, tanh, relu, linear (default: linear)" << endl;
    cout << endl;
    cout << "Training Parameters:" << endl;
    cout << "  --epochs N              Number of training epochs (default: 100)" << endl;
    cout << "  --lr RATE               Learning rate (default: 0.01)" << endl;
    cout << "  --clip VALUE            Gradient clipping value (default: 5.0)" << endl;
    cout << "  --val-split RATIO       Validation split ratio (default: 0.2)" << endl;
    cout << "  --loss TYPE             Loss function: mse, crossentropy (default: mse)" << endl;
    cout << "  --log-interval N        Log every N epochs (default: 10)" << endl;
    cout << "  --seed N                Random seed (default: random)" << endl;
    cout << endl;
    cout << "Inference:" << endl;
    cout << "  --predict               Predict mode (requires --input and --model)" << endl;
    cout << endl;
    cout << "Miscellaneous:" << endl;
    cout << "  --quiet                 Suppress progress output" << endl;
    cout << endl;
    cout << "Examples:" << endl;
    cout << "  rnn_cuda --input data.csv --target labels.csv --epochs 200 -m model.bin" << endl;
    cout << "  rnn_cuda --predict --input test.csv -m model.bin --output predictions.csv" << endl;
    cout << endl;
}

string GetArg(int argc, char* argv[], const string& Name) {
    for (int i = 1; i < argc - 1; i++)
        if (string(argv[i]) == Name)
            return string(argv[i + 1]);
    return "";
}

bool HasArg(int argc, char* argv[], const string& Name) {
    for (int i = 1; i < argc; i++)
        if (string(argv[i]) == Name)
            return true;
    return false;
}

int GetArgInt(int argc, char* argv[], const string& Name, int Default) {
    string S = GetArg(argc, argv, Name);
    if (S.empty()) return Default;
    return atoi(S.c_str());
}

double GetArgFloat(int argc, char* argv[], const string& Name, double Default) {
    string S = GetArg(argc, argv, Name);
    if (S.empty()) return Default;
    return atof(S.c_str());
}

TCellType ParseCellType(const string& S) {
    if (S == "rnn" || S == "simple") return ctSimpleRNN;
    else if (S == "gru") return ctGRU;
    else return ctLSTM;
}

TActivationType ParseActivation(const string& S) {
    if (S == "sigmoid") return atSigmoid;
    else if (S == "relu") return atReLU;
    else if (S == "linear") return atLinear;
    else return atTanh;
}

TLossType ParseLoss(const string& S) {
    if (S == "crossentropy") return ltCrossEntropy;
    else return ltMSE;
}

string CellTypeToStr(TCellType CT) {
    switch (CT) {
        case ctSimpleRNN: return "SimpleRNN";
        case ctLSTM: return "LSTM";
        case ctGRU: return "GRU";
        default: return "Unknown";
    }
}

TDArray2D LoadCSV(const string& FileName) {
    TDArray2D Result;
    ifstream F(FileName);
    if (!F.is_open()) {
        cerr << "Error: File not found: " << FileName << endl;
        exit(1);
    }

    string Line;
    while (getline(F, Line)) {
        if (Line.empty()) continue;

        DArray Row;
        stringstream SS(Line);
        string Token;
        while (getline(SS, Token, ',')) {
            double Val = atof(Token.c_str());
            Row.push_back(Val);
        }

        if (!Row.empty())
            Result.push_back(Row);
    }

    F.close();
    return Result;
}

void SaveCSV(const string& FileName, const TDArray2D& Data) {
    ofstream F(FileName);
    for (size_t i = 0; i < Data.size(); i++) {
        for (size_t j = 0; j < Data[i].size(); j++) {
            if (j > 0) F << ",";
            F << fixed << setprecision(6) << Data[i][j];
        }
        F << endl;
    }
    F.close();
}

// ========== Main ==========
int main(int argc, char* argv[]) {
    if (argc == 1 || HasArg(argc, argv, "-h") || HasArg(argc, argv, "--help")) {
        ShowHelp();
        return 0;
    }

    // Check CUDA device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "Error: No CUDA devices found" << endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Using CUDA device: " << prop.name << endl;

    string InputFile = GetArg(argc, argv, "-i");
    if (InputFile.empty()) InputFile = GetArg(argc, argv, "--input");
    string TargetFile = GetArg(argc, argv, "-t");
    if (TargetFile.empty()) TargetFile = GetArg(argc, argv, "--target");
    string ModelFile = GetArg(argc, argv, "-m");
    if (ModelFile.empty()) ModelFile = GetArg(argc, argv, "--model");
    string OutputFile = GetArg(argc, argv, "-o");
    if (OutputFile.empty()) OutputFile = GetArg(argc, argv, "--output");

    TCellType CellType = ParseCellType(GetArg(argc, argv, "--cell"));
    int HiddenSize = GetArgInt(argc, argv, "--hidden", 32);
    int NumLayers = GetArgInt(argc, argv, "--layers", 1);
    int Epochs = GetArgInt(argc, argv, "--epochs", 100);
    double LearningRate = GetArgFloat(argc, argv, "--lr", 0.01);
    double GradClip = GetArgFloat(argc, argv, "--clip", 5.0);
    double ValSplit = GetArgFloat(argc, argv, "--val-split", 0.2);
    int LogInterval = GetArgInt(argc, argv, "--log-interval", 10);
    TActivationType Activation = ParseActivation(GetArg(argc, argv, "--activation"));
    TActivationType OutActivation = ParseActivation(GetArg(argc, argv, "--out-activation"));
    if (GetArg(argc, argv, "--out-activation").empty()) OutActivation = atLinear;
    TLossType LossType = ParseLoss(GetArg(argc, argv, "--loss"));
    bool PredictMode = HasArg(argc, argv, "--predict");
    bool Quiet = HasArg(argc, argv, "--quiet");
    int Seed = GetArgInt(argc, argv, "--seed", -1);

    if (Seed >= 0)
        srand(Seed);
    else
        srand(time(NULL));

    vector<int> HiddenSizes(NumLayers, HiddenSize);

    if (InputFile.empty()) {
        cerr << "Error: --input is required" << endl;
        return 1;
    }

    TDArray2D Inputs = LoadCSV(InputFile);
    int InputSize = Inputs[0].size();

    if (PredictMode) {
        if (ModelFile.empty()) {
            cerr << "Error: --model is required for prediction" << endl;
            return 1;
        }

        TAdvancedRNN* RNN = new TAdvancedRNN(
            InputSize,
            HiddenSizes,
            10,  // dummy output size, will be overwritten by LoadModel
            CellType,
            Activation,
            OutActivation,
            LossType,
            LearningRate,
            GradClip,
            0
        );

        if (!RNN->LoadModel(ModelFile)) {
            cerr << "Error: Failed to load model from " << ModelFile << endl;
            delete RNN;
            return 1;
        }

        if (!Quiet)
            cout << "Model loaded from: " << ModelFile << endl;

        TDArray2D Predictions = RNN->Predict(Inputs);
        if (!OutputFile.empty()) {
            SaveCSV(OutputFile, Predictions);
            if (!Quiet)
                cout << "Predictions saved to: " << OutputFile << endl;
        }

        if (!Quiet) {
            cout << "Predictions:" << endl;
            for (size_t i = 0; i < Predictions.size() && i < 10; i++) {
                cout << "  Sample " << i << ": ";
                for (size_t j = 0; j < Predictions[i].size(); j++) {
                    cout << fixed << setprecision(6) << Predictions[i][j] << " ";
                }
                cout << endl;
            }
        }

        delete RNN;
    } else {
        if (TargetFile.empty()) {
            cerr << "Error: --target is required for training" << endl;
            return 1;
        }

        TDArray2D Targets = LoadCSV(TargetFile);
        int OutputSize = Targets[0].size();

        if (Inputs.size() != Targets.size()) {
            cerr << "Error: Input and target row counts do not match" << endl;
            return 1;
        }

        TDataSplit Split;
        SplitData(Inputs, Targets, ValSplit, Split);

        TAdvancedRNN* RNN = new TAdvancedRNN(
            InputSize,
            HiddenSizes,
            OutputSize,
            CellType,
            Activation,
            OutActivation,
            LossType,
            LearningRate,
            GradClip,
            0
        );

        if (!Quiet) {
            cout << "=== RNN Training (CUDA) ===" << endl;
            cout << "Cell Type:     " << CellTypeToStr(CellType) << endl;
            cout << "Hidden Size:   " << HiddenSize << endl;
            cout << "Layers:        " << NumLayers << endl;
            cout << "Input Size:    " << InputSize << endl;
            cout << "Output Size:   " << OutputSize << endl;
            cout << fixed << setprecision(4);
            cout << "Learning Rate: " << LearningRate << endl;
            cout << setprecision(2);
            cout << "Gradient Clip: " << GradClip << endl;
            cout << "Train samples: " << Split.TrainInputs.size() << endl;
            cout << "Val samples:   " << Split.ValInputs.size() << endl;
            cout << endl;
            cout << "Epoch | Train Loss | Val Loss" << endl;
            cout << "------+------------+-----------" << endl;
        }

        for (int Epoch = 1; Epoch <= Epochs; Epoch++) {
            double TrainLoss = 0;
            for (size_t b = 0; b < Split.TrainInputs.size(); b++) {
                TDArray2D SingleInput = { Split.TrainInputs[b] };
                TDArray2D SingleTarget = { Split.TrainTargets[b] };
                TrainLoss += RNN->TrainSequence(SingleInput, SingleTarget);
            }
            TrainLoss /= Split.TrainInputs.size();

            double ValLoss = 0;
            if (!Split.ValInputs.empty()) {
                for (size_t b = 0; b < Split.ValInputs.size(); b++) {
                    TDArray2D SingleInput = { Split.ValInputs[b] };
                    TDArray2D SingleTarget = { Split.ValTargets[b] };
                    ValLoss += RNN->ComputeLoss(SingleInput, SingleTarget);
                }
                ValLoss /= Split.ValInputs.size();
            }

            if (!Quiet) {
                if (Epoch % LogInterval == 0 || Epoch == Epochs) {
                    cout << setw(5) << Epoch << " | "
                         << fixed << setprecision(6) << setw(10) << TrainLoss << " | "
                         << setw(10) << ValLoss << endl;
                }
            }
        }

        if (!OutputFile.empty()) {
            TDArray2D Predictions = RNN->Predict(Inputs);
            SaveCSV(OutputFile, Predictions);
            if (!Quiet)
                cout << "Predictions saved to: " << OutputFile << endl;
        }

        if (!ModelFile.empty()) {
            if (RNN->SaveModel(ModelFile)) {
                if (!Quiet)
                    cout << "Model saved to: " << ModelFile << endl;
            } else {
                if (!Quiet)
                    cerr << "Warning: Failed to save model to " << ModelFile << endl;
            }
        }

        if (!Quiet)
            cout << "Training complete." << endl;

        delete RNN;
    }

    return 0;
}
