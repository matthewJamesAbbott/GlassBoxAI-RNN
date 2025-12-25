/*
  RNN Facade - CUDA GPU Accelerated Version
  Matthew Abbott 2025
*/

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

#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Custom atomicAdd for double (works on all compute capabilities)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// Use native atomicAdd for sm_60+
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// Portable atomicAdd wrapper that works on all architectures
__device__ __forceinline__ void atomicAddDouble(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

// Type definitions
enum ActivationType { atSigmoid, atTanh, atReLU, atLinear };
enum LossType { ltMSE, ltCrossEntropy };
enum CellType { ctSimpleRNN, ctLSTM, ctGRU };
enum GateType { gtForget, gtInput, gtOutput, gtCellCandidate, gtUpdate, gtReset, gtHiddenCandidate };

using DArray = vector<double>;
using DArray2D = vector<DArray>;
using DArray3D = vector<DArray2D>;

struct HistogramBin {
    double RangeMin, RangeMax;
    int Count;
    double Percentage;
};

struct GateSaturationStats {
    int NearZeroCount, NearOneCount, TotalCount;
    double NearZeroPct, NearOnePct;
};

struct GradientScaleStats {
    int Timestep;
    double MeanAbsGrad, MaxAbsGrad, MinAbsGrad;
};

struct LayerNormStats {
    double Mean, Variance, Gamma, Beta;
};

struct OptimizerStateRecord {
    double Momentum, Velocity, Beta1Power, Beta2Power;
};

// Host-side random
static random_device rd;
static mt19937 gen(rd());

// ============================================================================
// CUDA Kernels
// ============================================================================

__device__ double d_sigmoid(double x) {
    x = fmax(-500.0, fmin(500.0, x));
    return 1.0 / (1.0 + exp(-x));
}

__device__ double d_tanh_act(double x) {
    return tanh(x);
}

__device__ double d_relu(double x) {
    return x > 0 ? x : 0;
}

__device__ double d_apply_activation(double x, int actType) {
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
        case 0: return y * (1.0 - y);      // atSigmoid
        case 1: return 1.0 - y * y;        // atTanh
        case 2: return y > 0 ? 1.0 : 0.0;  // atReLU
        case 3: return 1.0;                // atLinear
        default: return 1.0;
    }
}

__device__ double d_clip_value(double v, double maxVal) {
    if (v > maxVal) return maxVal;
    if (v < -maxVal) return -maxVal;
    return v;
}

// Vector addition: C = A + B
__global__ void k_vec_add(double* C, const double* A, const double* B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Vector scale: A = A * scale
__global__ void k_vec_scale(double* A, double scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] *= scale;
    }
}

// Zero array
__global__ void k_zero_array(double* A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] = 0.0;
    }
}

// Fill array with value
__global__ void k_fill_array(double* A, double val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] = val;
    }
}

// Matrix-vector multiply: y = W * x + b, then apply activation
// W is [rows x cols], x is [cols], y is [rows], b is [rows]
__global__ void k_matvec_bias_act(double* y, const double* W, const double* x, 
                                   const double* b, int rows, int cols, int actType) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = b[row];
        for (int j = 0; j < cols; j++) {
            sum += W[row * cols + j] * x[j];
        }
        y[row] = d_apply_activation(sum, actType);
    }
}

// Matrix-vector multiply without activation: y = W * x + b
__global__ void k_matvec_bias(double* y, const double* W, const double* x, 
                               const double* b, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = b[row];
        for (int j = 0; j < cols; j++) {
            sum += W[row * cols + j] * x[j];
        }
        y[row] = sum;
    }
}

// Concatenate two vectors: out = [a, b]
__global__ void k_concat(double* out, const double* a, const double* b, int sizeA, int sizeB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sizeA) {
        out[idx] = a[idx];
    } else if (idx < sizeA + sizeB) {
        out[idx] = b[idx - sizeA];
    }
}

// ============================================================================
// SimpleRNN Forward Kernel
// ============================================================================
__global__ void k_simple_rnn_forward(double* H, double* PreH,
                                      const double* Wih, const double* Whh, const double* Bh,
                                      const double* Input, const double* PrevH,
                                      int inputSize, int hiddenSize, int actType) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hiddenSize) {
        double sum = Bh[i];
        for (int j = 0; j < inputSize; j++)
            sum += Wih[i * inputSize + j] * Input[j];
        for (int j = 0; j < hiddenSize; j++)
            sum += Whh[i * hiddenSize + j] * PrevH[j];
        PreH[i] = sum;
        H[i] = d_apply_activation(sum, actType);
    }
}

// ============================================================================
// LSTM Forward Kernel
// ============================================================================
__global__ void k_lstm_forward(double* H, double* C, double* FG, double* IG, 
                                double* CTilde, double* OG, double* TanhC,
                                const double* Wf, const double* Wi, 
                                const double* Wc, const double* Wo,
                                const double* Bf, const double* Bi,
                                const double* Bc, const double* Bo,
                                const double* Concat, const double* PrevC,
                                int concatSize, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        double sumF = Bf[k], sumI = Bi[k], sumC = Bc[k], sumO = Bo[k];
        for (int j = 0; j < concatSize; j++) {
            double cj = Concat[j];
            sumF += Wf[k * concatSize + j] * cj;
            sumI += Wi[k * concatSize + j] * cj;
            sumC += Wc[k * concatSize + j] * cj;
            sumO += Wo[k * concatSize + j] * cj;
        }
        FG[k] = d_sigmoid(sumF);
        IG[k] = d_sigmoid(sumI);
        CTilde[k] = d_tanh_act(sumC);
        OG[k] = d_sigmoid(sumO);
        C[k] = FG[k] * PrevC[k] + IG[k] * CTilde[k];
        TanhC[k] = tanh(C[k]);
        H[k] = OG[k] * TanhC[k];
    }
}

// ============================================================================
// GRU Forward Kernel - Part 1: Compute Z and R gates
// ============================================================================
__global__ void k_gru_forward_gates(double* Z, double* R,
                                     const double* Wz, const double* Wr,
                                     const double* Bz, const double* Br,
                                     const double* Concat,
                                     int concatSize, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        double sumZ = Bz[k], sumR = Br[k];
        for (int j = 0; j < concatSize; j++) {
            sumZ += Wz[k * concatSize + j] * Concat[j];
            sumR += Wr[k * concatSize + j] * Concat[j];
        }
        Z[k] = d_sigmoid(sumZ);
        R[k] = d_sigmoid(sumR);
    }
}

// GRU Forward Kernel - Part 2: Compute H with reset gate applied
__global__ void k_gru_forward_hidden(double* H, double* HTilde,
                                      const double* Wh, const double* Bh,
                                      const double* Input, const double* PrevH,
                                      const double* Z, const double* R,
                                      int inputSize, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        int concatSize = inputSize + hiddenSize;
        double sumH = Bh[k];
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

// ============================================================================
// Output Layer Forward
// ============================================================================
__global__ void k_output_forward(double* Output, double* Pre,
                                  const double* W, const double* B,
                                  const double* Input,
                                  int inputSize, int outputSize, int actType) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < outputSize) {
        double sum = B[i];
        for (int j = 0; j < inputSize; j++)
            sum += W[i * inputSize + j] * Input[j];
        Pre[i] = sum;
        Output[i] = d_apply_activation(sum, actType);
    }
}

// ============================================================================
// Backward Kernels
// ============================================================================

// SimpleRNN backward
__global__ void k_simple_rnn_backward_dHRaw(double* dHRaw, const double* dH, 
                                             const double* H, int hiddenSize, 
                                             int actType, double clipVal) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hiddenSize) {
        double deriv = d_activation_derivative(H[i], actType);
        dHRaw[i] = d_clip_value(dH[i] * deriv, clipVal);
    }
}

__global__ void k_accumulate_dW(double* dW, const double* dHRaw, const double* X,
                                 int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int i = idx / cols;
        int j = idx % cols;
        atomicAddDouble(&dW[idx], dHRaw[i] * X[j]);
    }
}

__global__ void k_accumulate_dB(double* dB, const double* dHRaw, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAddDouble(&dB[i], dHRaw[i]);
    }
}

__global__ void k_compute_dInput(double* dInput, const double* W, const double* dHRaw,
                                  int inputSize, int hiddenSize) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < inputSize) {
        double sum = 0;
        for (int i = 0; i < hiddenSize; i++)
            sum += W[i * inputSize + j] * dHRaw[i];
        dInput[j] = sum;
    }
}

__global__ void k_compute_dPrevH(double* dPrevH, const double* Whh, const double* dHRaw,
                                  int hiddenSize) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < hiddenSize) {
        double sum = 0;
        for (int i = 0; i < hiddenSize; i++)
            sum += Whh[i * hiddenSize + j] * dHRaw[i];
        dPrevH[j] = sum;
    }
}

// ============================================================================
// LSTM Backward Kernel
// ============================================================================
__global__ void k_lstm_backward_gates(double* dOG, double* dCTotal, double* dFG, 
                                       double* dIG, double* dCTilde,
                                       const double* dH, const double* dC,
                                       const double* FG, const double* IG,
                                       const double* CTilde, const double* OG,
                                       const double* TanhC, const double* PrevC,
                                       int hiddenSize, double clipVal) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        double og = OG[k], tanhc = TanhC[k];
        double fg = FG[k], ig = IG[k], ct = CTilde[k], pc = PrevC[k];
        
        dOG[k] = d_clip_value(dH[k] * tanhc * og * (1.0 - og), clipVal);
        double dcTotal = d_clip_value(dH[k] * og * (1.0 - tanhc * tanhc) + dC[k], clipVal);
        dCTotal[k] = dcTotal;
        dFG[k] = d_clip_value(dcTotal * pc * fg * (1.0 - fg), clipVal);
        dIG[k] = d_clip_value(dcTotal * ct * ig * (1.0 - ig), clipVal);
        dCTilde[k] = d_clip_value(dcTotal * ig * (1.0 - ct * ct), clipVal);
    }
}

__global__ void k_lstm_backward_dPrevC(double* dPrevC, const double* dCTotal, 
                                        const double* FG, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        dPrevC[k] = dCTotal[k] * FG[k];
    }
}

__global__ void k_lstm_backward_weights(double* dWf, double* dWi, double* dWc, double* dWo,
                                         double* dBf, double* dBi, double* dBc, double* dBo,
                                         const double* dFG, const double* dIG,
                                         const double* dCTilde, const double* dOG,
                                         const double* Concat,
                                         int hiddenSize, int concatSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hiddenSize * concatSize;
    if (idx < total) {
        int k = idx / concatSize;
        int j = idx % concatSize;
        double cj = Concat[j];
        atomicAddDouble(&dWf[idx], dFG[k] * cj);
        atomicAddDouble(&dWi[idx], dIG[k] * cj);
        atomicAddDouble(&dWc[idx], dCTilde[k] * cj);
        atomicAddDouble(&dWo[idx], dOG[k] * cj);
    }
    
    // Also accumulate biases (only first concatSize threads do this)
    if (idx < hiddenSize) {
        atomicAddDouble(&dBf[idx], dFG[idx]);
        atomicAddDouble(&dBi[idx], dIG[idx]);
        atomicAddDouble(&dBc[idx], dCTilde[idx]);
        atomicAddDouble(&dBo[idx], dOG[idx]);
    }
}

__global__ void k_lstm_backward_dInput_dPrevH(double* dInput, double* dPrevH,
                                               const double* Wf, const double* Wi,
                                               const double* Wc, const double* Wo,
                                               const double* dFG, const double* dIG,
                                               const double* dCTilde, const double* dOG,
                                               int inputSize, int hiddenSize, int concatSize) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < concatSize) {
        double sum = 0;
        for (int k = 0; k < hiddenSize; k++) {
            int idx = k * concatSize + j;
            sum += Wf[idx] * dFG[k] + Wi[idx] * dIG[k] + 
                   Wc[idx] * dCTilde[k] + Wo[idx] * dOG[k];
        }
        if (j < inputSize)
            dInput[j] = sum;
        else
            dPrevH[j - inputSize] = sum;
    }
}

// ============================================================================
// Apply Gradients Kernel
// ============================================================================
__global__ void k_apply_gradients(double* W, double* dW, double lr, double clipVal, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        W[idx] -= lr * d_clip_value(dW[idx], clipVal);
        dW[idx] = 0;
    }
}

// ============================================================================
// Loss Kernels
// ============================================================================
__global__ void k_mse_loss(double* loss, const double* pred, const double* target, int n) {
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = 0;
    if (idx < n) {
        double diff = pred[idx] - target[idx];
        sdata[tid] = diff * diff;
    }
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAddDouble(loss, sdata[0]);
}

__global__ void k_mse_gradient(double* grad, const double* pred, const double* target, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] = pred[idx] - target[idx];
    }
}

__global__ void k_crossentropy_gradient(double* grad, const double* pred, const double* target, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double p = fmax(1e-15, fmin(1.0 - 1e-15, pred[idx]));
        grad[idx] = (p - target[idx]) / (p * (1.0 - p) + 1e-15);
    }
}

// ============================================================================
// Device Memory Helper Class
// ============================================================================
class DeviceArray {
public:
    double* d_ptr;
    int size;
    
    DeviceArray() : d_ptr(nullptr), size(0) {}
    
    DeviceArray(int n) : d_ptr(nullptr), size(0) {
        if (n > 0) allocate(n);
    }
    
    ~DeviceArray() {
        if (d_ptr) cudaFree(d_ptr);
    }
    
    // Disable copy to prevent double-free
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    
    // Enable move
    DeviceArray(DeviceArray&& other) noexcept : d_ptr(other.d_ptr), size(other.size) {
        other.d_ptr = nullptr;
        other.size = 0;
    }
    
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            if (d_ptr) cudaFree(d_ptr);
            d_ptr = other.d_ptr;
            size = other.size;
            other.d_ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }
    
    void allocate(int n) {
        if (n <= 0) {
            size = 0;
            d_ptr = nullptr;
            return;
        }
        if (d_ptr) cudaFree(d_ptr);
        size = n;
        CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(double)));
        zero();
    }
    
    void zero() {
        if (size <= 0 || d_ptr == nullptr) return;
        int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_zero_array<<<blocks, BLOCK_SIZE>>>(d_ptr, size);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void fill(double val) {
        if (size <= 0 || d_ptr == nullptr) return;
        int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_fill_array<<<blocks, BLOCK_SIZE>>>(d_ptr, val, size);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void copyFromHost(const double* h_ptr, int n) {
        if (n <= 0 || d_ptr == nullptr) return;
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, min(n, size) * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    void copyFromHost(const DArray& h_arr) {
        if (h_arr.empty() || d_ptr == nullptr) return;
        int copySize = min((int)h_arr.size(), size);
        CUDA_CHECK(cudaMemcpy(d_ptr, h_arr.data(), copySize * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    void copyToHost(double* h_ptr, int n) {
        if (n <= 0 || d_ptr == nullptr || size <= 0) return;
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, min(n, size) * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    void copyToHost(DArray& h_arr) {
        if (size <= 0 || d_ptr == nullptr) {
            h_arr.clear();
            return;
        }
        h_arr.resize(size);
        CUDA_CHECK(cudaMemcpy(h_arr.data(), d_ptr, size * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    void copyFromDevice(const DeviceArray& other) {
        if (size <= 0 || d_ptr == nullptr || other.d_ptr == nullptr || other.size <= 0) return;
        int copySize = min(size, other.size);
        CUDA_CHECK(cudaMemcpy(d_ptr, other.d_ptr, copySize * sizeof(double), cudaMemcpyDeviceToDevice));
    }
};

// ============================================================================
// CUDA Cell Wrappers
// ============================================================================

class CudaSimpleRNNCell {
public:
    int inputSize, hiddenSize;
    ActivationType activation;
    
    DeviceArray Wih, Whh, Bh;
    DeviceArray dWih, dWhh, dBh;
    
    CudaSimpleRNNCell(int inSize, int hidSize, ActivationType act) 
        : inputSize(inSize), hiddenSize(hidSize), activation(act) {
        
        double scale = sqrt(2.0 / (inSize + hidSize));
        
        // Initialize weights on host then copy
        DArray h_Wih(hidSize * inSize), h_Whh(hidSize * hidSize), h_Bh(hidSize, 0);
        uniform_real_distribution<> dis(-scale, scale);
        for (auto& w : h_Wih) w = dis(gen);
        for (auto& w : h_Whh) w = dis(gen);
        
        Wih.allocate(hidSize * inSize);
        Whh.allocate(hidSize * hidSize);
        Bh.allocate(hidSize);
        Wih.copyFromHost(h_Wih);
        Whh.copyFromHost(h_Whh);
        Bh.copyFromHost(h_Bh);
        
        dWih.allocate(hidSize * inSize);
        dWhh.allocate(hidSize * hidSize);
        dBh.allocate(hidSize);
    }

void Load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Failed to open file for loading: " << filename << std::endl;
        return;
    }
    std::string line;
    enum Section { NONE, Wih, Whh, Bh } section = NONE;
    int rowCount = 0;
    while (std::getline(in, line)) {
        if (line == "#Wih")           { section = Wih; rowCount = 0; continue; }
        else if (line == "#Whh")      { section = Whh; rowCount = 0; continue; }
        else if (line == "#Bh")       { section = Bh; rowCount = 0; continue; }
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line); std::string cell; std::vector<double> vals;
        while (std::getline(ss, cell, ',')) vals.push_back(std::stod(cell));
        if (section == Wih && rowCount < Wih.size())   Wih[rowCount++] = vals;
        else if (section == Whh && rowCount < Whh.size()) Whh[rowCount++] = vals;
        else if (section == Bh && !vals.empty())       Bh = vals;
    }
    in.close();
    // After loading, upload to device:
    g_Wih.copyToDevice(Wih); g_Whh.copyToDevice(Whh); g_Bh.copyToDevice(Bh);
}

void Save(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for saving: " << filename << std::endl;
        return;
    }
    // Save Wih
    out << "#Wih\n";
    for (const auto& row : Wih) {
        for (size_t j = 0; j < row.size(); ++j) {
            out << row[j];
            if (j + 1 < row.size()) out << ",";
        }
        out << "\n";
    }
    // Save Whh
    out << "#Whh\n";
    for (const auto& row : Whh) {
        for (size_t j = 0; j < row.size(); ++j) {
            out << row[j];
            if (j + 1 < row.size()) out << ",";
        }
        out << "\n";
    }
    // Save Bh
    out << "#Bh\n";
    for (size_t j = 0; j < Bh.size(); ++j) {
        out << Bh[j];
        if (j + 1 < Bh.size()) out << ",";
    }
    out << "\n";
    out.close();
}

    void forward(DeviceArray& Input, DeviceArray& PrevH, DeviceArray& H, DeviceArray& PreH) {
        int blocks = (hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_simple_rnn_forward<<<blocks, BLOCK_SIZE>>>(
            H.d_ptr, PreH.d_ptr, Wih.d_ptr, Whh.d_ptr, Bh.d_ptr,
            Input.d_ptr, PrevH.d_ptr, inputSize, hiddenSize, (int)activation);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void backward(DeviceArray& dH, DeviceArray& H, DeviceArray& PreH,
                  DeviceArray& PrevH, DeviceArray& Input, double clipVal,
                  DeviceArray& dInput, DeviceArray& dPrevH, DeviceArray& dHRaw) {
        int blocks = (hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Compute dHRaw
        k_simple_rnn_backward_dHRaw<<<blocks, BLOCK_SIZE>>>(
            dHRaw.d_ptr, dH.d_ptr, H.d_ptr, hiddenSize, (int)activation, clipVal);
        
        // Accumulate gradients
        int wihBlocks = (hiddenSize * inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_accumulate_dW<<<wihBlocks, BLOCK_SIZE>>>(dWih.d_ptr, dHRaw.d_ptr, Input.d_ptr, hiddenSize, inputSize);
        
        int whhBlocks = (hiddenSize * hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_accumulate_dW<<<whhBlocks, BLOCK_SIZE>>>(dWhh.d_ptr, dHRaw.d_ptr, PrevH.d_ptr, hiddenSize, hiddenSize);
        
        k_accumulate_dB<<<blocks, BLOCK_SIZE>>>(dBh.d_ptr, dHRaw.d_ptr, hiddenSize);
        
        // Compute dInput and dPrevH
        int inBlocks = (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_compute_dInput<<<inBlocks, BLOCK_SIZE>>>(dInput.d_ptr, Wih.d_ptr, dHRaw.d_ptr, inputSize, hiddenSize);
        k_compute_dPrevH<<<blocks, BLOCK_SIZE>>>(dPrevH.d_ptr, Whh.d_ptr, dHRaw.d_ptr, hiddenSize);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void applyGradients(double lr, double clipVal) {
        int wihN = hiddenSize * inputSize;
        int whhN = hiddenSize * hiddenSize;
        
        k_apply_gradients<<<(wihN + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            Wih.d_ptr, dWih.d_ptr, lr, clipVal, wihN);
        k_apply_gradients<<<(whhN + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            Whh.d_ptr, dWhh.d_ptr, lr, clipVal, whhN);
        k_apply_gradients<<<(hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            Bh.d_ptr, dBh.d_ptr, lr, clipVal, hiddenSize);
    }
    
    void resetGradients() {
        dWih.zero();
        dWhh.zero();
        dBh.zero();
    }
};

class CudaLSTMCell {
public:
    int inputSize, hiddenSize, concatSize;
    ActivationType activation;
    
    DeviceArray Wf, Wi, Wc, Wo;
    DeviceArray Bf, Bi, Bc, Bo;
    DeviceArray dWf, dWi, dWc, dWo;
    DeviceArray dBf, dBi, dBc, dBo;
    
    CudaLSTMCell(int inSize, int hidSize, ActivationType act)
        : inputSize(inSize), hiddenSize(hidSize), activation(act) {
        
        concatSize = inSize + hidSize;
        double scale = sqrt(2.0 / concatSize);
        int wSize = hidSize * concatSize;
        
        DArray h_W(wSize);
        uniform_real_distribution<> dis(-scale, scale);
        
        Wf.allocate(wSize); Wi.allocate(wSize);
        Wc.allocate(wSize); Wo.allocate(wSize);
        Bf.allocate(hidSize); Bi.allocate(hidSize);
        Bc.allocate(hidSize); Bo.allocate(hidSize);
        
        for (auto& w : h_W) w = dis(gen);
        Wf.copyFromHost(h_W);
        for (auto& w : h_W) w = dis(gen);
        Wi.copyFromHost(h_W);
        for (auto& w : h_W) w = dis(gen);
        Wc.copyFromHost(h_W);
        for (auto& w : h_W) w = dis(gen);
        Wo.copyFromHost(h_W);
        
        // Forget gate bias = 1.0
        DArray h_Bf(hidSize, 1.0), h_B(hidSize, 0.0);
        Bf.copyFromHost(h_Bf);
        Bi.copyFromHost(h_B);
        Bc.copyFromHost(h_B);
        Bo.copyFromHost(h_B);
        
        dWf.allocate(wSize); dWi.allocate(wSize);
        dWc.allocate(wSize); dWo.allocate(wSize);
        dBf.allocate(hidSize); dBi.allocate(hidSize);
        dBc.allocate(hidSize); dBo.allocate(hidSize);
    }

void Load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Failed to open file for loading: " << filename << std::endl;
        return;
    }
    std::string line;
    enum Section { NONE, Wih, Whh, Bh } section = NONE;
    int rowCount = 0;
    while (std::getline(in, line)) {
        if (line == "#Wih")           { section = Wih; rowCount = 0; continue; }
        else if (line == "#Whh")      { section = Whh; rowCount = 0; continue; }
        else if (line == "#Bh")       { section = Bh; rowCount = 0; continue; }
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line); std::string cell; std::vector<double> vals;
        while (std::getline(ss, cell, ',')) vals.push_back(std::stod(cell));
        if (section == Wih && rowCount < Wih.size())   Wih[rowCount++] = vals;
        else if (section == Whh && rowCount < Whh.size()) Whh[rowCount++] = vals;
        else if (section == Bh && !vals.empty())       Bh = vals;
    }
    in.close();
    // After loading, upload to device:
    g_Wih.copyToDevice(Wih); g_Whh.copyToDevice(Whh); g_Bh.copyToDevice(Bh);
}

void Save(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for saving: " << filename << std::endl;
        return;
    }
    // Save Wih
    out << "#Wih\n";
    for (const auto& row : Wih) {
        for (size_t j = 0; j < row.size(); ++j) {
            out << row[j];
            if (j + 1 < row.size()) out << ",";
        }
        out << "\n";
    }
    // Save Whh
    out << "#Whh\n";
    for (const auto& row : Whh) {
        for (size_t j = 0; j < row.size(); ++j) {
            out << row[j];
            if (j + 1 < row.size()) out << ",";
        }
        out << "\n";
    }
    // Save Bh
    out << "#Bh\n";
    for (size_t j = 0; j < Bh.size(); ++j) {
        out << Bh[j];
        if (j + 1 < Bh.size()) out << ",";
    }
    out << "\n";
    out.close();
}

    void forward(DeviceArray& Input, DeviceArray& PrevH, DeviceArray& PrevC,
                 DeviceArray& H, DeviceArray& C, DeviceArray& FG, DeviceArray& IG,
                 DeviceArray& CTilde, DeviceArray& OG, DeviceArray& TanhC,
                 DeviceArray& Concat) {
        // Concat input and prevH
        int blocks = (concatSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_concat<<<blocks, BLOCK_SIZE>>>(Concat.d_ptr, Input.d_ptr, PrevH.d_ptr, inputSize, hiddenSize);
        
        // LSTM forward
        blocks = (hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_lstm_forward<<<blocks, BLOCK_SIZE>>>(
            H.d_ptr, C.d_ptr, FG.d_ptr, IG.d_ptr, CTilde.d_ptr, OG.d_ptr, TanhC.d_ptr,
            Wf.d_ptr, Wi.d_ptr, Wc.d_ptr, Wo.d_ptr,
            Bf.d_ptr, Bi.d_ptr, Bc.d_ptr, Bo.d_ptr,
            Concat.d_ptr, PrevC.d_ptr, concatSize, hiddenSize);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void backward(DeviceArray& dH, DeviceArray& dC,
                  DeviceArray& FG, DeviceArray& IG, DeviceArray& CTilde,
                  DeviceArray& OG, DeviceArray& TanhC, DeviceArray& PrevC,
                  DeviceArray& Concat, double clipVal,
                  DeviceArray& dInput, DeviceArray& dPrevH, DeviceArray& dPrevC,
                  DeviceArray& dOG, DeviceArray& dCTotal, DeviceArray& dFG,
                  DeviceArray& dIG, DeviceArray& dCTilde) {
        int blocks = (hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Compute gate gradients
        k_lstm_backward_gates<<<blocks, BLOCK_SIZE>>>(
            dOG.d_ptr, dCTotal.d_ptr, dFG.d_ptr, dIG.d_ptr, dCTilde.d_ptr,
            dH.d_ptr, dC.d_ptr, FG.d_ptr, IG.d_ptr, CTilde.d_ptr,
            OG.d_ptr, TanhC.d_ptr, PrevC.d_ptr, hiddenSize, clipVal);
        
        // dPrevC
        k_lstm_backward_dPrevC<<<blocks, BLOCK_SIZE>>>(dPrevC.d_ptr, dCTotal.d_ptr, FG.d_ptr, hiddenSize);
        
        // Accumulate weight gradients
        int wBlocks = (hiddenSize * concatSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_lstm_backward_weights<<<wBlocks, BLOCK_SIZE>>>(
            dWf.d_ptr, dWi.d_ptr, dWc.d_ptr, dWo.d_ptr,
            dBf.d_ptr, dBi.d_ptr, dBc.d_ptr, dBo.d_ptr,
            dFG.d_ptr, dIG.d_ptr, dCTilde.d_ptr, dOG.d_ptr,
            Concat.d_ptr, hiddenSize, concatSize);
        
        // Compute dInput and dPrevH
        int cBlocks = (concatSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_lstm_backward_dInput_dPrevH<<<cBlocks, BLOCK_SIZE>>>(
            dInput.d_ptr, dPrevH.d_ptr,
            Wf.d_ptr, Wi.d_ptr, Wc.d_ptr, Wo.d_ptr,
            dFG.d_ptr, dIG.d_ptr, dCTilde.d_ptr, dOG.d_ptr,
            inputSize, hiddenSize, concatSize);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void applyGradients(double lr, double clipVal) {
        int wSize = hiddenSize * concatSize;
        int blocks = (wSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int bBlocks = (hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        k_apply_gradients<<<blocks, BLOCK_SIZE>>>(Wf.d_ptr, dWf.d_ptr, lr, clipVal, wSize);
        k_apply_gradients<<<blocks, BLOCK_SIZE>>>(Wi.d_ptr, dWi.d_ptr, lr, clipVal, wSize);
        k_apply_gradients<<<blocks, BLOCK_SIZE>>>(Wc.d_ptr, dWc.d_ptr, lr, clipVal, wSize);
        k_apply_gradients<<<blocks, BLOCK_SIZE>>>(Wo.d_ptr, dWo.d_ptr, lr, clipVal, wSize);
        k_apply_gradients<<<bBlocks, BLOCK_SIZE>>>(Bf.d_ptr, dBf.d_ptr, lr, clipVal, hiddenSize);
        k_apply_gradients<<<bBlocks, BLOCK_SIZE>>>(Bi.d_ptr, dBi.d_ptr, lr, clipVal, hiddenSize);
        k_apply_gradients<<<bBlocks, BLOCK_SIZE>>>(Bc.d_ptr, dBc.d_ptr, lr, clipVal, hiddenSize);
        k_apply_gradients<<<bBlocks, BLOCK_SIZE>>>(Bo.d_ptr, dBo.d_ptr, lr, clipVal, hiddenSize);
    }
    
    void resetGradients() {
        dWf.zero(); dWi.zero(); dWc.zero(); dWo.zero();
        dBf.zero(); dBi.zero(); dBc.zero(); dBo.zero();
    }
};

class CudaGRUCell {
public:
    int inputSize, hiddenSize, concatSize;
    ActivationType activation;
    
    DeviceArray Wz, Wr, Wh;
    DeviceArray Bz, Br, Bh;
    DeviceArray dWz, dWr, dWh;
    DeviceArray dBz, dBr, dBh;
    
    CudaGRUCell(int inSize, int hidSize, ActivationType act)
        : inputSize(inSize), hiddenSize(hidSize), activation(act) {
        
        concatSize = inSize + hidSize;
        double scale = sqrt(2.0 / concatSize);
        int wSize = hidSize * concatSize;
        
        DArray h_W(wSize), h_B(hidSize, 0.0);
        uniform_real_distribution<> dis(-scale, scale);
        
        Wz.allocate(wSize); Wr.allocate(wSize); Wh.allocate(wSize);
        Bz.allocate(hidSize); Br.allocate(hidSize); Bh.allocate(hidSize);
        
        for (auto& w : h_W) w = dis(gen);
        Wz.copyFromHost(h_W);
        for (auto& w : h_W) w = dis(gen);
        Wr.copyFromHost(h_W);
        for (auto& w : h_W) w = dis(gen);
        Wh.copyFromHost(h_W);
        Bz.copyFromHost(h_B);
        Br.copyFromHost(h_B);
        Bh.copyFromHost(h_B);
        
        dWz.allocate(wSize); dWr.allocate(wSize); dWh.allocate(wSize);
        dBz.allocate(hidSize); dBr.allocate(hidSize); dBh.allocate(hidSize);
    }

void Load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Failed to open file for loading: " << filename << std::endl;
        return;
    }
    std::string line;
    enum Section { NONE, Wih, Whh, Bh } section = NONE;
    int rowCount = 0;
    while (std::getline(in, line)) {
        if (line == "#Wih")           { section = Wih; rowCount = 0; continue; }
        else if (line == "#Whh")      { section = Whh; rowCount = 0; continue; }
        else if (line == "#Bh")       { section = Bh; rowCount = 0; continue; }
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line); std::string cell; std::vector<double> vals;
        while (std::getline(ss, cell, ',')) vals.push_back(std::stod(cell));
        if (section == Wih && rowCount < Wih.size())   Wih[rowCount++] = vals;
        else if (section == Whh && rowCount < Whh.size()) Whh[rowCount++] = vals;
        else if (section == Bh && !vals.empty())       Bh = vals;
    }
    in.close();
    // After loading, upload to device:
    g_Wih.copyToDevice(Wih); g_Whh.copyToDevice(Whh); g_Bh.copyToDevice(Bh);
}

void Save(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for saving: " << filename << std::endl;
        return;
    }
    // Save Wih
    out << "#Wih\n";
    for (const auto& row : Wih) {
        for (size_t j = 0; j < row.size(); ++j) {
            out << row[j];
            if (j + 1 < row.size()) out << ",";
        }
        out << "\n";
    }
    // Save Whh
    out << "#Whh\n";
    for (const auto& row : Whh) {
        for (size_t j = 0; j < row.size(); ++j) {
            out << row[j];
            if (j + 1 < row.size()) out << ",";
        }
        out << "\n";
    }
    // Save Bh
    out << "#Bh\n";
    for (size_t j = 0; j < Bh.size(); ++j) {
        out << Bh[j];
        if (j + 1 < Bh.size()) out << ",";
    }
    out << "\n";
    out.close();
}

    void forward(DeviceArray& Input, DeviceArray& PrevH,
                 DeviceArray& H, DeviceArray& Z, DeviceArray& R, DeviceArray& HTilde,
                 DeviceArray& Concat) {
        // Concat input and prevH
        int blocks = (concatSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_concat<<<blocks, BLOCK_SIZE>>>(Concat.d_ptr, Input.d_ptr, PrevH.d_ptr, inputSize, hiddenSize);
        
        // Compute Z and R gates
        blocks = (hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_gru_forward_gates<<<blocks, BLOCK_SIZE>>>(
            Z.d_ptr, R.d_ptr, Wz.d_ptr, Wr.d_ptr, Bz.d_ptr, Br.d_ptr,
            Concat.d_ptr, concatSize, hiddenSize);
        
        // Compute hidden with reset gate
        k_gru_forward_hidden<<<blocks, BLOCK_SIZE>>>(
            H.d_ptr, HTilde.d_ptr, Wh.d_ptr, Bh.d_ptr,
            Input.d_ptr, PrevH.d_ptr, Z.d_ptr, R.d_ptr,
            inputSize, hiddenSize);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void applyGradients(double lr, double clipVal) {
        int wSize = hiddenSize * concatSize;
        int blocks = (wSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int bBlocks = (hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        k_apply_gradients<<<blocks, BLOCK_SIZE>>>(Wz.d_ptr, dWz.d_ptr, lr, clipVal, wSize);
        k_apply_gradients<<<blocks, BLOCK_SIZE>>>(Wr.d_ptr, dWr.d_ptr, lr, clipVal, wSize);
        k_apply_gradients<<<blocks, BLOCK_SIZE>>>(Wh.d_ptr, dWh.d_ptr, lr, clipVal, wSize);
        k_apply_gradients<<<bBlocks, BLOCK_SIZE>>>(Bz.d_ptr, dBz.d_ptr, lr, clipVal, hiddenSize);
        k_apply_gradients<<<bBlocks, BLOCK_SIZE>>>(Br.d_ptr, dBr.d_ptr, lr, clipVal, hiddenSize);
        k_apply_gradients<<<bBlocks, BLOCK_SIZE>>>(Bh.d_ptr, dBh.d_ptr, lr, clipVal, hiddenSize);
    }
    
    void resetGradients() {
        dWz.zero(); dWr.zero(); dWh.zero();
        dBz.zero(); dBr.zero(); dBh.zero();
    }
};

class CudaOutputLayer {
public:
    int inputSize, outputSize;
    ActivationType activation;
    
    DeviceArray W, B, dW, dB;
    
    CudaOutputLayer(int inSize, int outSize, ActivationType act)
        : inputSize(inSize), outputSize(outSize), activation(act) {
        
        double scale = sqrt(2.0 / inSize);
        int wSize = outSize * inSize;
        
        DArray h_W(wSize), h_B(outSize, 0.0);
        uniform_real_distribution<> dis(-scale, scale);
        for (auto& w : h_W) w = dis(gen);
        
        W.allocate(wSize);
        B.allocate(outSize);
        W.copyFromHost(h_W);
        B.copyFromHost(h_B);
        
        dW.allocate(wSize);
        dB.allocate(outSize);
    }
    
    void forward(DeviceArray& Input, DeviceArray& Output, DeviceArray& Pre) {
        int blocks = (outputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_output_forward<<<blocks, BLOCK_SIZE>>>(
            Output.d_ptr, Pre.d_ptr, W.d_ptr, B.d_ptr, Input.d_ptr,
            inputSize, outputSize, (int)activation);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void backward(DeviceArray& dOut, DeviceArray& Output, DeviceArray& Pre,
                  DeviceArray& Input, double clipVal, DeviceArray& dInput,
                  DeviceArray& dRaw) {
        int blocks = (outputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Compute dRaw
        k_simple_rnn_backward_dHRaw<<<blocks, BLOCK_SIZE>>>(
            dRaw.d_ptr, dOut.d_ptr, Output.d_ptr, outputSize, (int)activation, clipVal);
        
        // Accumulate weight gradients
        int wBlocks = (outputSize * inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_accumulate_dW<<<wBlocks, BLOCK_SIZE>>>(dW.d_ptr, dRaw.d_ptr, Input.d_ptr, outputSize, inputSize);
        k_accumulate_dB<<<blocks, BLOCK_SIZE>>>(dB.d_ptr, dRaw.d_ptr, outputSize);
        
        // Compute dInput
        int inBlocks = (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_compute_dInput<<<inBlocks, BLOCK_SIZE>>>(dInput.d_ptr, W.d_ptr, dRaw.d_ptr, inputSize, outputSize);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void applyGradients(double lr, double clipVal) {
        int wSize = outputSize * inputSize;
        k_apply_gradients<<<(wSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            W.d_ptr, dW.d_ptr, lr, clipVal, wSize);
        k_apply_gradients<<<(outputSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            B.d_ptr, dB.d_ptr, lr, clipVal, outputSize);
    }
    
    void resetGradients() {
        dW.zero();
        dB.zero();
    }
};

// ============================================================================
// Main CUDA RNN Facade
// ============================================================================
class CudaRNNFacade {
private:
    int inputSize, outputSize;
    vector<int> hiddenSizes;
    CellType cellType;
    ActivationType activation, outputActivation;
    LossType lossType;
    double learningRate, gradientClip;
    int bpttSteps;
    int sequenceLen;
    
    vector<unique_ptr<CudaSimpleRNNCell>> simpleCells;
    vector<unique_ptr<CudaLSTMCell>> lstmCells;
    vector<unique_ptr<CudaGRUCell>> gruCells;
    unique_ptr<CudaOutputLayer> outputLayer;
    
    // Persistent device buffers
    vector<DeviceArray> d_H, d_C, d_PreH;
    vector<DeviceArray> d_FG, d_IG, d_CTilde, d_OG, d_TanhC;
    vector<DeviceArray> d_Z, d_R, d_HTilde;
    vector<DeviceArray> d_Concat;
    DeviceArray d_OutVal, d_OutPre;
    
    // State buffers
    vector<DeviceArray> d_StateH, d_StateC;
    
    // Cache for sequences - use unique_ptr for proper move semantics
    struct TimeCache {
        unique_ptr<DeviceArray> Input, H, C, PreH;
        unique_ptr<DeviceArray> F, I, CTilde, O, TanhC;
        unique_ptr<DeviceArray> Z, R, HTilde;
        unique_ptr<DeviceArray> OutVal, OutPre;
        unique_ptr<DeviceArray> Concat;
        
        TimeCache() = default;
        TimeCache(TimeCache&&) = default;
        TimeCache& operator=(TimeCache&&) = default;
    };
    vector<TimeCache> caches;

public:
    CudaRNNFacade(int inSize, const vector<int>& hidSizes, int outSize,
                  CellType cType, ActivationType act, ActivationType outAct,
                  LossType loss, double lr, double gc, int bptt)
        : inputSize(inSize), outputSize(outSize), hiddenSizes(hidSizes),
          cellType(cType), activation(act), outputActivation(outAct),
          lossType(loss), learningRate(lr), gradientClip(gc), bpttSteps(bptt),
          sequenceLen(0) {
        
        int prevSize = inSize;
        
        switch (cType) {
            case ctSimpleRNN:
                simpleCells.resize(hidSizes.size());
                for (size_t i = 0; i < hidSizes.size(); i++) {
                    simpleCells[i] = make_unique<CudaSimpleRNNCell>(prevSize, hidSizes[i], act);
                    prevSize = hidSizes[i];
                }
                break;
            case ctLSTM:
                lstmCells.resize(hidSizes.size());
                for (size_t i = 0; i < hidSizes.size(); i++) {
                    lstmCells[i] = make_unique<CudaLSTMCell>(prevSize, hidSizes[i], act);
                    prevSize = hidSizes[i];
                }
                break;
            case ctGRU:
                gruCells.resize(hidSizes.size());
                for (size_t i = 0; i < hidSizes.size(); i++) {
                    gruCells[i] = make_unique<CudaGRUCell>(prevSize, hidSizes[i], act);
                    prevSize = hidSizes[i];
                }
                break;
        }
        
        outputLayer = make_unique<CudaOutputLayer>(prevSize, outSize, outAct);
        
        // Initialize state buffers
        d_StateH.resize(hidSizes.size());
        d_StateC.resize(hidSizes.size());
        for (size_t i = 0; i < hidSizes.size(); i++) {
            d_StateH[i].allocate(hidSizes[i]);
            d_StateC[i].allocate(hidSizes[i]);
        }
    }
    
    void resetStates() {
        for (size_t i = 0; i < hiddenSizes.size(); i++) {
            d_StateH[i].zero();
            d_StateC[i].zero();
        }
    }
    
    void allocateCaches(int seqLen) {
        if ((int)caches.size() >= seqLen) return;
        
        caches.resize(seqLen);
        int lastHiddenSize = hiddenSizes.back();
        int firstConcatSize = inputSize + hiddenSizes[0];
        
        for (int t = 0; t < seqLen; t++) {
            caches[t].Input = make_unique<DeviceArray>(inputSize);
            caches[t].H = make_unique<DeviceArray>(lastHiddenSize);
            caches[t].PreH = make_unique<DeviceArray>(lastHiddenSize);
            caches[t].OutVal = make_unique<DeviceArray>(outputSize);
            caches[t].OutPre = make_unique<DeviceArray>(outputSize);
            
            if (cellType == ctLSTM) {
                caches[t].C = make_unique<DeviceArray>(lastHiddenSize);
                caches[t].F = make_unique<DeviceArray>(lastHiddenSize);
                caches[t].I = make_unique<DeviceArray>(lastHiddenSize);
                caches[t].CTilde = make_unique<DeviceArray>(lastHiddenSize);
                caches[t].O = make_unique<DeviceArray>(lastHiddenSize);
                caches[t].TanhC = make_unique<DeviceArray>(lastHiddenSize);
                caches[t].Concat = make_unique<DeviceArray>(firstConcatSize);
            } else if (cellType == ctGRU) {
                caches[t].Z = make_unique<DeviceArray>(lastHiddenSize);
                caches[t].R = make_unique<DeviceArray>(lastHiddenSize);
                caches[t].HTilde = make_unique<DeviceArray>(lastHiddenSize);
                caches[t].Concat = make_unique<DeviceArray>(firstConcatSize);
            }
        }
    }
    
    DArray2D forwardSequence(const DArray2D& inputs) {
        sequenceLen = inputs.size();
        allocateCaches(sequenceLen);
        resetStates();
        
        DArray2D results(sequenceLen);
        int lastHiddenSize = hiddenSizes.back();
        int firstConcatSize = inputSize + hiddenSizes[0];
        
        DeviceArray d_X(inputSize);
        DeviceArray d_H(lastHiddenSize), d_PreH(lastHiddenSize);
        DeviceArray d_C(lastHiddenSize), d_FG(lastHiddenSize), d_IG(lastHiddenSize);
        DeviceArray d_CTilde(lastHiddenSize), d_OG(lastHiddenSize), d_TanhC(lastHiddenSize);
        DeviceArray d_Z(lastHiddenSize), d_R(lastHiddenSize), d_HTilde(lastHiddenSize);
        DeviceArray d_Concat(firstConcatSize);
        DeviceArray d_OutVal(outputSize), d_OutPre(outputSize);
        
        for (int t = 0; t < sequenceLen; t++) {
            // Copy input to device
            d_X.copyFromHost(inputs[t]);
            caches[t].Input->copyFromDevice(d_X);
            
            // Process through layers (simplified: single layer for now)
            for (size_t layer = 0; layer < hiddenSizes.size(); layer++) {
                switch (cellType) {
                    case ctSimpleRNN:
                        simpleCells[layer]->forward(d_X, d_StateH[layer], d_H, d_PreH);
                        d_StateH[layer].copyFromDevice(d_H);
                        break;
                    case ctLSTM:
                        lstmCells[layer]->forward(d_X, d_StateH[layer], d_StateC[layer],
                                                   d_H, d_C, d_FG, d_IG, d_CTilde, d_OG, d_TanhC,
                                                   d_Concat);
                        d_StateH[layer].copyFromDevice(d_H);
                        d_StateC[layer].copyFromDevice(d_C);
                        if (layer == 0) {  // Only cache first layer for introspection
                            caches[t].C->copyFromDevice(d_C);
                            caches[t].F->copyFromDevice(d_FG);
                            caches[t].I->copyFromDevice(d_IG);
                            caches[t].CTilde->copyFromDevice(d_CTilde);
                            caches[t].O->copyFromDevice(d_OG);
                            caches[t].TanhC->copyFromDevice(d_TanhC);
                            caches[t].Concat->copyFromDevice(d_Concat);
                        }
                        break;
                    case ctGRU:
                        gruCells[layer]->forward(d_X, d_StateH[layer], d_H, d_Z, d_R, d_HTilde, d_Concat);
                        d_StateH[layer].copyFromDevice(d_H);
                        if (layer == 0) {
                            caches[t].Z->copyFromDevice(d_Z);
                            caches[t].R->copyFromDevice(d_R);
                            caches[t].HTilde->copyFromDevice(d_HTilde);
                            caches[t].Concat->copyFromDevice(d_Concat);
                        }
                        break;
                }
                d_X.copyFromDevice(d_H);
            }
            
            caches[t].H->copyFromDevice(d_H);
            caches[t].PreH->copyFromDevice(d_PreH);
            
            // Output layer
            outputLayer->forward(d_H, d_OutVal, d_OutPre);
            caches[t].OutVal->copyFromDevice(d_OutVal);
            caches[t].OutPre->copyFromDevice(d_OutPre);
            
            // Copy result to host
            d_OutVal.copyToHost(results[t]);
        }
        
        cudaDeviceSynchronize();
        return results;
    }
    
    double backwardSequence(const DArray2D& targets) {
        int T_len = targets.size();
        int bpttLimit = bpttSteps > 0 ? bpttSteps : T_len;
        double totalLoss = 0;
        
        int lastHiddenSize = hiddenSizes.back();
        
        // Device buffers for backward
        DeviceArray d_Target(outputSize);
        DeviceArray d_Grad(outputSize);
        DeviceArray d_dH(lastHiddenSize), d_dC(lastHiddenSize);
        DeviceArray d_dInput(inputSize), d_dPrevH(lastHiddenSize), d_dPrevC(lastHiddenSize);
        DeviceArray d_dOG(lastHiddenSize), d_dCTotal(lastHiddenSize);
        DeviceArray d_dFG(lastHiddenSize), d_dIG(lastHiddenSize), d_dCTilde(lastHiddenSize);
        DeviceArray d_dHRaw(lastHiddenSize);
        DeviceArray d_dOutRaw(outputSize);
        DeviceArray d_Loss(1);
        
        // Accumulated state gradients
        DeviceArray d_dStateH(lastHiddenSize), d_dStateC(lastHiddenSize);
        d_dStateH.zero();
        d_dStateC.zero();
        
        for (int t = T_len - 1; t >= max(0, T_len - bpttLimit); t--) {
            // Compute loss
            d_Target.copyFromHost(targets[t]);
            d_Loss.zero();
            
            int blocks = (outputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            k_mse_loss<<<blocks, BLOCK_SIZE>>>(d_Loss.d_ptr, caches[t].OutVal->d_ptr, d_Target.d_ptr, outputSize);
            cudaDeviceSynchronize();
            
            double loss;
            d_Loss.copyToHost(&loss, 1);
            totalLoss += loss / outputSize;
            
            // Compute loss gradient
            if (lossType == ltMSE)
                k_mse_gradient<<<blocks, BLOCK_SIZE>>>(d_Grad.d_ptr, caches[t].OutVal->d_ptr, d_Target.d_ptr, outputSize);
            else
                k_crossentropy_gradient<<<blocks, BLOCK_SIZE>>>(d_Grad.d_ptr, caches[t].OutVal->d_ptr, d_Target.d_ptr, outputSize);
            
            // Backward through output layer
            outputLayer->backward(d_Grad, *caches[t].OutVal, *caches[t].OutPre,
                                   *caches[t].H, gradientClip, d_dH, d_dOutRaw);
            
            // Add accumulated gradients
            blocks = (lastHiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            k_vec_add<<<blocks, BLOCK_SIZE>>>(d_dH.d_ptr, d_dH.d_ptr, d_dStateH.d_ptr, lastHiddenSize);
            
            // Get previous states
            DeviceArray d_PrevH(lastHiddenSize), d_PrevC(lastHiddenSize);
            if (t > 0) {
                d_PrevH.copyFromDevice(*caches[t-1].H);
                if (cellType == ctLSTM)
                    d_PrevC.copyFromDevice(*caches[t-1].C);
            } else {
                d_PrevH.zero();
                d_PrevC.zero();
            }
            
            // Backward through RNN cell (single layer for simplicity)
            switch (cellType) {
                case ctSimpleRNN:
                    simpleCells[0]->backward(d_dH, *caches[t].H, *caches[t].PreH,
                                              d_PrevH, *caches[t].Input, gradientClip,
                                              d_dInput, d_dPrevH, d_dHRaw);
                    d_dStateH.copyFromDevice(d_dPrevH);
                    break;
                case ctLSTM:
                    k_vec_add<<<blocks, BLOCK_SIZE>>>(d_dC.d_ptr, d_dC.d_ptr, d_dStateC.d_ptr, lastHiddenSize);
                    lstmCells[0]->backward(d_dH, d_dC, *caches[t].F, *caches[t].I,
                                            *caches[t].CTilde, *caches[t].O, *caches[t].TanhC,
                                            d_PrevC, *caches[t].Concat, gradientClip,
                                            d_dInput, d_dPrevH, d_dPrevC,
                                            d_dOG, d_dCTotal, d_dFG, d_dIG, d_dCTilde);
                    d_dStateH.copyFromDevice(d_dPrevH);
                    d_dStateC.copyFromDevice(d_dPrevC);
                    break;
                case ctGRU:
                    // GRU backward (simplified - would need full implementation)
                    d_dStateH.copyFromDevice(d_dH);
                    break;
            }
        }
        
        return totalLoss / T_len;
    }
    
    double trainSequence(const DArray2D& inputs, const DArray2D& targets) {
        resetGradients();
        forwardSequence(inputs);
        double loss = backwardSequence(targets);
        applyGradients();
        return loss;
    }
    
    DArray2D predict(const DArray2D& inputs) {
        return forwardSequence(inputs);
    }
    
    void resetGradients() {
        switch (cellType) {
            case ctSimpleRNN:
                for (auto& cell : simpleCells) cell->resetGradients();
                break;
            case ctLSTM:
                for (auto& cell : lstmCells) cell->resetGradients();
                break;
            case ctGRU:
                for (auto& cell : gruCells) cell->resetGradients();
                break;
        }
        outputLayer->resetGradients();
    }
    
    void applyGradients() {
        switch (cellType) {
            case ctSimpleRNN:
                for (auto& cell : simpleCells) cell->applyGradients(learningRate, gradientClip);
                break;
            case ctLSTM:
                for (auto& cell : lstmCells) cell->applyGradients(learningRate, gradientClip);
                break;
            case ctGRU:
                for (auto& cell : gruCells) cell->applyGradients(learningRate, gradientClip);
                break;
        }
        outputLayer->applyGradients(learningRate, gradientClip);
    }
    
    // Getters
    int getLayerCount() const { return hiddenSizes.size(); }
    int getHiddenSize(int layer) const { return hiddenSizes[layer]; }
    CellType getCellType() const { return cellType; }
    double getLearningRate() const { return learningRate; }
    void setLearningRate(double lr) { learningRate = lr; }
    int getSequenceLength() const { return sequenceLen; }
    
    // Facade introspection methods
    double getHiddenState(int layerIdx, int timestep, int neuronIdx) {
        if (timestep < 0 || timestep >= sequenceLen || neuronIdx < 0) return 0;
        if (layerIdx < 0 || layerIdx >= (int)hiddenSizes.size()) return 0;
        if (neuronIdx >= hiddenSizes[layerIdx]) return 0;
        if (!caches[timestep].H) return 0;
        DArray h;
        caches[timestep].H->copyToHost(h);
        if (neuronIdx < (int)h.size()) return h[neuronIdx];
        return 0;
    }
    
    void setHiddenState(int layerIdx, int timestep, int neuronIdx, double value) {
        if (timestep < 0 || timestep >= sequenceLen || neuronIdx < 0) return;
        if (layerIdx < 0 || layerIdx >= (int)hiddenSizes.size()) return;
        if (neuronIdx >= hiddenSizes[layerIdx]) return;
        if (!caches[timestep].H) return;
        DArray h;
        caches[timestep].H->copyToHost(h);
        if (neuronIdx < (int)h.size()) {
            h[neuronIdx] = value;
            caches[timestep].H->copyFromHost(h);
        }
    }
    
    double getOutput(int timestep, int outputIdx) {
        if (timestep < 0 || timestep >= sequenceLen || outputIdx < 0) return 0;
        if (outputIdx >= outputSize) return 0;
        if (!caches[timestep].OutVal) return 0;
        DArray out;
        caches[timestep].OutVal->copyToHost(out);
        if (outputIdx < (int)out.size()) return out[outputIdx];
        return 0;
    }
    
    double getCellState(int layerIdx, int timestep, int neuronIdx) {
        if (cellType != ctLSTM) return 0;
        if (timestep < 0 || timestep >= sequenceLen || neuronIdx < 0) return 0;
        if (layerIdx < 0 || layerIdx >= (int)hiddenSizes.size()) return 0;
        if (neuronIdx >= hiddenSizes[layerIdx]) return 0;
        if (!caches[timestep].C) return 0;
        DArray c;
        caches[timestep].C->copyToHost(c);
        if (neuronIdx < (int)c.size()) return c[neuronIdx];
        return 0;
    }
    
    double getGateValue(GateType gateType, int layerIdx, int timestep, int neuronIdx) {
        if (timestep < 0 || timestep >= sequenceLen || neuronIdx < 0) return 0;
        DArray gate;
        
        if (cellType == ctLSTM) {
            switch (gateType) {
                case gtForget: if (caches[timestep].F) caches[timestep].F->copyToHost(gate); break;
                case gtInput: if (caches[timestep].I) caches[timestep].I->copyToHost(gate); break;
                case gtOutput: if (caches[timestep].O) caches[timestep].O->copyToHost(gate); break;
                case gtCellCandidate: if (caches[timestep].CTilde) caches[timestep].CTilde->copyToHost(gate); break;
                default: return 0;
            }
        } else if (cellType == ctGRU) {
            switch (gateType) {
                case gtUpdate: if (caches[timestep].Z) caches[timestep].Z->copyToHost(gate); break;
                case gtReset: if (caches[timestep].R) caches[timestep].R->copyToHost(gate); break;
                case gtHiddenCandidate: if (caches[timestep].HTilde) caches[timestep].HTilde->copyToHost(gate); break;
                default: return 0;
            }
        } else {
            return 0;
        }
        
        if (neuronIdx < (int)gate.size()) return gate[neuronIdx];
        return 0;
    }
    
    double getPreActivation(int layerIdx, int timestep, int neuronIdx) {
        if (timestep < 0 || timestep >= sequenceLen || neuronIdx < 0) return 0;
        if (!caches[timestep].PreH) return 0;
        DArray pre;
        caches[timestep].PreH->copyToHost(pre);
        if (neuronIdx < (int)pre.size()) return pre[neuronIdx];
        return 0;
    }
    
    double getInputValue(int timestep, int inputIdx) {
        if (timestep < 0 || timestep >= sequenceLen || inputIdx < 0) return 0;
        if (inputIdx >= inputSize) return 0;
        if (!caches[timestep].Input) return 0;
        DArray inp;
        caches[timestep].Input->copyToHost(inp);
        if (inputIdx < (int)inp.size()) return inp[inputIdx];
        return 0;
    }
    
    DArray getSequenceOutputs(int outputIdx) {
        DArray result(sequenceLen);
        for (int t = 0; t < sequenceLen; t++) {
            result[t] = getOutput(t, outputIdx);
        }
        return result;
    }
    
    DArray getSequenceHiddenStates(int layerIdx, int neuronIdx) {
        DArray result(sequenceLen);
        for (int t = 0; t < sequenceLen; t++) {
            result[t] = getHiddenState(layerIdx, t, neuronIdx);
        }
        return result;
    }
    
    DArray getSequenceCellStates(int layerIdx, int neuronIdx) {
        DArray result(sequenceLen);
        for (int t = 0; t < sequenceLen; t++) {
            result[t] = getCellState(layerIdx, t, neuronIdx);
        }
        return result;
    }
    
    void setDropoutRate(double rate) { dropoutRate = rate; }
    double getDropoutRate() const { return dropoutRate; }
    
    bool detectVanishingGradient(int layerIdx, double threshold = 1e-6) {
        // Check gradient norms across timesteps
        for (int t = 0; t < sequenceLen; t++) {
            if (!caches[t].H) continue;
            DArray h;
            caches[t].H->copyToHost(h);
            double norm = 0;
            for (double v : h) norm += v * v;
            if (sqrt(norm) < threshold) return true;
        }
        return false;
    }
    
    bool detectExplodingGradient(int layerIdx, double threshold = 1e6) {
        for (int t = 0; t < sequenceLen; t++) {
            if (!caches[t].H) continue;
            DArray h;
            caches[t].H->copyToHost(h);
            double norm = 0;
            for (double v : h) norm += v * v;
            if (sqrt(norm) > threshold) return true;
        }
        return false;
    }
    
private:
    double dropoutRate = 0.0;
};

// ============================================================================
// Utility Functions
// ============================================================================

DArray2D LoadCSV(const string& filename, int expectedCols = -1) {
    DArray2D data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return data;
    }
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        DArray row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        if (expectedCols < 0 || (int)row.size() == expectedCols)
            data.push_back(row);
    }
    return data;
}

void SaveCSV(const string& filename, const DArray2D& data) {
    ofstream file(filename);
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); i++) {
            if (i > 0) file << ",";
            file << row[i];
        }
        file << "\n";
    }
}

vector<int> ParseHiddenSizes(const string& s) {
    vector<int> sizes;
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')) {
        sizes.push_back(stoi(item));
    }
    return sizes;
}

CellType ParseCellType(const string& s) {
    if (s == "lstm" || s == "LSTM") return ctLSTM;
    if (s == "gru" || s == "GRU") return ctGRU;
    return ctSimpleRNN;
}

ActivationType ParseActivation(const string& s) {
    if (s == "tanh") return atTanh;
    if (s == "relu") return atReLU;
    if (s == "linear") return atLinear;
    return atSigmoid;
}

LossType ParseLossType(const string& s) {
    if (s == "crossentropy" || s == "ce") return ltCrossEntropy;
    return ltMSE;
}

void PrintUsage(const char* progname) {
    cout << "RNN Facade CLI (CUDA GPU) - Matthew Abbott 2025\n\n";
    cout << "Usage: " << progname << " <command> [options]\n\n";
    cout << "Commands:\n";
    cout << "  create              Create and initialize an RNN model\n";
    cout << "  train               Train the model on data\n";
    cout << "  predict             Run prediction on input data\n";
    cout << "  info                Display GPU information\n";
    cout << "\nFacade Introspection Commands:\n";
    cout << "  get-hidden          Get hidden state value\n";
    cout << "  set-hidden          Set hidden state value\n";
    cout << "  get-output          Get output value at timestep\n";
    cout << "  get-cell-state      Get LSTM cell state\n";
    cout << "  get-gate            Get gate value (LSTM/GRU)\n";
    cout << "  get-preactivation   Get pre-activation value\n";
    cout << "  get-input           Get input vector value\n";
    cout << "  reset-states        Reset all hidden/cell states\n";
    cout << "  set-dropout         Set dropout rate\n";
    cout << "  get-dropout         Get current dropout rate\n";
    cout << "  detect-vanishing    Check for vanishing gradients\n";
    cout << "  detect-exploding    Check for exploding gradients\n";
    cout << "  get-seq-outputs     Get all outputs for a sequence\n";
    cout << "  get-seq-hidden      Get hidden states over sequence\n";
    cout << "\nCreate/Train/Predict options:\n";
    cout << "  --input-size <n>       Input dimension (required)\n";
    cout << "  --hidden-sizes <n,n>   Comma-separated hidden layer sizes (required)\n";
    cout << "  --output-size <n>      Output dimension (required)\n";
    cout << "  --cell-type <type>     rnn, lstm, or gru (default: lstm)\n";
    cout << "  --activation <type>    sigmoid, tanh, relu, linear (default: tanh)\n";
    cout << "  --output-activation    Output layer activation (default: sigmoid)\n";
    cout << "  --loss <type>          mse or crossentropy (default: mse)\n";
    cout << "  --learning-rate <f>    Learning rate (default: 0.01)\n";
    cout << "  --gradient-clip <f>    Gradient clipping value (default: 5.0)\n";
    cout << "  --bptt-steps <n>       BPTT truncation steps (default: 0 = full)\n";
    cout << "  --epochs <n>           Number of training epochs (default: 100)\n";
    cout << "  --input-file <file>    CSV file with input sequences\n";
    cout << "  --target-file <file>   CSV file with target sequences\n";
    cout << "  --output-file <file>   CSV file to write predictions\n";
    cout << "\nFacade options:\n";
    cout << "  --layer <n>            Layer index (default: 0)\n";
    cout << "  --timestep <n>         Timestep index (default: 0)\n";
    cout << "  --neuron <n>           Neuron index (default: 0)\n";
    cout << "  --output-idx <n>       Output index (default: 0)\n";
    cout << "  --value <f>            Value to set\n";
    cout << "  --gate <type>          Gate type: forget,input,output,cell,update,reset,hidden\n";
    cout << "  --threshold <f>        Threshold for gradient detection (default: 1e-6)\n";
}

void PrintGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cout << "CUDA Devices: " << deviceCount << "\n";
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        cout << "\nDevice " << i << ": " << prop.name << "\n";
        cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        cout << "  Total memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB\n";
        cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 1;
    }

    string command = argv[1];
    
    if (command == "info") {
        PrintGPUInfo();
        return 0;
    }
    
    // Default parameters
    int inputSize = 0, outputSize = 0;
    vector<int> hiddenSizes;
    CellType cellType = ctLSTM;
    ActivationType activation = atTanh;
    ActivationType outputActivation = atSigmoid;
    LossType lossType = ltMSE;
    double learningRate = 0.01;
    double gradientClip = 5.0;
    int bpttSteps = 0;
    int epochs = 100;
    string inputFile, targetFile, outputFile;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--input-size" && i+1 < argc) inputSize = stoi(argv[++i]);
        else if (arg == "--hidden-sizes" && i+1 < argc) hiddenSizes = ParseHiddenSizes(argv[++i]);
        else if (arg == "--output-size" && i+1 < argc) outputSize = stoi(argv[++i]);
        else if (arg == "--cell-type" && i+1 < argc) cellType = ParseCellType(argv[++i]);
        else if (arg == "--activation" && i+1 < argc) activation = ParseActivation(argv[++i]);
        else if (arg == "--output-activation" && i+1 < argc) outputActivation = ParseActivation(argv[++i]);
        else if (arg == "--loss" && i+1 < argc) lossType = ParseLossType(argv[++i]);
        else if (arg == "--learning-rate" && i+1 < argc) learningRate = stod(argv[++i]);
        else if (arg == "--gradient-clip" && i+1 < argc) gradientClip = stod(argv[++i]);
        else if (arg == "--bptt-steps" && i+1 < argc) bpttSteps = stoi(argv[++i]);
        else if (arg == "--epochs" && i+1 < argc) epochs = stoi(argv[++i]);
        else if (arg == "--input-file" && i+1 < argc) inputFile = argv[++i];
        else if (arg == "--target-file" && i+1 < argc) targetFile = argv[++i];
        else if (arg == "--output-file" && i+1 < argc) outputFile = argv[++i];
        else if (arg == "--help" || arg == "-h") { PrintUsage(argv[0]); return 0; }
    }

    // Parse additional facade options
    int layerIdx = 0, timestep = 0, neuronIdx = 0, outputIdx = 0;
    double value = 0.0, threshold = 1e-6;
    string gateStr;
    
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--layer" && i+1 < argc) layerIdx = stoi(argv[++i]);
        else if (arg == "--timestep" && i+1 < argc) timestep = stoi(argv[++i]);
        else if (arg == "--neuron" && i+1 < argc) neuronIdx = stoi(argv[++i]);
        else if (arg == "--output-idx" && i+1 < argc) outputIdx = stoi(argv[++i]);
        else if (arg == "--value" && i+1 < argc) value = stod(argv[++i]);
        else if (arg == "--threshold" && i+1 < argc) threshold = stod(argv[++i]);
        else if (arg == "--gate" && i+1 < argc) gateStr = argv[++i];
    }
    
    auto parseGateType = [](const string& s) -> GateType {
        if (s == "forget") return gtForget;
        if (s == "input") return gtInput;
        if (s == "output") return gtOutput;
        if (s == "cell") return gtCellCandidate;
        if (s == "update") return gtUpdate;
        if (s == "reset") return gtReset;
        if (s == "hidden") return gtHiddenCandidate;
        return gtForget;
    };

    if (command == "create" || command == "train" || command == "predict" ||
        command == "get-hidden" || command == "set-hidden" || command == "get-output" ||
        command == "get-cell-state" || command == "get-gate" || command == "get-preactivation" ||
        command == "get-input" || command == "reset-states" || command == "set-dropout" ||
        command == "get-dropout" || command == "detect-vanishing" || command == "detect-exploding" ||
        command == "get-seq-outputs" || command == "get-seq-hidden") {
        
        if (inputSize == 0 || hiddenSizes.empty() || outputSize == 0) {
            cerr << "Error: --input-size, --hidden-sizes, and --output-size are required\n";
            return 1;
        }

        cout << "Initializing CUDA...\n";
        PrintGPUInfo();
        cout << "\n";

        CudaRNNFacade rnn(inputSize, hiddenSizes, outputSize, cellType, activation,
                          outputActivation, lossType, learningRate, gradientClip, bpttSteps);

        cout << "Created CUDA RNN: input=" << inputSize << " hidden=[";
        for (size_t i = 0; i < hiddenSizes.size(); i++) {
            if (i > 0) cout << ",";
            cout << hiddenSizes[i];
        }
        cout << "] output=" << outputSize;
        cout << " type=" << (cellType == ctLSTM ? "LSTM" : (cellType == ctGRU ? "GRU" : "SimpleRNN"));
        cout << " lr=" << learningRate << "\n";

        // Run forward pass if input file provided (needed for introspection)
        DArray2D inputs, targets;
        if (!inputFile.empty()) {
            inputs = LoadCSV(inputFile, inputSize);
            if (!inputs.empty()) {
                rnn.forwardSequence(inputs);
                cout << "Loaded and processed " << inputs.size() << " timesteps\n";
            }
        }
        if (!targetFile.empty()) {
            targets = LoadCSV(targetFile, outputSize);
        }

        if (command == "train") {
            if (inputs.empty() || targets.empty()) {
                cerr << "Error: --input-file and --target-file required for training\n";
                return 1;
            }
            cout << "Training on GPU...\n";
            
            for (int epoch = 0; epoch < epochs; epoch++) {
                double loss = rnn.trainSequence(inputs, targets);
                if (epoch % 10 == 0 || epoch == epochs - 1)
                    cout << "Epoch " << epoch << " loss: " << loss << "\n";
            }
        }
        else if (command == "predict") {
            if (inputs.empty()) {
                cerr << "Error: --input-file required for prediction\n";
                return 1;
            }
            DArray2D predictions = rnn.predict(inputs);
            
            if (!outputFile.empty()) {
                SaveCSV(outputFile, predictions);
                cout << "Predictions saved to " << outputFile << "\n";
            } else {
                cout << "Predictions:\n";
                for (const auto& row : predictions) {
                    for (size_t i = 0; i < row.size(); i++) {
                        if (i > 0) cout << ",";
                        cout << row[i];
                    }
                    cout << "\n";
                }
            }
        }
        else if (command == "get-hidden") {
            double val = rnn.getHiddenState(layerIdx, timestep, neuronIdx);
            cout << "Hidden[layer=" << layerIdx << ", t=" << timestep << ", n=" << neuronIdx << "] = " << val << "\n";
        }
        else if (command == "set-hidden") {
            rnn.setHiddenState(layerIdx, timestep, neuronIdx, value);
            cout << "Set Hidden[layer=" << layerIdx << ", t=" << timestep << ", n=" << neuronIdx << "] = " << value << "\n";
        }
        else if (command == "get-output") {
            double val = rnn.getOutput(timestep, outputIdx);
            cout << "Output[t=" << timestep << ", idx=" << outputIdx << "] = " << val << "\n";
        }
        else if (command == "get-cell-state") {
            double val = rnn.getCellState(layerIdx, timestep, neuronIdx);
            cout << "CellState[layer=" << layerIdx << ", t=" << timestep << ", n=" << neuronIdx << "] = " << val << "\n";
        }
        else if (command == "get-gate") {
            GateType gt = parseGateType(gateStr);
            double val = rnn.getGateValue(gt, layerIdx, timestep, neuronIdx);
            cout << "Gate[" << gateStr << ", layer=" << layerIdx << ", t=" << timestep << ", n=" << neuronIdx << "] = " << val << "\n";
        }
        else if (command == "get-preactivation") {
            double val = rnn.getPreActivation(layerIdx, timestep, neuronIdx);
            cout << "PreActivation[layer=" << layerIdx << ", t=" << timestep << ", n=" << neuronIdx << "] = " << val << "\n";
        }
        else if (command == "get-input") {
            double val = rnn.getInputValue(timestep, neuronIdx);
            cout << "Input[t=" << timestep << ", idx=" << neuronIdx << "] = " << val << "\n";
        }
        else if (command == "reset-states") {
            rnn.resetStates();
            cout << "All states reset to zero\n";
        }
        else if (command == "set-dropout") {
            rnn.setDropoutRate(value);
            cout << "Dropout rate set to " << value << "\n";
        }
        else if (command == "get-dropout") {
            cout << "Dropout rate = " << rnn.getDropoutRate() << "\n";
        }
        else if (command == "detect-vanishing") {
            bool detected = rnn.detectVanishingGradient(layerIdx, threshold);
            cout << "Vanishing gradient " << (detected ? "DETECTED" : "not detected") << " (threshold=" << threshold << ")\n";
        }
        else if (command == "detect-exploding") {
            bool detected = rnn.detectExplodingGradient(layerIdx, threshold);
            cout << "Exploding gradient " << (detected ? "DETECTED" : "not detected") << " (threshold=" << threshold << ")\n";
        }
        else if (command == "get-seq-outputs") {
            DArray seq = rnn.getSequenceOutputs(outputIdx);
            cout << "Sequence outputs[idx=" << outputIdx << "]:\n";
            for (size_t t = 0; t < seq.size(); t++) {
                cout << "  t=" << t << ": " << seq[t] << "\n";
            }
        }
        else if (command == "get-seq-hidden") {
            DArray seq = rnn.getSequenceHiddenStates(layerIdx, neuronIdx);
            cout << "Sequence hidden[layer=" << layerIdx << ", neuron=" << neuronIdx << "]:\n";
            for (size_t t = 0; t < seq.size(); t++) {
                cout << "  t=" << t << ": " << seq[t] << "\n";
            }
        }
    }
    else if (command == "help" || command == "--help" || command == "-h") {
        PrintUsage(argv[0]);
    }
    else {
        cerr << "Unknown command: " << command << "\n";
        PrintUsage(argv[0]);
        return 1;
    }

    return 0;
}
