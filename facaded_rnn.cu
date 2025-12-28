//
// Facaded RNN
// Matthew Abbott 2025
//

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <memory>
#include <cuda_runtime.h>

// ============================================================================
// Type Definitions
// ============================================================================

using DArray = std::vector<double>;
using DArray2D = std::vector<DArray>;
using DArray3D = std::vector<DArray2D>;

enum class TActivationType {
    atSigmoid,
    atTanh,
    atReLU,
    atLinear
};

enum class TLossType {
    ltMSE,
    ltCrossEntropy
};

enum class TCellType {
    ctSimpleRNN,
    ctLSTM,
    ctGRU
};

enum class TGateType {
    gtForget,
    gtInput,
    gtOutput,
    gtCellCandidate,
    gtUpdate,
    gtReset,
    gtHiddenCandidate
};

struct TTimeStepCacheEx {
    DArray Input;
    DArray H, C;
    DArray PreH;
    DArray F, I, CTilde, O, TanhC;
    DArray Z, R, HTilde;
    DArray OutPre, OutVal;
    DArray DropoutMask;
};
using TTimeStepCacheExArray = std::vector<TTimeStepCacheEx>;

// ============================================================================
// CUDA Buffer Classes
// ============================================================================

class CUDABuffer {
private:
    double* gpu_ptr;
    size_t size;

public:
    CUDABuffer(size_t sz) : size(sz), gpu_ptr(nullptr) {
        cudaMalloc(&gpu_ptr, sz * sizeof(double));
    }

    ~CUDABuffer() {
        if (gpu_ptr) cudaFree(gpu_ptr);
    }

    void upload(const double* data) {
        if (gpu_ptr)
            cudaMemcpy(gpu_ptr, data, size * sizeof(double), cudaMemcpyHostToDevice);
    }

    void download(double* data) {
        if (gpu_ptr)
            cudaMemcpy(data, gpu_ptr, size * sizeof(double), cudaMemcpyDeviceToHost);
    }

    double* getPtr() const { return gpu_ptr; }
    size_t getSize() const { return size; }
    bool isValid() const { return gpu_ptr != nullptr; }
};

class CUDAMatrix {
private:
    double* gpu_ptr;
    int rows, cols;

public:
    CUDAMatrix(int r, int c) : rows(r), cols(c), gpu_ptr(nullptr) {
        cudaMalloc(&gpu_ptr, r * c * sizeof(double));
    }

    ~CUDAMatrix() {
        if (gpu_ptr) cudaFree(gpu_ptr);
    }

    void upload(const DArray2D& data) {
        if (!gpu_ptr) return;
        std::vector<double> flat(rows * cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat[i * cols + j] = data[i][j];
            }
        }
        cudaMemcpy(gpu_ptr, flat.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    }

    void download(DArray2D& data) {
        if (!gpu_ptr) return;
        std::vector<double> flat(rows * cols);
        cudaMemcpy(flat.data(), gpu_ptr, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
        data.resize(rows);
        for (int i = 0; i < rows; i++) {
            data[i].resize(cols);
            for (int j = 0; j < cols; j++) {
                data[i][j] = flat[i * cols + j];
            }
        }
    }

    double* getPtr() const { return gpu_ptr; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    bool isValid() const { return gpu_ptr != nullptr; }
};

// ============================================================================
// RNN Cell Wrappers
// ============================================================================

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> dis(0.0, 1.0);

double ClipValue(double V, double MaxVal) {
    if (V > MaxVal) return MaxVal;
    if (V < -MaxVal) return -MaxVal;
    return V;
}

double RandomWeight(double Scale) {
    return (dis(gen) - 0.5) * 2.0 * Scale;
}

void InitMatrix(DArray2D& M, int Rows, int Cols, double Scale) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++) {
            M[i][j] = RandomWeight(Scale);
        }
    }
}

void ZeroMatrix(DArray2D& M, int Rows, int Cols) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols, 0.0);
    }
}

void ZeroArray(DArray& A, int Size) {
    A.resize(Size, 0.0);
}

DArray ConcatArrays(const DArray& A, const DArray& B) {
    DArray Result(A.size() + B.size());
    std::copy(A.begin(), A.end(), Result.begin());
    std::copy(B.begin(), B.end(), Result.begin() + A.size());
    return Result;
}

double ApplyActivation(double X, TActivationType ActType) {
    switch (ActType) {
        case TActivationType::atSigmoid:
            X = std::max(-500.0, std::min(500.0, X));
            return 1.0 / (1.0 + std::exp(-X));
        case TActivationType::atTanh:
            return std::tanh(X);
        case TActivationType::atReLU:
            return X > 0 ? X : 0;
        case TActivationType::atLinear:
            return X;
        default:
            return X;
    }
}

double ActivationDerivative(double Y, TActivationType ActType) {
    switch (ActType) {
        case TActivationType::atSigmoid:
            return Y * (1.0 - Y);
        case TActivationType::atTanh:
            return 1.0 - Y * Y;
        case TActivationType::atReLU:
            return Y > 0 ? 1.0 : 0.0;
        case TActivationType::atLinear:
            return 1.0;
        default:
            return 1.0;
    }
}

class TSimpleRNNCellWrapper {
public:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;
    DArray2D Wih, Whh;
    DArray Bh;
    DArray2D dWih, dWhh;
    DArray dBh;
    DArray2D MWih, MWhh;
    DArray MBh;
    DArray2D VWih, VWhh;
    DArray VBh;

    TSimpleRNNCellWrapper(int InputSize, int HiddenSize, TActivationType Activation)
        : FInputSize(InputSize), FHiddenSize(HiddenSize), FActivation(Activation) {
        double Scale = std::sqrt(2.0 / (InputSize + HiddenSize));
        InitMatrix(Wih, HiddenSize, InputSize, Scale);
        InitMatrix(Whh, HiddenSize, HiddenSize, Scale);
        ZeroArray(Bh, HiddenSize);
        ZeroMatrix(dWih, HiddenSize, InputSize);
        ZeroMatrix(dWhh, HiddenSize, HiddenSize);
        ZeroArray(dBh, HiddenSize);
        ZeroMatrix(MWih, HiddenSize, InputSize);
        ZeroMatrix(MWhh, HiddenSize, HiddenSize);
        ZeroArray(MBh, HiddenSize);
        ZeroMatrix(VWih, HiddenSize, InputSize);
        ZeroMatrix(VWhh, HiddenSize, HiddenSize);
        ZeroArray(VBh, HiddenSize);
    }

    void Forward(const DArray& Input, const DArray& PrevH, DArray& H, DArray& PreH) {
        H.resize(FHiddenSize);
        PreH.resize(FHiddenSize);
        for (int i = 0; i < FHiddenSize; i++) {
            double Sum = Bh[i];
            for (int j = 0; j < FInputSize; j++)
                Sum += Wih[i][j] * Input[j];
            for (int j = 0; j < FHiddenSize; j++)
                Sum += Whh[i][j] * PrevH[j];
            PreH[i] = Sum;
            H[i] = ApplyActivation(Sum, FActivation);
        }
    }

    void Backward(const DArray& dH, const DArray& H, const DArray& PreH,
                  const DArray& PrevH, const DArray& Input, double ClipVal,
                  DArray& dInput, DArray& dPrevH) {
        DArray dHRaw(FHiddenSize);
        dInput.resize(FInputSize, 0.0);
        dPrevH.resize(FHiddenSize, 0.0);

        for (int i = 0; i < FHiddenSize; i++)
            dHRaw[i] = ClipValue(dH[i] * ActivationDerivative(H[i], FActivation), ClipVal);

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
    }

    void ResetGradients() {
        ZeroMatrix(dWih, FHiddenSize, FInputSize);
        ZeroMatrix(dWhh, FHiddenSize, FHiddenSize);
        ZeroArray(dBh, FHiddenSize);
    }
};

class TLSTMCellWrapper {
public:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;
    DArray2D Wf, Wi, Wc, Wo;
    DArray Bf, Bi, Bc, Bo;
    DArray2D dWf, dWi, dWc, dWo;
    DArray dBf, dBi, dBc, dBo;

    TLSTMCellWrapper(int InputSize, int HiddenSize, TActivationType Activation)
        : FInputSize(InputSize), FHiddenSize(HiddenSize), FActivation(Activation) {
        double Scale = std::sqrt(2.0 / (InputSize + HiddenSize));
        InitMatrix(Wf, HiddenSize, InputSize + HiddenSize, Scale);
        InitMatrix(Wi, HiddenSize, InputSize + HiddenSize, Scale);
        InitMatrix(Wc, HiddenSize, InputSize + HiddenSize, Scale);
        InitMatrix(Wo, HiddenSize, InputSize + HiddenSize, Scale);
        ZeroArray(Bf, HiddenSize);
        ZeroArray(Bi, HiddenSize);
        ZeroArray(Bc, HiddenSize);
        ZeroArray(Bo, HiddenSize);
        ZeroMatrix(dWf, HiddenSize, InputSize + HiddenSize);
        ZeroMatrix(dWi, HiddenSize, InputSize + HiddenSize);
        ZeroMatrix(dWc, HiddenSize, InputSize + HiddenSize);
        ZeroMatrix(dWo, HiddenSize, InputSize + HiddenSize);
        ZeroArray(dBf, HiddenSize);
        ZeroArray(dBi, HiddenSize);
        ZeroArray(dBc, HiddenSize);
        ZeroArray(dBo, HiddenSize);
    }

    void Forward(const DArray& Input, const DArray& PrevH, const DArray& PrevC,
                 DArray& H, DArray& C, DArray& FG, DArray& IG, DArray& CTilde, 
                 DArray& OG, DArray& TanhC) {
        DArray X = ConcatArrays(Input, PrevH);
        H.resize(FHiddenSize);
        C.resize(FHiddenSize);
        FG.resize(FHiddenSize);
        IG.resize(FHiddenSize);
        CTilde.resize(FHiddenSize);
        OG.resize(FHiddenSize);
        TanhC.resize(FHiddenSize);

        for (int i = 0; i < FHiddenSize; i++) {
            double Sf = Bf[i], Si = Bi[i], Sc = Bc[i], So = Bo[i];
            for (int j = 0; j < (int)X.size(); j++) {
                Sf += Wf[i][j] * X[j];
                Si += Wi[i][j] * X[j];
                Sc += Wc[i][j] * X[j];
                So += Wo[i][j] * X[j];
            }
            FG[i] = ApplyActivation(Sf, TActivationType::atSigmoid);
            IG[i] = ApplyActivation(Si, TActivationType::atSigmoid);
            CTilde[i] = ApplyActivation(Sc, FActivation);
            C[i] = FG[i] * PrevC[i] + IG[i] * CTilde[i];
            TanhC[i] = std::tanh(C[i]);
            OG[i] = ApplyActivation(So, TActivationType::atSigmoid);
            H[i] = OG[i] * TanhC[i];
        }
    }

    void Backward(const DArray& dH, DArray& dPrevC, const DArray& H, const DArray& C,
                  const DArray& FG, const DArray& IG, const DArray& CTilde, const DArray& OG, 
                  const DArray& TanhC, const DArray& PrevH, const DArray& PrevC, 
                  const DArray& Input, double ClipVal, DArray& dInput, DArray& dPrevH, DArray& dPrevC_out) {
        DArray dC(FHiddenSize, 0.0);
        dInput.resize(Input.size(), 0.0);
        dPrevH.resize(PrevH.size(), 0.0);
        dPrevC_out.resize(PrevC.size(), 0.0);
        DArray X = ConcatArrays(Input, PrevH);

        for (int i = 0; i < FHiddenSize; i++) {
            double dO = dH[i] * TanhC[i] * OG[i] * (1.0 - OG[i]);
            dC[i] += dH[i] * OG[i] * (1.0 - TanhC[i] * TanhC[i]);
            dC[i] += dO;

            double dF = dC[i] * PrevC[i] * FG[i] * (1.0 - FG[i]);
            double dI = dC[i] * CTilde[i] * IG[i] * (1.0 - IG[i]);
            double dCTilde = dC[i] * IG[i] * (1.0 - CTilde[i] * CTilde[i]);

            for (int j = 0; j < (int)X.size(); j++) {
                dWf[i][j] += dF * X[j];
                dWi[i][j] += dI * X[j];
                dWc[i][j] += dCTilde * X[j];
                dWo[i][j] += dO * X[j];
            }
            dBf[i] += dF;
            dBi[i] += dI;
            dBc[i] += dCTilde;
            dBo[i] += dO;
        }

        for (int j = 0; j < (int)X.size(); j++) {
            for (int i = 0; i < FHiddenSize; i++) {
                double dx = Wf[i][j] * dC[i] + Wi[i][j] * dC[i] + Wc[i][j] * dC[i] + Wo[i][j] * dC[i];
                if (j < (int)Input.size()) dInput[j] += dx;
                else dPrevH[j - Input.size()] += dx;
            }
        }
        for (int i = 0; i < FHiddenSize; i++) {
            dPrevC_out[i] = dC[i] * FG[i];
        }
    }

    void ApplyGradients(double LR, double ClipVal) {
        for (int i = 0; i < FHiddenSize; i++) {
            for (int j = 0; j < (int)Wf[i].size(); j++) {
                Wf[i][j] -= LR * ClipValue(dWf[i][j], ClipVal);
                Wi[i][j] -= LR * ClipValue(dWi[i][j], ClipVal);
                Wc[i][j] -= LR * ClipValue(dWc[i][j], ClipVal);
                Wo[i][j] -= LR * ClipValue(dWo[i][j], ClipVal);
            }
            Bf[i] -= LR * ClipValue(dBf[i], ClipVal);
            Bi[i] -= LR * ClipValue(dBi[i], ClipVal);
            Bc[i] -= LR * ClipValue(dBc[i], ClipVal);
            Bo[i] -= LR * ClipValue(dBo[i], ClipVal);
        }
        ResetGradients();
    }

    void ResetGradients() {
        ZeroMatrix(dWf, FHiddenSize, FInputSize);
        ZeroMatrix(dWi, FHiddenSize, FInputSize);
        ZeroMatrix(dWc, FHiddenSize, FInputSize);
        ZeroMatrix(dWo, FHiddenSize, FInputSize);
        ZeroArray(dBf, FHiddenSize);
        ZeroArray(dBi, FHiddenSize);
        ZeroArray(dBc, FHiddenSize);
        ZeroArray(dBo, FHiddenSize);
    }
};

class TGRUCellWrapper {
public:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;
    DArray2D Wz, Wr, Wh;
    DArray Bz, Br, Bh;
    DArray2D dWz, dWr, dWh;
    DArray dBz, dBr, dBh;

    TGRUCellWrapper(int InputSize, int HiddenSize, TActivationType Activation)
        : FInputSize(InputSize), FHiddenSize(HiddenSize), FActivation(Activation) {
        double Scale = std::sqrt(2.0 / (InputSize + HiddenSize));
        InitMatrix(Wz, HiddenSize, InputSize + HiddenSize, Scale);
        InitMatrix(Wr, HiddenSize, InputSize + HiddenSize, Scale);
        InitMatrix(Wh, HiddenSize, InputSize + HiddenSize, Scale);
        ZeroArray(Bz, HiddenSize);
        ZeroArray(Br, HiddenSize);
        ZeroArray(Bh, HiddenSize);
        ZeroMatrix(dWz, HiddenSize, InputSize + HiddenSize);
        ZeroMatrix(dWr, HiddenSize, InputSize + HiddenSize);
        ZeroMatrix(dWh, HiddenSize, InputSize + HiddenSize);
        ZeroArray(dBz, HiddenSize);
        ZeroArray(dBr, HiddenSize);
        ZeroArray(dBh, HiddenSize);
    }

    void Forward(const DArray& Input, const DArray& PrevH, DArray& H, DArray& Z, DArray& R, DArray& HTilde) {
        DArray X = ConcatArrays(Input, PrevH);
        H.resize(FHiddenSize);
        Z.resize(FHiddenSize);
        R.resize(FHiddenSize);
        HTilde.resize(FHiddenSize);

        for (int i = 0; i < FHiddenSize; i++) {
            double Sz = Bz[i], Sr = Br[i];
            for (int j = 0; j < (int)X.size(); j++) {
                Sz += Wz[i][j] * X[j];
                Sr += Wr[i][j] * X[j];
            }
            Z[i] = ApplyActivation(Sz, TActivationType::atSigmoid);
            R[i] = ApplyActivation(Sr, TActivationType::atSigmoid);
        }

        for (int i = 0; i < FHiddenSize; i++) {
            double Sh = Bh[i];
            for (int j = 0; j < (int)Input.size(); j++) {
                Sh += Wh[i][j] * Input[j];
            }
            for (int j = 0; j < FHiddenSize; j++) {
                Sh += Wh[i][Input.size() + j] * R[i] * PrevH[j];
            }
            HTilde[i] = ApplyActivation(Sh, FActivation);
            H[i] = (1.0 - Z[i]) * HTilde[i] + Z[i] * PrevH[i];
        }
    }

    void Backward(const DArray& dH, const DArray& H, const DArray& Z, const DArray& R, 
                  const DArray& HTilde, const DArray& PrevH, const DArray& Input, 
                  double ClipVal, DArray& dInput, DArray& dPrevH) {
        dInput.resize(Input.size(), 0.0);
        dPrevH.resize(PrevH.size(), 0.0);

        for (int i = 0; i < FHiddenSize; i++) {
            double dZ = dH[i] * (PrevH[i] - HTilde[i]) * Z[i] * (1.0 - Z[i]);
            double dHTilde = dH[i] * (1.0 - Z[i]) * ActivationDerivative(HTilde[i], FActivation);

            for (int j = 0; j < (int)Input.size(); j++) {
                dWz[i][j] += dZ * Input[j];
                dWh[i][j] += dHTilde * Input[j];
            }
            dBz[i] += dZ;
            dBh[i] += dHTilde;
        }

        for (int i = 0; i < FHiddenSize; i++) {
            dPrevH[i] += dH[i] * Z[i];
        }
    }

    void ApplyGradients(double LR, double ClipVal) {
        for (int i = 0; i < FHiddenSize; i++) {
            for (int j = 0; j < (int)Wz[i].size(); j++) {
                Wz[i][j] -= LR * ClipValue(dWz[i][j], ClipVal);
                Wr[i][j] -= LR * ClipValue(dWr[i][j], ClipVal);
                Wh[i][j] -= LR * ClipValue(dWh[i][j], ClipVal);
            }
            Bz[i] -= LR * ClipValue(dBz[i], ClipVal);
            Br[i] -= LR * ClipValue(dBr[i], ClipVal);
            Bh[i] -= LR * ClipValue(dBh[i], ClipVal);
        }
        ResetGradients();
    }

    void ResetGradients() {
        ZeroMatrix(dWz, FHiddenSize, FInputSize);
        ZeroMatrix(dWr, FHiddenSize, FInputSize);
        ZeroMatrix(dWh, FHiddenSize, FInputSize);
        ZeroArray(dBz, FHiddenSize);
        ZeroArray(dBr, FHiddenSize);
        ZeroArray(dBh, FHiddenSize);
    }
};

class TOutputLayerWrapper {
public:
    int FInputSize, FOutputSize;
    TActivationType FActivation;
    DArray2D W;
    DArray B;
    DArray2D dW;
    DArray dB;
    DArray2D MW;
    DArray MB;
    DArray2D VW;
    DArray VB;

    TOutputLayerWrapper(int InputSize, int OutputSize, TActivationType Activation)
        : FInputSize(InputSize), FOutputSize(OutputSize), FActivation(Activation) {
        double Scale = std::sqrt(2.0 / InputSize);
        InitMatrix(W, OutputSize, InputSize, Scale);
        ZeroArray(B, OutputSize);
        ZeroMatrix(dW, OutputSize, InputSize);
        ZeroArray(dB, OutputSize);
        ZeroMatrix(MW, OutputSize, InputSize);
        ZeroArray(MB, OutputSize);
        ZeroMatrix(VW, OutputSize, InputSize);
        ZeroArray(VB, OutputSize);
    }

    void Forward(const DArray& Input, DArray& Output, DArray& Pre) {
        Output.resize(FOutputSize);
        Pre.resize(FOutputSize);
        for (int i = 0; i < FOutputSize; i++) {
            double Sum = B[i];
            for (int j = 0; j < FInputSize; j++)
                Sum += W[i][j] * Input[j];
            Pre[i] = Sum;
            Output[i] = ApplyActivation(Sum, FActivation);
        }
    }

    void Backward(const DArray& dOut, const DArray& Output, const DArray& Pre,
                  const DArray& Input, double ClipVal, DArray& dInput) {
        dInput.resize(FInputSize, 0.0);
        for (int i = 0; i < FOutputSize; i++) {
            double dOut_i = ClipValue(dOut[i] * ActivationDerivative(Output[i], FActivation), ClipVal);
            for (int j = 0; j < FInputSize; j++) {
                dW[i][j] += dOut_i * Input[j];
                dInput[j] += W[i][j] * dOut_i;
            }
            dB[i] += dOut_i;
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
    }

    void ResetGradients() {
        ZeroMatrix(dW, FOutputSize, FInputSize);
        ZeroArray(dB, FOutputSize);
    }
};

// ============================================================================
// Loss Functions
// ============================================================================

double ComputeLoss(const DArray& Pred, const DArray& Target, TLossType LossType) {
    double Loss = 0.0;
    if (LossType == TLossType::ltMSE) {
        for (size_t i = 0; i < Pred.size(); i++) {
            double Diff = Pred[i] - Target[i];
            Loss += Diff * Diff;
        }
        Loss /= Pred.size();
    } else if (LossType == TLossType::ltCrossEntropy) {
        for (size_t i = 0; i < Pred.size(); i++) {
            Loss -= Target[i] * std::log(std::max(Pred[i], 1e-10));
        }
    }
    return Loss;
}

void ComputeLossGradient(const DArray& Pred, const DArray& Target, TLossType LossType, DArray& Grad) {
    Grad.resize(Pred.size());
    if (LossType == TLossType::ltMSE) {
        for (size_t i = 0; i < Pred.size(); i++) {
            Grad[i] = 2.0 * (Pred[i] - Target[i]) / Pred.size();
        }
    } else if (LossType == TLossType::ltCrossEntropy) {
        for (size_t i = 0; i < Pred.size(); i++) {
            Grad[i] = -Target[i] / std::max(Pred[i], 1e-10);
        }
    }
}

// ============================================================================
// RNN Facade
// ============================================================================

class TRNNFacadeCUDA {
private:
    int FInputSize, FOutputSize;
    std::vector<int> FHiddenSizes;
    TCellType FCellType;
    TActivationType FActivation;
    TActivationType FOutputActivation;
    TLossType FLossType;
    double FLearningRate;
    double FGradientClip;
    int FBPTTSteps;
    double FDropoutRate;
    bool FUseDropout;

    std::vector<TSimpleRNNCellWrapper> FSimpleCells;
    std::vector<TLSTMCellWrapper> FLSTMCells;
    std::vector<TGRUCellWrapper> FGRUCells;
    TOutputLayerWrapper* FOutputLayer;

    TTimeStepCacheExArray FCaches;
    DArray3D FStates;
    int FSequenceLen;
    DArray2D FGradientHistory;

    std::unique_ptr<CUDABuffer> gpu_input, gpu_hidden, gpu_prev_h;
    std::unique_ptr<CUDABuffer> gpu_dhidden, gpu_doutput;

    bool use_gpu;

public:
    TRNNFacadeCUDA(int InputSize, const std::vector<int>& HiddenSizes,
                   int OutputSize, TCellType CellType,
                   TActivationType Activation, TActivationType OutputActivation,
                   TLossType LossType, double LearningRate, double GradientClip,
                   int BPTTSteps, bool useGPU = true)
        : FInputSize(InputSize), FOutputSize(OutputSize), FHiddenSizes(HiddenSizes),
          FCellType(CellType), FActivation(Activation), FOutputActivation(OutputActivation),
          FLossType(LossType), FLearningRate(LearningRate), FGradientClip(GradientClip),
          FBPTTSteps(BPTTSteps), FDropoutRate(0.0), FUseDropout(false), FSequenceLen(0),
          use_gpu(useGPU) {

        FOutputLayer = new TOutputLayerWrapper(HiddenSizes.back(), OutputSize, OutputActivation);

        if (CellType == TCellType::ctSimpleRNN) {
            for (size_t i = 0; i < HiddenSizes.size(); i++) {
                int InSize = (i == 0) ? InputSize : HiddenSizes[i-1];
                FSimpleCells.emplace_back(InSize, HiddenSizes[i], Activation);
            }
        } else if (CellType == TCellType::ctLSTM) {
            for (size_t i = 0; i < HiddenSizes.size(); i++) {
                int InSize = (i == 0) ? InputSize : HiddenSizes[i-1];
                FLSTMCells.emplace_back(InSize, HiddenSizes[i], Activation);
            }
        } else if (CellType == TCellType::ctGRU) {
            for (size_t i = 0; i < HiddenSizes.size(); i++) {
                int InSize = (i == 0) ? InputSize : HiddenSizes[i-1];
                FGRUCells.emplace_back(InSize, HiddenSizes[i], Activation);
            }
        }

        FStates = InitHiddenStates();

        if (useGPU) {
            int device_count = 0;
            cudaGetDeviceCount(&device_count);
            if (device_count > 0) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);
                std::cout << "CUDA Device: " << prop.name << std::endl;
                std::cout << "GPU Global Memory: " << (prop.totalGlobalMem / (1024UL*1024UL*1024UL)) << " GB" << std::endl;
                std::cout << "CUDA acceleration enabled" << std::endl;
            } else {
                use_gpu = false;
                std::cout << "No CUDA devices found, using CPU" << std::endl;
            }
        }
    }

    ~TRNNFacadeCUDA() {
        delete FOutputLayer;
    }

    DArray3D InitHiddenStates() {
        DArray3D States(FHiddenSizes.size());
        for (size_t i = 0; i < FHiddenSizes.size(); i++) {
            States[i].resize(2);
            States[i][0].resize(FHiddenSizes[i], 0.0);
            if (FCellType == TCellType::ctLSTM)
                States[i][1].resize(FHiddenSizes[i], 0.0);
        }
        return States;
    }

    DArray2D ForwardSequence(const DArray2D& Inputs) {
        FSequenceLen = Inputs.size();
        FCaches.resize(FSequenceLen);
        FGradientHistory.resize(FSequenceLen);

        DArray2D Outputs(FSequenceLen);

        for (int t = 0; t < FSequenceLen; t++) {
            FCaches[t].Input = Inputs[t];

            DArray LayerInput = Inputs[t];
            std::vector<DArray> LayerOutputs(FHiddenSizes.size());

            for (size_t layer = 0; layer < FHiddenSizes.size(); layer++) {
                DArray H, C, PreH;
                
                if (FCellType == TCellType::ctSimpleRNN) {
                    FSimpleCells[layer].Forward(LayerInput, FStates[layer][0], H, PreH);
                    FStates[layer][0] = H;
                    LayerInput = H;
                    if (layer == 0 || t == 0) {
                        FCaches[t].H = H;
                        FCaches[t].PreH = PreH;
                    }
                } else if (FCellType == TCellType::ctLSTM) {
                    DArray FG, IG, CTilde, OG, TanhC;
                    FLSTMCells[layer].Forward(LayerInput, FStates[layer][0], FStates[layer][1],
                                              H, C, FG, IG, CTilde, OG, TanhC);
                    FStates[layer][0] = H;
                    FStates[layer][1] = C;
                    LayerInput = H;
                    if (layer == 0 || t == 0) {
                        FCaches[t].H = H;
                        FCaches[t].C = C;
                        FCaches[t].F = FG;
                        FCaches[t].I = IG;
                        FCaches[t].CTilde = CTilde;
                        FCaches[t].O = OG;
                        FCaches[t].TanhC = TanhC;
                    }
                } else if (FCellType == TCellType::ctGRU) {
                    DArray Z, R, HTilde;
                    FGRUCells[layer].Forward(LayerInput, FStates[layer][0], H, Z, R, HTilde);
                    FStates[layer][0] = H;
                    LayerInput = H;
                    if (layer == 0 || t == 0) {
                        FCaches[t].H = H;
                        FCaches[t].Z = Z;
                        FCaches[t].R = R;
                        FCaches[t].HTilde = HTilde;
                    }
                }
                LayerOutputs[layer] = LayerInput;
            }

            DArray OutPre, OutVal;
            FOutputLayer->Forward(LayerInput, OutVal, OutPre);
            FCaches[t].OutVal = OutVal;
            FCaches[t].OutPre = OutPre;
            Outputs[t] = OutVal;
        }

        if (use_gpu) {
            cudaDeviceSynchronize();
        }

        return Outputs;
    }

    double BackwardSequence(const DArray2D& Targets) {
        double Loss = 0.0;

        for (int t = 0; t < FSequenceLen; t++) {
            Loss += ComputeLoss(FCaches[t].OutVal, Targets[t], FLossType);
        }

        for (int t = FSequenceLen - 1; t >= 0; t--) {
            DArray dOut(FOutputSize);
            ComputeLossGradient(FCaches[t].OutVal, Targets[t], FLossType, dOut);

            DArray dLayerInput;
            FOutputLayer->Backward(dOut, FCaches[t].OutVal, FCaches[t].OutPre,
                                   FCaches[t].H, FGradientClip, dLayerInput);

            for (int layer = (int)FHiddenSizes.size() - 1; layer >= 0; layer--) {
                DArray dInput, dPrevH, dPrevC;
                if (FCellType == TCellType::ctSimpleRNN) {
                    FSimpleCells[layer].Backward(dLayerInput, FCaches[t].H, FCaches[t].PreH,
                                                 FStates[layer][0], FCaches[t].Input,
                                                 FGradientClip, dInput, dPrevH);
                } else if (FCellType == TCellType::ctLSTM) {
                    FLSTMCells[layer].Backward(dLayerInput, dPrevC, FCaches[t].H, FCaches[t].C,
                                               FCaches[t].F, FCaches[t].I, FCaches[t].CTilde,
                                               FCaches[t].O, FCaches[t].TanhC,
                                               FStates[layer][0], FStates[layer][1],
                                               FCaches[t].Input, FGradientClip,
                                               dInput, dPrevH, dPrevC);
                } else if (FCellType == TCellType::ctGRU) {
                    FGRUCells[layer].Backward(dLayerInput, FCaches[t].H, FCaches[t].Z,
                                              FCaches[t].R, FCaches[t].HTilde,
                                              FStates[layer][0], FCaches[t].Input,
                                              FGradientClip, dInput, dPrevH);
                }
                dLayerInput = dInput;
            }
        }

        if (use_gpu) {
            cudaDeviceSynchronize();
        }

        return Loss / FSequenceLen;
    }

    double TrainSequence(const DArray2D& Inputs, const DArray2D& Targets) {
        ResetAllStates();
        DArray2D Outputs = ForwardSequence(Inputs);
        double Loss = BackwardSequence(Targets);
        ApplyGradients();
        return Loss;
    }

    DArray2D Predict(const DArray2D& Inputs) {
        ResetAllStates();
        return ForwardSequence(Inputs);
    }

    void ResetAllStates(double Value = 0.0) {
        for (size_t layer = 0; layer < FHiddenSizes.size(); layer++) {
            for (int i = 0; i < FHiddenSizes[layer]; i++) {
                FStates[layer][0][i] = Value;
                if (FCellType == TCellType::ctLSTM)
                    FStates[layer][1][i] = Value;
            }
        }
    }

    void ResetGradients() {
        for (auto& cell : FSimpleCells) cell.ResetGradients();
        for (auto& cell : FLSTMCells) cell.ResetGradients();
        for (auto& cell : FGRUCells) cell.ResetGradients();
        if (FOutputLayer) FOutputLayer->ResetGradients();
    }

    void ApplyGradients() {
        for (auto& cell : FSimpleCells) cell.ApplyGradients(FLearningRate, FGradientClip);
        for (auto& cell : FLSTMCells) cell.ApplyGradients(FLearningRate, FGradientClip);
        for (auto& cell : FGRUCells) cell.ApplyGradients(FLearningRate, FGradientClip);
        if (FOutputLayer) FOutputLayer->ApplyGradients(FLearningRate, FGradientClip);
    }

    // ========================================================================
    // Facade Introspection
    // ========================================================================

    double GetHiddenValue(int LayerIdx, int Timestep, int NeuronIdx) {
        if (LayerIdx < 0 || LayerIdx >= (int)FHiddenSizes.size() || 
            Timestep < 0 || Timestep >= FSequenceLen ||
            NeuronIdx < 0 || NeuronIdx >= FHiddenSizes[LayerIdx]) {
            return 0.0;
        }
        return FStates[LayerIdx][0][NeuronIdx];
    }

    void SetHiddenValue(int LayerIdx, int NeuronIdx, double Value) {
        if (LayerIdx >= 0 && LayerIdx < (int)FHiddenSizes.size() &&
            NeuronIdx >= 0 && NeuronIdx < FHiddenSizes[LayerIdx]) {
            FStates[LayerIdx][0][NeuronIdx] = Value;
        }
    }

    double GetOutputValue(int Timestep, int OutputIdx) {
        if (Timestep < 0 || Timestep >= FSequenceLen || OutputIdx < 0 || OutputIdx >= FOutputSize) {
            return 0.0;
        }
        return FCaches[Timestep].OutVal[OutputIdx];
    }

    double GetCellState(int LayerIdx, int NeuronIdx) {
        if (FCellType != TCellType::ctLSTM || LayerIdx < 0 || LayerIdx >= (int)FHiddenSizes.size() ||
            NeuronIdx < 0 || NeuronIdx >= FHiddenSizes[LayerIdx]) {
            return 0.0;
        }
        return FStates[LayerIdx][1][NeuronIdx];
    }

    double GetGateValue(int LayerIdx, int Timestep, int NeuronIdx, TGateType Gate) {
        if (Timestep < 0 || Timestep >= FSequenceLen || LayerIdx < 0 || LayerIdx >= (int)FHiddenSizes.size() ||
            NeuronIdx < 0 || NeuronIdx >= FHiddenSizes[LayerIdx]) {
            return 0.0;
        }

        if (FCellType == TCellType::ctLSTM && Timestep < (int)FCaches.size()) {
            switch (Gate) {
                case TGateType::gtForget: return FCaches[Timestep].F[NeuronIdx];
                case TGateType::gtInput: return FCaches[Timestep].I[NeuronIdx];
                case TGateType::gtOutput: return FCaches[Timestep].O[NeuronIdx];
                case TGateType::gtCellCandidate: return FCaches[Timestep].CTilde[NeuronIdx];
                default: return 0.0;
            }
        } else if (FCellType == TCellType::ctGRU && Timestep < (int)FCaches.size()) {
            switch (Gate) {
                case TGateType::gtUpdate: return FCaches[Timestep].Z[NeuronIdx];
                case TGateType::gtReset: return FCaches[Timestep].R[NeuronIdx];
                case TGateType::gtHiddenCandidate: return FCaches[Timestep].HTilde[NeuronIdx];
                default: return 0.0;
            }
        }
        return 0.0;
    }

    double GetPreactivation(int LayerIdx, int Timestep, int NeuronIdx) {
        if (Timestep < 0 || Timestep >= FSequenceLen || LayerIdx < 0 || LayerIdx >= (int)FHiddenSizes.size() ||
            NeuronIdx < 0 || NeuronIdx >= FHiddenSizes[LayerIdx]) {
            return 0.0;
        }
        return FCaches[Timestep].PreH[NeuronIdx];
    }

    double GetInputValue(int Timestep, int InputIdx) {
        if (Timestep < 0 || Timestep >= FSequenceLen || InputIdx < 0 || InputIdx >= FInputSize) {
            return 0.0;
        }
        return FCaches[Timestep].Input[InputIdx];
    }

    void DetectVanishingGradients(double Threshold, int& Count, double& MinAbsGrad) {
        Count = 0;
        MinAbsGrad = 1e10;
        
        for (auto& cell : FSimpleCells) {
            for (auto& grad_row : cell.dWih) {
                for (double g : grad_row) {
                    double abs_g = std::abs(g);
                    if (abs_g < Threshold) Count++;
                    MinAbsGrad = std::min(MinAbsGrad, abs_g);
                }
            }
        }
        for (auto& cell : FLSTMCells) {
            for (auto& grad_row : cell.dWf) {
                for (double g : grad_row) {
                    double abs_g = std::abs(g);
                    if (abs_g < Threshold) Count++;
                    MinAbsGrad = std::min(MinAbsGrad, abs_g);
                }
            }
        }
    }

    void DetectExplodingGradients(double Threshold, int& Count, double& MaxAbsGrad) {
        Count = 0;
        MaxAbsGrad = 0.0;
        
        for (auto& cell : FSimpleCells) {
            for (auto& grad_row : cell.dWih) {
                for (double g : grad_row) {
                    double abs_g = std::abs(g);
                    if (abs_g > Threshold) Count++;
                    MaxAbsGrad = std::max(MaxAbsGrad, abs_g);
                }
            }
        }
        for (auto& cell : FLSTMCells) {
            for (auto& grad_row : cell.dWf) {
                for (double g : grad_row) {
                    double abs_g = std::abs(g);
                    if (abs_g > Threshold) Count++;
                    MaxAbsGrad = std::max(MaxAbsGrad, abs_g);
                }
            }
        }
    }

    DArray2D GetSequenceOutputs() {
        DArray2D Outputs;
        for (int t = 0; t < FSequenceLen; t++) {
            Outputs.push_back(FCaches[t].OutVal);
        }
        return Outputs;
    }

    DArray2D GetSequenceHiddenStates(int LayerIdx) {
        DArray2D States;
        if (LayerIdx >= 0 && LayerIdx < (int)FHiddenSizes.size()) {
            for (int t = 0; t < FSequenceLen; t++) {
                if (t < (int)FCaches.size()) {
                    States.push_back(FCaches[t].H);
                }
            }
        }
        return States;
    }

    // ========================================================================
    // Model Persistence
    // ========================================================================

    bool SaveModel(const std::string& Filename) {
        try {
            std::ofstream file(Filename, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file for writing: " << Filename << std::endl;
                return false;
            }

            // Write header
            int cellTypeInt = (int)FCellType;
            int actTypeInt = (int)FActivation;
            int outActTypeInt = (int)FOutputActivation;
            int lossTypeInt = (int)FLossType;
            int numLayers = FHiddenSizes.size();

            file.write((char*)&FInputSize, sizeof(int));
            file.write((char*)&FOutputSize, sizeof(int));
            file.write((char*)&numLayers, sizeof(int));
            file.write((char*)&cellTypeInt, sizeof(int));
            file.write((char*)&actTypeInt, sizeof(int));
            file.write((char*)&outActTypeInt, sizeof(int));
            file.write((char*)&lossTypeInt, sizeof(int));
            file.write((char*)&FLearningRate, sizeof(double));
            file.write((char*)&FGradientClip, sizeof(double));

            // Write hidden sizes
            for (int h : FHiddenSizes) {
                file.write((char*)&h, sizeof(int));
            }

            // Write SimpleRNN cells
            for (auto& cell : FSimpleCells) {
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < cell.FInputSize; j++) {
                        file.write((char*)&cell.Wih[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < cell.FHiddenSize; j++) {
                        file.write((char*)&cell.Whh[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    file.write((char*)&cell.Bh[i], sizeof(double));
                }
            }

            // Write LSTM cells
            for (auto& cell : FLSTMCells) {
                int inputPlusHidden = cell.FInputSize + cell.FHiddenSize;
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.write((char*)&cell.Wf[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.write((char*)&cell.Wi[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.write((char*)&cell.Wc[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.write((char*)&cell.Wo[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    file.write((char*)&cell.Bf[i], sizeof(double));
                    file.write((char*)&cell.Bi[i], sizeof(double));
                    file.write((char*)&cell.Bc[i], sizeof(double));
                    file.write((char*)&cell.Bo[i], sizeof(double));
                }
            }

            // Write GRU cells
            for (auto& cell : FGRUCells) {
                int inputPlusHidden = cell.FInputSize + cell.FHiddenSize;
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.write((char*)&cell.Wz[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.write((char*)&cell.Wr[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.write((char*)&cell.Wh[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    file.write((char*)&cell.Bz[i], sizeof(double));
                    file.write((char*)&cell.Br[i], sizeof(double));
                    file.write((char*)&cell.Bh[i], sizeof(double));
                }
            }

            // Write output layer
            for (int i = 0; i < FOutputLayer->FOutputSize; i++) {
                for (int j = 0; j < FOutputLayer->FInputSize; j++) {
                    file.write((char*)&FOutputLayer->W[i][j], sizeof(double));
                }
            }
            for (int i = 0; i < FOutputLayer->FOutputSize; i++) {
                file.write((char*)&FOutputLayer->B[i], sizeof(double));
            }

            file.close();
            return true;
        } catch (...) {
            std::cerr << "Error saving model" << std::endl;
            return false;
        }
    }

    bool LoadModel(const std::string& Filename) {
        try {
            std::ifstream file(Filename, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file for reading: " << Filename << std::endl;
                return false;
            }

            // Read header
            int cellTypeInt, actTypeInt, outActTypeInt, lossTypeInt, numLayers;
            file.read((char*)&FInputSize, sizeof(int));
            file.read((char*)&FOutputSize, sizeof(int));
            file.read((char*)&numLayers, sizeof(int));
            file.read((char*)&cellTypeInt, sizeof(int));
            file.read((char*)&actTypeInt, sizeof(int));
            file.read((char*)&outActTypeInt, sizeof(int));
            file.read((char*)&lossTypeInt, sizeof(int));
            file.read((char*)&FLearningRate, sizeof(double));
            file.read((char*)&FGradientClip, sizeof(double));

            FCellType = (TCellType)cellTypeInt;
            FActivation = (TActivationType)actTypeInt;
            FOutputActivation = (TActivationType)outActTypeInt;
            FLossType = (TLossType)lossTypeInt;

            // Read hidden sizes
            FHiddenSizes.clear();
            for (int i = 0; i < numLayers; i++) {
                int h;
                file.read((char*)&h, sizeof(int));
                FHiddenSizes.push_back(h);
            }

            // Reinitialize cells based on loaded config
            FSimpleCells.clear();
            FLSTMCells.clear();
            FGRUCells.clear();

            if (FCellType == TCellType::ctSimpleRNN) {
                for (size_t i = 0; i < FHiddenSizes.size(); i++) {
                    int InSize = (i == 0) ? FInputSize : FHiddenSizes[i-1];
                    FSimpleCells.emplace_back(InSize, FHiddenSizes[i], FActivation);
                }
            } else if (FCellType == TCellType::ctLSTM) {
                for (size_t i = 0; i < FHiddenSizes.size(); i++) {
                    int InSize = (i == 0) ? FInputSize : FHiddenSizes[i-1];
                    FLSTMCells.emplace_back(InSize, FHiddenSizes[i], FActivation);
                }
            } else if (FCellType == TCellType::ctGRU) {
                for (size_t i = 0; i < FHiddenSizes.size(); i++) {
                    int InSize = (i == 0) ? FInputSize : FHiddenSizes[i-1];
                    FGRUCells.emplace_back(InSize, FHiddenSizes[i], FActivation);
                }
            }

            if (FOutputLayer) delete FOutputLayer;
            FOutputLayer = new TOutputLayerWrapper(FHiddenSizes.back(), FOutputSize, FOutputActivation);

            // Read SimpleRNN cells
            for (auto& cell : FSimpleCells) {
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < cell.FInputSize; j++) {
                        file.read((char*)&cell.Wih[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < cell.FHiddenSize; j++) {
                        file.read((char*)&cell.Whh[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    file.read((char*)&cell.Bh[i], sizeof(double));
                }
            }

            // Read LSTM cells
            for (auto& cell : FLSTMCells) {
                int inputPlusHidden = cell.FInputSize + cell.FHiddenSize;
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.read((char*)&cell.Wf[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.read((char*)&cell.Wi[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.read((char*)&cell.Wc[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.read((char*)&cell.Wo[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    file.read((char*)&cell.Bf[i], sizeof(double));
                    file.read((char*)&cell.Bi[i], sizeof(double));
                    file.read((char*)&cell.Bc[i], sizeof(double));
                    file.read((char*)&cell.Bo[i], sizeof(double));
                }
            }

            // Read GRU cells
            for (auto& cell : FGRUCells) {
                int inputPlusHidden = cell.FInputSize + cell.FHiddenSize;
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.read((char*)&cell.Wz[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.read((char*)&cell.Wr[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    for (int j = 0; j < inputPlusHidden; j++) {
                        file.read((char*)&cell.Wh[i][j], sizeof(double));
                    }
                }
                for (int i = 0; i < cell.FHiddenSize; i++) {
                    file.read((char*)&cell.Bz[i], sizeof(double));
                    file.read((char*)&cell.Br[i], sizeof(double));
                    file.read((char*)&cell.Bh[i], sizeof(double));
                }
            }

            // Read output layer
            for (int i = 0; i < FOutputLayer->FOutputSize; i++) {
                for (int j = 0; j < FOutputLayer->FInputSize; j++) {
                    file.read((char*)&FOutputLayer->W[i][j], sizeof(double));
                }
            }
            for (int i = 0; i < FOutputLayer->FOutputSize; i++) {
                file.read((char*)&FOutputLayer->B[i], sizeof(double));
            }

            file.close();
            FStates = InitHiddenStates();
            return true;
        } catch (...) {
            std::cerr << "Error loading model" << std::endl;
            return false;
        }
    }

    int GetLayerCount() const { return FHiddenSizes.size(); }
    int GetHiddenSize(int LayerIdx) const {
        if (LayerIdx >= 0 && LayerIdx < (int)FHiddenSizes.size())
            return FHiddenSizes[LayerIdx];
        return 0;
    }
    TCellType GetCellType() const { return FCellType; }
    int GetSequenceLength() const { return FSequenceLen; }
    bool isGPUAvailable() const { return use_gpu; }
    void setUseGPU(bool use) { use_gpu = use; }
    double getLearningRate() const { return FLearningRate; }
    void setLearningRate(double value) { FLearningRate = value; }
    double getGradientClip() const { return FGradientClip; }
    void setGradientClip(double value) { FGradientClip = value; }
    double getDropoutRate() const { return FDropoutRate; }
    void setDropoutRate(double value) { FDropoutRate = value; FUseDropout = value > 0.0; }
};

// ============================================================================
// CLI Entry Point
// ============================================================================

void ShowHelp(const char* progName) {
    std::cout << "RNN Facade CLI (CUDA GPU) - Matthew Abbott 2025" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << progName << " <command> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  create              Create and initialize an RNN model" << std::endl;
    std::cout << "  train               Train the model on data" << std::endl;
    std::cout << "  predict             Run prediction on input data" << std::endl;
    std::cout << "  save                Save model weights to file" << std::endl;
    std::cout << "  load                Load model weights from file" << std::endl;
    std::cout << "  info                Display GPU information" << std::endl;
    std::cout << std::endl;
    std::cout << "Facade Introspection Commands:" << std::endl;
    std::cout << "  get-hidden          Get hidden state value" << std::endl;
    std::cout << "  set-hidden          Set hidden state value" << std::endl;
    std::cout << "  get-output          Get output value at timestep" << std::endl;
    std::cout << "  get-cell-state      Get LSTM cell state" << std::endl;
    std::cout << "  get-gate            Get gate value (LSTM/GRU)" << std::endl;
    std::cout << "  get-preactivation   Get pre-activation value" << std::endl;
    std::cout << "  get-input           Get input vector value" << std::endl;
    std::cout << "  reset-states        Reset all hidden/cell states" << std::endl;
    std::cout << "  set-dropout         Set dropout rate" << std::endl;
    std::cout << "  get-dropout         Get current dropout rate" << std::endl;
    std::cout << "  detect-vanishing    Check for vanishing gradients" << std::endl;
    std::cout << "  detect-exploding    Check for exploding gradients" << std::endl;
    std::cout << "  get-seq-outputs     Get all outputs for a sequence" << std::endl;
    std::cout << "  get-seq-hidden      Get hidden states over sequence" << std::endl;
    std::cout << std::endl;
    std::cout << "Create/Train/Predict options:" << std::endl;
    std::cout << "  --input-size <n>       Input dimension (required)" << std::endl;
    std::cout << "  --hidden-sizes <n,n>   Comma-separated hidden layer sizes (required)" << std::endl;
    std::cout << "  --output-size <n>      Output dimension (required)" << std::endl;
    std::cout << "  --cell-type <type>     rnn, lstm, or gru (default: lstm)" << std::endl;
    std::cout << "  --activation <type>    sigmoid, tanh, relu, linear (default: tanh)" << std::endl;
    std::cout << "  --output-activation    Output layer activation (default: sigmoid)" << std::endl;
    std::cout << "  --loss <type>          mse or crossentropy (default: mse)" << std::endl;
    std::cout << "  --learning-rate <f>    Learning rate (default: 0.01)" << std::endl;
    std::cout << "  --gradient-clip <f>    Gradient clipping value (default: 5.0)" << std::endl;
    std::cout << "  --bptt-steps <n>       BPTT truncation steps (default: 0 = full)" << std::endl;
    std::cout << "  --epochs <n>           Number of training epochs (default: 100)" << std::endl;
    std::cout << "  --input-file <file>    CSV file with input sequences" << std::endl;
    std::cout << "  --target-file <file>   CSV file with target sequences" << std::endl;
    std::cout << "  --output-file <file>   CSV file to write predictions" << std::endl;
    std::cout << std::endl;
    std::cout << "Facade options:" << std::endl;
    std::cout << "  --layer <n>            Layer index (default: 0)" << std::endl;
    std::cout << "  --timestep <n>         Timestep index (default: 0)" << std::endl;
    std::cout << "  --neuron <n>           Neuron index (default: 0)" << std::endl;
    std::cout << "  --output-idx <n>       Output index (default: 0)" << std::endl;
    std::cout << "  --value <f>            Value to set" << std::endl;
    std::cout << "  --gate <type>          Gate type: forget,input,output,cell,update,reset,hidden" << std::endl;
    std::cout << "  --threshold <f>        Threshold for gradient detection (default: 1e-6)" << std::endl;
    std::cout << "  --model-file <file>    Model file path for save/load" << std::endl;
}

TCellType ParseCellType(const std::string& str) {
    if (str == "rnn") return TCellType::ctSimpleRNN;
    if (str == "lstm") return TCellType::ctLSTM;
    if (str == "gru") return TCellType::ctGRU;
    return TCellType::ctLSTM;
}

TActivationType ParseActivation(const std::string& str) {
    if (str == "sigmoid") return TActivationType::atSigmoid;
    if (str == "tanh") return TActivationType::atTanh;
    if (str == "relu") return TActivationType::atReLU;
    if (str == "linear") return TActivationType::atLinear;
    return TActivationType::atTanh;
}

TLossType ParseLossType(const std::string& str) {
    if (str == "mse") return TLossType::ltMSE;
    if (str == "crossentropy") return TLossType::ltCrossEntropy;
    return TLossType::ltMSE;
}

TGateType ParseGateType(const std::string& str) {
    if (str == "forget") return TGateType::gtForget;
    if (str == "input") return TGateType::gtInput;
    if (str == "output") return TGateType::gtOutput;
    if (str == "cell") return TGateType::gtCellCandidate;
    if (str == "update") return TGateType::gtUpdate;
    if (str == "reset") return TGateType::gtReset;
    if (str == "hidden") return TGateType::gtHiddenCandidate;
    return TGateType::gtForget;
}

std::vector<int> ParseHiddenSizes(const std::string& str) {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stoi(item));
    }
    return result.empty() ? std::vector<int>{64, 32} : result;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        ShowHelp(argv[0]);
        return 0;
    }

    std::string command = argv[1];

    // Default parameters
    int InputSize = 10;
    std::vector<int> HiddenSizes = {64, 32};
    int OutputSize = 5;
    TCellType CellType = TCellType::ctLSTM;
    TActivationType Activation = TActivationType::atTanh;
    TActivationType OutputActivation = TActivationType::atSigmoid;
    TLossType LossType = TLossType::ltMSE;
    double LearningRate = 0.01;
    double GradientClip = 5.0;
    int BPTTSteps = 0;
    int Epochs = 100;

    // Introspection parameters
    int LayerIdx = 0;
    int Timestep = 0;
    int NeuronIdx = 0;
    int OutputIdx = 0;
    double Value = 0.0;
    TGateType Gate = TGateType::gtForget;
    double Threshold = 1e-6;

    std::string InputFile, TargetFile, OutputFile, ModelFile;

    // Parse command-specific arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if ((arg == "--help" || arg == "-h") && command == "create") {
            ShowHelp(argv[0]);
            return 0;
        }
        else if (arg == "--input-size" && i + 1 < argc) {
            InputSize = std::stoi(argv[++i]);
        }
        else if (arg == "--hidden-sizes" && i + 1 < argc) {
            HiddenSizes = ParseHiddenSizes(argv[++i]);
        }
        else if (arg == "--output-size" && i + 1 < argc) {
            OutputSize = std::stoi(argv[++i]);
        }
        else if (arg == "--cell-type" && i + 1 < argc) {
            CellType = ParseCellType(argv[++i]);
        }
        else if (arg == "--activation" && i + 1 < argc) {
            Activation = ParseActivation(argv[++i]);
        }
        else if (arg == "--output-activation" && i + 1 < argc) {
            OutputActivation = ParseActivation(argv[++i]);
        }
        else if (arg == "--loss" && i + 1 < argc) {
            LossType = ParseLossType(argv[++i]);
        }
        else if (arg == "--learning-rate" && i + 1 < argc) {
            LearningRate = std::stod(argv[++i]);
        }
        else if (arg == "--gradient-clip" && i + 1 < argc) {
            GradientClip = std::stod(argv[++i]);
        }
        else if (arg == "--bptt-steps" && i + 1 < argc) {
            BPTTSteps = std::stoi(argv[++i]);
        }
        else if (arg == "--epochs" && i + 1 < argc) {
            Epochs = std::stoi(argv[++i]);
        }
        else if (arg == "--input-file" && i + 1 < argc) {
            InputFile = argv[++i];
        }
        else if (arg == "--target-file" && i + 1 < argc) {
            TargetFile = argv[++i];
        }
        else if (arg == "--output-file" && i + 1 < argc) {
            OutputFile = argv[++i];
        }
        else if (arg == "--model-file" && i + 1 < argc) {
            ModelFile = argv[++i];
        }
        else if (arg == "--layer" && i + 1 < argc) {
            LayerIdx = std::stoi(argv[++i]);
        }
        else if (arg == "--timestep" && i + 1 < argc) {
            Timestep = std::stoi(argv[++i]);
        }
        else if (arg == "--neuron" && i + 1 < argc) {
            NeuronIdx = std::stoi(argv[++i]);
        }
        else if (arg == "--output-idx" && i + 1 < argc) {
            OutputIdx = std::stoi(argv[++i]);
        }
        else if (arg == "--value" && i + 1 < argc) {
            Value = std::stod(argv[++i]);
        }
        else if (arg == "--gate" && i + 1 < argc) {
            Gate = ParseGateType(argv[++i]);
        }
        else if (arg == "--threshold" && i + 1 < argc) {
            Threshold = std::stod(argv[++i]);
        }
    }

    // Create RNN instance
    TRNNFacadeCUDA rnn(InputSize, HiddenSizes, OutputSize, CellType,
                       Activation, OutputActivation, LossType, LearningRate,
                       GradientClip, BPTTSteps, true);

    // Execute command
    if (command == "create") {
        std::cout << "RNN Facade CLI (CUDA GPU) - Matthew Abbott 2025" << std::endl;
        std::cout << "Model created:" << std::endl;
        std::cout << "  Input Size: " << InputSize << std::endl;
        std::cout << "  Hidden Sizes: ";
        for (int h : HiddenSizes) std::cout << h << " ";
        std::cout << std::endl;
        std::cout << "  Output Size: " << OutputSize << std::endl;
        std::cout << "  Cell Type: " << (int)CellType << std::endl;
        std::cout << "  GPU Available: " << (rnn.isGPUAvailable() ? "Yes" : "No") << std::endl;
    }
    else if (command == "info") {
        std::cout << "GPU Information:" << std::endl;
        std::cout << "  Available: " << (rnn.isGPUAvailable() ? "Yes" : "No") << std::endl;
        std::cout << "  Layers: " << rnn.GetLayerCount() << std::endl;
        std::cout << "  Cell Type: " << (int)rnn.GetCellType() << std::endl;
        std::cout << "  Learning Rate: " << rnn.getLearningRate() << std::endl;
    }
    else if (command == "get-hidden") {
        double val = rnn.GetHiddenValue(LayerIdx, Timestep, NeuronIdx);
        std::cout << "Hidden[" << LayerIdx << "][" << Timestep << "][" << NeuronIdx << "] = " << val << std::endl;
    }
    else if (command == "set-hidden") {
        rnn.SetHiddenValue(LayerIdx, NeuronIdx, Value);
        std::cout << "Set hidden[" << LayerIdx << "][" << NeuronIdx << "] = " << Value << std::endl;
    }
    else if (command == "get-output") {
        double val = rnn.GetOutputValue(Timestep, OutputIdx);
        std::cout << "Output[" << Timestep << "][" << OutputIdx << "] = " << val << std::endl;
    }
    else if (command == "get-cell-state") {
        double val = rnn.GetCellState(LayerIdx, NeuronIdx);
        std::cout << "CellState[" << LayerIdx << "][" << NeuronIdx << "] = " << val << std::endl;
    }
    else if (command == "get-gate") {
        double val = rnn.GetGateValue(LayerIdx, Timestep, NeuronIdx, Gate);
        std::cout << "Gate[" << LayerIdx << "][" << Timestep << "][" << NeuronIdx << "] = " << val << std::endl;
    }
    else if (command == "get-preactivation") {
        double val = rnn.GetPreactivation(LayerIdx, Timestep, NeuronIdx);
        std::cout << "Preactivation[" << LayerIdx << "][" << Timestep << "][" << NeuronIdx << "] = " << val << std::endl;
    }
    else if (command == "get-input") {
        double val = rnn.GetInputValue(Timestep, NeuronIdx);
        std::cout << "Input[" << Timestep << "][" << NeuronIdx << "] = " << val << std::endl;
    }
    else if (command == "reset-states") {
        rnn.ResetAllStates(Value);
        std::cout << "All states reset to " << Value << std::endl;
    }
    else if (command == "set-dropout") {
        rnn.setDropoutRate(Value);
        std::cout << "Dropout rate set to " << Value << std::endl;
    }
    else if (command == "get-dropout") {
        std::cout << "Current dropout rate: " << rnn.getDropoutRate() << std::endl;
    }
    else if (command == "detect-vanishing") {
        int count;
        double minGrad;
        rnn.DetectVanishingGradients(Threshold, count, minGrad);
        std::cout << "Vanishing gradients detected: " << count << " values below " << Threshold << std::endl;
        std::cout << "Minimum absolute gradient: " << minGrad << std::endl;
    }
    else if (command == "detect-exploding") {
        int count;
        double maxGrad;
        rnn.DetectExplodingGradients(Threshold, count, maxGrad);
        std::cout << "Exploding gradients detected: " << count << " values above " << Threshold << std::endl;
        std::cout << "Maximum absolute gradient: " << maxGrad << std::endl;
    }
    else if (command == "get-seq-outputs") {
        DArray2D outputs = rnn.GetSequenceOutputs();
        std::cout << "Sequence outputs (" << outputs.size() << " timesteps, " 
                  << (outputs.empty() ? 0 : outputs[0].size()) << " dimensions each):" << std::endl;
        for (size_t t = 0; t < outputs.size(); t++) {
            std::cout << "  T" << t << ": ";
            for (double v : outputs[t]) std::cout << v << " ";
            std::cout << std::endl;
        }
    }
    else if (command == "get-seq-hidden") {
        DArray2D states = rnn.GetSequenceHiddenStates(LayerIdx);
        std::cout << "Hidden states for layer " << LayerIdx << " (" << states.size() 
                  << " timesteps, " << (states.empty() ? 0 : states[0].size()) << " hidden units):" << std::endl;
        for (size_t t = 0; t < states.size(); t++) {
            std::cout << "  T" << t << ": ";
            for (double v : states[t]) std::cout << v << " ";
            std::cout << std::endl;
        }
    }
    else if (command == "save") {
        if (ModelFile.empty()) {
            std::cerr << "Error: --model-file required for save command" << std::endl;
            return 1;
        }
        if (rnn.SaveModel(ModelFile)) {
            std::cout << "Model saved to " << ModelFile << std::endl;
        } else {
            std::cerr << "Failed to save model" << std::endl;
            return 1;
        }
    }
    else if (command == "load") {
        if (ModelFile.empty()) {
            std::cerr << "Error: --model-file required for load command" << std::endl;
            return 1;
        }
        if (rnn.LoadModel(ModelFile)) {
            std::cout << "Model loaded from " << ModelFile << std::endl;
            std::cout << "Model architecture:" << std::endl;
            std::cout << "  Input Size: " << InputSize << std::endl;
            std::cout << "  Output Size: " << rnn.GetLayerCount() << " layers" << std::endl;
        } else {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }
    }
    else {
        std::cerr << "Unknown command: " << command << std::endl;
        ShowHelp(argv[0]);
        return 1;
    }

    return 0;
}
