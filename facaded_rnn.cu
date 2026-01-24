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
#include <memory>
#include <cuda_runtime.h>

// ========== Type Aliases ==========
using DArray = std::vector<double>;
using TDArray2D = std::vector<DArray>;
using TDArray3D = std::vector<TDArray2D>;
using TIntArray = std::vector<int>;

// ========== Enums ==========
enum TActivationType { atSigmoid, atTanh, atReLU, atLinear };
enum TLossType { ltMSE, ltCrossEntropy };
enum TCellType { ctSimpleRNN, ctLSTM, ctGRU };
enum TGateType { gtForget, gtInput, gtOutput, gtCellCandidate, gtUpdate, gtReset, gtHiddenCandidate };
enum TCommand { cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdInfo, cmdQuery, cmdHelp };

// ========== Data Structures ==========
struct THistogramBin {
    double RangeMin, RangeMax;
    int Count;
    double Percentage;
};

struct TGateSaturationStats {
    int NearZeroCount;
    int NearOneCount;
    int TotalCount;
    double NearZeroPct;
    double NearOnePct;
};

struct TGradientScaleStats {
    int Timestep;
    double MeanAbsGrad;
    double MaxAbsGrad;
    double MinAbsGrad;
};

struct TLayerNormStats {
    double Mean;
    double Variance;
    double Gamma;
    double Beta;
};

struct TOptimizerStateRecord {
    double Momentum;
    double Velocity;
    double Beta1Power;
    double Beta2Power;
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

// ========== Forward Declarations ==========
std::string CellTypeToStr(TCellType ct);
std::string ActivationToStr(TActivationType act);
std::string LossToStr(TLossType loss);
TCellType ParseCellType(const std::string& s);
TActivationType ParseActivation(const std::string& s);
TLossType ParseLoss(const std::string& s);
bool ParseIntArrayHelper(const std::string& s, TIntArray& arr);
void ParseDoubleArrayHelper(const std::string& s, DArray& arr);
void LoadDataFromCSV(const std::string& filename, TDArray2D& inputs, TDArray2D& targets);
void PrintUsage();

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

    void upload(const TDArray2D& data) {
        if (!gpu_ptr) return;
        std::vector<double> flat(rows * cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat[i * cols + j] = data[i][j];
            }
        }
        cudaMemcpy(gpu_ptr, flat.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    }

    void download(TDArray2D& data) {
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

void InitMatrix(TDArray2D& M, int Rows, int Cols, double Scale) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++) {
            M[i][j] = RandomWeight(Scale);
        }
    }
}

void ZeroMatrix(TDArray2D& M, int Rows, int Cols) {
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

// ========== Activation Functions ==========
class TActivation {
public:
    static double Apply(double X, TActivationType ActType) {
        switch (ActType) {
            case atSigmoid:
                return 1.0 / (1.0 + std::exp(-std::max(-500.0, std::min(500.0, X))));
            case atTanh:
                return std::tanh(X);
            case atReLU:
                return X > 0 ? X : 0;
            case atLinear:
                return X;
            default:
                return X;
        }
    }

    static double Derivative(double Y, TActivationType ActType) {
        switch (ActType) {
            case atSigmoid:
                return Y * (1.0 - Y);
            case atTanh:
                return 1.0 - Y * Y;
            case atReLU:
                return Y > 0 ? 1.0 : 0.0;
            case atLinear:
                return 1.0;
            default:
                return 1.0;
        }
    }

    static void ApplySoftmax(DArray& Arr) {
        if (Arr.empty()) return;
        
        double MaxVal = Arr[0];
        for (size_t i = 1; i < Arr.size(); ++i) {
            if (Arr[i] > MaxVal) MaxVal = Arr[i];
        }
        
        double Sum = 0;
        for (size_t i = 0; i < Arr.size(); ++i) {
            Arr[i] = std::exp(Arr[i] - MaxVal);
            Sum += Arr[i];
        }
        
        for (size_t i = 0; i < Arr.size(); ++i) {
            Arr[i] /= Sum;
        }
    }
};

// ========== Loss Functions ==========
class TLoss {
public:
    static double Compute(const DArray& Pred, const DArray& Target, TLossType LossType) {
        double Result = 0;
        switch (LossType) {
            case ltMSE:
                for (size_t i = 0; i < Pred.size(); ++i) {
                    Result += std::pow(Pred[i] - Target[i], 2);
                }
                break;
            case ltCrossEntropy:
                for (size_t i = 0; i < Pred.size(); ++i) {
                    double P = std::max(1e-15, std::min(1 - 1e-15, Pred[i]));
                    Result -= (Target[i] * std::log(P) + (1 - Target[i]) * std::log(1 - P));
                }
                break;
        }
        return Result / Pred.size();
    }

    static void Gradient(const DArray& Pred, const DArray& Target, TLossType LossType, DArray& Grad) {
        Grad.resize(Pred.size());
        switch (LossType) {
            case ltMSE:
                for (size_t i = 0; i < Pred.size(); ++i) {
                    Grad[i] = Pred[i] - Target[i];
                }
                break;
            case ltCrossEntropy:
                for (size_t i = 0; i < Pred.size(); ++i) {
                    double P = std::max(1e-15, std::min(1 - 1e-15, Pred[i]));
                    Grad[i] = (P - Target[i]) / (P * (1 - P) + 1e-15);
                }
                break;
        }
    }
};

class TSimpleRNNCellWrapper {
public:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;
    TDArray2D Wih, Whh;
    DArray Bh;
    TDArray2D dWih, dWhh;
    DArray dBh;
    TDArray2D MWih, MWhh;
    DArray MBh;
    TDArray2D VWih, VWhh;
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
            H[i] = TActivation::Apply(Sum, FActivation);
        }
    }

    void Backward(const DArray& dH, const DArray& H, const DArray& PreH,
                  const DArray& PrevH, const DArray& Input, double ClipVal,
                  DArray& dInput, DArray& dPrevH) {
        DArray dHRaw(FHiddenSize);
        dInput.resize(FInputSize, 0.0);
        dPrevH.resize(FHiddenSize, 0.0);

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
    TDArray2D Wf, Wi, Wc, Wo;
    DArray Bf, Bi, Bc, Bo;
    TDArray2D dWf, dWi, dWc, dWo;
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
            FG[i] = TActivation::Apply(Sf, atSigmoid);
            IG[i] = TActivation::Apply(Si, atSigmoid);
            CTilde[i] = TActivation::Apply(Sc, FActivation);
            C[i] = FG[i] * PrevC[i] + IG[i] * CTilde[i];
            TanhC[i] = std::tanh(C[i]);
            OG[i] = TActivation::Apply(So, atSigmoid);
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
    TDArray2D Wz, Wr, Wh;
    DArray Bz, Br, Bh;
    TDArray2D dWz, dWr, dWh;
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
            Z[i] = TActivation::Apply(Sz, atSigmoid);
            R[i] = TActivation::Apply(Sr, atSigmoid);
        }

        for (int i = 0; i < FHiddenSize; i++) {
            double Sh = Bh[i];
            for (int j = 0; j < (int)Input.size(); j++) {
                Sh += Wh[i][j] * Input[j];
            }
            for (int j = 0; j < FHiddenSize; j++) {
                Sh += Wh[i][Input.size() + j] * R[i] * PrevH[j];
            }
            HTilde[i] = TActivation::Apply(Sh, FActivation);
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
            double dHTilde = dH[i] * (1.0 - Z[i]) * TActivation::Derivative(HTilde[i], FActivation);

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
    TDArray2D W;
    DArray B;
    TDArray2D dW;
    DArray dB;
    TDArray2D MW;
    DArray MB;
    TDArray2D VW;
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
            Output[i] = TActivation::Apply(Sum, FActivation);
        }
    }

    void Backward(const DArray& dOut, const DArray& Output, const DArray& Pre,
                  const DArray& Input, double ClipVal, DArray& dInput) {
        dInput.resize(FInputSize, 0.0);
        for (int i = 0; i < FOutputSize; i++) {
            double dOut_i = ClipValue(dOut[i] * TActivation::Derivative(Output[i], FActivation), ClipVal);
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
    if (LossType == ltMSE) {
        for (size_t i = 0; i < Pred.size(); i++) {
            double Diff = Pred[i] - Target[i];
            Loss += Diff * Diff;
        }
        Loss /= Pred.size();
    } else if (LossType == ltCrossEntropy) {
        for (size_t i = 0; i < Pred.size(); i++) {
            Loss -= Target[i] * std::log(std::max(Pred[i], 1e-10));
        }
    }
    return Loss;
}

void ComputeLossGradient(const DArray& Pred, const DArray& Target, TLossType LossType, DArray& Grad) {
    Grad.resize(Pred.size());
    if (LossType == ltMSE) {
        for (size_t i = 0; i < Pred.size(); i++) {
            Grad[i] = 2.0 * (Pred[i] - Target[i]) / Pred.size();
        }
    } else if (LossType == ltCrossEntropy) {
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
    TDArray3D FStates;
    int FSequenceLen;
    TDArray2D FGradientHistory;

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

        if (CellType == ctSimpleRNN) {
            for (size_t i = 0; i < HiddenSizes.size(); i++) {
                int InSize = (i == 0) ? InputSize : HiddenSizes[i-1];
                FSimpleCells.emplace_back(InSize, HiddenSizes[i], Activation);
            }
        } else if (CellType == ctLSTM) {
            for (size_t i = 0; i < HiddenSizes.size(); i++) {
                int InSize = (i == 0) ? InputSize : HiddenSizes[i-1];
                FLSTMCells.emplace_back(InSize, HiddenSizes[i], Activation);
            }
        } else if (CellType == ctGRU) {
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

    TDArray3D InitHiddenStates() {
        TDArray3D States(FHiddenSizes.size());
        for (size_t i = 0; i < FHiddenSizes.size(); i++) {
            States[i].resize(2);
            States[i][0].resize(FHiddenSizes[i], 0.0);
            if (FCellType == ctLSTM)
                States[i][1].resize(FHiddenSizes[i], 0.0);
        }
        return States;
    }

    TDArray2D ForwardSequence(const TDArray2D& Inputs) {
        FSequenceLen = Inputs.size();
        FCaches.resize(FSequenceLen);
        FGradientHistory.resize(FSequenceLen);

        TDArray2D Outputs(FSequenceLen);

        for (int t = 0; t < FSequenceLen; t++) {
            FCaches[t].Input = Inputs[t];

            DArray LayerInput = Inputs[t];
            std::vector<DArray> LayerOutputs(FHiddenSizes.size());

            for (size_t layer = 0; layer < FHiddenSizes.size(); layer++) {
                DArray H, C, PreH;
                
                if (FCellType == ctSimpleRNN) {
                    FSimpleCells[layer].Forward(LayerInput, FStates[layer][0], H, PreH);
                    FStates[layer][0] = H;
                    LayerInput = H;
                    if (layer == 0 || t == 0) {
                        FCaches[t].H = H;
                        FCaches[t].PreH = PreH;
                    }
                } else if (FCellType == ctLSTM) {
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
                } else if (FCellType == ctGRU) {
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

    double BackwardSequence(const TDArray2D& Targets) {
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
                if (FCellType == ctSimpleRNN) {
                    FSimpleCells[layer].Backward(dLayerInput, FCaches[t].H, FCaches[t].PreH,
                                                 FStates[layer][0], FCaches[t].Input,
                                                 FGradientClip, dInput, dPrevH);
                } else if (FCellType == ctLSTM) {
                    FLSTMCells[layer].Backward(dLayerInput, dPrevC, FCaches[t].H, FCaches[t].C,
                                               FCaches[t].F, FCaches[t].I, FCaches[t].CTilde,
                                               FCaches[t].O, FCaches[t].TanhC,
                                               FStates[layer][0], FStates[layer][1],
                                               FCaches[t].Input, FGradientClip,
                                               dInput, dPrevH, dPrevC);
                } else if (FCellType == ctGRU) {
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

    double TrainSequence(const TDArray2D& Inputs, const TDArray2D& Targets) {
        ResetAllStates();
        TDArray2D Outputs = ForwardSequence(Inputs);
        double Loss = BackwardSequence(Targets);
        ApplyGradients();
        return Loss;
    }

    TDArray2D Predict(const TDArray2D& Inputs) {
        ResetAllStates();
        return ForwardSequence(Inputs);
    }

    void ResetAllStates(double Value = 0.0) {
        for (size_t layer = 0; layer < FHiddenSizes.size(); layer++) {
            for (int i = 0; i < FHiddenSizes[layer]; i++) {
                FStates[layer][0][i] = Value;
                if (FCellType == ctLSTM)
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
        if (FCellType != ctLSTM || LayerIdx < 0 || LayerIdx >= (int)FHiddenSizes.size() ||
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

        if (FCellType == ctLSTM && Timestep < (int)FCaches.size()) {
            switch (Gate) {
                case gtForget: return FCaches[Timestep].F[NeuronIdx];
                case gtInput: return FCaches[Timestep].I[NeuronIdx];
                case gtOutput: return FCaches[Timestep].O[NeuronIdx];
                case gtCellCandidate: return FCaches[Timestep].CTilde[NeuronIdx];
                default: return 0.0;
            }
        } else if (FCellType == ctGRU && Timestep < (int)FCaches.size()) {
            switch (Gate) {
                case gtUpdate: return FCaches[Timestep].Z[NeuronIdx];
                case gtReset: return FCaches[Timestep].R[NeuronIdx];
                case gtHiddenCandidate: return FCaches[Timestep].HTilde[NeuronIdx];
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

    TDArray2D GetSequenceOutputs() {
        TDArray2D Outputs;
        for (int t = 0; t < FSequenceLen; t++) {
            Outputs.push_back(FCaches[t].OutVal);
        }
        return Outputs;
    }

    TDArray2D GetSequenceHiddenStates(int LayerIdx) {
        TDArray2D States;
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
    // JSON Helper Methods
    // ========================================================================

    std::string Array1DToJSON(const DArray& Arr) {
        std::string Result = "[";
        for (size_t i = 0; i < Arr.size(); ++i) {
            if (i > 0) Result += ",";
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.17g", Arr[i]);
            Result += buf;
        }
        Result += "]";
        return Result;
    }

    std::string Array2DToJSON(const TDArray2D& Arr) {
        std::string Result = "[";
        for (size_t i = 0; i < Arr.size(); ++i) {
            if (i > 0) Result += ",";
            Result += Array1DToJSON(Arr[i]);
        }
        Result += "]";
        return Result;
    }

    // ========================================================================
    // Model Persistence (JSON format)
    // ========================================================================

    void SaveModel(const std::string& Filename) {
        std::ofstream File(Filename);
        if (!File.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << Filename << "\n";
            return;
        }

        File << "{\n";
        File << "  \"input_size\": " << FInputSize << ",\n";
        File << "  \"output_size\": " << FOutputSize << ",\n";
        File << "  \"hidden_sizes\": [\n";
        for (size_t i = 0; i < FHiddenSizes.size(); ++i) {
            if (i > 0) File << ",\n";
            File << "    " << FHiddenSizes[i];
        }
        File << "\n  ],\n";

        std::string CellTypeStr = CellTypeToStr(FCellType);
        File << "  \"cell_type\": \"" << CellTypeStr << "\",\n";
        File << "  \"activation\": \"" << ActivationToStr(FActivation) << "\",\n";
        File << "  \"output_activation\": \"" << ActivationToStr(FOutputActivation) << "\",\n";
        File << "  \"loss_type\": \"" << LossToStr(FLossType) << "\",\n";
        File << "  \"learning_rate\": " << FLearningRate << ",\n";
        File << "  \"gradient_clip\": " << FGradientClip << ",\n";
        File << "  \"bptt_steps\": " << FBPTTSteps << ",\n";
        File << "  \"dropout_rate\": " << FDropoutRate << ",\n";

        switch (FCellType) {
            case ctSimpleRNN:
                File << "  \"cells\": [\n";
                for (size_t i = 0; i < FSimpleCells.size(); ++i) {
                    if (i > 0) File << ",\n";
                    File << "    {\n";
                    File << "      \"Wih\": " << Array2DToJSON(FSimpleCells[i].Wih) << ",\n";
                    File << "      \"Whh\": " << Array2DToJSON(FSimpleCells[i].Whh) << ",\n";
                    File << "      \"Bh\": " << Array1DToJSON(FSimpleCells[i].Bh) << "\n";
                    File << "    }";
                }
                File << "\n  ]\n";
                break;
            case ctLSTM:
                File << "  \"cells\": [\n";
                for (size_t i = 0; i < FLSTMCells.size(); ++i) {
                    if (i > 0) File << ",\n";
                    File << "    {\n";
                    File << "      \"Wf\": " << Array2DToJSON(FLSTMCells[i].Wf) << ",\n";
                    File << "      \"Wi\": " << Array2DToJSON(FLSTMCells[i].Wi) << ",\n";
                    File << "      \"Wc\": " << Array2DToJSON(FLSTMCells[i].Wc) << ",\n";
                    File << "      \"Wo\": " << Array2DToJSON(FLSTMCells[i].Wo) << ",\n";
                    File << "      \"Bf\": " << Array1DToJSON(FLSTMCells[i].Bf) << ",\n";
                    File << "      \"Bi\": " << Array1DToJSON(FLSTMCells[i].Bi) << ",\n";
                    File << "      \"Bc\": " << Array1DToJSON(FLSTMCells[i].Bc) << ",\n";
                    File << "      \"Bo\": " << Array1DToJSON(FLSTMCells[i].Bo) << "\n";
                    File << "    }";
                }
                File << "\n  ]\n";
                break;
            case ctGRU:
                File << "  \"cells\": [\n";
                for (size_t i = 0; i < FGRUCells.size(); ++i) {
                    if (i > 0) File << ",\n";
                    File << "    {\n";
                    File << "      \"Wz\": " << Array2DToJSON(FGRUCells[i].Wz) << ",\n";
                    File << "      \"Wr\": " << Array2DToJSON(FGRUCells[i].Wr) << ",\n";
                    File << "      \"Wh\": " << Array2DToJSON(FGRUCells[i].Wh) << ",\n";
                    File << "      \"Bz\": " << Array1DToJSON(FGRUCells[i].Bz) << ",\n";
                    File << "      \"Br\": " << Array1DToJSON(FGRUCells[i].Br) << ",\n";
                    File << "      \"Bh\": " << Array1DToJSON(FGRUCells[i].Bh) << "\n";
                    File << "    }";
                }
                File << "\n  ]\n";
                break;
        }

        File << ",\n";
        File << "  \"output_layer\": {\n";
        File << "    \"W\": " << Array2DToJSON(FOutputLayer->W) << ",\n";
        File << "    \"B\": " << Array1DToJSON(FOutputLayer->B) << "\n";
        File << "  }\n";
        File << "}\n";
        File.close();
        std::cout << "Model saved to JSON: " << Filename << "\n";
    }

    void LoadModel(const std::string& Filename) {
        std::ifstream File(Filename);
        if (!File.is_open()) {
            std::cerr << "Error: Could not open file for reading: " << Filename << "\n";
            return;
        }

        std::stringstream Buffer;
        Buffer << File.rdbuf();
        std::string Content = Buffer.str();
        File.close();

        auto ExtractJSONValue = [](const std::string& json, const std::string& key) -> std::string {
            std::string searchKey = "\"" + key + "\"";
            size_t keyPos = json.find(searchKey);
            if (keyPos == std::string::npos) return "";
            size_t colonPos = json.find(':', keyPos);
            if (colonPos == std::string::npos) return "";
            size_t startPos = colonPos + 1;
            while (startPos < json.length() && (json[startPos] == ' ' || json[startPos] == '\t' 
                   || json[startPos] == '\n' || json[startPos] == '\r')) {
                ++startPos;
            }
            if (startPos < json.length() && json[startPos] == '"') {
                size_t quotePos1 = startPos;
                size_t quotePos2 = json.find('"', quotePos1 + 1);
                if (quotePos2 != std::string::npos) {
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
            if (endPos == std::string::npos) endPos = json.find('}', startPos);
            if (endPos == std::string::npos) endPos = json.find(']', startPos);
            std::string result = json.substr(startPos, endPos - startPos);
            size_t end = result.find_last_not_of(" \t\n\r");
            if (end != std::string::npos) {
                result = result.substr(0, end + 1);
            }
            return result;
        };
        
        std::string inputStr = ExtractJSONValue(Content, "input_size");
        int inputSize = inputStr.empty() ? 1 : std::stoi(inputStr);
        
        std::string outputStr = ExtractJSONValue(Content, "output_size");
        int outputSize = outputStr.empty() ? 1 : std::stoi(outputStr);
        
        std::string cellTypeStr = ExtractJSONValue(Content, "cell_type");
        TCellType cellType = ParseCellType(cellTypeStr);
        
        std::string hiddenStr = ExtractJSONValue(Content, "hidden_sizes");
        
        std::string activationStr = ExtractJSONValue(Content, "hidden_activation");
        if (activationStr.empty()) activationStr = ExtractJSONValue(Content, "activation");
        if (activationStr.empty()) activationStr = "tanh";
        
        std::string outputActStr = ExtractJSONValue(Content, "output_activation");
        if (outputActStr.empty()) outputActStr = "linear";
        
        std::string lossStr = ExtractJSONValue(Content, "loss_type");
        if (lossStr.empty()) lossStr = "mse";
        
        std::string lrStr = ExtractJSONValue(Content, "learning_rate");
        double learningRate = lrStr.empty() ? 0.01 : std::stod(lrStr);
        
        std::string clipStr = ExtractJSONValue(Content, "gradient_clip");
        double gradientClip = clipStr.empty() ? 5.0 : std::stod(clipStr);
        
        std::string bpttStr = ExtractJSONValue(Content, "bptt_steps");
        int bpttSteps = bpttStr.empty() ? 0 : std::stoi(bpttStr);
        
        std::string dropoutStr = ExtractJSONValue(Content, "dropout_rate");
        double dropoutRate = dropoutStr.empty() ? 0.0 : std::stod(dropoutStr);
        
        TIntArray hiddenSizes;
        size_t openBracket = hiddenStr.find('[');
        size_t closeBracket = hiddenStr.rfind(']');
        if (openBracket != std::string::npos && closeBracket != std::string::npos) {
            std::string arrayContent = hiddenStr.substr(openBracket + 1, closeBracket - openBracket - 1);
            std::stringstream ss(arrayContent);
            std::string token;
            while (std::getline(ss, token, ',')) {
                size_t start = token.find_first_not_of(" \t\n\r");
                size_t end = token.find_last_not_of(" \t\n\r");
                if (start != std::string::npos && end != std::string::npos) {
                    token = token.substr(start, end - start + 1);
                    if (!token.empty()) {
                        hiddenSizes.push_back(std::stoi(token));
                    }
                }
            }
        } else {
            if (!hiddenStr.empty()) {
                hiddenSizes.push_back(std::stoi(hiddenStr));
            }
        }
        
        FInputSize = inputSize;
        FOutputSize = outputSize;
        FHiddenSizes = std::vector<int>(hiddenSizes.begin(), hiddenSizes.end());
        FCellType = cellType;
        FActivation = ParseActivation(activationStr);
        FOutputActivation = ParseActivation(outputActStr);
        FLossType = ParseLoss(lossStr);
        FLearningRate = learningRate;
        FGradientClip = gradientClip;
        FBPTTSteps = bpttSteps;
        FDropoutRate = dropoutRate;
        
        FSimpleCells.clear();
        FLSTMCells.clear();
        FGRUCells.clear();
        if (FOutputLayer) delete FOutputLayer;
        
        int PrevSize = inputSize;
        switch (cellType) {
            case ctSimpleRNN:
                for (size_t i = 0; i < hiddenSizes.size(); ++i) {
                    FSimpleCells.emplace_back(PrevSize, hiddenSizes[i], FActivation);
                    PrevSize = hiddenSizes[i];
                }
                break;
            case ctLSTM:
                for (size_t i = 0; i < hiddenSizes.size(); ++i) {
                    FLSTMCells.emplace_back(PrevSize, hiddenSizes[i], FActivation);
                    PrevSize = hiddenSizes[i];
                }
                break;
            case ctGRU:
                for (size_t i = 0; i < hiddenSizes.size(); ++i) {
                    FGRUCells.emplace_back(PrevSize, hiddenSizes[i], FActivation);
                    PrevSize = hiddenSizes[i];
                }
                break;
        }
        
        FOutputLayer = new TOutputLayerWrapper(PrevSize, outputSize, FOutputActivation);
        FStates = InitHiddenStates();
        
        std::cout << "Model loaded from JSON: " << Filename << "\n";
    }

    int GetInputSize() const { return FInputSize; }
    int GetOutputSize() const { return FOutputSize; }
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

// ========== Helper Functions ==========
std::string CellTypeToStr(TCellType ct) {
    switch (ct) {
        case ctSimpleRNN: return "simplernn";
        case ctLSTM: return "lstm";
        case ctGRU: return "gru";
        default: return "simplernn";
    }
}

std::string ActivationToStr(TActivationType act) {
    switch (act) {
        case atSigmoid: return "sigmoid";
        case atTanh: return "tanh";
        case atReLU: return "relu";
        case atLinear: return "linear";
        default: return "linear";
    }
}

std::string LossToStr(TLossType loss) {
    switch (loss) {
        case ltMSE: return "mse";
        case ltCrossEntropy: return "crossentropy";
        default: return "mse";
    }
}

TCellType ParseCellType(const std::string& s) {
    std::string s_lower = s;
    std::transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);
    if (s_lower == "lstm") return ctLSTM;
    if (s_lower == "gru") return ctGRU;
    return ctSimpleRNN;
}

TActivationType ParseActivation(const std::string& s) {
    std::string s_lower = s;
    std::transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);
    if (s_lower == "sigmoid") return atSigmoid;
    if (s_lower == "tanh") return atTanh;
    if (s_lower == "relu") return atReLU;
    return atLinear;
}

TLossType ParseLoss(const std::string& s) {
    std::string s_lower = s;
    std::transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);
    if (s_lower == "crossentropy") return ltCrossEntropy;
    return ltMSE;
}

bool ParseIntArrayHelper(const std::string& s, TIntArray& arr) {
    arr.clear();
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        try {
            arr.push_back(std::stoi(item));
        } catch (...) {
            return false;
        }
    }
    return true;
}

void ParseDoubleArrayHelper(const std::string& s, DArray& arr) {
    arr.clear();
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        arr.push_back(std::stod(item));
    }
}

void LoadDataFromCSV(const std::string& Filename, TDArray2D& Inputs, TDArray2D& Targets) {
    std::ifstream File(Filename);
    if (!File.is_open()) {
        std::cerr << "Error: Could not open CSV file: " << Filename << "\n";
        return;
    }

    std::string Line;
    while (std::getline(File, Line)) {
        if (Line.empty()) continue;

        std::stringstream ss(Line);
        std::string Item;
        std::vector<double> Values;

        while (std::getline(ss, Item, ',')) {
            Item.erase(0, Item.find_first_not_of(" \t"));
            Item.erase(Item.find_last_not_of(" \t") + 1);
            try {
                Values.push_back(std::stod(Item));
            } catch (...) {
                continue;
            }
        }

        if (Values.size() >= 2) {
            size_t SplitPoint = Values.size() / 2;
            DArray Input(Values.begin(), Values.begin() + SplitPoint);
            DArray Target(Values.begin() + SplitPoint, Values.end());
            Inputs.push_back(Input);
            Targets.push_back(Target);
        }
    }

    File.close();
}

void PrintUsage() {
    std::cout << "Facaded RNN (CUDA Accelerated)\n\n";
    std::cout << "Commands:\n";
    std::cout << "  create   Create a new RNN model and save to JSON\n";
    std::cout << "  train    Train an existing model with data from JSON\n";
    std::cout << "  predict  Make predictions with a trained model from JSON\n";
    std::cout << "  info     Display model information from JSON\n";
    std::cout << "  query    Query model state and internals (facade functions)\n";
    std::cout << "  help     Show this help message\n\n";
    std::cout << "Create Options:\n";
    std::cout << "  --input=N              Input layer size (required)\n";
    std::cout << "  --hidden=N,N,...       Hidden layer sizes (required)\n";
    std::cout << "  --output=N             Output layer size (required)\n";
    std::cout << "  --save=FILE.json       Save model to JSON file (required)\n";
    std::cout << "  --cell=TYPE            simplernn|lstm|gru (default: lstm)\n";
    std::cout << "  --lr=VALUE             Learning rate (default: 0.01)\n";
    std::cout << "  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: tanh)\n";
    std::cout << "  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)\n";
    std::cout << "  --loss=TYPE            mse|crossentropy (default: mse)\n";
    std::cout << "  --clip=VALUE           Gradient clipping (default: 5.0)\n";
    std::cout << "  --bptt=N               BPTT steps (default: 0 = full)\n\n";
    std::cout << "Train Options:\n";
    std::cout << "  --model=FILE.json      Load model from JSON file (required)\n";
    std::cout << "  --data=FILE.csv        Training data CSV file (required)\n";
    std::cout << "  --save=FILE.json       Save trained model to JSON (required)\n";
    std::cout << "  --epochs=N             Number of training epochs (default: 100)\n";
    std::cout << "  --batch=N              Batch size (default: 1)\n";
    std::cout << "  --lr=VALUE             Override learning rate\n";
    std::cout << "  --seq-len=N            Sequence length (default: auto-detect)\n\n";
    std::cout << "Predict Options:\n";
    std::cout << "  --model=FILE.json      Load model from JSON file (required)\n";
    std::cout << "  --input=v1,v2,...      Input values as CSV (required)\n\n";
    std::cout << "Info Options:\n";
    std::cout << "  --model=FILE.json      Load model from JSON file (required)\n\n";
    std::cout << "Query Options (Facade Functions):\n";
    std::cout << "  --model=FILE.json      Load model from JSON file (required)\n";
    std::cout << "  --query-type=TYPE      Query type (required)\n";
    std::cout << "                         Valid types: input-size, output-size, hidden-size,\n";
    std::cout << "                                      cell-type, sequence-length, dropout-rate,\n";
    std::cout << "                                      hidden-state\n";
    std::cout << "  --layer=N              Layer index\n";
    std::cout << "  --timestep=N           Timestep index\n";
    std::cout << "  --neuron=N             Neuron index\n";
    std::cout << "  --index=N              Generic index parameter\n";
    std::cout << "  --dropout-rate=VALUE   Set dropout rate (0.0-1.0)\n";
    std::cout << "  --enable-dropout       Enable dropout\n";
    std::cout << "  --disable-dropout      Disable dropout\n\n";
    std::cout << "Examples:\n";
    std::cout << "  facaded_rnn create --input=2 --hidden=16 --output=2 --cell=lstm --save=seq.json\n";
    std::cout << "  facaded_rnn train --model=seq.json --data=seq.csv --epochs=200 --save=seq_trained.json\n";
    std::cout << "  facaded_rnn predict --model=seq_trained.json --input=0.5,0.5\n";
    std::cout << "  facaded_rnn info --model=seq_trained.json\n";
}

// ========== Main Program ==========
int main(int argc, char* argv[]) {
    if (argc < 2) {
        PrintUsage();
        return 1;
    }

    std::string CmdStr = argv[1];
    TCommand Command = cmdNone;

    if (CmdStr == "create") Command = cmdCreate;
    else if (CmdStr == "train") Command = cmdTrain;
    else if (CmdStr == "predict") Command = cmdPredict;
    else if (CmdStr == "info") Command = cmdInfo;
    else if (CmdStr == "query") Command = cmdQuery;
    else if (CmdStr == "help" || CmdStr == "--help" || CmdStr == "-h") Command = cmdHelp;
    else {
        std::cerr << "Unknown command: " << CmdStr << "\n";
        PrintUsage();
        return 1;
    }

    if (Command == cmdHelp) {
        PrintUsage();
        return 0;
    }

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
    std::string modelFile, saveFile, dataFile;
    DArray inputValues;

    std::string queryType, gateTypeStr;
    int layer = 0, timestep = 0, neuron = 0, index = 0, param = 0;
    double dropoutValue = 0.0;
    bool enableDropoutFlag = false, disableDropoutFlag = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--enable-dropout") {
            enableDropoutFlag = true;
        } else if (arg == "--disable-dropout") {
            disableDropoutFlag = true;
        } else {
            size_t eqPos = arg.find('=');
            if (eqPos == std::string::npos) {
                std::cerr << "Invalid argument: " << arg << "\n";
                continue;
            }

            std::string key = arg.substr(0, eqPos);
            std::string value = arg.substr(eqPos + 1);

            if (key == "--input") {
                if (Command == cmdPredict) {
                    ParseDoubleArrayHelper(value, inputValues);
                } else {
                    inputSize = std::stoi(value);
                }
            } else if (key == "--hidden") {
                ParseIntArrayHelper(value, hiddenSizes);
            } else if (key == "--output") {
                outputSize = std::stoi(value);
            } else if (key == "--save") {
                saveFile = value;
            } else if (key == "--model") {
                modelFile = value;
            } else if (key == "--data") {
                dataFile = value;
            } else if (key == "--lr") {
                learningRate = std::stod(value);
            } else if (key == "--cell") {
                cellType = ParseCellType(value);
            } else if (key == "--hidden-act") {
                hiddenAct = ParseActivation(value);
            } else if (key == "--output-act") {
                outputAct = ParseActivation(value);
            } else if (key == "--loss") {
                lossType = ParseLoss(value);
            } else if (key == "--clip") {
                gradientClip = std::stod(value);
            } else if (key == "--bptt") {
                bpttSteps = std::stoi(value);
            } else if (key == "--epochs") {
                epochs = std::stoi(value);
            } else if (key == "--batch") {
                batchSize = std::stoi(value);
            } else if (key == "--seq-len") {
                seqLen = std::stoi(value);
            } else if (key == "--query-type") {
                queryType = value;
            } else if (key == "--layer") {
                layer = std::stoi(value);
            } else if (key == "--timestep") {
                timestep = std::stoi(value);
            } else if (key == "--neuron") {
                neuron = std::stoi(value);
            } else if (key == "--index") {
                index = std::stoi(value);
            } else if (key == "--gate") {
                gateTypeStr = value;
            } else if (key == "--param") {
                param = std::stoi(value);
            } else if (key == "--dropout-rate") {
                dropoutValue = std::stod(value);
            } else {
                std::cerr << "Unknown option: " << key << "\n";
            }
        }
    }

    if (Command == cmdCreate) {
        if (inputSize <= 0) { std::cerr << "Error: --input is required\n"; return 1; }
        if (hiddenSizes.empty()) { std::cerr << "Error: --hidden is required\n"; return 1; }
        if (outputSize <= 0) { std::cerr << "Error: --output is required\n"; return 1; }
        if (saveFile.empty()) { std::cerr << "Error: --save is required\n"; return 1; }

        TRNNFacadeCUDA* RNN = new TRNNFacadeCUDA(inputSize, std::vector<int>(hiddenSizes.begin(), hiddenSizes.end()), 
                                                  outputSize, cellType, hiddenAct, outputAct, lossType, 
                                                  learningRate, gradientClip, bpttSteps, true);

        std::cout << "Created RNN model:\n";
        std::cout << "  Input size: " << inputSize << "\n";
        std::cout << "  Hidden sizes: ";
        for (size_t i = 0; i < hiddenSizes.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << hiddenSizes[i];
        }
        std::cout << "\n";
        std::cout << "  Output size: " << outputSize << "\n";
        std::cout << "  Cell type: " << CellTypeToStr(cellType) << "\n";
        std::cout << "  Hidden activation: " << ActivationToStr(hiddenAct) << "\n";
        std::cout << "  Output activation: " << ActivationToStr(outputAct) << "\n";
        std::cout << "  Loss function: " << LossToStr(lossType) << "\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "  Learning rate: " << learningRate << "\n";
        std::cout << std::fixed << std::setprecision(2)
                  << "  Gradient clip: " << gradientClip << "\n";
        std::cout << "  BPTT steps: " << bpttSteps << "\n";
        std::cout << "  GPU Available: " << (RNN->isGPUAvailable() ? "Yes" : "No") << "\n";

        RNN->SaveModel(saveFile);
        delete RNN;
    }
    else if (Command == cmdTrain) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required\n"; return 1; }
        if (dataFile.empty()) { std::cerr << "Error: --data is required\n"; return 1; }
        if (saveFile.empty()) { std::cerr << "Error: --save is required\n"; return 1; }

        std::cout << "Loading model from JSON: " << modelFile << "\n";
        TRNNFacadeCUDA* RNN = new TRNNFacadeCUDA(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0, true);
        RNN->LoadModel(modelFile);
        std::cout << "Model loaded successfully.\n";

        std::cout << "Loading training data from: " << dataFile << "\n";
        TDArray2D Inputs, Targets;
        LoadDataFromCSV(dataFile, Inputs, Targets);

        if (Inputs.empty()) {
            std::cerr << "Error: No data loaded from CSV file\n";
            delete RNN;
            return 1;
        }

        std::cout << "Loaded " << Inputs.size() << " timesteps of training data\n";
        std::cout << "Starting training for " << epochs << " epochs...\n";

        for (int Epoch = 1; Epoch <= epochs; ++Epoch) {
            double TrainLoss = RNN->TrainSequence(Inputs, Targets);

            if (!std::isnan(TrainLoss) && !std::isinf(TrainLoss)) {
                if (verbose || (Epoch % 10 == 0) || (Epoch == epochs)) {
                    std::cout << "Epoch " << std::setw(4) << Epoch << "/"
                              << epochs << " - Loss: "
                              << std::fixed << std::setprecision(6) << TrainLoss << "\n";
                }
            }
        }

        std::cout << "Training completed.\n";
        std::cout << "Saving trained model to: " << saveFile << "\n";
        RNN->SaveModel(saveFile);

        delete RNN;
    }
    else if (Command == cmdPredict) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required\n"; return 1; }
        if (inputValues.empty()) { std::cerr << "Error: --input is required\n"; return 1; }

        TRNNFacadeCUDA* RNN = new TRNNFacadeCUDA(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0, true);
        RNN->LoadModel(modelFile);

        TDArray2D Inputs(1);
        Inputs[0] = inputValues;

        TDArray2D Predictions = RNN->Predict(Inputs);

        std::cout << "Input: ";
        for (size_t i = 0; i < inputValues.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << inputValues[i];
        }
        std::cout << "\n";

        if (!Predictions.empty() && !Predictions.back().empty()) {
            std::cout << "Output: ";
            for (size_t i = 0; i < Predictions.back().size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(6) << Predictions.back()[i];
            }
            std::cout << "\n";

            if (Predictions.back().size() > 1) {
                size_t maxIdx = 0;
                for (size_t i = 1; i < Predictions.back().size(); ++i) {
                    if (Predictions.back()[i] > Predictions.back()[maxIdx]) {
                        maxIdx = i;
                    }
                }
                std::cout << "Max index: " << maxIdx << "\n";
            }
        }

        delete RNN;
    }
    else if (Command == cmdInfo) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required\n"; return 1; }
        std::cout << "Loading model from JSON: " << modelFile << "\n";
        TRNNFacadeCUDA* RNN = new TRNNFacadeCUDA(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0, true);
        RNN->LoadModel(modelFile);
        
        std::cout << "Model Information:\n";
        std::cout << "  Layers: " << RNN->GetLayerCount() << "\n";
        std::cout << "  Hidden sizes: ";
        for (int i = 0; i < RNN->GetLayerCount(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << RNN->GetHiddenSize(i);
        }
        std::cout << "\n";
        std::cout << "  Cell type: " << CellTypeToStr(RNN->GetCellType()) << "\n";
        std::cout << "  Learning rate: " << std::fixed << std::setprecision(6) << RNN->getLearningRate() << "\n";
        std::cout << "  Gradient clip: " << std::fixed << std::setprecision(2) << RNN->getGradientClip() << "\n";
        std::cout << "  Dropout rate: " << std::fixed << std::setprecision(6) << RNN->getDropoutRate() << "\n";
        std::cout << "  GPU Available: " << (RNN->isGPUAvailable() ? "Yes" : "No") << "\n";
        delete RNN;
    }
    else if (Command == cmdQuery) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required\n"; return 1; }
        if (queryType.empty()) { std::cerr << "Error: --query-type is required\n"; return 1; }

        std::cout << "Loading model from JSON: " << modelFile << "\n";
        TRNNFacadeCUDA* RNN = new TRNNFacadeCUDA(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0, true);
        RNN->LoadModel(modelFile);

        std::cout << "Executing query: " << queryType << "\n\n";

        if (queryType == "input-size") {
            std::cout << "Input size: " << RNN->GetInputSize() << "\n";
        } else if (queryType == "output-size") {
            std::cout << "Output size: " << RNN->GetOutputSize() << "\n";
        } else if (queryType == "hidden-size") {
            std::cout << "Hidden size (layer " << layer << "): " << RNN->GetHiddenSize(layer) << "\n";
        } else if (queryType == "cell-type") {
            std::cout << "Cell type: " << CellTypeToStr(RNN->GetCellType()) << "\n";
        } else if (queryType == "sequence-length") {
            std::cout << "Sequence length: " << RNN->GetSequenceLength() << "\n";
        } else if (queryType == "dropout-rate") {
            std::cout << std::fixed << std::setprecision(6)
                      << "Current dropout rate: " << RNN->getDropoutRate() << "\n";
        } else if (queryType == "hidden-state") {
            std::cout << std::fixed << std::setprecision(6)
                      << "Hidden state at [" << layer << "," << timestep << "," << neuron << "]: "
                      << RNN->GetHiddenValue(layer, timestep, neuron) << "\n";
        } else {
            std::cout << "Unknown query type: " << queryType << "\n";
        }

        if (dropoutValue > 0) {
            RNN->setDropoutRate(dropoutValue);
            std::cout << std::fixed << std::setprecision(6)
                      << "Dropout rate set to: " << dropoutValue << "\n";
        }

        delete RNN;
    }

    return 0;
}
