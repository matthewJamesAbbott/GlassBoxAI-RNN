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

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// ========== Type Aliases ==========
using DArray = std::vector<double>;
using TDArray2D = std::vector<DArray>;
using TDArray3D = std::vector<TDArray2D>;
using TIntArray = std::vector<int>;

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

// ========== OpenCL Context ==========
class TOpenCLContext {
public:
    cl_platform_id Platform;
    cl_device_id Device;
    cl_context Context;
    cl_command_queue Queue;
    bool Initialized;

    TOpenCLContext() : Initialized(false) {
        InitializeOpenCL();
    }

    ~TOpenCLContext() {
        if (Initialized) {
            clReleaseCommandQueue(Queue);
            clReleaseContext(Context);
        }
    }

    void InitializeOpenCL() {
        cl_int Error;
        
        clGetPlatformIDs(1, &Platform, nullptr);
        clGetDeviceIDs(Platform, CL_DEVICE_TYPE_GPU, 1, &Device, nullptr);
        
        Context = clCreateContext(nullptr, 1, &Device, nullptr, nullptr, &Error);
        if (Error != CL_SUCCESS) {
            std::cerr << "Failed to create OpenCL context: " << Error << "\n";
            Initialized = false;
            return;
        }
        
        Queue = clCreateCommandQueue(Context, Device, 0, &Error);
        if (Error != CL_SUCCESS) {
            std::cerr << "Failed to create command queue: " << Error << "\n";
            Initialized = false;
            return;
        }
        
        Initialized = true;
    }

    cl_mem CreateBuffer(size_t Size, const void* HostData = nullptr) {
        cl_int Error;
        cl_mem Buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE | 
                                       (HostData ? CL_MEM_COPY_HOST_PTR : 0),
                                       Size, const_cast<void*>(HostData), &Error);
        if (Error != CL_SUCCESS) {
            std::cerr << "Failed to create buffer: " << Error << "\n";
        }
        return Buffer;
    }

    void ReleaseBuffer(cl_mem Buffer) {
        clReleaseMemObject(Buffer);
    }

    void WriteBuffer(cl_mem Buffer, size_t Size, const void* HostData) {
        clEnqueueWriteBuffer(Queue, Buffer, CL_TRUE, 0, Size, const_cast<void*>(HostData), 0, nullptr, nullptr);
    }

    void ReadBuffer(cl_mem Buffer, size_t Size, void* HostData) {
        clEnqueueReadBuffer(Queue, Buffer, CL_TRUE, 0, Size, HostData, 0, nullptr, nullptr);
    }

    void Finish() {
        clFinish(Queue);
    }
};

static TOpenCLContext gOpenCLContext;

// ========== Utility Functions ==========
double ClipValue(double V, double MaxVal) {
    if (V > MaxVal) return MaxVal;
    if (V < -MaxVal) return -MaxVal;
    return V;
}

double RandomWeight(double Scale) {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    return (dis(gen) - 0.5) * 2.0 * Scale;
}

void InitMatrix(TDArray2D& M, int Rows, int Cols, double Scale) {
    M.resize(Rows);
    for (int i = 0; i < Rows; ++i) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; ++j) {
            M[i][j] = RandomWeight(Scale);
        }
    }
}

void ZeroMatrix(TDArray2D& M, int Rows, int Cols) {
    M.resize(Rows);
    for (int i = 0; i < Rows; ++i) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; ++j) {
            M[i][j] = 0.0;
        }
    }
}

void ZeroArray(DArray& A, int Size) {
    A.resize(Size);
    for (int i = 0; i < Size; ++i) {
        A[i] = 0.0;
    }
}

DArray ConcatArrays(const DArray& A, const DArray& B) {
    DArray Result(A.size() + B.size());
    for (size_t i = 0; i < A.size(); ++i) {
        Result[i] = A[i];
    }
    for (size_t i = 0; i < B.size(); ++i) {
        Result[A.size() + i] = B[i];
    }
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

// ========== Simple RNN Cell ==========
class TSimpleRNNCell {
private:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;

public:
    TDArray2D Wih, Whh;
    DArray Bh;
    TDArray2D dWih, dWhh;
    DArray dBh;

    TSimpleRNNCell(int InputSize, int HiddenSize, TActivationType Activation)
        : FInputSize(InputSize), FHiddenSize(HiddenSize), FActivation(Activation) {
        double Scale = std::sqrt(2.0 / (InputSize + HiddenSize));
        InitMatrix(Wih, HiddenSize, InputSize, Scale);
        InitMatrix(Whh, HiddenSize, HiddenSize, Scale);
        ZeroArray(Bh, HiddenSize);
        ZeroMatrix(dWih, HiddenSize, InputSize);
        ZeroMatrix(dWhh, HiddenSize, HiddenSize);
        ZeroArray(dBh, HiddenSize);
    }

    void Forward(const DArray& Input, const DArray& PrevH, DArray& H, DArray& PreH) {
        H.resize(FHiddenSize);
        PreH.resize(FHiddenSize);
        
        for (int i = 0; i < FHiddenSize; ++i) {
            double Sum = Bh[i];
            for (int j = 0; j < FInputSize; ++j) {
                Sum += Wih[i][j] * Input[j];
            }
            for (int j = 0; j < FHiddenSize; ++j) {
                Sum += Whh[i][j] * PrevH[j];
            }
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
        
        for (int i = 0; i < FInputSize; ++i) dInput[i] = 0;
        for (int i = 0; i < FHiddenSize; ++i) dPrevH[i] = 0;

        for (int i = 0; i < FHiddenSize; ++i) {
            dHRaw[i] = ClipValue(dH[i] * TActivation::Derivative(H[i], FActivation), ClipVal);
        }

        for (int i = 0; i < FHiddenSize; ++i) {
            for (int j = 0; j < FInputSize; ++j) {
                dWih[i][j] += dHRaw[i] * Input[j];
                dInput[j] += Wih[i][j] * dHRaw[i];
            }
            for (int j = 0; j < FHiddenSize; ++j) {
                dWhh[i][j] += dHRaw[i] * PrevH[j];
                dPrevH[j] += Whh[i][j] * dHRaw[i];
            }
            dBh[i] += dHRaw[i];
        }
    }

    void ApplyGradients(double LR, double ClipVal) {
        for (int i = 0; i < FHiddenSize; ++i) {
            for (int j = 0; j < FInputSize; ++j) {
                Wih[i][j] -= LR * ClipValue(dWih[i][j], ClipVal);
                dWih[i][j] = 0;
            }
            for (int j = 0; j < FHiddenSize; ++j) {
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

    int GetHiddenSize() const { return FHiddenSize; }
};

// ========== LSTM Cell ==========
class TLSTMCell {
private:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;

public:
    TDArray2D Wf, Wi, Wc, Wo;
    DArray Bf, Bi, Bc, Bo;
    TDArray2D dWf, dWi, dWc, dWo;
    DArray dBf, dBi, dBc, dBo;

    TLSTMCell(int InputSize, int HiddenSize, TActivationType Activation)
        : FInputSize(InputSize), FHiddenSize(HiddenSize), FActivation(Activation) {
        int ConcatSize = InputSize + HiddenSize;
        double Scale = 0.01;

        InitMatrix(Wf, HiddenSize, ConcatSize, Scale);
        InitMatrix(Wi, HiddenSize, ConcatSize, Scale);
        InitMatrix(Wc, HiddenSize, ConcatSize, Scale);
        InitMatrix(Wo, HiddenSize, ConcatSize, Scale);

        ZeroArray(Bf, HiddenSize);
        ZeroArray(Bi, HiddenSize);
        ZeroArray(Bc, HiddenSize);
        ZeroArray(Bo, HiddenSize);

        ZeroMatrix(dWf, HiddenSize, ConcatSize);
        ZeroMatrix(dWi, HiddenSize, ConcatSize);
        ZeroMatrix(dWc, HiddenSize, ConcatSize);
        ZeroMatrix(dWo, HiddenSize, ConcatSize);
        ZeroArray(dBf, HiddenSize);
        ZeroArray(dBi, HiddenSize);
        ZeroArray(dBc, HiddenSize);
        ZeroArray(dBo, HiddenSize);
    }

    void Forward(const DArray& Input, const DArray& PrevH, const DArray& PrevC,
                 DArray& H, DArray& C, DArray& F, DArray& I, DArray& CTilde,
                 DArray& O, DArray& TanhC) {
        DArray Concat = ConcatArrays(Input, PrevH);
        
        H.resize(FHiddenSize);
        C.resize(FHiddenSize);
        F.resize(FHiddenSize);
        I.resize(FHiddenSize);
        CTilde.resize(FHiddenSize);
        O.resize(FHiddenSize);
        TanhC.resize(FHiddenSize);

        for (int k = 0; k < FHiddenSize; ++k) {
            double SumF = Bf[k], SumI = Bi[k], SumC = Bc[k], SumO = Bo[k];
            for (size_t j = 0; j < Concat.size(); ++j) {
                SumF += Wf[k][j] * Concat[j];
                SumI += Wi[k][j] * Concat[j];
                SumC += Wc[k][j] * Concat[j];
                SumO += Wo[k][j] * Concat[j];
            }
            F[k] = TActivation::Apply(SumF, atSigmoid);
            I[k] = TActivation::Apply(SumI, atSigmoid);
            CTilde[k] = TActivation::Apply(SumC, atTanh);
            O[k] = TActivation::Apply(SumO, atSigmoid);
            C[k] = F[k] * PrevC[k] + I[k] * CTilde[k];
            TanhC[k] = std::tanh(C[k]);
            H[k] = O[k] * TanhC[k];
        }
    }

    void Backward(const DArray& dH, const DArray& dC, const DArray& H, const DArray& C,
                  const DArray& F, const DArray& I, const DArray& CTilde, const DArray& O,
                  const DArray& TanhC, const DArray& PrevH, const DArray& PrevC,
                  const DArray& Input, double ClipVal, DArray& dInput, DArray& dPrevH, DArray& dPrevC) {
        DArray Concat = ConcatArrays(Input, PrevH);
        int ConcatSize = Concat.size();
        
        DArray d0(FHiddenSize), dCTotal(FHiddenSize), dF(FHiddenSize), dI(FHiddenSize), dCTilde(FHiddenSize);
        dInput.resize(FInputSize);
        dPrevH.resize(FHiddenSize);
        dPrevC.resize(FHiddenSize);

        for (int k = 0; k < FInputSize; ++k) dInput[k] = 0;
        for (int k = 0; k < FHiddenSize; ++k) {
            dPrevH[k] = 0;
            dPrevC[k] = 0;
        }

        for (int k = 0; k < FHiddenSize; ++k) {
            d0[k] = ClipValue(dH[k] * TanhC[k] * TActivation::Derivative(O[k], atSigmoid), ClipVal);
            dCTotal[k] = ClipValue(dH[k] * O[k] * (1 - TanhC[k] * TanhC[k]) + dC[k], ClipVal);
            dF[k] = ClipValue(dCTotal[k] * PrevC[k] * TActivation::Derivative(F[k], atSigmoid), ClipVal);
            dI[k] = ClipValue(dCTotal[k] * CTilde[k] * TActivation::Derivative(I[k], atSigmoid), ClipVal);
            dCTilde[k] = ClipValue(dCTotal[k] * I[k] * TActivation::Derivative(CTilde[k], atTanh), ClipVal);
            dPrevC[k] = dCTotal[k] * F[k];
        }

        for (int k = 0; k < FHiddenSize; ++k) {
            for (int j = 0; j < ConcatSize; ++j) {
                dWf[k][j] += dF[k] * Concat[j];
                dWi[k][j] += dI[k] * Concat[j];
                dWc[k][j] += dCTilde[k] * Concat[j];
                dWo[k][j] += d0[k] * Concat[j];

                if (j < FInputSize) {
                    dInput[j] += Wf[k][j] * dF[k] + Wi[k][j] * dI[k] +
                                Wc[k][j] * dCTilde[k] + Wo[k][j] * d0[k];
                } else {
                    dPrevH[j - FInputSize] += Wf[k][j] * dF[k] + Wi[k][j] * dI[k] +
                                             Wc[k][j] * dCTilde[k] + Wo[k][j] * d0[k];
                }
            }
            dBf[k] += dF[k];
            dBi[k] += dI[k];
            dBc[k] += dCTilde[k];
            dBo[k] += d0[k];
        }
    }

    void ApplyGradients(double LR, double ClipVal) {
        int ConcatSize = FInputSize + FHiddenSize;
        for (int k = 0; k < FHiddenSize; ++k) {
            for (int j = 0; j < ConcatSize; ++j) {
                Wf[k][j] -= LR * ClipValue(dWf[k][j], ClipVal);
                Wi[k][j] -= LR * ClipValue(dWi[k][j], ClipVal);
                Wc[k][j] -= LR * ClipValue(dWc[k][j], ClipVal);
                Wo[k][j] -= LR * ClipValue(dWo[k][j], ClipVal);
                dWf[k][j] = 0;
                dWi[k][j] = 0;
                dWc[k][j] = 0;
                dWo[k][j] = 0;
            }
            Bf[k] -= LR * ClipValue(dBf[k], ClipVal);
            Bi[k] -= LR * ClipValue(dBi[k], ClipVal);
            Bc[k] -= LR * ClipValue(dBc[k], ClipVal);
            Bo[k] -= LR * ClipValue(dBo[k], ClipVal);
            dBf[k] = 0;
            dBi[k] = 0;
            dBc[k] = 0;
            dBo[k] = 0;
        }
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

    int GetHiddenSize() const { return FHiddenSize; }
};

// ========== GRU Cell ==========
class TGRUCell {
private:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;

public:
    TDArray2D Wz, Wr, Wh;
    DArray Bz, Br, Bh;
    TDArray2D dWz, dWr, dWh;
    DArray dBz, dBr, dBh;

    TGRUCell(int InputSize, int HiddenSize, TActivationType Activation)
        : FInputSize(InputSize), FHiddenSize(HiddenSize), FActivation(Activation) {
        int ConcatSize = InputSize + HiddenSize;
        double Scale = std::sqrt(2.0 / ConcatSize);

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
    }

    void Forward(const DArray& Input, const DArray& PrevH, DArray& H,
                 DArray& Z, DArray& R, DArray& HTilde) {
        DArray Concat = ConcatArrays(Input, PrevH);
        
        H.resize(FHiddenSize);
        Z.resize(FHiddenSize);
        R.resize(FHiddenSize);
        HTilde.resize(FHiddenSize);

        for (int k = 0; k < FHiddenSize; ++k) {
            double SumZ = Bz[k], SumR = Br[k];
            for (size_t j = 0; j < Concat.size(); ++j) {
                SumZ += Wz[k][j] * Concat[j];
                SumR += Wr[k][j] * Concat[j];
            }
            Z[k] = TActivation::Apply(SumZ, atSigmoid);
            R[k] = TActivation::Apply(SumR, atSigmoid);
        }

        DArray ConcatR(FInputSize + FHiddenSize);
        for (int k = 0; k < FInputSize; ++k) {
            ConcatR[k] = Input[k];
        }
        for (int k = 0; k < FHiddenSize; ++k) {
            ConcatR[FInputSize + k] = R[k] * PrevH[k];
        }

        for (int k = 0; k < FHiddenSize; ++k) {
            double SumH = Bh[k];
            for (size_t j = 0; j < ConcatR.size(); ++j) {
                SumH += Wh[k][j] * ConcatR[j];
            }
            HTilde[k] = TActivation::Apply(SumH, atTanh);
            H[k] = (1 - Z[k]) * PrevH[k] + Z[k] * HTilde[k];
        }
    }

    void Backward(const DArray& dH, const DArray& H, const DArray& Z, const DArray& R,
                  const DArray& HTilde, const DArray& PrevH, const DArray& Input,
                  double ClipVal, DArray& dInput, DArray& dPrevH) {
        DArray Concat = ConcatArrays(Input, PrevH);
        int ConcatSize = Concat.size();

        DArray ConcatR(ConcatSize);
        for (int k = 0; k < FInputSize; ++k) {
            ConcatR[k] = Input[k];
        }
        for (int k = 0; k < FHiddenSize; ++k) {
            ConcatR[FInputSize + k] = R[k] * PrevH[k];
        }

        DArray dZ(FHiddenSize), dR(FHiddenSize), dHTilde(FHiddenSize);
        dInput.resize(FInputSize);
        dPrevH.resize(FHiddenSize);

        for (int k = 0; k < FInputSize; ++k) dInput[k] = 0;
        for (int k = 0; k < FHiddenSize; ++k) dPrevH[k] = dH[k] * (1 - Z[k]);

        for (int k = 0; k < FHiddenSize; ++k) {
            dHTilde[k] = ClipValue(dH[k] * Z[k] * TActivation::Derivative(HTilde[k], atTanh), ClipVal);
            dZ[k] = ClipValue(dH[k] * (HTilde[k] - PrevH[k]) * TActivation::Derivative(Z[k], atSigmoid), ClipVal);
        }

        for (int k = 0; k < FHiddenSize; ++k) {
            for (int j = 0; j < ConcatSize; ++j) {
                dWh[k][j] += dHTilde[k] * ConcatR[j];
                if (j < FInputSize) {
                    dInput[j] += Wh[k][j] * dHTilde[k];
                } else {
                    dR[j - FInputSize] = (dR[j - FInputSize] + Wh[k][j] * dHTilde[k] * PrevH[j - FInputSize]);
                    dPrevH[j - FInputSize] += Wh[k][j] * dHTilde[k] * R[j - FInputSize];
                }
            }
            dBh[k] += dHTilde[k];
        }

        for (int k = 0; k < FHiddenSize; ++k) {
            dR[k] = ClipValue(dR[k] * TActivation::Derivative(R[k], atSigmoid), ClipVal);
        }

        for (int k = 0; k < FHiddenSize; ++k) {
            for (int j = 0; j < ConcatSize; ++j) {
                dWz[k][j] += dZ[k] * Concat[j];
                dWr[k][j] += dR[k] * Concat[j];
                if (j < FInputSize) {
                    dInput[j] += Wz[k][j] * dZ[k] + Wr[k][j] * dR[k];
                } else {
                    dPrevH[j - FInputSize] += Wz[k][j] * dZ[k] + Wr[k][j] * dR[k];
                }
            }
            dBz[k] += dZ[k];
            dBr[k] += dR[k];
        }
    }

    void ApplyGradients(double LR, double ClipVal) {
        int ConcatSize = FInputSize + FHiddenSize;
        for (int k = 0; k < FHiddenSize; ++k) {
            for (int j = 0; j < ConcatSize; ++j) {
                Wz[k][j] -= LR * ClipValue(dWz[k][j], ClipVal);
                Wr[k][j] -= LR * ClipValue(dWr[k][j], ClipVal);
                Wh[k][j] -= LR * ClipValue(dWh[k][j], ClipVal);
                dWz[k][j] = 0;
                dWr[k][j] = 0;
                dWh[k][j] = 0;
            }
            Bz[k] -= LR * ClipValue(dBz[k], ClipVal);
            Br[k] -= LR * ClipValue(dBr[k], ClipVal);
            Bh[k] -= LR * ClipValue(dBh[k], ClipVal);
            dBz[k] = 0;
            dBr[k] = 0;
            dBh[k] = 0;
        }
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

    int GetHiddenSize() const { return FHiddenSize; }
};

// ========== Output Layer ==========
class TOutputLayer {
private:
    int FInputSize, FOutputSize;
    TActivationType FActivation;

public:
    TDArray2D W;
    DArray B;
    TDArray2D dW;
    DArray dB;

    TOutputLayer(int InputSize, int OutputSize, TActivationType Activation)
        : FInputSize(InputSize), FOutputSize(OutputSize), FActivation(Activation) {
        double Scale = std::sqrt(2.0 / InputSize);
        InitMatrix(W, OutputSize, InputSize, Scale);
        ZeroArray(B, OutputSize);
        ZeroMatrix(dW, OutputSize, InputSize);
        ZeroArray(dB, OutputSize);
    }

    void Forward(const DArray& Input, DArray& Output, DArray& Pre) {
        Pre.resize(FOutputSize);
        Output.resize(FOutputSize);
        
        for (int i = 0; i < FOutputSize; ++i) {
            double Sum = B[i];
            for (int j = 0; j < FInputSize; ++j) {
                Sum += W[i][j] * Input[j];
            }
            Pre[i] = Sum;
        }

        if (FActivation == atLinear) {
            for (int i = 0; i < FOutputSize; ++i) {
                Output[i] = Pre[i];
            }
        } else {
            for (int i = 0; i < FOutputSize; ++i) {
                Output[i] = TActivation::Apply(Pre[i], FActivation);
            }
        }
    }

    void Backward(const DArray& d0ut, const DArray& Output, const DArray& Pre,
                  const DArray& Input, double ClipVal, DArray& dInput) {
        DArray dPre(FOutputSize);
        dInput.resize(FInputSize);
        
        for (int j = 0; j < FInputSize; ++j) dInput[j] = 0;

        for (int i = 0; i < FOutputSize; ++i) {
            dPre[i] = ClipValue(d0ut[i] * TActivation::Derivative(Output[i], FActivation), ClipVal);
        }

        for (int i = 0; i < FOutputSize; ++i) {
            for (int j = 0; j < FInputSize; ++j) {
                dW[i][j] += dPre[i] * Input[j];
                dInput[j] += W[i][j] * dPre[i];
            }
            dB[i] += dPre[i];
        }
    }

    void ApplyGradients(double LR, double ClipVal) {
        for (int i = 0; i < FOutputSize; ++i) {
            for (int j = 0; j < FInputSize; ++j) {
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
        default: return "sigmoid";
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
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "lstm") return ctLSTM;
    if (lower == "gru") return ctGRU;
    return ctSimpleRNN;
}

TActivationType ParseActivation(const std::string& s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "tanh") return atTanh;
    if (lower == "relu") return atReLU;
    if (lower == "linear") return atLinear;
    return atSigmoid;
}

TLossType ParseLoss(const std::string& s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "crossentropy") return ltCrossEntropy;
    return ltMSE;
}

void ParseIntArrayHelper(const std::string& s, TIntArray& result) {
    result.clear();
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        result.push_back(std::stoi(token));
    }
}

void ParseDoubleArrayHelper(const std::string& s, DArray& result) {
    result.clear();
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        result.push_back(std::stod(token));
    }
}

void LoadDataFromCSV(const std::string& Filename, TDArray2D& Inputs, TDArray2D& Targets) {
    Inputs.clear();
    Targets.clear();
    
    std::ifstream file(Filename);
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        DArray InputsArr, TargetsArr;
        std::stringstream ss(line);
        std::string token;
        
        std::vector<double> tokens;
        while (std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);
            tokens.push_back(std::stod(token));
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

void SplitData(const TDArray2D& Inputs, const TDArray2D& Targets, double ValSplit, TDataSplit& Split) {
    size_t N = Inputs.size();
    size_t ValCount = static_cast<size_t>(N * ValSplit);
    size_t TrainCount = N - ValCount;

    TIntArray Indices(N);
    for (size_t i = 0; i < N; ++i) {
        Indices[i] = i;
    }

    for (int i = N - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        std::swap(Indices[i], Indices[j]);
    }

    Split.TrainInputs.resize(TrainCount);
    Split.TrainTargets.resize(TrainCount);
    Split.ValInputs.resize(ValCount);
    Split.ValTargets.resize(ValCount);

    for (size_t i = 0; i < TrainCount; ++i) {
        Split.TrainInputs[i] = Inputs[Indices[i]];
        Split.TrainTargets[i] = Targets[Indices[i]];
    }

    for (size_t i = 0; i < ValCount; ++i) {
        Split.ValInputs[i] = Inputs[Indices[TrainCount + i]];
        Split.ValTargets[i] = Targets[Indices[TrainCount + i]];
    }
}

static std::string ExtractJSONValue(const std::string& json, const std::string& key) {
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
     
     // Handle string values
     if (startPos < json.length() && json[startPos] == '"') {
         size_t quotePos1 = startPos;
         size_t quotePos2 = json.find('"', quotePos1 + 1);
         if (quotePos2 != std::string::npos) {
             return json.substr(quotePos1 + 1, quotePos2 - quotePos1 - 1);
         }
         return "";
     }
     
     // Handle arrays
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
     
     // Handle numeric values
     size_t endPos = json.find(',', startPos);
     if (endPos == std::string::npos) endPos = json.find('}', startPos);
     if (endPos == std::string::npos) endPos = json.find(']', startPos);
     
     std::string result = json.substr(startPos, endPos - startPos);
     size_t end = result.find_last_not_of(" \t\n\r");
     if (end != std::string::npos) {
         result = result.substr(0, end + 1);
     }
     return result;
 }

// ========== Main RNN Class ==========
class TRNN {
private:
    int FInputSize, FOutputSize;
    TIntArray FHiddenSizes;
    TCellType FCellType;
    TActivationType FActivation, FOutputActivation;
    TLossType FLossType;
    double FLearningRate, FGradientClip;
    int FBPTTSteps;

    std::vector<TSimpleRNNCell*> FSimpleCells;
    std::vector<TLSTMCell*> FLSTMCells;
    std::vector<TGRUCell*> FGRUCells;
    TOutputLayer* FOutputLayer;

public:
    TRNN(int InputSize, const TIntArray& HiddenSizes, int OutputSize, TCellType CellType,
         TActivationType Activation, TActivationType OutputActivation, TLossType LossType,
         double LearningRate, double GradientClip, int BPTTSteps)
        : FInputSize(InputSize), FOutputSize(OutputSize), FCellType(CellType),
          FActivation(Activation), FOutputActivation(OutputActivation),
          FLossType(LossType), FLearningRate(LearningRate), FGradientClip(GradientClip),
          FBPTTSteps(BPTTSteps) {
        
        FHiddenSizes = HiddenSizes;
        
        int PrevSize = InputSize;
        switch (CellType) {
            case ctSimpleRNN:
                for (size_t i = 0; i < HiddenSizes.size(); ++i) {
                    FSimpleCells.push_back(new TSimpleRNNCell(PrevSize, HiddenSizes[i], Activation));
                    PrevSize = HiddenSizes[i];
                }
                break;
            case ctLSTM:
                for (size_t i = 0; i < HiddenSizes.size(); ++i) {
                    FLSTMCells.push_back(new TLSTMCell(PrevSize, HiddenSizes[i], Activation));
                    PrevSize = HiddenSizes[i];
                }
                break;
            case ctGRU:
                for (size_t i = 0; i < HiddenSizes.size(); ++i) {
                    FGRUCells.push_back(new TGRUCell(PrevSize, HiddenSizes[i], Activation));
                    PrevSize = HiddenSizes[i];
                }
                break;
        }

        FOutputLayer = new TOutputLayer(PrevSize, OutputSize, OutputActivation);
    }

    ~TRNN() {
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
        for (size_t i = 0; i < FHiddenSizes.size(); ++i) {
            Result[i].resize(2);
            ZeroArray(Result[i][0], FHiddenSizes[i]);
            ZeroArray(Result[i][1], FHiddenSizes[i]);
        }
        return Result;
    }

    TDArray2D ForwardSequence(const TDArray2D& Inputs, std::vector<TTimeStepCache>& Caches,
                              TDArray3D& States) {
        TDArray2D Result(Inputs.size());
        TDArray3D NewStates = InitHiddenStates();

        for (size_t t = 0; t < Inputs.size(); ++t) {
            DArray X = Inputs[t];
            Caches[t].Input = X;
            Caches[t].LayerInputs.resize(FHiddenSizes.size() + 1);

            for (size_t layer = 0; layer < FHiddenSizes.size(); ++layer) {
                Caches[t].LayerInputs[layer] = X;
                
                DArray H, C, PreH, F, I, CTilde, O, TanhC, Z, R, HTilde;
                
                switch (FCellType) {
                    case ctSimpleRNN:
                        FSimpleCells[layer]->Forward(X, States[layer][0], H, PreH);
                        NewStates[layer][0] = H;
                        Caches[t].H = H;
                        Caches[t].PreH = PreH;
                        break;
                    case ctLSTM:
                        FLSTMCells[layer]->Forward(X, States[layer][0], States[layer][1], H, C, F, I, CTilde, O, TanhC);
                        NewStates[layer][0] = H;
                        NewStates[layer][1] = C;
                        Caches[t].H = H;
                        Caches[t].C = C;
                        Caches[t].F = F;
                        Caches[t].I = I;
                        Caches[t].CTilde = CTilde;
                        Caches[t].O = O;
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

            Caches[t].LayerInputs[FHiddenSizes.size()] = X;
            DArray OutVal, OutPre;
            FOutputLayer->Forward(X, OutVal, OutPre);
            Caches[t].OutVal = OutVal;
            Caches[t].OutPre = OutPre;
            Result[t] = OutVal;

            States = NewStates;
        }

        return Result;
    }

    double BackwardSequence(const TDArray2D& Targets, const std::vector<TTimeStepCache>& Caches,
                            const TDArray3D& States) {
        int T_len = Targets.size();
        int BPTTLimit = (FBPTTSteps > 0) ? FBPTTSteps : T_len;

        double TotalLoss = 0;

        TDArray2D dStatesH(FHiddenSizes.size());
        TDArray2D dStatesC(FHiddenSizes.size());
        for (size_t layer = 0; layer < FHiddenSizes.size(); ++layer) {
            ZeroArray(dStatesH[layer], FHiddenSizes[layer]);
            ZeroArray(dStatesC[layer], FHiddenSizes[layer]);
        }

        for (int t = T_len - 1; t >= std::max(0, T_len - BPTTLimit); --t) {
            TotalLoss += TLoss::Compute(Caches[t].OutVal, Targets[t], FLossType);
            DArray Grad;
            TLoss::Gradient(Caches[t].OutVal, Targets[t], FLossType, Grad);

            DArray dH;
            FOutputLayer->Backward(Grad, Caches[t].OutVal, Caches[t].OutPre,
                                  Caches[t].LayerInputs[FHiddenSizes.size()], FGradientClip, dH);

            for (int layer = static_cast<int>(FHiddenSizes.size()) - 1; layer >= 0; --layer) {
                DArray d0ut(FHiddenSizes[layer]);
                for (int k = 0; k < FHiddenSizes[layer]; ++k) {
                    d0ut[k] = dH[k] + dStatesH[layer][k];
                }

                DArray PrevH;
                if (t > 0) {
                    PrevH = Caches[t-1].H;
                } else {
                    ZeroArray(PrevH, FHiddenSizes[layer]);
                }

                DArray dInput, dPrevH, dPrevC;
                
                switch (FCellType) {
                    case ctSimpleRNN:
                        FSimpleCells[layer]->Backward(d0ut, Caches[t].H, Caches[t].PreH, PrevH,
                                                       Caches[t].LayerInputs[layer], FGradientClip, dInput, dPrevH);
                        dStatesH[layer] = dPrevH;
                        break;
                    case ctLSTM: {
                        DArray PrevC;
                        if (t > 0) {
                            PrevC = Caches[t-1].C;
                        } else {
                            ZeroArray(PrevC, FHiddenSizes[layer]);
                        }

                        DArray dC(FHiddenSizes[layer]);
                        for (int k = 0; k < FHiddenSizes[layer]; ++k) {
                            dC[k] = dStatesC[layer][k];
                        }

                        FLSTMCells[layer]->Backward(d0ut, dC, Caches[t].H, Caches[t].C,
                                                   Caches[t].F, Caches[t].I, Caches[t].CTilde,
                                                   Caches[t].O, Caches[t].TanhC,
                                                   PrevH, PrevC, Caches[t].LayerInputs[layer],
                                                   FGradientClip, dInput, dPrevH, dPrevC);
                        dStatesH[layer] = dPrevH;
                        dStatesC[layer] = dPrevC;
                        break;
                    }
                    case ctGRU:
                        FGRUCells[layer]->Backward(d0ut, Caches[t].H, Caches[t].Z, Caches[t].R,
                                                   Caches[t].HTilde, PrevH, Caches[t].LayerInputs[layer],
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
        std::vector<TTimeStepCache> Caches(Inputs.size());
        TDArray3D States = InitHiddenStates();
        ForwardSequence(Inputs, Caches, States);
        double Loss = BackwardSequence(Targets, Caches, States);
        ApplyGradients();
        return Loss;
    }

    double TrainBatch(const TDArray3D& BatchInputs, const TDArray3D& BatchTargets) {
        ResetGradients();
        double BatchLoss = 0;

        for (size_t b = 0; b < BatchInputs.size(); ++b) {
            std::vector<TTimeStepCache> Caches(BatchInputs[b].size());
            TDArray3D States = InitHiddenStates();
            ForwardSequence(BatchInputs[b], Caches, States);
            BatchLoss += BackwardSequence(BatchTargets[b], Caches, States);
        }

        ApplyGradients();
        return BatchLoss / BatchInputs.size();
    }

    TDArray2D Predict(const TDArray2D& Inputs) {
        std::vector<TTimeStepCache> Caches(Inputs.size());
        TDArray3D States = InitHiddenStates();
        return ForwardSequence(Inputs, Caches, States);
    }

    double ComputeLoss(const TDArray2D& Inputs, const TDArray2D& Targets) {
        TDArray2D Outputs = Predict(Inputs);
        double Result = 0;
        for (size_t t = 0; t < Outputs.size(); ++t) {
            Result += TLoss::Compute(Outputs[t], Targets[t], FLossType);
        }
        return Result / Outputs.size();
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

    void SaveModelToJSON(const std::string& Filename) {
        std::ofstream file(Filename);
        
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
        file << std::fixed << std::setprecision(17);
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
        std::cout << "Model saved to JSON: " << Filename << "\n";
    }

    void LoadModelFromJSON(const std::string& Filename) {
        std::ifstream file(Filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << Filename << "\n";
            return;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string Content = buffer.str();
        file.close();
        
        // Extract model parameters from JSON
        std::string inputStr = ExtractJSONValue(Content, "input_size");
        if (inputStr.empty()) { std::cerr << "Error: Could not parse input_size from JSON\n"; return; }
        int inputSize = std::stoi(inputStr);
        
        std::string outputStr = ExtractJSONValue(Content, "output_size");
        if (outputStr.empty()) { std::cerr << "Error: Could not parse output_size from JSON\n"; return; }
        int outputSize = std::stoi(outputStr);
        
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
        
        // Parse hidden sizes (array notation [5,3])
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
        } else if (!hiddenStr.empty()) {
            hiddenSizes.push_back(std::stoi(hiddenStr));
        }
        
        // Reinitialize the model with proper dimensions
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
        
        // Clear old cells
        for (auto cell : FSimpleCells) delete cell;
        for (auto cell : FLSTMCells) delete cell;
        for (auto cell : FGRUCells) delete cell;
        FSimpleCells.clear();
        FLSTMCells.clear();
        FGRUCells.clear();
        if (FOutputLayer) delete FOutputLayer;
        
        // Create new cells with proper dimensions
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
        
        std::cout << "Model loaded from JSON: " << Filename << "\n";
    }
};

// ========== Utility Functions ==========
void PrintUsage() {
    std::cout << "RNN (OpenCL Accelerated)\n\n";
    std::cout << "Commands:\n";
    std::cout << "  create   Create a new RNN model and save to JSON\n";
    std::cout << "  train    Train an existing model with data from JSON\n";
    std::cout << "  predict  Make predictions with a trained model from JSON\n";
    std::cout << "  info     Display model information from JSON\n";
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
    std::cout << "Options:\n";
    std::cout << "  --input       Input size (create) or input values (predict)\n";
    std::cout << "  --hidden      Hidden layer sizes (comma-separated)\n";
    std::cout << "  --output      Output size\n";
    std::cout << "  --cell        Cell type: simplernn, lstm, gru (default: lstm)\n";
    std::cout << "  --hidden-act  Hidden activation: sigmoid, tanh, relu, linear (default: tanh)\n";
    std::cout << "  --output-act  Output activation: sigmoid, tanh, relu, linear (default: linear)\n";
    std::cout << "  --loss        Loss function: mse, crossentropy (default: mse)\n";
    std::cout << "  --lr          Learning rate (default: 0.01)\n";
    std::cout << "  --clip        Gradient clipping value (default: 5.0)\n";
    std::cout << "  --bptt        BPTT steps (default: 0 = full sequence)\n";
    std::cout << "  --epochs      Training epochs (default: 100)\n";
    std::cout << "  --batch       Batch size (default: 1)\n";
    std::cout << "  --model       Model file path\n";
    std::cout << "  --data        Data file path\n";
    std::cout << "  --save        Save file path\n";
    std::cout << "  --verbose     Verbose output\n";
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

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--verbose") {
            verbose = true;
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

        TRNN* RNNModel = new TRNN(inputSize, hiddenSizes, outputSize, cellType,
                                   hiddenAct, outputAct, lossType, learningRate,
                                   gradientClip, bpttSteps);

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

        RNNModel->SaveModelToJSON(saveFile);
        std::cout << "Model saved to: " << saveFile << "\n";

        delete RNNModel;
    }
    else if (Command == cmdTrain) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required\n"; return 1; }
        if (dataFile.empty()) { std::cerr << "Error: --data is required\n"; return 1; }
        if (saveFile.empty()) { std::cerr << "Error: --save is required\n"; return 1; }

        std::cout << "Loading model from JSON: " << modelFile << "\n";
        TRNN* RNNModel = new TRNN(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
        RNNModel->LoadModelFromJSON(modelFile);
        std::cout << "Model loaded successfully.\n";

        std::cout << "Loading training data from: " << dataFile << "\n";
        TDArray2D Inputs, Targets;
        LoadDataFromCSV(dataFile, Inputs, Targets);

        if (Inputs.empty()) {
            std::cerr << "Error: No data loaded from CSV file\n";
            delete RNNModel;
            return 1;
        }

        std::cout << "Loaded " << Inputs.size() << " timesteps of training data\n";
        std::cout << "Starting training for " << epochs << " epochs...\n";

        for (int Epoch = 1; Epoch <= epochs; ++Epoch) {
            double TrainLoss = RNNModel->TrainSequence(Inputs, Targets);

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
        RNNModel->SaveModelToJSON(saveFile);

        delete RNNModel;
    }
    else if (Command == cmdPredict) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required\n"; return 1; }
        if (inputValues.empty()) { std::cerr << "Error: --input is required\n"; return 1; }

        TRNN* RNNModel = new TRNN(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
        RNNModel->LoadModelFromJSON(modelFile);
        if (RNNModel == nullptr) { std::cerr << "Error: Failed to load model\n"; return 1; }

        TDArray2D Inputs(1);
        Inputs[0] = inputValues;

        TDArray2D Predictions = RNNModel->Predict(Inputs);

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

        delete RNNModel;
    }
    else if (Command == cmdInfo) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required\n"; return 1; }
        std::cout << "Loading model from JSON: " << modelFile << "\n";
        TRNN* RNNModel = new TRNN(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
        RNNModel->LoadModelFromJSON(modelFile);
        
        std::ifstream file(modelFile);
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string Content = buffer.str();
        file.close();
        
        std::cout << "Model Information:\n";
        std::cout << "  Input size: " << ExtractJSONValue(Content, "input_size") << "\n";
        std::cout << "  Output size: " << ExtractJSONValue(Content, "output_size") << "\n";
        std::cout << "  Hidden sizes: " << ExtractJSONValue(Content, "hidden_sizes") << "\n";
        std::cout << "  Cell type: " << ExtractJSONValue(Content, "cell_type") << "\n";
        std::cout << "  Hidden activation: " << ExtractJSONValue(Content, "hidden_activation") << "\n";
        std::cout << "  Output activation: " << ExtractJSONValue(Content, "output_activation") << "\n";
        std::cout << "  Loss function: " << ExtractJSONValue(Content, "loss_type") << "\n";
        std::cout << "  Learning rate: " << ExtractJSONValue(Content, "learning_rate") << "\n";
        std::cout << "  Gradient clip: " << ExtractJSONValue(Content, "gradient_clip") << "\n";
        std::cout << "  BPTT steps: " << ExtractJSONValue(Content, "bptt_steps") << "\n";
        delete RNNModel;
    }

    return 0;
}
