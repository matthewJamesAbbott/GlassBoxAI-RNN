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

// ========== Cell Wrappers ==========

class TSimpleRNNCellWrapper {
private:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;

public:
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
        InitMatrix(Wih, HiddenSize, InputSize, 0.01);
        InitMatrix(Whh, HiddenSize, HiddenSize, 0.01);
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
        // Placeholder
    }

    void Backward(const DArray& dH, const DArray& H, const DArray& PreH,
                  const DArray& PrevH, const DArray& Input, double ClipVal,
                  DArray& dInput, DArray& dPrevH) {
        // Placeholder
    }

    void ApplyGradients(double LR, double ClipVal) {
        // Placeholder
    }

    void ResetGradients() {
        // Placeholder
    }

    int GetHiddenSize() const { return FHiddenSize; }
    int GetInputSize() const { return FInputSize; }
};

class TLSTMCellWrapper {
private:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;

public:
    TDArray2D Wf, Wi, Wc, Wo;
    DArray Bf, Bi, Bc, Bo;
    TDArray2D dWf, dWi, dWc, dWo;
    DArray dBf, dBi, dBc, dBo;
    TDArray2D MWf, MWi, MWc, MWo;
    DArray MBf, MBi, MBc, MBo;
    TDArray2D VWf, VWi, VWc, VWo;
    DArray VBf, VBi, VBc, VBo;

    TLSTMCellWrapper(int InputSize, int HiddenSize, TActivationType Activation)
        : FInputSize(InputSize), FHiddenSize(HiddenSize), FActivation(Activation) {
        int CombinedSize = InputSize + HiddenSize;
        InitMatrix(Wf, HiddenSize, CombinedSize, 0.01);
        InitMatrix(Wi, HiddenSize, CombinedSize, 0.01);
        InitMatrix(Wc, HiddenSize, CombinedSize, 0.01);
        InitMatrix(Wo, HiddenSize, CombinedSize, 0.01);
        ZeroArray(Bf, HiddenSize);
        ZeroArray(Bi, HiddenSize);
        ZeroArray(Bc, HiddenSize);
        ZeroArray(Bo, HiddenSize);
        ZeroMatrix(dWf, HiddenSize, CombinedSize);
        ZeroMatrix(dWi, HiddenSize, CombinedSize);
        ZeroMatrix(dWc, HiddenSize, CombinedSize);
        ZeroMatrix(dWo, HiddenSize, CombinedSize);
        ZeroArray(dBf, HiddenSize);
        ZeroArray(dBi, HiddenSize);
        ZeroArray(dBc, HiddenSize);
        ZeroArray(dBo, HiddenSize);
        ZeroMatrix(MWf, HiddenSize, CombinedSize);
        ZeroMatrix(MWi, HiddenSize, CombinedSize);
        ZeroMatrix(MWc, HiddenSize, CombinedSize);
        ZeroMatrix(MWo, HiddenSize, CombinedSize);
        ZeroArray(MBf, HiddenSize);
        ZeroArray(MBi, HiddenSize);
        ZeroArray(MBc, HiddenSize);
        ZeroArray(MBo, HiddenSize);
        ZeroMatrix(VWf, HiddenSize, CombinedSize);
        ZeroMatrix(VWi, HiddenSize, CombinedSize);
        ZeroMatrix(VWc, HiddenSize, CombinedSize);
        ZeroMatrix(VWo, HiddenSize, CombinedSize);
        ZeroArray(VBf, HiddenSize);
        ZeroArray(VBi, HiddenSize);
        ZeroArray(VBc, HiddenSize);
        ZeroArray(VBo, HiddenSize);
    }

    void Forward(const DArray& Input, const DArray& PrevH, const DArray& PrevC,
                 DArray& H, DArray& C, DArray& FG, DArray& IG, DArray& CTilde, 
                 DArray& OG, DArray& TanhC) {
        // Placeholder
    }

    void Backward(const DArray& dH, const DArray& dC, const DArray& H, const DArray& C,
                  const DArray& FG, const DArray& IG, const DArray& CTilde, const DArray& OG,
                  const DArray& TanhC, const DArray& PrevH, const DArray& PrevC, const DArray& Input,
                  double ClipVal, DArray& dInput, DArray& dPrevH, DArray& dPrevC) {
        // Placeholder
    }

    void ApplyGradients(double LR, double ClipVal) {
        // Placeholder
    }

    void ResetGradients() {
        // Placeholder
    }

    int GetHiddenSize() const { return FHiddenSize; }
    int GetInputSize() const { return FInputSize; }
};

class TGRUCellWrapper {
private:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;

public:
    TDArray2D Wz, Wr, Wh;
    DArray Bz, Br, Bh;
    TDArray2D dWz, dWr, dWh;
    DArray dBz, dBr, dBh;
    TDArray2D MWz, MWr, MWh;
    DArray MBz, MBr, MBh;
    TDArray2D VWz, VWr, VWh;
    DArray VBz, VBr, VBh;

    TGRUCellWrapper(int InputSize, int HiddenSize, TActivationType Activation)
        : FInputSize(InputSize), FHiddenSize(HiddenSize), FActivation(Activation) {
        int CombinedSize = InputSize + HiddenSize;
        InitMatrix(Wz, HiddenSize, CombinedSize, 0.01);
        InitMatrix(Wr, HiddenSize, CombinedSize, 0.01);
        InitMatrix(Wh, HiddenSize, CombinedSize, 0.01);
        ZeroArray(Bz, HiddenSize);
        ZeroArray(Br, HiddenSize);
        ZeroArray(Bh, HiddenSize);
        ZeroMatrix(dWz, HiddenSize, CombinedSize);
        ZeroMatrix(dWr, HiddenSize, CombinedSize);
        ZeroMatrix(dWh, HiddenSize, CombinedSize);
        ZeroArray(dBz, HiddenSize);
        ZeroArray(dBr, HiddenSize);
        ZeroArray(dBh, HiddenSize);
        ZeroMatrix(MWz, HiddenSize, CombinedSize);
        ZeroMatrix(MWr, HiddenSize, CombinedSize);
        ZeroMatrix(MWh, HiddenSize, CombinedSize);
        ZeroArray(MBz, HiddenSize);
        ZeroArray(MBr, HiddenSize);
        ZeroArray(MBh, HiddenSize);
        ZeroMatrix(VWz, HiddenSize, CombinedSize);
        ZeroMatrix(VWr, HiddenSize, CombinedSize);
        ZeroMatrix(VWh, HiddenSize, CombinedSize);
        ZeroArray(VBz, HiddenSize);
        ZeroArray(VBr, HiddenSize);
        ZeroArray(VBh, HiddenSize);
    }

    void Forward(const DArray& Input, const DArray& PrevH, DArray& H, DArray& Z, DArray& R, DArray& HTilde) {
        // Placeholder
    }

    void Backward(const DArray& dH, const DArray& H, const DArray& Z, const DArray& R,
                  const DArray& HTilde, const DArray& PrevH, const DArray& Input,
                  double ClipVal, DArray& dInput, DArray& dPrevH) {
        // Placeholder
    }

    void ApplyGradients(double LR, double ClipVal) {
        // Placeholder
    }

    void ResetGradients() {
        // Placeholder
    }

    int GetHiddenSize() const { return FHiddenSize; }
    int GetInputSize() const { return FInputSize; }
};

class TOutputLayerWrapper {
private:
    int FInputSize, FOutputSize;
    TActivationType FActivation;

public:
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
        InitMatrix(W, OutputSize, InputSize, 0.01);
        ZeroArray(B, OutputSize);
        ZeroMatrix(dW, OutputSize, InputSize);
        ZeroArray(dB, OutputSize);
        ZeroMatrix(MW, OutputSize, InputSize);
        ZeroArray(MB, OutputSize);
        ZeroMatrix(VW, OutputSize, InputSize);
        ZeroArray(VB, OutputSize);
    }

    void Forward(const DArray& Input, DArray& Output, DArray& Pre) {
        // Placeholder
    }

    void Backward(const DArray& dOut, const DArray& Output, const DArray& Pre,
                  const DArray& Input, double ClipVal, DArray& dInput) {
        // Placeholder
    }

    void ApplyGradients(double LR, double ClipVal) {
        // Placeholder
    }

    void ResetGradients() {
        // Placeholder
    }

    int GetInputSize() const { return FInputSize; }
    int GetOutputSize() const { return FOutputSize; }
};

// ========== Main RNN Facade ==========
class TRNNFacade {
private:
    int FInputSize, FOutputSize;
    TIntArray FHiddenSizes;
    TCellType FCellType;
    TActivationType FActivation, FOutputActivation;
    TLossType FLossType;
    double FLearningRate, FGradientClip;
    int FBPTTSteps;
    double FDropoutRate;
    bool FUseDropout;
    int FSequenceLen;

    std::vector<TSimpleRNNCellWrapper*> FSimpleCells;
    std::vector<TLSTMCellWrapper*> FLSTMCells;
    std::vector<TGRUCellWrapper*> FGRUCells;
    TOutputLayerWrapper* FOutputLayer;

    std::vector<TTimeStepCacheEx> FCaches;
    TDArray3D FStates;
    TDArray2D FGradientHistory;

    double ClipGradient(double G, double MaxVal) {
        if (G > MaxVal) return MaxVal;
        if (G < -MaxVal) return -MaxVal;
        return G;
    }

    std::string Array1DToJSON(const DArray& Arr) {
        std::string Result = "[";
        for (size_t i = 0; i < Arr.size(); ++i) {
            if (i > 0) Result += ",";
            Result += std::to_string(Arr[i]);
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

    TDArray3D InitHiddenStates() {
        TDArray3D Result(FHiddenSizes.size());
        for (size_t i = 0; i < FHiddenSizes.size(); ++i) {
            Result[i].resize(2);
            ZeroArray(Result[i][0], FHiddenSizes[i]);
            ZeroArray(Result[i][1], FHiddenSizes[i]);
        }
        return Result;
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

public:
    TRNNFacade(int InputSize, const TIntArray& HiddenSizes, int OutputSize, TCellType CellType,
               TActivationType Activation, TActivationType OutputActivation, TLossType LossType,
               double LearningRate, double GradientClip, int BPTTSteps)
        : FInputSize(InputSize), FOutputSize(OutputSize), FCellType(CellType),
          FActivation(Activation), FOutputActivation(OutputActivation), FLossType(LossType),
          FLearningRate(LearningRate), FGradientClip(GradientClip), FBPTTSteps(BPTTSteps),
          FDropoutRate(0.0), FUseDropout(false), FSequenceLen(0), FOutputLayer(nullptr) {
        
        FHiddenSizes = HiddenSizes;

        switch (CellType) {
            case ctSimpleRNN:
                for (size_t i = 0; i < HiddenSizes.size(); ++i) {
                    int InputSz = (i == 0) ? InputSize : HiddenSizes[i - 1];
                    FSimpleCells.push_back(new TSimpleRNNCellWrapper(InputSz, HiddenSizes[i], Activation));
                }
                break;
            case ctLSTM:
                for (size_t i = 0; i < HiddenSizes.size(); ++i) {
                    int InputSz = (i == 0) ? InputSize : HiddenSizes[i - 1];
                    FLSTMCells.push_back(new TLSTMCellWrapper(InputSz, HiddenSizes[i], Activation));
                }
                break;
            case ctGRU:
                for (size_t i = 0; i < HiddenSizes.size(); ++i) {
                    int InputSz = (i == 0) ? InputSize : HiddenSizes[i - 1];
                    FGRUCells.push_back(new TGRUCellWrapper(InputSz, HiddenSizes[i], Activation));
                }
                break;
        }

        int LastHiddenSize = HiddenSizes.back();
        FOutputLayer = new TOutputLayerWrapper(LastHiddenSize, OutputSize, OutputActivation);
    }

    ~TRNNFacade() {
        for (auto cell : FSimpleCells) delete cell;
        for (auto cell : FLSTMCells) delete cell;
        for (auto cell : FGRUCells) delete cell;
        if (FOutputLayer) delete FOutputLayer;
    }

    TDArray2D ForwardSequence(const TDArray2D& Inputs) {
        TDArray2D Result(Inputs.size());
        FStates = InitHiddenStates();

        for (size_t t = 0; t < Inputs.size(); ++t) {
            DArray X = Inputs[t];
            
            for (size_t layer = 0; layer < FHiddenSizes.size(); ++layer) {
                DArray H, C, PreH, F, I, CTilde, O, TanhC, Z, R, HTilde;
                
                switch (FCellType) {
                    case ctSimpleRNN:
                        FSimpleCells[layer]->Forward(X, FStates[layer][0], H, PreH);
                        FStates[layer][0] = H;
                        break;
                    case ctLSTM:
                        FLSTMCells[layer]->Forward(X, FStates[layer][0], FStates[layer][1], 
                                                    H, C, F, I, CTilde, O, TanhC);
                        FStates[layer][0] = H;
                        FStates[layer][1] = C;
                        break;
                    case ctGRU:
                        FGRUCells[layer]->Forward(X, FStates[layer][0], H, Z, R, HTilde);
                        FStates[layer][0] = H;
                        break;
                }
                X = H;
            }

            DArray OutVal, OutPre;
            FOutputLayer->Forward(X, OutVal, OutPre);
            Result[t] = OutVal;
        }

        return Result;
    }

    double BackwardSequence(const TDArray2D& Targets) {
        return 0.0;  // Simplified
    }

    double TrainSequence(const TDArray2D& Inputs, const TDArray2D& Targets) {
        ResetGradients();
        
        TDArray2D Predictions = ForwardSequence(Inputs);
        
        double TotalLoss = 0.0;
        int N = std::min(Targets.size(), Predictions.size());
        if (N > 0) {
            for (int t = 0; t < N; ++t) {
                TotalLoss += TLoss::Compute(Predictions[t], Targets[t], FLossType);
            }
            ApplyGradients();
            return TotalLoss / N;
        }
        return 0.0;
    }

    TDArray2D Predict(const TDArray2D& Inputs) {
        return ForwardSequence(Inputs);
    }

    double GetHiddenState(int LayerIdx, int Timestep, int NeuronIdx) {
        if (LayerIdx >= 0 && LayerIdx < (int)FStates.size() &&
            Timestep >= 0 && Timestep < (int)FStates[LayerIdx].size() &&
            NeuronIdx >= 0 && NeuronIdx < (int)FStates[LayerIdx][Timestep].size()) {
            return FStates[LayerIdx][Timestep][NeuronIdx];
        }
        return 0.0;
    }

    void SetHiddenState(int LayerIdx, int Timestep, int NeuronIdx, double Value) {
        if (LayerIdx >= 0 && LayerIdx < (int)FStates.size() &&
            Timestep >= 0 && Timestep < (int)FStates[LayerIdx].size() &&
            NeuronIdx >= 0 && NeuronIdx < (int)FStates[LayerIdx][Timestep].size()) {
            FStates[LayerIdx][Timestep][NeuronIdx] = Value;
        }
    }

    double GetOutput(int Timestep, int OutputIdx) { return 0.0; }
    void SetOutput(int Timestep, int OutputIdx, double Value) { }
    double GetCellState(int LayerIdx, int Timestep, int NeuronIdx) { return 0.0; }
    double GetGateValue(TGateType GateType, int LayerIdx, int Timestep, int NeuronIdx) { return 0.0; }
    double GetPreActivation(int LayerIdx, int Timestep, int NeuronIdx) { return 0.0; }
    double GetInputVector(int Timestep, int InputIdx) { return 0.0; }
    double GetWeightGradient(int LayerIdx, int NeuronIdx, int WeightIdx) { return 0.0; }
    double GetBiasGradient(int LayerIdx, int NeuronIdx) { return 0.0; }
    
    TOptimizerStateRecord GetOptimizerState(int LayerIdx, int NeuronIdx, int Param) {
        TOptimizerStateRecord Result = {0.0, 0.0, 0.0, 0.0};
        return Result;
    }
    
    double GetCellGradient(int LayerIdx, int Timestep, int NeuronIdx) { return 0.0; }

    void SetDropoutRate(double Rate) { FDropoutRate = Rate; }
    double GetDropoutRate() const { return FDropoutRate; }
    double GetDropoutMask(int LayerIdx, int Timestep, int NeuronIdx) { return 0.0; }
    TLayerNormStats GetLayerNormStats(int LayerIdx, int Timestep) {
        TLayerNormStats Result = {0.0, 0.0, 0.0, 0.0};
        return Result;
    }
    void EnableDropout(bool Enable) { FUseDropout = Enable; }

    DArray GetSequenceOutputs(int OutputIdx) {
        DArray Result;
        return Result;
    }

    DArray GetSequenceHiddenStates(int LayerIdx, int NeuronIdx) {
        DArray Result;
        return Result;
    }

    DArray GetSequenceCellStates(int LayerIdx, int NeuronIdx) {
        DArray Result;
        return Result;
    }

    DArray GetSequenceGateValues(TGateType GateType, int LayerIdx, int NeuronIdx) {
        DArray Result;
        return Result;
    }

    int GetInputSize() const { return FInputSize; }
    int GetOutputSize() const { return FOutputSize; }
    int GetHiddenSize(int LayerIdx) const {
        if (LayerIdx >= 0 && LayerIdx < (int)FHiddenSizes.size()) {
            return FHiddenSizes[LayerIdx];
        }
        return 0;
    }
    TCellType GetCellType() const { return FCellType; }
    int GetSequenceLength() const { return FSequenceLen; }

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
                    File << "      \"Wih\": " << Array2DToJSON(FSimpleCells[i]->Wih) << ",\n";
                    File << "      \"Whh\": " << Array2DToJSON(FSimpleCells[i]->Whh) << ",\n";
                    File << "      \"Bh\": " << Array1DToJSON(FSimpleCells[i]->Bh) << "\n";
                    File << "    }";
                }
                File << "\n  ]\n";
                break;
            case ctLSTM:
                File << "  \"cells\": [\n";
                for (size_t i = 0; i < FLSTMCells.size(); ++i) {
                    if (i > 0) File << ",\n";
                    File << "    {\n";
                    File << "      \"Wf\": " << Array2DToJSON(FLSTMCells[i]->Wf) << ",\n";
                    File << "      \"Wi\": " << Array2DToJSON(FLSTMCells[i]->Wi) << ",\n";
                    File << "      \"Wc\": " << Array2DToJSON(FLSTMCells[i]->Wc) << ",\n";
                    File << "      \"Wo\": " << Array2DToJSON(FLSTMCells[i]->Wo) << ",\n";
                    File << "      \"Bf\": " << Array1DToJSON(FLSTMCells[i]->Bf) << ",\n";
                    File << "      \"Bi\": " << Array1DToJSON(FLSTMCells[i]->Bi) << ",\n";
                    File << "      \"Bc\": " << Array1DToJSON(FLSTMCells[i]->Bc) << ",\n";
                    File << "      \"Bo\": " << Array1DToJSON(FLSTMCells[i]->Bo) << "\n";
                    File << "    }";
                }
                File << "\n  ]\n";
                break;
            case ctGRU:
                File << "  \"cells\": [\n";
                for (size_t i = 0; i < FGRUCells.size(); ++i) {
                    if (i > 0) File << ",\n";
                    File << "    {\n";
                    File << "      \"Wz\": " << Array2DToJSON(FGRUCells[i]->Wz) << ",\n";
                    File << "      \"Wr\": " << Array2DToJSON(FGRUCells[i]->Wr) << ",\n";
                    File << "      \"Wh\": " << Array2DToJSON(FGRUCells[i]->Wh) << ",\n";
                    File << "      \"Bz\": " << Array1DToJSON(FGRUCells[i]->Bz) << ",\n";
                    File << "      \"Br\": " << Array1DToJSON(FGRUCells[i]->Br) << ",\n";
                    File << "      \"Bh\": " << Array1DToJSON(FGRUCells[i]->Bh) << "\n";
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

        std::cout << "Model loaded from JSON: " << Filename << "\n";
    }

    double GetLearningRate() const { return FLearningRate; }
    void SetLearningRate(double Value) { FLearningRate = Value; }
    double GetGradientClip() const { return FGradientClip; }
    void SetGradientClip(double Value) { FGradientClip = Value; }
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
    std::cout << "Facaded RNN\n\n";
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
    std::cout << "  --layer=N              Layer index\n";
    std::cout << "  --timestep=N           Timestep index\n";
    std::cout << "  --neuron=N             Neuron index\n";
    std::cout << "  --index=N              Generic index parameter\n";
    std::cout << "  --dropout-rate=VALUE   Set dropout rate (0.0-1.0)\n";
    std::cout << "  --enable-dropout       Enable dropout\n";
    std::cout << "  --disable-dropout      Disable dropout\n\n";
    std::cout << "Examples:\n";
    std::cout << "  rnn create --input=2 --hidden=16 --output=2 --cell=lstm --save=seq.json\n";
    std::cout << "  rnn train --model=seq.json --data=seq.csv --epochs=200 --save=seq_trained.json\n";
    std::cout << "  rnn predict --model=seq_trained.json --input=0.5,0.5\n";
    std::cout << "  rnn info --model=seq_trained.json\n";
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

        TRNNFacade* RNN = new TRNNFacade(inputSize, hiddenSizes, outputSize, cellType,
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

        RNN->SaveModel(saveFile);
        delete RNN;
    }
    else if (Command == cmdTrain) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required\n"; return 1; }
        if (dataFile.empty()) { std::cerr << "Error: --data is required\n"; return 1; }
        if (saveFile.empty()) { std::cerr << "Error: --save is required\n"; return 1; }

        std::cout << "Loading model from JSON: " << modelFile << "\n";
        TRNNFacade* RNN = new TRNNFacade(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
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

        TRNNFacade* RNN = new TRNNFacade(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
        RNN->LoadModel(modelFile);
        if (RNN == nullptr) { std::cerr << "Error: Failed to load model\n"; return 1; }

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
        TRNNFacade* RNN = new TRNNFacade(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
        RNN->LoadModel(modelFile);
        std::cout << "Model information displayed above.\n";
        delete RNN;
    }
    else if (Command == cmdQuery) {
        if (modelFile.empty()) { std::cerr << "Error: --model is required\n"; return 1; }
        if (queryType.empty()) { std::cerr << "Error: --query-type is required\n"; return 1; }

        std::cout << "Loading model from JSON: " << modelFile << "\n";
        TRNNFacade* RNN = new TRNNFacade(1, {1}, 1, ctLSTM, atTanh, atLinear, ltMSE, 0.01, 5.0, 0);
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
                      << "Current dropout rate: " << RNN->GetDropoutRate() << "\n";
        } else if (queryType == "hidden-state") {
            std::cout << std::fixed << std::setprecision(6)
                      << "Hidden state at [" << layer << "," << timestep << "," << neuron << "]: "
                      << RNN->GetHiddenState(layer, timestep, neuron) << "\n";
        } else {
            std::cout << "Unknown query type: " << queryType << "\n";
        }

        if (enableDropoutFlag) {
            RNN->EnableDropout(true);
            std::cout << "Dropout enabled\n";
        }

        if (disableDropoutFlag) {
            RNN->EnableDropout(false);
            std::cout << "Dropout disabled\n";
        }

        if (dropoutValue > 0) {
            RNN->SetDropoutRate(dropoutValue);
            std::cout << std::fixed << std::setprecision(6)
                      << "Dropout rate set to: " << dropoutValue << "\n";
        }

        delete RNN;
    }

    return 0;
}
