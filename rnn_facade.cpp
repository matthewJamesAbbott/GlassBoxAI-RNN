#include "rnn_facade.hpp"
#include <cmath>
#include <algorithm>

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> dis(0.0, 1.0);

// Utility functions
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

// TSimpleRNNCellWrapper
TSimpleRNNCellWrapper::TSimpleRNNCellWrapper(int InputSize, int HiddenSize, TActivationType Activation)
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

void TSimpleRNNCellWrapper::Forward(const DArray& Input, const DArray& PrevH, DArray& H, DArray& PreH) {
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

void TSimpleRNNCellWrapper::Backward(const DArray& dH, const DArray& H, const DArray& PreH,
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

void TSimpleRNNCellWrapper::ApplyGradients(double LR, double ClipVal) {
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

void TSimpleRNNCellWrapper::ResetGradients() {
    ZeroMatrix(dWih, FHiddenSize, FInputSize);
    ZeroMatrix(dWhh, FHiddenSize, FHiddenSize);
    ZeroArray(dBh, FHiddenSize);
}

int TSimpleRNNCellWrapper::GetHiddenSize() const { return FHiddenSize; }
int TSimpleRNNCellWrapper::GetInputSize() const { return FInputSize; }

// TLSTMCellWrapper
TLSTMCellWrapper::TLSTMCellWrapper(int InputSize, int HiddenSize, TActivationType Activation)
    : FInputSize(InputSize), FHiddenSize(HiddenSize), FActivation(Activation) {
    int ConcatSize = InputSize + HiddenSize;
    double Scale = std::sqrt(2.0 / ConcatSize);
    InitMatrix(Wf, HiddenSize, ConcatSize, Scale);
    InitMatrix(Wi, HiddenSize, ConcatSize, Scale);
    InitMatrix(Wc, HiddenSize, ConcatSize, Scale);
    InitMatrix(Wo, HiddenSize, ConcatSize, Scale);
    Bf.resize(HiddenSize); Bi.resize(HiddenSize);
    Bc.resize(HiddenSize); Bo.resize(HiddenSize);
    for (int i = 0; i < HiddenSize; i++) {
        Bf[i] = 1.0; Bi[i] = 0; Bc[i] = 0; Bo[i] = 0;
    }
    ZeroMatrix(dWf, HiddenSize, ConcatSize); ZeroMatrix(dWi, HiddenSize, ConcatSize);
    ZeroMatrix(dWc, HiddenSize, ConcatSize); ZeroMatrix(dWo, HiddenSize, ConcatSize);
    ZeroArray(dBf, HiddenSize); ZeroArray(dBi, HiddenSize);
    ZeroArray(dBc, HiddenSize); ZeroArray(dBo, HiddenSize);
    ZeroMatrix(MWf, HiddenSize, ConcatSize); ZeroMatrix(MWi, HiddenSize, ConcatSize);
    ZeroMatrix(MWc, HiddenSize, ConcatSize); ZeroMatrix(MWo, HiddenSize, ConcatSize);
    ZeroArray(MBf, HiddenSize); ZeroArray(MBi, HiddenSize);
    ZeroArray(MBc, HiddenSize); ZeroArray(MBo, HiddenSize);
    ZeroMatrix(VWf, HiddenSize, ConcatSize); ZeroMatrix(VWi, HiddenSize, ConcatSize);
    ZeroMatrix(VWc, HiddenSize, ConcatSize); ZeroMatrix(VWo, HiddenSize, ConcatSize);
    ZeroArray(VBf, HiddenSize); ZeroArray(VBi, HiddenSize);
    ZeroArray(VBc, HiddenSize); ZeroArray(VBo, HiddenSize);
}

void TLSTMCellWrapper::Forward(const DArray& Input, const DArray& PrevH, const DArray& PrevC,
                                DArray& H, DArray& C, DArray& FG, DArray& IG, DArray& CTilde,
                                DArray& OG, DArray& TanhC) {
    DArray Concat = ConcatArrays(Input, PrevH);
    H.resize(FHiddenSize); C.resize(FHiddenSize);
    FG.resize(FHiddenSize); IG.resize(FHiddenSize);
    CTilde.resize(FHiddenSize); OG.resize(FHiddenSize);
    TanhC.resize(FHiddenSize);

    for (int i = 0; i < FHiddenSize; i++) {
        double SumF = Bf[i], SumI = Bi[i], SumC = Bc[i], SumO = Bo[i];
        for (int j = 0; j < (int)Concat.size(); j++) {
            SumF += Wf[i][j] * Concat[j];
            SumI += Wi[i][j] * Concat[j];
            SumC += Wc[i][j] * Concat[j];
            SumO += Wo[i][j] * Concat[j];
        }
        FG[i] = ApplyActivation(SumF, TActivationType::atSigmoid);
        IG[i] = ApplyActivation(SumI, TActivationType::atSigmoid);
        CTilde[i] = ApplyActivation(SumC, TActivationType::atTanh);
        C[i] = FG[i] * PrevC[i] + IG[i] * CTilde[i];
        TanhC[i] = std::tanh(C[i]);
        OG[i] = ApplyActivation(SumO, TActivationType::atSigmoid);
        H[i] = OG[i] * TanhC[i];
    }
}

void TLSTMCellWrapper::Backward(const DArray& dH, const DArray& dC, const DArray& H, const DArray& C,
                                 const DArray& FG, const DArray& IG, const DArray& CTilde,
                                 const DArray& OG, const DArray& TanhC, const DArray& PrevH,
                                 const DArray& PrevC, const DArray& Input, double ClipVal,
                                 DArray& dInput, DArray& dPrevH, DArray& dPrevC) {
    int ConcatSize = Input.size() + PrevH.size();
    dInput.resize(Input.size(), 0.0);
    dPrevH.resize(PrevH.size(), 0.0);
    dPrevC.resize(PrevC.size(), 0.0);

    for (int i = 0; i < FHiddenSize; i++) {
        double dO = dH[i] * TanhC[i] * ActivationDerivative(OG[i], TActivationType::atSigmoid);
        double dTanhC = dH[i] * OG[i] * (1.0 - TanhC[i] * TanhC[i]);
        double dC_i = dC[i] + dTanhC;
        double dF = dC_i * PrevC[i] * ActivationDerivative(FG[i], TActivationType::atSigmoid);
        double dI = dC_i * CTilde[i] * ActivationDerivative(IG[i], TActivationType::atSigmoid);
        double dCTilde = dC_i * IG[i] * ActivationDerivative(CTilde[i], TActivationType::atTanh);

        DArray dConcat(ConcatSize);
        for (int j = 0; j < ConcatSize; j++)
            dConcat[j] = Wf[i][j] * dF + Wi[i][j] * dI + Wc[i][j] * dCTilde + Wo[i][j] * dO;

        for (int j = 0; j < (int)Input.size(); j++)
            dInput[j] += dConcat[j];
        for (int j = 0; j < (int)PrevH.size(); j++)
            dPrevH[j] += dConcat[Input.size() + j];

        dPrevC[i] = dC_i * FG[i];

        for (int j = 0; j < ConcatSize; j++) {
            dWf[i][j] += dF * (j < (int)Input.size() ? Input[j] : PrevH[j - Input.size()]);
            dWi[i][j] += dI * (j < (int)Input.size() ? Input[j] : PrevH[j - Input.size()]);
            dWc[i][j] += dCTilde * (j < (int)Input.size() ? Input[j] : PrevH[j - Input.size()]);
            dWo[i][j] += dO * (j < (int)Input.size() ? Input[j] : PrevH[j - Input.size()]);
        }
        dBf[i] += dF;
        dBi[i] += dI;
        dBc[i] += dCTilde;
        dBo[i] += dO;
    }
}

void TLSTMCellWrapper::ApplyGradients(double LR, double ClipVal) {
    for (int i = 0; i < FHiddenSize; i++) {
        for (int j = 0; j < (int)Wf[i].size(); j++) {
            Wf[i][j] -= LR * ClipValue(dWf[i][j], ClipVal);
            Wi[i][j] -= LR * ClipValue(dWi[i][j], ClipVal);
            Wc[i][j] -= LR * ClipValue(dWc[i][j], ClipVal);
            Wo[i][j] -= LR * ClipValue(dWo[i][j], ClipVal);
            dWf[i][j] = 0; dWi[i][j] = 0; dWc[i][j] = 0; dWo[i][j] = 0;
        }
        Bf[i] -= LR * ClipValue(dBf[i], ClipVal);
        Bi[i] -= LR * ClipValue(dBi[i], ClipVal);
        Bc[i] -= LR * ClipValue(dBc[i], ClipVal);
        Bo[i] -= LR * ClipValue(dBo[i], ClipVal);
        dBf[i] = 0; dBi[i] = 0; dBc[i] = 0; dBo[i] = 0;
    }
}

void TLSTMCellWrapper::ResetGradients() {
    int ConcatSize = Wf[0].size();
    ZeroMatrix(dWf, FHiddenSize, ConcatSize);
    ZeroMatrix(dWi, FHiddenSize, ConcatSize);
    ZeroMatrix(dWc, FHiddenSize, ConcatSize);
    ZeroMatrix(dWo, FHiddenSize, ConcatSize);
    ZeroArray(dBf, FHiddenSize);
    ZeroArray(dBi, FHiddenSize);
    ZeroArray(dBc, FHiddenSize);
    ZeroArray(dBo, FHiddenSize);
}

int TLSTMCellWrapper::GetHiddenSize() const { return FHiddenSize; }
int TLSTMCellWrapper::GetInputSize() const { return FInputSize; }

// TGRUCellWrapper
TGRUCellWrapper::TGRUCellWrapper(int InputSize, int HiddenSize, TActivationType Activation)
    : FInputSize(InputSize), FHiddenSize(HiddenSize), FActivation(Activation) {
    int ConcatSize = InputSize + HiddenSize;
    double Scale = std::sqrt(2.0 / ConcatSize);
    InitMatrix(Wz, HiddenSize, ConcatSize, Scale);
    InitMatrix(Wr, HiddenSize, ConcatSize, Scale);
    InitMatrix(Wh, HiddenSize, ConcatSize, Scale);
    ZeroArray(Bz, HiddenSize);
    ZeroArray(Br, HiddenSize);
    ZeroArray(Bh, HiddenSize);
    ZeroMatrix(dWz, HiddenSize, ConcatSize); ZeroMatrix(dWr, HiddenSize, ConcatSize); ZeroMatrix(dWh, HiddenSize, ConcatSize);
    ZeroArray(dBz, HiddenSize); ZeroArray(dBr, HiddenSize); ZeroArray(dBh, HiddenSize);
    ZeroMatrix(MWz, HiddenSize, ConcatSize); ZeroMatrix(MWr, HiddenSize, ConcatSize); ZeroMatrix(MWh, HiddenSize, ConcatSize);
    ZeroArray(MBz, HiddenSize); ZeroArray(MBr, HiddenSize); ZeroArray(MBh, HiddenSize);
    ZeroMatrix(VWz, HiddenSize, ConcatSize); ZeroMatrix(VWr, HiddenSize, ConcatSize); ZeroMatrix(VWh, HiddenSize, ConcatSize);
    ZeroArray(VBz, HiddenSize); ZeroArray(VBr, HiddenSize); ZeroArray(VBh, HiddenSize);
}

void TGRUCellWrapper::Forward(const DArray& Input, const DArray& PrevH,
                               DArray& H, DArray& Z, DArray& R, DArray& HTilde) {
    DArray Concat = ConcatArrays(Input, PrevH);
    H.resize(FHiddenSize); Z.resize(FHiddenSize);
    R.resize(FHiddenSize); HTilde.resize(FHiddenSize);

    for (int i = 0; i < FHiddenSize; i++) {
        double SumZ = Bz[i], SumR = Br[i];
        for (int j = 0; j < (int)Concat.size(); j++) {
            SumZ += Wz[i][j] * Concat[j];
            SumR += Wr[i][j] * Concat[j];
        }
        Z[i] = ApplyActivation(SumZ, TActivationType::atSigmoid);
        R[i] = ApplyActivation(SumR, TActivationType::atSigmoid);
    }

    for (int i = 0; i < FHiddenSize; i++) {
        double SumH = Bh[i];
        for (int j = 0; j < (int)Input.size(); j++)
            SumH += Wh[i][j] * Input[j];
        for (int j = 0; j < (int)PrevH.size(); j++)
            SumH += Wh[i][Input.size() + j] * R[i] * PrevH[j];
        HTilde[i] = ApplyActivation(SumH, FActivation);
        H[i] = Z[i] * PrevH[i] + (1.0 - Z[i]) * HTilde[i];
    }
}

void TGRUCellWrapper::Backward(const DArray& dH, const DArray& H, const DArray& Z,
                                const DArray& R, const DArray& HTilde, const DArray& PrevH,
                                const DArray& Input, double ClipVal,
                                DArray& dInput, DArray& dPrevH) {
    dInput.resize(Input.size(), 0.0);
    dPrevH.resize(PrevH.size(), 0.0);

    for (int i = 0; i < FHiddenSize; i++) {
        double dHTilde = dH[i] * (1.0 - Z[i]) * ActivationDerivative(HTilde[i], FActivation);
        double dZ = dH[i] * (PrevH[i] - HTilde[i]) * ActivationDerivative(Z[i], TActivationType::atSigmoid);

        for (int j = 0; j < (int)Input.size(); j++) {
            dWz[i][j] += dZ * Input[j];
            dWh[i][j] += dHTilde * Input[j];
            dInput[j] += Wz[i][j] * dZ + Wh[i][j] * dHTilde;
        }
        for (int j = 0; j < (int)PrevH.size(); j++) {
            dWz[i][Input.size() + j] += dZ * PrevH[j];
            dWh[i][Input.size() + j] += dHTilde * R[i] * PrevH[j];
            dPrevH[j] += Wz[i][Input.size() + j] * dZ + Wh[i][Input.size() + j] * dHTilde * R[i];
            dPrevH[j] += dH[i] * Z[i];
        }
        dBz[i] += dZ;
        dBh[i] += dHTilde;
    }
}

void TGRUCellWrapper::ApplyGradients(double LR, double ClipVal) {
    for (int i = 0; i < FHiddenSize; i++) {
        for (int j = 0; j < (int)Wz[i].size(); j++) {
            Wz[i][j] -= LR * ClipValue(dWz[i][j], ClipVal);
            Wr[i][j] -= LR * ClipValue(dWr[i][j], ClipVal);
            Wh[i][j] -= LR * ClipValue(dWh[i][j], ClipVal);
            dWz[i][j] = 0; dWr[i][j] = 0; dWh[i][j] = 0;
        }
        Bz[i] -= LR * ClipValue(dBz[i], ClipVal);
        Br[i] -= LR * ClipValue(dBr[i], ClipVal);
        Bh[i] -= LR * ClipValue(dBh[i], ClipVal);
        dBz[i] = 0; dBr[i] = 0; dBh[i] = 0;
    }
}

void TGRUCellWrapper::ResetGradients() {
    int ConcatSize = Wz[0].size();
    ZeroMatrix(dWz, FHiddenSize, ConcatSize);
    ZeroMatrix(dWr, FHiddenSize, ConcatSize);
    ZeroMatrix(dWh, FHiddenSize, ConcatSize);
    ZeroArray(dBz, FHiddenSize);
    ZeroArray(dBr, FHiddenSize);
    ZeroArray(dBh, FHiddenSize);
}

int TGRUCellWrapper::GetHiddenSize() const { return FHiddenSize; }
int TGRUCellWrapper::GetInputSize() const { return FInputSize; }

// TOutputLayerWrapper
TOutputLayerWrapper::TOutputLayerWrapper(int InputSize, int OutputSize, TActivationType Activation)
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

void TOutputLayerWrapper::Forward(const DArray& Input, DArray& Output, DArray& Pre) {
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

void TOutputLayerWrapper::Backward(const DArray& dOut, const DArray& Output, const DArray& Pre,
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

void TOutputLayerWrapper::ApplyGradients(double LR, double ClipVal) {
    for (int i = 0; i < FOutputSize; i++) {
        for (int j = 0; j < FInputSize; j++) {
            W[i][j] -= LR * ClipValue(dW[i][j], ClipVal);
            dW[i][j] = 0;
        }
        B[i] -= LR * ClipValue(dB[i], ClipVal);
        dB[i] = 0;
    }
}

void TOutputLayerWrapper::ResetGradients() {
    ZeroMatrix(dW, FOutputSize, FInputSize);
    ZeroArray(dB, FOutputSize);
}

int TOutputLayerWrapper::GetInputSize() const { return FInputSize; }
int TOutputLayerWrapper::GetOutputSize() const { return FOutputSize; }

// TRNNFacade
TRNNFacade::TRNNFacade(int InputSize, const std::vector<int>& HiddenSizes,
                       int OutputSize, TCellType CellType,
                       TActivationType Activation, TActivationType OutputActivation,
                       TLossType LossType, double LearningRate, double GradientClip,
                       int BPTTSteps)
    : FInputSize(InputSize), FOutputSize(OutputSize), FHiddenSizes(HiddenSizes),
      FCellType(CellType), FActivation(Activation), FOutputActivation(OutputActivation),
      FLossType(LossType), FLearningRate(LearningRate), FGradientClip(GradientClip),
      FBPTTSteps(BPTTSteps), FDropoutRate(0.0), FUseDropout(false), FSequenceLen(0) {

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
}

TRNNFacade::~TRNNFacade() {
    delete FOutputLayer;
}

DArray3D TRNNFacade::InitHiddenStates() {
    DArray3D States(FHiddenSizes.size());
    for (size_t i = 0; i < FHiddenSizes.size(); i++) {
        States[i].resize(2);
        States[i][0].resize(FHiddenSizes[i], 0.0);
        if (FCellType == TCellType::ctLSTM)
            States[i][1].resize(FHiddenSizes[i], 0.0);
    }
    return States;
}

DArray2D TRNNFacade::ForwardSequence(const DArray2D& Inputs) {
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

    return Outputs;
}

double TRNNFacade::BackwardSequence(const DArray2D& Targets) {
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

    return Loss / FSequenceLen;
}

double TRNNFacade::TrainSequence(const DArray2D& Inputs, const DArray2D& Targets) {
    ResetAllStates();
    DArray2D Outputs = ForwardSequence(Inputs);
    double Loss = BackwardSequence(Targets);
    ApplyGradients();
    return Loss;
}

DArray2D TRNNFacade::Predict(const DArray2D& Inputs) {
    ResetAllStates();
    return ForwardSequence(Inputs);
}

// Accessor methods
double TRNNFacade::GetHiddenState(int LayerIdx, int Timestep, int NeuronIdx) const {
    if (LayerIdx < 0 || LayerIdx >= (int)FHiddenSizes.size()) return 0;
    if (Timestep < 0 || Timestep >= FSequenceLen) return 0;
    if (NeuronIdx < 0 || NeuronIdx >= (int)FCaches[Timestep].H.size()) return 0;
    return FCaches[Timestep].H[NeuronIdx];
}

void TRNNFacade::SetHiddenState(int LayerIdx, int Timestep, int NeuronIdx, double Value) {
    if (LayerIdx < 0 || LayerIdx >= (int)FHiddenSizes.size()) return;
    if (Timestep < 0 || Timestep >= FSequenceLen) return;
    if (NeuronIdx < 0 || NeuronIdx >= (int)FCaches[Timestep].H.size()) return;
    FCaches[Timestep].H[NeuronIdx] = Value;
}

double TRNNFacade::GetOutput(int Timestep, int OutputIdx) const {
    if (Timestep < 0 || Timestep >= FSequenceLen) return 0;
    if (OutputIdx < 0 || OutputIdx >= (int)FCaches[Timestep].OutVal.size()) return 0;
    return FCaches[Timestep].OutVal[OutputIdx];
}

void TRNNFacade::SetOutput(int Timestep, int OutputIdx, double Value) {
    if (Timestep < 0 || Timestep >= FSequenceLen) return;
    if (OutputIdx < 0 || OutputIdx >= (int)FCaches[Timestep].OutVal.size()) return;
    FCaches[Timestep].OutVal[OutputIdx] = Value;
}

double TRNNFacade::GetCellState(int LayerIdx, int Timestep, int NeuronIdx) const {
    if (FCellType != TCellType::ctLSTM) return 0;
    if (Timestep < 0 || Timestep >= FSequenceLen) return 0;
    if (NeuronIdx < 0 || NeuronIdx >= (int)FCaches[Timestep].C.size()) return 0;
    return FCaches[Timestep].C[NeuronIdx];
}

double TRNNFacade::GetGateValue(TGateType GateType, int LayerIdx, int Timestep, int NeuronIdx) const {
    if (Timestep < 0 || Timestep >= FSequenceLen || NeuronIdx < 0) return 0;

    switch (FCellType) {
        case TCellType::ctLSTM:
            switch (GateType) {
                case TGateType::gtForget: if (NeuronIdx < (int)FCaches[Timestep].F.size()) return FCaches[Timestep].F[NeuronIdx]; break;
                case TGateType::gtInput: if (NeuronIdx < (int)FCaches[Timestep].I.size()) return FCaches[Timestep].I[NeuronIdx]; break;
                case TGateType::gtOutput: if (NeuronIdx < (int)FCaches[Timestep].O.size()) return FCaches[Timestep].O[NeuronIdx]; break;
                case TGateType::gtCellCandidate: if (NeuronIdx < (int)FCaches[Timestep].CTilde.size()) return FCaches[Timestep].CTilde[NeuronIdx]; break;
                default: break;
            }
            break;
        case TCellType::ctGRU:
            switch (GateType) {
                case TGateType::gtUpdate: if (NeuronIdx < (int)FCaches[Timestep].Z.size()) return FCaches[Timestep].Z[NeuronIdx]; break;
                case TGateType::gtReset: if (NeuronIdx < (int)FCaches[Timestep].R.size()) return FCaches[Timestep].R[NeuronIdx]; break;
                case TGateType::gtHiddenCandidate: if (NeuronIdx < (int)FCaches[Timestep].HTilde.size()) return FCaches[Timestep].HTilde[NeuronIdx]; break;
                default: break;
            }
            break;
        default: break;
    }
    return 0;
}

double TRNNFacade::GetPreActivation(int LayerIdx, int Timestep, int NeuronIdx) const {
    if (Timestep < 0 || Timestep >= FSequenceLen) return 0;
    if (NeuronIdx < 0 || NeuronIdx >= (int)FCaches[Timestep].PreH.size()) return 0;
    return FCaches[Timestep].PreH[NeuronIdx];
}

double TRNNFacade::GetInputVector(int Timestep, int InputIdx) const {
    if (Timestep < 0 || Timestep >= FSequenceLen) return 0;
    if (InputIdx < 0 || InputIdx >= (int)FCaches[Timestep].Input.size()) return 0;
    return FCaches[Timestep].Input[InputIdx];
}

double TRNNFacade::GetWeightGradient(int LayerIdx, int NeuronIdx, int WeightIdx) const {
    if (LayerIdx < 0 || LayerIdx >= (int)FHiddenSizes.size()) return 0;
    switch (FCellType) {
        case TCellType::ctSimpleRNN:
            if (NeuronIdx >= 0 && NeuronIdx < FHiddenSizes[LayerIdx])
                if (WeightIdx >= 0 && WeightIdx < (int)FSimpleCells[LayerIdx].dWih[NeuronIdx].size())
                    return FSimpleCells[LayerIdx].dWih[NeuronIdx][WeightIdx];
            break;
        case TCellType::ctLSTM:
            if (NeuronIdx >= 0 && NeuronIdx < FHiddenSizes[LayerIdx])
                if (WeightIdx >= 0 && WeightIdx < (int)FLSTMCells[LayerIdx].dWf[NeuronIdx].size())
                    return FLSTMCells[LayerIdx].dWf[NeuronIdx][WeightIdx];
            break;
        case TCellType::ctGRU:
            if (NeuronIdx >= 0 && NeuronIdx < FHiddenSizes[LayerIdx])
                if (WeightIdx >= 0 && WeightIdx < (int)FGRUCells[LayerIdx].dWz[NeuronIdx].size())
                    return FGRUCells[LayerIdx].dWz[NeuronIdx][WeightIdx];
            break;
    }
    return 0;
}

double TRNNFacade::GetBiasGradient(int LayerIdx, int NeuronIdx) const {
    if (LayerIdx < 0 || LayerIdx >= (int)FHiddenSizes.size()) return 0;
    switch (FCellType) {
        case TCellType::ctSimpleRNN:
            if (NeuronIdx >= 0 && NeuronIdx < (int)FSimpleCells[LayerIdx].dBh.size())
                return FSimpleCells[LayerIdx].dBh[NeuronIdx];
            break;
        case TCellType::ctLSTM:
            if (NeuronIdx >= 0 && NeuronIdx < (int)FLSTMCells[LayerIdx].dBf.size())
                return FLSTMCells[LayerIdx].dBf[NeuronIdx];
            break;
        case TCellType::ctGRU:
            if (NeuronIdx >= 0 && NeuronIdx < (int)FGRUCells[LayerIdx].dBz.size())
                return FGRUCells[LayerIdx].dBz[NeuronIdx];
            break;
    }
    return 0;
}

TOptimizerStateRecord TRNNFacade::GetOptimizerState(int LayerIdx, int NeuronIdx, int Param) const {
    TOptimizerStateRecord Result = {0, 0, 0, 0};
    if (LayerIdx < 0 || LayerIdx >= (int)FHiddenSizes.size()) return Result;
    switch (FCellType) {
        case TCellType::ctSimpleRNN:
            if (NeuronIdx >= 0 && NeuronIdx < FHiddenSizes[LayerIdx])
                if (Param >= 0 && Param < (int)FSimpleCells[LayerIdx].MWih[NeuronIdx].size()) {
                    Result.Momentum = FSimpleCells[LayerIdx].MWih[NeuronIdx][Param];
                    Result.Velocity = FSimpleCells[LayerIdx].VWih[NeuronIdx][Param];
                }
            break;
        case TCellType::ctLSTM:
            if (NeuronIdx >= 0 && NeuronIdx < FHiddenSizes[LayerIdx])
                if (Param >= 0 && Param < (int)FLSTMCells[LayerIdx].MWf[NeuronIdx].size()) {
                    Result.Momentum = FLSTMCells[LayerIdx].MWf[NeuronIdx][Param];
                    Result.Velocity = FLSTMCells[LayerIdx].VWf[NeuronIdx][Param];
                }
            break;
        case TCellType::ctGRU:
            if (NeuronIdx >= 0 && NeuronIdx < FHiddenSizes[LayerIdx])
                if (Param >= 0 && Param < (int)FGRUCells[LayerIdx].MWz[NeuronIdx].size()) {
                    Result.Momentum = FGRUCells[LayerIdx].MWz[NeuronIdx][Param];
                    Result.Velocity = FGRUCells[LayerIdx].VWz[NeuronIdx][Param];
                }
            break;
    }
    return Result;
}

double TRNNFacade::GetCellGradient(int LayerIdx, int Timestep, int NeuronIdx) const {
    if (Timestep >= 0 && Timestep < (int)FGradientHistory.size())
        if (FGradientHistory[Timestep].size() > 0)
            return FGradientHistory[Timestep][0];
    return 0;
}

void TRNNFacade::SetDropoutRate(double Rate) {
    FDropoutRate = Rate;
    FUseDropout = Rate > 0;
}

double TRNNFacade::GetDropoutRate() const {
    return FDropoutRate;
}

double TRNNFacade::GetDropoutMask(int LayerIdx, int Timestep, int NeuronIdx) const {
    if (Timestep >= 0 && Timestep < FSequenceLen && NeuronIdx >= 0)
        if (NeuronIdx < (int)FCaches[Timestep].DropoutMask.size())
            return FCaches[Timestep].DropoutMask[NeuronIdx];
    return 1.0;
}

TLayerNormStats TRNNFacade::GetLayerNormStats(int LayerIdx, int Timestep) const {
    TLayerNormStats Result = {0, 1, 1, 0};
    if (Timestep < 0 || Timestep >= FSequenceLen) return Result;
    if (FCaches[Timestep].H.empty()) return Result;

    double Sum = 0, SumSq = 0;
    for (double h : FCaches[Timestep].H) {
        Sum += h;
        SumSq += h * h;
    }
    double Mean = Sum / FCaches[Timestep].H.size();
    double Variance = SumSq / FCaches[Timestep].H.size() - Mean * Mean;
    Result.Mean = Mean;
    Result.Variance = Variance;
    return Result;
}

void TRNNFacade::EnableDropout(bool Enable) {
    FUseDropout = Enable;
}

DArray TRNNFacade::GetSequenceOutputs(int OutputIdx) const {
    DArray Result(FSequenceLen);
    for (int t = 0; t < FSequenceLen; t++) {
        if (OutputIdx >= 0 && OutputIdx < (int)FCaches[t].OutVal.size())
            Result[t] = FCaches[t].OutVal[OutputIdx];
        else
            Result[t] = 0;
    }
    return Result;
}

DArray TRNNFacade::GetSequenceHiddenStates(int LayerIdx, int NeuronIdx) const {
    DArray Result(FSequenceLen);
    for (int t = 0; t < FSequenceLen; t++) {
        if (NeuronIdx >= 0 && NeuronIdx < (int)FCaches[t].H.size())
            Result[t] = FCaches[t].H[NeuronIdx];
        else
            Result[t] = 0;
    }
    return Result;
}

DArray TRNNFacade::GetSequenceCellStates(int LayerIdx, int NeuronIdx) const {
    DArray Result(FSequenceLen);
    if (FCellType != TCellType::ctLSTM) return Result;
    for (int t = 0; t < FSequenceLen; t++) {
        if (NeuronIdx >= 0 && NeuronIdx < (int)FCaches[t].C.size())
            Result[t] = FCaches[t].C[NeuronIdx];
        else
            Result[t] = 0;
    }
    return Result;
}

DArray TRNNFacade::GetSequenceGateValues(TGateType GateType, int LayerIdx, int NeuronIdx) const {
    DArray Result(FSequenceLen);
    for (int t = 0; t < FSequenceLen; t++)
        Result[t] = GetGateValue(GateType, LayerIdx, t, NeuronIdx);
    return Result;
}

void TRNNFacade::ResetHiddenState(int LayerIdx, double Value) {
    if (LayerIdx >= 0 && LayerIdx < (int)FHiddenSizes.size()) {
        for (int i = 0; i < FHiddenSizes[LayerIdx]; i++)
            FStates[LayerIdx][0][i] = Value;
    }
}

void TRNNFacade::ResetCellState(int LayerIdx, double Value) {
    if (FCellType != TCellType::ctLSTM) return;
    if (LayerIdx >= 0 && LayerIdx < (int)FHiddenSizes.size()) {
        for (int i = 0; i < FHiddenSizes[LayerIdx]; i++)
            FStates[LayerIdx][1][i] = Value;
    }
}

void TRNNFacade::ResetAllStates(double Value) {
    for (size_t layer = 0; layer < FHiddenSizes.size(); layer++) {
        for (int i = 0; i < FHiddenSizes[layer]; i++) {
            FStates[layer][0][i] = Value;
            if (FCellType == TCellType::ctLSTM)
                FStates[layer][1][i] = Value;
        }
    }
}

void TRNNFacade::InjectHiddenState(int LayerIdx, const DArray& ValuesArray) {
    if (LayerIdx < 0 || LayerIdx > (int)FHiddenSizes.size() - 1) return;
    for (int i = 0; i < std::min((int)ValuesArray.size(), FHiddenSizes[LayerIdx]); i++)
        FStates[LayerIdx][0][i] = ValuesArray[i];
}

void TRNNFacade::InjectCellState(int LayerIdx, const DArray& ValuesArray) {
    if (FCellType != TCellType::ctLSTM) return;
    if (LayerIdx < 0 || LayerIdx > (int)FHiddenSizes.size() - 1) return;
    for (int i = 0; i < std::min((int)ValuesArray.size(), FHiddenSizes[LayerIdx]); i++)
        FStates[LayerIdx][1][i] = ValuesArray[i];
}

DArray TRNNFacade::GetAttentionWeights(int Timestep) const {
    return DArray();
}

DArray TRNNFacade::GetAttentionContext(int Timestep) const {
    return DArray();
}

THistogram TRNNFacade::GetHiddenStateHistogram(int LayerIdx, int Timestep, int NumBins) const {
    THistogram Result(NumBins);
    if (Timestep < 0 || Timestep >= FSequenceLen) return Result;
    if (FCaches[Timestep].H.empty()) return Result;

    double MinVal = FCaches[Timestep].H[0];
    double MaxVal = FCaches[Timestep].H[0];
    for (double h : FCaches[Timestep].H) {
        MinVal = std::min(MinVal, h);
        MaxVal = std::max(MaxVal, h);
    }
    if (MaxVal == MinVal) MaxVal = MinVal + 1;
    double BinWidth = (MaxVal - MinVal) / NumBins;

    for (int i = 0; i < NumBins; i++) {
        Result[i].RangeMin = MinVal + i * BinWidth;
        Result[i].RangeMax = MinVal + (i + 1) * BinWidth;
        Result[i].Count = 0;
        Result[i].Percentage = 0;
    }

    for (double h : FCaches[Timestep].H) {
        int BinIdx = (int)((h - MinVal) / BinWidth);
        if (BinIdx >= NumBins) BinIdx = NumBins - 1;
        if (BinIdx < 0) BinIdx = 0;
        Result[BinIdx].Count++;
    }

    for (int i = 0; i < NumBins; i++)
        Result[i].Percentage = Result[i].Count / (double)FCaches[Timestep].H.size() * 100;

    return Result;
}

THistogram TRNNFacade::GetActivationHistogramOverTime(int LayerIdx, int NeuronIdx, int NumBins) const {
    THistogram Result(NumBins);
    if (FSequenceLen == 0) return Result;

    DArray Values(FSequenceLen);
    for (int t = 0; t < FSequenceLen; t++) {
        if (NeuronIdx >= 0 && NeuronIdx < (int)FCaches[t].H.size())
            Values[t] = FCaches[t].H[NeuronIdx];
        else
            Values[t] = 0;
    }

    double MinVal = Values[0], MaxVal = Values[0];
    for (double v : Values) {
        MinVal = std::min(MinVal, v);
        MaxVal = std::max(MaxVal, v);
    }
    if (MaxVal == MinVal) MaxVal = MinVal + 1;
    double BinWidth = (MaxVal - MinVal) / NumBins;

    for (int i = 0; i < NumBins; i++) {
        Result[i].RangeMin = MinVal + i * BinWidth;
        Result[i].RangeMax = MinVal + (i + 1) * BinWidth;
        Result[i].Count = 0;
    }

    for (double v : Values) {
        int BinIdx = (int)((v - MinVal) / BinWidth);
        if (BinIdx >= NumBins) BinIdx = NumBins - 1;
        if (BinIdx < 0) BinIdx = 0;
        Result[BinIdx].Count++;
    }

    for (int i = 0; i < NumBins; i++)
        Result[i].Percentage = Result[i].Count / (double)FSequenceLen * 100;

    return Result;
}

TGateSaturationStats TRNNFacade::GetGateSaturation(TGateType GateType, int LayerIdx, int Timestep, double Threshold) const {
    TGateSaturationStats Result = {0, 0, 0, 0, 0};
    if (Timestep < 0 || Timestep >= FSequenceLen) return Result;

    DArray GateArr;
    switch (FCellType) {
        case TCellType::ctLSTM:
            switch (GateType) {
                case TGateType::gtForget: GateArr = FCaches[Timestep].F; break;
                case TGateType::gtInput: GateArr = FCaches[Timestep].I; break;
                case TGateType::gtOutput: GateArr = FCaches[Timestep].O; break;
                default: return Result;
            }
            break;
        case TCellType::ctGRU:
            switch (GateType) {
                case TGateType::gtUpdate: GateArr = FCaches[Timestep].Z; break;
                case TGateType::gtReset: GateArr = FCaches[Timestep].R; break;
                default: return Result;
            }
            break;
        default: return Result;
    }

    Result.TotalCount = GateArr.size();
    for (double val : GateArr) {
        if (val < Threshold) Result.NearZeroCount++;
        if (val > (1 - Threshold)) Result.NearOneCount++;
    }
    if (Result.TotalCount > 0) {
        Result.NearZeroPct = Result.NearZeroCount / (double)Result.TotalCount * 100;
        Result.NearOnePct = Result.NearOneCount / (double)Result.TotalCount * 100;
    }
    return Result;
}

TGradientScaleArray TRNNFacade::GetGradientScalesOverTime(int LayerIdx) const {
    TGradientScaleArray Result(FSequenceLen);
    for (int t = 0; t < FSequenceLen; t++) {
        Result[t].Timestep = t;
        if (t < (int)FGradientHistory.size() && FGradientHistory[t].size() > 0) {
            Result[t].MeanAbsGrad = FGradientHistory[t][0];
            Result[t].MaxAbsGrad = FGradientHistory[t][0];
            Result[t].MinAbsGrad = FGradientHistory[t][0];
        } else {
            Result[t].MeanAbsGrad = 0;
            Result[t].MaxAbsGrad = 0;
            Result[t].MinAbsGrad = 0;
        }
    }
    return Result;
}

bool TRNNFacade::DetectVanishingGradient(int LayerIdx, double Threshold) const {
    for (const auto& hist : FGradientHistory)
        if (hist.size() > 0 && hist[0] < Threshold)
            return true;
    return false;
}

bool TRNNFacade::DetectExplodingGradient(int LayerIdx, double Threshold) const {
    for (const auto& hist : FGradientHistory)
        if (hist.size() > 0 && hist[0] > Threshold)
            return true;
    return false;
}

void TRNNFacade::ResetGradients() {
    for (auto& cell : FSimpleCells) cell.ResetGradients();
    for (auto& cell : FLSTMCells) cell.ResetGradients();
    for (auto& cell : FGRUCells) cell.ResetGradients();
    if (FOutputLayer) FOutputLayer->ResetGradients();
}

void TRNNFacade::ApplyGradients() {
    for (auto& cell : FSimpleCells) cell.ApplyGradients(FLearningRate, FGradientClip);
    for (auto& cell : FLSTMCells) cell.ApplyGradients(FLearningRate, FGradientClip);
    for (auto& cell : FGRUCells) cell.ApplyGradients(FLearningRate, FGradientClip);
    if (FOutputLayer) FOutputLayer->ApplyGradients(FLearningRate, FGradientClip);
}

int TRNNFacade::GetLayerCount() const {
    return FHiddenSizes.size();
}

int TRNNFacade::GetHiddenSize(int LayerIdx) const {
    if (LayerIdx >= 0 && LayerIdx < (int)FHiddenSizes.size())
        return FHiddenSizes[LayerIdx];
    return 0;
}

TCellType TRNNFacade::GetCellType() const {
    return FCellType;
}

int TRNNFacade::GetSequenceLength() const {
    return FSequenceLen;
}

void TRNNFacade::SaveModel(const std::string& Filename) {
    std::ofstream F(Filename, std::ios::binary);
    F.write((char*)&FInputSize, sizeof(int));
    F.write((char*)&FOutputSize, sizeof(int));
}

void TRNNFacade::LoadModel(const std::string& Filename) {
    std::ifstream F(Filename, std::ios::binary);
    F.read((char*)&FInputSize, sizeof(int));
    F.read((char*)&FOutputSize, sizeof(int));
}

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
