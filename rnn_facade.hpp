#ifndef RNN_FACADE_HPP
#define RNN_FACADE_HPP

#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <fstream>

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

struct THistogramBin {
    double RangeMin;
    double RangeMax;
    int Count;
    double Percentage;
};
using THistogram = std::vector<THistogramBin>;

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
using TGradientScaleArray = std::vector<TGradientScaleStats>;

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

class TRNNFacade;

class TSimpleRNNCellWrapper {
private:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;

public:
    DArray2D Wih, Whh;
    DArray Bh;
    DArray2D dWih, dWhh;
    DArray dBh;
    DArray2D MWih, MWhh;
    DArray MBh;
    DArray2D VWih, VWhh;
    DArray VBh;

    TSimpleRNNCellWrapper(int InputSize, int HiddenSize, TActivationType Activation);
    void Forward(const DArray& Input, const DArray& PrevH, DArray& H, DArray& PreH);
    void Backward(const DArray& dH, const DArray& H, const DArray& PreH,
                  const DArray& PrevH, const DArray& Input, double ClipVal,
                  DArray& dInput, DArray& dPrevH);
    void ApplyGradients(double LR, double ClipVal);
    void ResetGradients();
    int GetHiddenSize() const;
    int GetInputSize() const;
};

class TLSTMCellWrapper {
private:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;

public:
    DArray2D Wf, Wi, Wc, Wo;
    DArray Bf, Bi, Bc, Bo;
    DArray2D dWf, dWi, dWc, dWo;
    DArray dBf, dBi, dBc, dBo;
    DArray2D MWf, MWi, MWc, MWo;
    DArray MBf, MBi, MBc, MBo;
    DArray2D VWf, VWi, VWc, VWo;
    DArray VBf, VBi, VBc, VBo;

    TLSTMCellWrapper(int InputSize, int HiddenSize, TActivationType Activation);
    void Forward(const DArray& Input, const DArray& PrevH, const DArray& PrevC,
                 DArray& H, DArray& C, DArray& FG, DArray& IG, DArray& CTilde,
                 DArray& OG, DArray& TanhC);
    void Backward(const DArray& dH, const DArray& dC, const DArray& H, const DArray& C,
                  const DArray& FG, const DArray& IG, const DArray& CTilde,
                  const DArray& OG, const DArray& TanhC, const DArray& PrevH,
                  const DArray& PrevC, const DArray& Input, double ClipVal,
                  DArray& dInput, DArray& dPrevH, DArray& dPrevC);
    void ApplyGradients(double LR, double ClipVal);
    void ResetGradients();
    int GetHiddenSize() const;
    int GetInputSize() const;
};

class TGRUCellWrapper {
private:
    int FInputSize, FHiddenSize;
    TActivationType FActivation;

public:
    DArray2D Wz, Wr, Wh;
    DArray Bz, Br, Bh;
    DArray2D dWz, dWr, dWh;
    DArray dBz, dBr, dBh;
    DArray2D MWz, MWr, MWh;
    DArray MBz, MBr, MBh;
    DArray2D VWz, VWr, VWh;
    DArray VBz, VBr, VBh;

    TGRUCellWrapper(int InputSize, int HiddenSize, TActivationType Activation);
    void Forward(const DArray& Input, const DArray& PrevH,
                 DArray& H, DArray& Z, DArray& R, DArray& HTilde);
    void Backward(const DArray& dH, const DArray& H, const DArray& Z,
                  const DArray& R, const DArray& HTilde, const DArray& PrevH,
                  const DArray& Input, double ClipVal,
                  DArray& dInput, DArray& dPrevH);
    void ApplyGradients(double LR, double ClipVal);
    void ResetGradients();
    int GetHiddenSize() const;
    int GetInputSize() const;
};

class TOutputLayerWrapper {
private:
    int FInputSize, FOutputSize;
    TActivationType FActivation;

public:
    DArray2D W;
    DArray B;
    DArray2D dW;
    DArray dB;
    DArray2D MW;
    DArray MB;
    DArray2D VW;
    DArray VB;

    TOutputLayerWrapper(int InputSize, int OutputSize, TActivationType Activation);
    void Forward(const DArray& Input, DArray& Output, DArray& Pre);
    void Backward(const DArray& dOut, const DArray& Output, const DArray& Pre,
                  const DArray& Input, double ClipVal, DArray& dInput);
    void ApplyGradients(double LR, double ClipVal);
    void ResetGradients();
    int GetInputSize() const;
    int GetOutputSize() const;
};

class TRNNFacade {
    // Allow OpenCL derived class to access private members
    friend class TRNNFacadeOpenCL;

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

    double ClipGradient(double G, double MaxVal) const;

public:
    TRNNFacade(int InputSize, const std::vector<int>& HiddenSizes,
               int OutputSize, TCellType CellType,
               TActivationType Activation, TActivationType OutputActivation,
               TLossType LossType, double LearningRate, double GradientClip,
               int BPTTSteps);
    ~TRNNFacade();

    void SaveModel(const std::string& Filename);
    void LoadModel(const std::string& Filename);

    DArray2D ForwardSequence(const DArray2D& Inputs);
    double BackwardSequence(const DArray2D& Targets);
    double TrainSequence(const DArray2D& Inputs, const DArray2D& Targets);
    DArray2D Predict(const DArray2D& Inputs);

    // 1. Time-Step and Sequence Access
    double GetHiddenState(int LayerIdx, int Timestep, int NeuronIdx) const;
    void SetHiddenState(int LayerIdx, int Timestep, int NeuronIdx, double Value);
    double GetOutput(int Timestep, int OutputIdx) const;
    void SetOutput(int Timestep, int OutputIdx, double Value);

    // 2. Cell State and Gate Access
    double GetCellState(int LayerIdx, int Timestep, int NeuronIdx) const;
    double GetGateValue(TGateType GateType, int LayerIdx, int Timestep, int NeuronIdx) const;

    // 3. Cached Pre-Activations and Inputs
    double GetPreActivation(int LayerIdx, int Timestep, int NeuronIdx) const;
    double GetInputVector(int Timestep, int InputIdx) const;

    // 4. Gradients and Optimizer States
    double GetWeightGradient(int LayerIdx, int NeuronIdx, int WeightIdx) const;
    double GetBiasGradient(int LayerIdx, int NeuronIdx) const;
    TOptimizerStateRecord GetOptimizerState(int LayerIdx, int NeuronIdx, int Param) const;
    double GetCellGradient(int LayerIdx, int Timestep, int NeuronIdx) const;

    // 5. Dropout, LayerNorm, Regularization
    void SetDropoutRate(double Rate);
    double GetDropoutRate() const;
    double GetDropoutMask(int LayerIdx, int Timestep, int NeuronIdx) const;
    TLayerNormStats GetLayerNormStats(int LayerIdx, int Timestep) const;
    void EnableDropout(bool Enable);

    // 6. Sequence-to-Sequence APIs
    DArray GetSequenceOutputs(int OutputIdx) const;
    DArray GetSequenceHiddenStates(int LayerIdx, int NeuronIdx) const;
    DArray GetSequenceCellStates(int LayerIdx, int NeuronIdx) const;
    DArray GetSequenceGateValues(TGateType GateType, int LayerIdx, int NeuronIdx) const;

    // 7. Reset and Manipulate Hidden States
    void ResetHiddenState(int LayerIdx, double Value = 0.0);
    void ResetCellState(int LayerIdx, double Value = 0.0);
    void ResetAllStates(double Value = 0.0);
    void InjectHiddenState(int LayerIdx, const DArray& ValuesArray);
    void InjectCellState(int LayerIdx, const DArray& ValuesArray);

    // 8. Attention Introspection
    DArray GetAttentionWeights(int Timestep) const;
    DArray GetAttentionContext(int Timestep) const;

    // 9. Time-Series Diagnostics
    THistogram GetHiddenStateHistogram(int LayerIdx, int Timestep, int NumBins = 10) const;
    THistogram GetActivationHistogramOverTime(int LayerIdx, int NeuronIdx, int NumBins = 10) const;
    TGateSaturationStats GetGateSaturation(TGateType GateType, int LayerIdx, int Timestep,
                                            double Threshold = 0.05) const;
    TGradientScaleArray GetGradientScalesOverTime(int LayerIdx) const;
    bool DetectVanishingGradient(int LayerIdx, double Threshold = 1e-6) const;
    bool DetectExplodingGradient(int LayerIdx, double Threshold = 1e6) const;

    // Utility
    void ResetGradients();
    void ApplyGradients();
    DArray3D InitHiddenStates();
    int GetLayerCount() const;
    int GetHiddenSize(int LayerIdx) const;
    TCellType GetCellType() const;
    int GetSequenceLength() const;

    // Properties
    double getLearningRate() const { return FLearningRate; }
    void setLearningRate(double value) { FLearningRate = value; }

    double getGradientClip() const { return FGradientClip; }
    void setGradientClip(double value) { FGradientClip = value; }

    double getDropoutRate() const { return FDropoutRate; }
    void setDropoutRate(double value) { FDropoutRate = value; }
};

double ComputeLoss(const DArray& Pred, const DArray& Target, TLossType LossType);
void ComputeLossGradient(const DArray& Pred, const DArray& Target, TLossType LossType, DArray& Grad);

#endif
