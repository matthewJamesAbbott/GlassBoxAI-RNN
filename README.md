# GlassBoxAI-RNN

**Author:** Matthew Abbott (2025)

This repository contains modern, transparent CUDA implementations of Recurrent Neural Networks (RNNs) for deep learning on the GPU. There are two main modules:

- **rnn.cu:** Direct, advanced CUDA implementation supporting SimpleRNN, LSTM, and GRU with full BPTT and gradient control.
- **facaded_rnn.cu:** Research/teaching-friendly C++ CUDA RNN facade, exposing high-level yet hackable APIs for analysis, extension, and custom models.

Both modules are designed with **reproducibility, extensibility, and visibility** in mind, offering advanced internals for introspection and gradient debugging.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [rnn.cu](#rnncu)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [Public Methods](#public-methods)
- [facaded_rnn.cu (RNN Facade)](#facaded_rnncu-rnn-facade)
  - [Usage](#usage-1)
  - [Arguments](#arguments-1)
  - [Public Methods](#public-methods-1)
- [Data Structures](#data-structures)
- [Overview & Notes](#overview--notes)

---

## Features

- Supports **SimpleRNN**, **LSTM**, and **GRU** cells on the GPU.
- Full backpropagation through time (BPTT).
- Gradient clipping and numerical stability controls.
- Multiple activation and loss types (sigmoid, tanh, ReLU, MSE, cross-entropy).
- Batch and sequence processing.
- Save/load model weights and configurations.
- Designed for extensibility and introspection for research/education.
- No external deep learning libraries needed.

---

## Requirements

- NVIDIA GPU with CUDA support (compute 5.0 or above recommended)
- CUDA toolkit (tested with CUDA 11+)
- C++11 or later
- Optional: CMake (for integrating into larger projects)

---

## rnn.cu

### High-Level Description

A direct, "bare-metal" RNN/LSTM/GRU implementation with CUDA for training and inference. Includes:

- Support for variable architecture and sequence length.
- Matrix and vector math, custom CUDA kernels for all major operations.
- Flexible selection of activation types, loss types, and cell types at runtime.
- Gradient accumulation and application, both CPU and GPU.
- Utilities for I/O and model inspection.

### Usage

```bash
nvcc -o rnn_cuda rnn.cu

# (Integration into your program or as library code; no built-in CLI driver)
```
You are expected to instantiate and drive the RNN (and its cells) via the provided classes and APIs.

### Arguments

#### Model/Constructor Arguments

- `inputSize`: Size of each input vector (**int**)
- `outputSize`: Output vector size  (**int**)
- `hiddenSizes`: Vector of ints; arbitrary depths/sizes per hidden layer
- `cellType`: `ctSimpleRNN`, `ctLSTM`, or `ctGRU`
- `activation`, `outputActivation`: Activation type enums (`atSigmoid`, `atTanh`, `atReLU`, `atLinear`)
- `lossType`: `ltMSE` or `ltCrossEntropy`
- `learningRate`: Training step size (**double**)
- `gradientClip`: Maximum norm for clipping gradients
- `bpttSteps`: Number of backpropagation steps
- `sequenceLen`: Length of input sequence per sample (**int**)

#### Public Methods

- See below per-class descriptions for API details.

### Public Classes and Main Functions

#### Main Classes

- **TSimpleRNNCell**: Basic RNN cell with GPU/CPU forward and backward passes.
- **TLSTMCell**: LSTM cell with fully GPU-accelerated forward; CPU and GPU hybrid backward.
- **TGRUCell**: GRU cell, similar architecture to LSTM.
- **TOutputLayer**: Dense output with selectable activation.
- **GPUArray/GPUMatrix**: C++ CUDA wrappers for managing device memory.

#### Major Functions (Per Cell Type)

All cells implement at least:

- **Forward**: Calculates hidden/cell state from input and previous state(s).
  - `void Forward(const DArray& Input, const DArray& PrevH, ...)`
  - (LSTM/GRU have additional arguments for cell or candidate states.)
- **Backward**: Computes gradients through the cell given output delta.
  - `void Backward(...)`
- **ApplyGradients**: Applies accumulated gradients with clipping and learning rate.
- **ResetGradients**: Clears gradient accumulators.

#### Other Key API:

- Classes for model persistence, GPU memory management, and activations/loss are included and accessible.

---

## facaded_rnn.cu (RNN Facade)

### High-Level Description

A feature-rich, modern C++/CUDA interface to recurrent neural networks, with attention to usability, analysis, and transparency. Supports:

- SimpleRNN, LSTM, GRU cell types.
- Easy extension to multiple layers and inspection of intermediate results.
- CUDA-accelerated forward and backward passes.
- Experimentation with activations, losses, and optimizer states.

### Usage

```bash
nvcc -O2 -o facaded_rnn_cuda facaded_rnn.cu
```
(intended as a library; see class below for interface)

### Arguments

#### Constructor Arguments (Class: `CudaRNNFacade`)

- `inputSize`: Input vector size
- `outputSize`: Output vector size
- `hiddenSizes`: `vector<int>`; arbitrary per-layer sizes supported
- `cellType`: See enum (`ctSimpleRNN`, `ctLSTM`, `ctGRU`)
- `activation`, `outputActivation`: Activation function types
- `lossType`: Loss function type
- `learningRate`: Training rate (default: 0.001)
- `gradientClip`: Gradient norm clip max (default: 1.0)
- `bpttSteps`: Steps of BPTT
- `sequenceLen`: Maximum sequence length

#### Main Methods

- **Forward**: Propagate a sequence/timestep through the network
  - `Forward(sequence)` / `ForwardStep(input, timestep)`
- **Backward**: Accumulate gradients and run optimizer
  - `Backward(sequence, targets)`
  - `ApplyGradients()`
- **ResetGradients**
- **SaveModel(filename) / LoadModel(filename)**
- **SetTrainingMode(mode: bool)**
- **Get/Set hidden or cell states**
- **Introspection/statistics**:
  - `GetLayerNormStats()`, `GetGradientStats()`, `GetGateStats()`
  - Access to per-step values, activations, and gates

#### Data Structures

- `DeviceArray`: Utility CUDA host class for array allocation/copy/zero.
- `CudaSimpleRNNCell`, `CudaLSTMCell`, `CudaGRUCell`, `CudaOutputLayer`: CUDA cell subtypes.
- `TimeCache`: Keeps history/buffers for sequence passes.

---

## Data Structures

- **Enumerated types:** ActivationType, LossType, CellType, GateType.
- **DeviceArray / GPUArray / GPUMatrix**: Host-side and device-side (CUDA) memory wrappers.
- **Structs:** For statistics—gate saturations, gradient scale, histogram bins, layer normalization.

---

## License

MIT License, Copyright © 2025 Matthew Abbott

---
