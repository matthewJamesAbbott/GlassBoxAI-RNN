# GlassBoxAI-RNN

**Author:** Matthew Abbott (2025)

GlassBoxAI-RNN is a transparent, research-grade recurrent neural network (RNN) toolkit offering high-performance CUDA and OpenCL implementations for SimpleRNN, LSTM, and GRU cells. The repository includes both a direct, low-level kernel-based implementation and a higher-level facade architecture for advanced introspection and teaching. All four main files can be built from source and run from the command line with abundant options and diagnostics.

---

## Table of Contents

- [Features](#features)
- [Overview of Module Types](#overview-of-module-types)
- [Requirements](#requirements)
- [Quickstart: Compiling & Running](#quickstart-compiling--running)
- [CLI Usage and Help](#cli-usage-and-help)
  - [1. CUDA Command-Line Model (rnn.cu)](#1-cuda-command-line-model-rnncu)
  - [2. OpenCL Command-Line Model (rnn_opencl.cpp)](#2-opencl-command-line-model-rnn_openclcpp)
  - [3. CUDA Facade (facaded_rnn.cu)](#3-cuda-facade-facaded_rnncu)
  - [4. OpenCL Facade (facaded_rnn_opencl.cpp)](#4-opencl-facade-facaded_rnn_openclcpp)
    - [OpenCL Facade CLI Example](#opencl-facade-cli-example)
    - [All Facade Introspection Options](#all-facade-introspection-options)
- [Architecture Notes](#architecture-notes)
- [Data Structures & Internals](#data-structures--internals)
- [License](#license)

---

## Features

- Pure, dependency-free CUDA (and OpenCL) RNNs: SimpleRNN, LSTM, and GRU
- **Two styles**:  
  - **Direct/Kernel RNN** (`rnn.cu`, `rnn_opencl.cpp`): Command-line "raw" models and training
  - **Facade/Introspectable RNN** (`facaded_rnn.cu`, `facaded_rnn_opencl.cpp`): Research/teaching CLI with advanced inspection tools
- Support for all popular activation and loss functions, gradient clipping, and introspection
- Model save/load, flexible sequence length learning, and thorough command-line arguments
- No external DL frameworks required
- Designed for transparency, extension, and education

---

## Overview of Module Types

There are **2 x 2 = 4 modes**:

| Type      | Direct/Kernels    | Facade/Introspectable     |
|-----------|-------------------|---------------------------|
| CUDA      | `rnn.cu`          | `facaded_rnn.cu`          |
| OpenCL    | `rnn_opencl.cpp`  | `facaded_rnn_opencl.cpp`  |

**Direct** = command-line, CSV-driven, minimal API  
**Facade** = research/education, CLI with introspection, model hacking, advanced state control

---

## Requirements

- **CUDA** (for `rnn.cu`, `facaded_rnn.cu`): NVIDIA GPU, CUDA Toolkit 11+, C++11
- **OpenCL** (for `rnn_opencl.cpp`, `facaded_rnn_opencl.cpp`): Any OpenCL 1.2+ device, C++11
- **C++ build tools:** g++, nvcc or clang++
- **Optional:** CMake

---

## Quickstart: Compiling & Running

**CUDA:**
```bash
# rnn.cu (vanilla CUDA)
nvcc -O2 -o rnn_cuda rnn.cu

# facaded_rnn.cu (facade CUDA)
nvcc -O2 -o facaded_rnn_cuda facaded_rnn.cu
```

**OpenCL:**
```bash
# rnn_opencl.cpp (vanilla OpenCL)
g++ -O2 -std=c++11 -o rnn_opencl rnn_opencl.cpp -lOpenCL

# facaded_rnn_opencl.cpp (facade OpenCL)
g++ -O2 -std=c++11 -o facaded_rnn_opencl facaded_rnn_opencl.cpp -lOpenCL
```

---

## CLI Usage and Help

Below are usage templates and help for each of the four modes (see also their `--help` output):

---

### 1. CUDA Command-Line Model (`rnn.cu`)

This module provides a basic (but advanced) CLI trainer/inferencer for RNNs, LSTMs, and GRUs using CUDA.

There is **no builtin help command**, but arguments typically follow:
- `--input input.csv` (input data file)
- `--target target.csv` (target data file)
- `--output preds.csv` (optional output file)
- `--hidden N` (hidden size), `--cell lstm|gru|rnn`
- `--epochs 100`, `--lr 0.01`, `--clip 5.0`
- `--model weights.bin` (save/load model)
- `--predict`, `--quiet`

Example (training):
```bash
nvcc -O2 -o rnn_cuda rnn.cu
./rnn_cuda --input train_x.csv --target train_y.csv --hidden 32 --cell lstm --epochs 100 --output preds.csv --model my_model.bin
```

Example (predict):
```bash
./rnn_cuda --input test_x.csv --model my_model.bin --predict --output preds.csv
```

---

### 2. OpenCL Command-Line Model (`rnn_opencl.cpp`)

#### Print built-in help:
```bash
./rnn_opencl --help
# or just run with no arguments
```

#### Example use:
```bash
# Train an LSTM with OpenCL:
./rnn_opencl --input train_x.csv --target train_y.csv --hidden 32 --cell lstm --epochs 100 --output preds.csv --model my_model_opencl.bin

# Predict
./rnn_opencl --input test_x.csv --model my_model_opencl.bin --predict --output preds.csv
```

#### Sample Help Output (abridged):
```
RNN OpenCL - Command-line Sequence Model (SimpleRNN/LSTM/GRU)

Commands:
  create      Create a new model
  train       Train model with data
  predict     Predict output sequence
  info        Print model info
  help        Print usage
Options:
  --input=N            Input size
  --hidden=N           Hidden size
  --output=N           Output size
  --cell=simple|lstm|gru   Cell type
  --loss=mse|ce        Loss function
  --save=FILE          Save model to file
  --model=FILE         Model file to load
  --data=FILE          CSV data file
  --epochs=N           Training epochs
  --lr=VALUE           Learning rate
  --clip=VALUE         Gradient clip value
  --normalize          Normalize input data
```

---

### 3. CUDA Facade (`facaded_rnn.cu`)

Facaded, object-oriented C++/CUDA RNN with high-visibility internals for step-by-step hacking, API access, and custom downstream code. Intended for direct integration or scientific research/teaching.

_No stand-alone CLI is bundled by default; typical use is through user C++ driver code, or as a library from an interactive/analysis tool._

**To integrate:**
```cpp
#include "facaded_rnn.cu"

// Construct and use the TRNNFacadeCUDA class
TRNNFacadeCUDA rnn(...);
rnn.TrainSequence(inputs, targets);
rnn.Predict(inputs);
rnn.GetHiddenValue(...);
```

---

### 4. OpenCL Facade (`facaded_rnn_opencl.cpp`)

A command-line introspectable OpenCL RNN CLI with a rich set of subcommands for model engineering, debugging, and research.

#### Print help:
```bash
./facaded_rnn_opencl help
# or with no args for summary
./facaded_rnn_opencl
```

#### Example CLI commands:
- Create a model:
    ```bash
    ./facaded_rnn_opencl create --input-size 10 --hidden-sizes 32,32 --output-size 2 --cell-type gru
    ```
- Info:
    ```bash
    ./facaded_rnn_opencl info
    ```
- Save/load weights:
    ```bash
    ./facaded_rnn_opencl save --model-file my_facade.bin
    ./facaded_rnn_opencl load --model-file my_facade.bin
    ```

#### OpenCL Facade CLI Example Output:
```
RNN Facade CLI (OpenCL GPU) - Matthew Abbott 2025

Usage: facaded_rnn_opencl <command> [options]

Commands:
  create              Create and initialize an RNN model
  train               Train the model on data
  predict             Run prediction on input data
  save                Save model weights to file
  load                Load model weights from file
  info                Display GPU information

Facade Introspection Commands:
  get-hidden          Get hidden state value
  set-hidden          Set hidden state value
  get-output          Get output value at timestep
  get-cell-state      Get LSTM cell state
  get-gate            Get gate value (LSTM/GRU)
  get-preactivation   Get pre-activation value
  get-input           Get input vector value
  reset-states        Reset all hidden/cell states
  set-dropout         Set dropout rate
  get-dropout         Get current dropout rate
  detect-vanishing    Check for vanishing gradients
  detect-exploding    Check for exploding gradients
  get-seq-outputs     Get all outputs for a sequence
  get-seq-hidden      Get hidden states over sequence

Create/Train/Predict options:
  --input-size <n>       Input dimension (required)
  --hidden-sizes <n,n>   Comma-separated hidden layer sizes (required)
  --output-size <n>      Output dimension (required)
  --cell-type <type>     rnn, lstm, or gru (default: lstm)
  --activation <type>    sigmoid, tanh, relu, linear (default: tanh)
  --output-activation    Output layer activation (default: sigmoid)
  --loss <type>          mse or crossentropy (default: mse)
  --learning-rate <f>    Learning rate (default: 0.01)
  --gradient-clip <f>    Gradient clipping value (default: 5.0)
  --bptt-steps <n>       BPTT truncation steps (default: 0 = full)
  --epochs <n>           Number of training epochs (default: 100)
  --input-file <file>    CSV file with input sequences
  --target-file <file>   CSV file with target sequences
  --output-file <file>   CSV file to write predictions

Facade options:
  --layer <n>            Layer index (default: 0)
  --timestep <n>         Timestep index (default: 0)
  --neuron <n>           Neuron index (default: 0)
  --output-idx <n>       Output index (default: 0)
  --value <f>            Value to set
  --gate <type>          Gate type: forget,input,output,cell,update,reset,hidden
  --threshold <f>        Threshold for gradient detection (default: 1e-6)
```

---

#### All Facade Introspection Options

- `get-hidden`, `set-hidden` — Probe/set hidden state at any layer/timestep/neuron
- `get-output` — Retrieve output at any timestep and index
- `get-cell-state` — Retrieve LSTM cell state
- `get-gate` — Inspect value of a gate (LSTM: forget/input/output/cell; GRU: update/reset/hidden)
- `detect-vanishing`, `detect-exploding` — Test for vanishing/exploding gradients
- `reset-states`, `set-dropout`, `get-dropout` — Reset/init model state or set dropout
- `get-seq-outputs` — Print the list of all model outputs for a sequence
- `get-seq-hidden` — Print the hidden state for a layer across a sequence

---

## Architecture Notes

- **Direct** (`rnn.cu`, `rnn_opencl.cpp`): Lower-abstraction, classic training loops, best for simple or "production" custom scripts.
- **Facade** (`facaded_rnn.cu`, `facaded_rnn_opencl.cpp`): Designed for deep research, teaching, and introspection—run, hack, and inspect.

All versions are meant to be highly readable and extensible, prioritizing clarity and learning.

---

## Data Structures & Internals

- **Enums:** `TCellType`, `TActivationType`, `TLossType`
- **Cell classes:** Separate for SimpleRNN, LSTM, GRU, and Output
- **Memory utilities:** `DeviceArray`, `GPUMatrix`, `CLArray`, etc. for easy CUDA/OpenCL host-device access
- **Introspection:** All activations, gradients, and network states are accessible to users (especially in the facade builds)

---

## License

MIT License  
© 2025 Matthew Abbott

---
