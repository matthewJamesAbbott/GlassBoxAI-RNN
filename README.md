# GlassBoxAI-RNN

## **Recurrent Neural Network Suite**

### *GPU-Accelerated RNN Implementations with Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-RNN is a comprehensive, production-ready Recurrent Neural Network implementation suite featuring:

- **Multiple GPU backends**: CUDA and OpenCL acceleration
- **Multiple language implementations**: C++ and Rust
- **Facade pattern architecture**: Clean API separation with deep introspection capabilities
- **Formal verification**: Kani-verified Rust implementation for memory safety guarantees
- **Multiple cell types**: SimpleRNN, LSTM, and GRU architectures
- **CISA/NSA Secure by Design compliance**: Built following government cybersecurity standards

This project demonstrates enterprise-grade software engineering practices including comprehensive testing, formal verification, cross-platform compatibility, and security-first development.

---

## **Table of Contents**

1. [Features](#features)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Prerequisites](#prerequisites)
5. [Installation & Compilation](#installation--compilation)
6. [CLI Reference](#cli-reference)
   - [Standard RNN Commands](#standard-rnn-commands)
   - [Facade RNN Commands](#facade-rnn-commands)
7. [Testing](#testing)
8. [Formal Verification with Kani](#formal-verification-with-kani)
9. [CISA/NSA Compliance](#cisansa-compliance)
10. [License](#license)
11. [Author](#author)

---

## **Features**

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Cell Types** | SimpleRNN, LSTM, and GRU architectures |
| **Multi-Layer Support** | Configurable stacked hidden layers |
| **Training** | Backpropagation Through Time (BPTT) with gradient clipping |
| **Activation Functions** | Sigmoid, Tanh, ReLU, Linear |
| **Loss Functions** | MSE, Cross-Entropy with stable softmax |
| **Model Persistence** | JSON serialization for model save/load |
| **Dropout** | Regularization support during training |
| **Sequence Learning** | Variable-length sequence processing |

### GPU Acceleration

| Backend | Implementation | Performance |
|---------|---------------|-------------|
| **CUDA** | Native CUDA kernels | Optimal for NVIDIA GPUs |
| **OpenCL** | Cross-platform GPU | AMD, Intel, NVIDIA support |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership model |
| **Formal Verification** | Kani proof harnesses |
| **Bounds Checking** | Verified array access |
| **Input Validation** | CLI argument validation |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GlassBoxAI-RNN                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │   C++ CUDA  │  │ C++ OpenCL  │  │         Rust CUDA           │  │
│  ├─────────────┤  ├─────────────┤  ├─────────────────────────────┤  │
│  │ • rnn.cu    │  │ • rnn_      │  │ • rust_cuda/                │  │
│  │ • facaded_  │  │   opencl.cpp│  │ • facaded_rust_cuda/        │  │
│  │   rnn.cu    │  │ • facaded_  │  │   └─ kani/                  │  │
│  │             │  │   rnn_      │  │      (Formal Verification)  │  │
│  │             │  │   opencl.cpp│  │                             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     Shared Features                             ││
│  │  • Consistent CLI interface across all implementations          ││
│  │  • JSON compatible model formats                                ││
│  │  • Comprehensive test suites                                    ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
GlassBoxAI-RNN/
│
├── rnn.cpp                     # C++ CPU RNN implementation
├── rnn.cu                      # C++ CUDA RNN implementation
├── rnn_opencl.cpp              # C++ OpenCL RNN implementation
├── facaded_rnn.cpp             # C++ CPU RNN with Facade pattern
├── facaded_rnn.cu              # C++ CUDA RNN with Facade pattern
├── facaded_rnn_opencl.cpp      # C++ OpenCL RNN with Facade pattern
│
├── rust_cuda/                  # Rust CUDA RNN implementation
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
│
├── facaded_rust_cuda/          # Rust CUDA RNN with Facade pattern
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       └── kani/               # Kani proof harnesses
│
├── rnn_cuda_tests.sh           # CUDA test suite
├── rnn_opencl_tests.sh         # OpenCL test suite
├── rnn_cpp_tests.sh            # C++ test suite
│
├── license.md                  # MIT License
└── README.md                   # This file
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **GCC/G++** | 11+ | C++ compilation |
| **CUDA Toolkit** | 12.0+ | CUDA compilation |
| **Rust** | 1.75+ | Rust compilation |

### Optional

| Dependency | Version | Purpose |
|------------|---------|---------|
| **OpenCL SDK** | 3.0 | OpenCL compilation |
| **Kani** | 0.67+ | Formal verification |

---

## **Installation & Compilation**

### **C++ CUDA Implementation**

```bash
# Standard RNN
nvcc -O2 -o rnn_cuda rnn.cu

# Facade RNN
nvcc -O2 -o facaded_rnn_cuda facaded_rnn.cu
```

### **C++ OpenCL Implementation**

```bash
# Standard RNN
g++ -O2 -std=c++11 -o rnn_opencl rnn_opencl.cpp -lOpenCL

# Facade RNN
g++ -O2 -std=c++11 -o facaded_rnn_opencl facaded_rnn_opencl.cpp -lOpenCL
```

### **C++ CPU Implementation**

```bash
# Standard RNN
g++ -O2 -std=c++11 -o rnn rnn.cpp

# Facade RNN
g++ -O2 -std=c++11 -o facaded_rnn facaded_rnn.cpp
```

### **Rust CUDA Implementation**

```bash
# Standard RNN
cd rust_cuda
cargo build --release

# Facade RNN
cd facaded_rust_cuda
cargo build --release
```

### **Build All**

```bash
# Build everything
nvcc -O2 -o rnn_cuda rnn.cu
nvcc -O2 -o facaded_rnn_cuda facaded_rnn.cu
g++ -O2 -std=c++11 -o rnn_opencl rnn_opencl.cpp -lOpenCL
g++ -O2 -std=c++11 -o facaded_rnn_opencl facaded_rnn_opencl.cpp -lOpenCL
g++ -O2 -std=c++11 -o rnn rnn.cpp
g++ -O2 -std=c++11 -o facaded_rnn facaded_rnn.cpp
(cd rust_cuda && cargo build --release)
(cd facaded_rust_cuda && cargo build --release)
```

---

## **CLI Reference**

### **Standard RNN Commands**

The standard RNN implementations provide core recurrent neural network functionality.

#### Usage

```
rnn_cuda <command> [options]
rnn_opencl <command> [options]
rust_cuda/target/release/rnn_cuda <command> [options]
```

#### Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new RNN model |
| `train` | Train the model with data |
| `predict` | Make predictions with a trained model |
| `info` | Display model information |
| `help` | Show help message |

#### Create Options

| Option | Description |
|--------|-------------|
| `--input=N` | Input layer size (required) |
| `--hidden=N,N,...` | Hidden layer sizes (required) |
| `--output=N` | Output layer size (required) |
| `--save=FILE.json` | Save model to JSON file (required) |
| `--cell=TYPE` | simplernn\|lstm\|gru (default: lstm) |
| `--lr=VALUE` | Learning rate (default: 0.01) |
| `--hidden-act=TYPE` | sigmoid\|tanh\|relu\|linear (default: tanh) |
| `--output-act=TYPE` | sigmoid\|tanh\|relu\|linear (default: linear) |
| `--loss=TYPE` | mse\|crossentropy (default: mse) |
| `--clip=VALUE` | Gradient clipping (default: 5.0) |
| `--bptt=N` | BPTT steps (default: 0 = full sequence) |

#### Train Options

| Option | Description |
|--------|-------------|
| `--model=FILE.json` | Load model from JSON file (required) |
| `--data=FILE.csv` | Training data CSV file (required) |
| `--save=FILE.json` | Save trained model to JSON (required) |
| `--epochs=N` | Number of training epochs (default: 100) |
| `--batch=N` | Batch size (default: 1) |
| `--lr=VALUE` | Override learning rate |
| `--seq-len=N` | Sequence length (default: auto-detect) |
| `--verbose` | Verbose output |

#### Predict Options

| Option | Description |
|--------|-------------|
| `--model=FILE.json` | Load model from JSON file (required) |
| `--input=v1,v2,...` | Input values as CSV (required) |

#### Info Options

| Option | Description |
|--------|-------------|
| `--model=FILE.json` | Load model from JSON file (required) |

#### Examples

```bash
# Create an LSTM model
rnn_cuda create --input=2 --hidden=16,16 --output=2 --cell=lstm --save=model.json

# Train the model
rnn_cuda train --model=model.json --data=train.csv --epochs=100 --save=model_trained.json

# Make predictions
rnn_cuda predict --model=model_trained.json --input=0.5,0.5

# Display model info
rnn_cuda info --model=model.json
```

---

### **Facade RNN Commands**

The Facade implementations include all standard commands plus deep introspection capabilities for research and debugging.

#### Additional Command

| Command | Description |
|---------|-------------|
| `query` | Query model state and internals |

#### Query Options

| Option | Description |
|--------|-------------|
| `--model=FILE.json` | Load model from JSON file (required) |
| `--query-type=TYPE` | Query type (required) |
| `--layer=N` | Layer index |
| `--timestep=N` | Timestep index |
| `--neuron=N` | Neuron index |
| `--index=N` | Generic index parameter |
| `--dropout-rate=VALUE` | Set dropout rate (0.0-1.0) |
| `--enable-dropout` | Enable dropout |
| `--disable-dropout` | Disable dropout |

#### Available Query Types

| Query Type | Description |
|------------|-------------|
| `input-size` | Get input layer size |
| `output-size` | Get output layer size |
| `hidden-size` | Get hidden layer size (requires --layer) |
| `cell-type` | Get cell type (simplernn/lstm/gru) |
| `sequence-length` | Get current sequence length |
| `dropout-rate` | Get current dropout rate |
| `hidden-state` | Get hidden state value (requires --layer, --timestep, --neuron) |

#### Facade Examples

```bash
# Create a new model
facaded_rnn_cuda create --input=2 --hidden=16 --output=2 --cell=lstm --save=model.json

# Get model information
facaded_rnn_cuda info --model=model.json

# Query input size
facaded_rnn_cuda query --model=model.json --query-type=input-size

# Query hidden size for layer 0
facaded_rnn_cuda query --model=model.json --query-type=hidden-size --layer=0

# Query cell type
facaded_rnn_cuda query --model=model.json --query-type=cell-type

# Query hidden state at specific location
facaded_rnn_cuda query --model=model.json --query-type=hidden-state --layer=0 --timestep=0 --neuron=0

# Set dropout rate
facaded_rnn_cuda query --model=model.json --query-type=dropout-rate --dropout-rate=0.5
```

---

## **Testing**

### Running All Tests

```bash
# Run CUDA tests
./rnn_cuda_tests.sh

# Run OpenCL tests
./rnn_opencl_tests.sh

# Run C++ tests
./rnn_cpp_tests.sh

# Run Rust tests
cd rust_cuda && cargo test
cd facaded_rust_cuda && cargo test
```

### Test Categories

Each test suite covers:

| Category | Tests |
|----------|-------|
| **Help & Usage** | Command-line interface verification |
| **Model Creation** | Various architecture configurations |
| **Cell Types** | SimpleRNN, LSTM, GRU |
| **Hyperparameters** | Learning rate, activation, loss functions |
| **Model Info** | Metadata retrieval |
| **Save & Load** | Model persistence |
| **Query/Introspection** | Hidden state, gate values inspection |
| **Error Handling** | Invalid input handling |
| **Cross-Implementation** | API compatibility |
| **Train & Predict** | End-to-end workflows |

### Test Output Example

```
=========================================
RNN CUDA Comprehensive Test Suite
=========================================

Group: Help & Usage
Test 1: RNN help command... PASS
Test 2: RNN --help flag... PASS
Test 3: RNN -h flag... PASS
...

=========================================
Test Summary
=========================================
Total tests: 50
Passed: 50
Failed: 0

All tests passed!
```

---

## **Formal Verification with Kani**

### Overview

The Rust Facade implementation includes **Kani formal verification proofs** that mathematically prove the absence of certain classes of bugs. This goes beyond traditional testing to provide **mathematical guarantees** about code correctness.

### Verification Categories

The test suite covers security verification categories:

| Category | Description |
|----------|-------------|
| **Strict Bound Checks** | Array/collection indexing safety |
| **Pointer Validity** | Slice-to-pointer conversion safety |
| **No-Panic Guarantee** | Enum and command handling safety |
| **Integer Overflow Prevention** | Weight size, dimension calculations |
| **Division-by-Zero Exclusion** | Launch config, sequence processing |
| **Global State Consistency** | Training mode state tracking |
| **Deadlock-Free Logic** | Arc reference counting |
| **Input Sanitization Bounds** | Loop iteration limits |
| **Result Coverage Audit** | Error handling completeness |
| **Memory Leak Prevention** | Vector allocation bounds |
| **Constant-Time Execution** | Timing-safe operations |
| **State Machine Integrity** | Training state transitions |
| **Enum Exhaustion** | Match statement completeness |
| **Floating-Point Sanity** | NaN/Infinity prevention |
| **Resource Limit Compliance** | Memory budget enforcement |

### Running Kani Verification

```bash
# Facade RNN
cd facaded_rust_cuda
cargo kani

# Run specific proof
cargo kani --harness verify_hidden_indexing
```

### Why Formal Verification Matters

Traditional testing can only verify specific test cases. Formal verification with Kani:

- **Exhaustively checks all possible inputs** within defined bounds
- **Mathematically proves** absence of panics, buffer overflows, and undefined behavior
- **Catches edge cases** that random testing might miss
- **Provides cryptographic-level assurance** for safety-critical code

---

## **CISA/NSA Compliance**

### Secure by Design

This project follows **CISA (Cybersecurity and Infrastructure Security Agency)** and **NSA (National Security Agency)** Secure by Design principles:

| Principle | Implementation |
|-----------|---------------|
| **Memory Safety** | Rust ownership model eliminates buffer overflows, use-after-free, and data races |
| **Formal Verification** | Kani proofs mathematically verify absence of critical bugs |
| **Input Validation** | All CLI inputs validated before processing |
| **Defense in Depth** | Multiple layers of safety (language, compiler, runtime checks) |
| **Secure Defaults** | Safe default configurations throughout |
| **Transparency** | Open source with full code visibility |

### Compliance Checklist

- [x] **Memory-safe language** (Rust implementation)
- [x] **Static analysis** (Rust compiler + Clippy)
- [x] **Formal verification** (Kani proof harnesses)
- [x] **Comprehensive testing** (Unit tests + integration tests)
- [x] **Bounds checking** (Verified array access)
- [x] **Input validation** (CLI argument parsing)
- [x] **No unsafe code in critical paths** (Where possible)
- [x] **Documentation** (Inline docs + README)
- [x] **Version control** (Git)
- [x] **License clarity** (MIT License)

### Attestation

This codebase has been developed following secure software development lifecycle (SSDLC) practices and demonstrates:

- **Formal verification proofs passed** (Kani proofs)
- **Zero warnings** compilation across all implementations
- **Consistent API** across all language/backend combinations
- **Production-ready** code quality

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## **Author**

**Matthew Abbott**  
Email: mattbachg@gmail.com

---

*Built with precision. Verified with rigor. Secured by design.*
