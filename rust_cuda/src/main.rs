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

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::manual_strip)]

use clap::{Parser, Subcommand};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;

const BLOCK_SIZE: u32 = 256;

type DArray = Vec<f64>;
type TDArray2D = Vec<DArray>;
type TDArray3D = Vec<TDArray2D>;
#[allow(dead_code)]
type TIntArray = Vec<i32>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationType {
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
}

impl ActivationType {
    #[allow(dead_code)]
    fn as_int(self) -> i32 {
        match self {
            ActivationType::Sigmoid => 0,
            ActivationType::Tanh => 1,
            ActivationType::ReLU => 2,
            ActivationType::Linear => 3,
        }
    }
}

impl std::str::FromStr for ActivationType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sigmoid" => Ok(ActivationType::Sigmoid),
            "tanh" => Ok(ActivationType::Tanh),
            "relu" => Ok(ActivationType::ReLU),
            "linear" => Ok(ActivationType::Linear),
            _ => Err(format!("Unknown activation: {}", s)),
        }
    }
}

impl std::fmt::Display for ActivationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationType::Sigmoid => write!(f, "sigmoid"),
            ActivationType::Tanh => write!(f, "tanh"),
            ActivationType::ReLU => write!(f, "relu"),
            ActivationType::Linear => write!(f, "linear"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LossType {
    MSE,
    CrossEntropy,
}

impl std::str::FromStr for LossType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mse" => Ok(LossType::MSE),
            "crossentropy" => Ok(LossType::CrossEntropy),
            _ => Err(format!("Unknown loss: {}", s)),
        }
    }
}

impl std::fmt::Display for LossType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LossType::MSE => write!(f, "mse"),
            LossType::CrossEntropy => write!(f, "crossentropy"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CellType {
    SimpleRNN,
    LSTM,
    GRU,
}

impl std::str::FromStr for CellType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "simplernn" => Ok(CellType::SimpleRNN),
            "lstm" => Ok(CellType::LSTM),
            "gru" => Ok(CellType::GRU),
            _ => Err(format!("Unknown cell type: {}", s)),
        }
    }
}

impl std::fmt::Display for CellType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CellType::SimpleRNN => write!(f, "simplernn"),
            CellType::LSTM => write!(f, "lstm"),
            CellType::GRU => write!(f, "gru"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LayerCache {
    pub h: DArray,
    pub c: DArray,
    pub pre_h: DArray,
    pub f: DArray,
    pub i: DArray,
    pub c_tilde: DArray,
    pub o: DArray,
    pub tanh_c: DArray,
    pub z: DArray,
    pub r: DArray,
    pub h_tilde: DArray,
    pub input: DArray,
}

#[derive(Debug, Clone, Default)]
pub struct TimeStepCache {
    pub input: DArray,
    pub layers: Vec<LayerCache>,
    pub out_val: DArray,
    pub out_pre: DArray,
}

const CUDA_KERNEL_SRC: &str = r#"
extern "C" {

__device__ double d_sigmoid(double x) {
    double clamped = fmax(-500.0, fmin(500.0, x));
    return 1.0 / (1.0 + exp(-clamped));
}

__device__ double d_tanh_act(double x) {
    return tanh(x);
}

__device__ double d_relu(double x) {
    return x > 0 ? x : 0;
}

__device__ double d_activation(double x, int actType) {
    switch (actType) {
        case 0: return d_sigmoid(x);
        case 1: return d_tanh_act(x);
        case 2: return d_relu(x);
        case 3: return x;
        default: return x;
    }
}

__device__ double d_activation_derivative(double y, int actType) {
    switch (actType) {
        case 0: return y * (1.0 - y);
        case 1: return 1.0 - y * y;
        case 2: return y > 0 ? 1.0 : 0.0;
        case 3: return 1.0;
        default: return 1.0;
    }
}

__device__ double d_clip(double v, double maxVal) {
    if (v > maxVal) return maxVal;
    else if (v < -maxVal) return -maxVal;
    else return v;
}

__global__ void k_matvec_add(double* y, const double* W, const double* x, const double* b,
                              int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        double sum = b[i];
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

__global__ void k_activate(double* y, const double* x, int n, int actType) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = d_activation(x[i], actType);
    }
}

// Split LSTM into two kernels to stay within cudarc's 12-param limit
// Step 1: Compute gates from sums
__global__ void k_lstm_gates(double* Fg, double* Ig, double* CTilde, double* Og,
                              const double* SumF, const double* SumI,
                              const double* SumC, const double* SumO, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        Fg[k] = d_sigmoid(SumF[k]);
        Ig[k] = d_sigmoid(SumI[k]);
        CTilde[k] = tanh(SumC[k]);
        Og[k] = d_sigmoid(SumO[k]);
    }
}

// Step 2: Compute cell state and hidden state
__global__ void k_lstm_state(double* H, double* C, double* TanhC,
                              const double* Fg, const double* Ig, const double* CTilde,
                              const double* Og, const double* PrevC, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        C[k] = Fg[k] * PrevC[k] + Ig[k] * CTilde[k];
        TanhC[k] = tanh(C[k]);
        H[k] = Og[k] * TanhC[k];
    }
}

__global__ void k_gru_gates(double* Z, double* R, const double* SumZ, const double* SumR, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        Z[k] = d_sigmoid(SumZ[k]);
        R[k] = d_sigmoid(SumR[k]);
    }
}

__global__ void k_gru_hidden(double* H, double* HTilde, const double* SumH,
                              const double* Z, const double* PrevH, int hiddenSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < hiddenSize) {
        HTilde[k] = tanh(SumH[k]);
        H[k] = (1.0 - Z[k]) * PrevH[k] + Z[k] * HTilde[k];
    }
}

__global__ void k_simple_rnn_forward(double* H, double* PreH, const double* Sum, int hiddenSize, int actType) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hiddenSize) {
        PreH[i] = Sum[i];
        H[i] = d_activation(Sum[i], actType);
    }
}

__global__ void k_zero(double* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = 0.0;
    }
}

}
"#;

pub struct CudaContext {
    pub device: Arc<CudaDevice>,
}

impl CudaContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;
        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SRC)?;
        device.load_ptx(ptx, "rnn_kernels", &[
            "k_matvec_add",
            "k_activate",
            "k_lstm_gates",
            "k_lstm_state",
            "k_gru_gates",
            "k_gru_hidden",
            "k_simple_rnn_forward",
            "k_zero",
        ])?;
        Ok(Self { device })
    }

    pub fn get_launch_config(&self, n: usize) -> LaunchConfig {
        let blocks = (n as u32).div_ceil(BLOCK_SIZE);
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

fn clip_value(v: f64, max_val: f64) -> f64 {
    if v > max_val {
        max_val
    } else if v < -max_val {
        -max_val
    } else {
        v
    }
}

fn random_weight(scale: f64) -> f64 {
    let mut rng = rand::thread_rng();
    (rng.gen::<f64>() - 0.5) * 2.0 * scale
}

fn init_matrix(rows: usize, cols: usize, scale: f64) -> TDArray2D {
    (0..rows)
        .map(|_| (0..cols).map(|_| random_weight(scale)).collect())
        .collect()
}

fn zero_matrix(rows: usize, cols: usize) -> TDArray2D {
    vec![vec![0.0; cols]; rows]
}

fn zero_array(size: usize) -> DArray {
    vec![0.0; size]
}

fn concat_arrays(a: &DArray, b: &DArray) -> DArray {
    let mut result = a.clone();
    result.extend(b.iter());
    result
}

fn flatten_matrix(m: &TDArray2D) -> Vec<f64> {
    m.iter().flat_map(|row| row.iter().copied()).collect()
}

pub struct Activation;

impl Activation {
    pub fn apply(x: f64, act_type: ActivationType) -> f64 {
        match act_type {
            ActivationType::Sigmoid => 1.0 / (1.0 + (-x.clamp(-500.0, 500.0)).exp()),
            ActivationType::Tanh => x.tanh(),
            ActivationType::ReLU => if x > 0.0 { x } else { 0.0 },
            ActivationType::Linear => x,
        }
    }

    pub fn derivative(y: f64, act_type: ActivationType) -> f64 {
        match act_type {
            ActivationType::Sigmoid => y * (1.0 - y),
            ActivationType::Tanh => 1.0 - y * y,
            ActivationType::ReLU => if y > 0.0 { 1.0 } else { 0.0 },
            ActivationType::Linear => 1.0,
        }
    }
}

pub struct Loss;

impl Loss {
    pub fn compute(pred: &DArray, target: &DArray, loss_type: LossType) -> f64 {
        let mut result = 0.0;
        match loss_type {
            LossType::MSE => {
                for i in 0..pred.len() {
                    result += (pred[i] - target[i]).powi(2);
                }
            }
            LossType::CrossEntropy => {
                for i in 0..pred.len() {
                    let p = pred[i].clamp(1e-15, 1.0 - 1e-15);
                    result -= target[i] * p.ln() + (1.0 - target[i]) * (1.0 - p).ln();
                }
            }
        }
        result / pred.len() as f64
    }

    pub fn gradient(pred: &DArray, target: &DArray, loss_type: LossType) -> DArray {
        match loss_type {
            LossType::MSE => pred.iter().zip(target.iter()).map(|(p, t)| p - t).collect(),
            LossType::CrossEntropy => {
                pred.iter()
                    .zip(target.iter())
                    .map(|(p, t)| {
                        let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
                        (p_clamped - t) / (p_clamped * (1.0 - p_clamped) + 1e-15)
                    })
                    .collect()
            }
        }
    }
}

pub struct SimpleRNNCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub activation: ActivationType,
    pub wih: TDArray2D,
    pub whh: TDArray2D,
    pub bh: DArray,
    pub d_wih: TDArray2D,
    pub d_whh: TDArray2D,
    pub d_bh: DArray,
}

impl SimpleRNNCell {
    pub fn new(input_size: usize, hidden_size: usize, activation: ActivationType) -> Self {
        let scale = (2.0 / (input_size + hidden_size) as f64).sqrt();
        Self {
            input_size,
            hidden_size,
            activation,
            wih: init_matrix(hidden_size, input_size, scale),
            whh: init_matrix(hidden_size, hidden_size, scale),
            bh: zero_array(hidden_size),
            d_wih: zero_matrix(hidden_size, input_size),
            d_whh: zero_matrix(hidden_size, hidden_size),
            d_bh: zero_array(hidden_size),
        }
    }

    pub fn forward(&self, input: &DArray, prev_h: &DArray) -> (DArray, DArray) {
        let mut h = vec![0.0; self.hidden_size];
        let mut pre_h = vec![0.0; self.hidden_size];

        for i in 0..self.hidden_size {
            let mut sum = self.bh[i];
            for j in 0..self.input_size {
                sum += self.wih[i][j] * input[j];
            }
            for j in 0..self.hidden_size {
                sum += self.whh[i][j] * prev_h[j];
            }
            pre_h[i] = sum;
            h[i] = Activation::apply(sum, self.activation);
        }
        (h, pre_h)
    }

    pub fn backward(
        &mut self,
        d_h: &DArray,
        h: &DArray,
        _pre_h: &DArray,
        prev_h: &DArray,
        input: &DArray,
        clip_val: f64,
    ) -> (DArray, DArray) {
        let mut d_h_raw = vec![0.0; self.hidden_size];
        let mut d_input = vec![0.0; self.input_size];
        let mut d_prev_h = vec![0.0; self.hidden_size];

        for i in 0..self.hidden_size {
            d_h_raw[i] = clip_value(d_h[i] * Activation::derivative(h[i], self.activation), clip_val);
        }

        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                self.d_wih[i][j] += d_h_raw[i] * input[j];
                d_input[j] += self.wih[i][j] * d_h_raw[i];
            }
            for j in 0..self.hidden_size {
                self.d_whh[i][j] += d_h_raw[i] * prev_h[j];
                d_prev_h[j] += self.whh[i][j] * d_h_raw[i];
            }
            self.d_bh[i] += d_h_raw[i];
        }
        (d_input, d_prev_h)
    }

    pub fn apply_gradients(&mut self, lr: f64, clip_val: f64) {
        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                self.wih[i][j] -= lr * clip_value(self.d_wih[i][j], clip_val);
                self.d_wih[i][j] = 0.0;
            }
            for j in 0..self.hidden_size {
                self.whh[i][j] -= lr * clip_value(self.d_whh[i][j], clip_val);
                self.d_whh[i][j] = 0.0;
            }
            self.bh[i] -= lr * clip_value(self.d_bh[i], clip_val);
            self.d_bh[i] = 0.0;
        }
    }

    pub fn reset_gradients(&mut self) {
        self.d_wih = zero_matrix(self.hidden_size, self.input_size);
        self.d_whh = zero_matrix(self.hidden_size, self.hidden_size);
        self.d_bh = zero_array(self.hidden_size);
    }
}

pub struct LSTMCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub activation: ActivationType,
    pub wf: TDArray2D,
    pub wi: TDArray2D,
    pub wc: TDArray2D,
    pub wo: TDArray2D,
    pub bf: DArray,
    pub bi: DArray,
    pub bc: DArray,
    pub bo: DArray,
    pub d_wf: TDArray2D,
    pub d_wi: TDArray2D,
    pub d_wc: TDArray2D,
    pub d_wo: TDArray2D,
    pub d_bf: DArray,
    pub d_bi: DArray,
    pub d_bc: DArray,
    pub d_bo: DArray,
    ctx: Option<Arc<CudaDevice>>,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize, activation: ActivationType) -> Self {
        let concat_size = input_size + hidden_size;
        let scale = (2.0 / concat_size as f64).sqrt();

        let mut bf = vec![0.0; hidden_size];
        for i in 0..hidden_size {
            bf[i] = 1.0;
        }

        Self {
            input_size,
            hidden_size,
            activation,
            wf: init_matrix(hidden_size, concat_size, scale),
            wi: init_matrix(hidden_size, concat_size, scale),
            wc: init_matrix(hidden_size, concat_size, scale),
            wo: init_matrix(hidden_size, concat_size, scale),
            bf,
            bi: zero_array(hidden_size),
            bc: zero_array(hidden_size),
            bo: zero_array(hidden_size),
            d_wf: zero_matrix(hidden_size, concat_size),
            d_wi: zero_matrix(hidden_size, concat_size),
            d_wc: zero_matrix(hidden_size, concat_size),
            d_wo: zero_matrix(hidden_size, concat_size),
            d_bf: zero_array(hidden_size),
            d_bi: zero_array(hidden_size),
            d_bc: zero_array(hidden_size),
            d_bo: zero_array(hidden_size),
            ctx: None,
        }
    }

    pub fn set_cuda_context(&mut self, device: Arc<CudaDevice>) {
        self.ctx = Some(device);
    }

    pub fn forward(
        &self,
        input: &DArray,
        prev_h: &DArray,
        prev_c: &DArray,
    ) -> (DArray, DArray, DArray, DArray, DArray, DArray, DArray) {
        let concat = concat_arrays(input, prev_h);
        let concat_size = self.input_size + self.hidden_size;

        if let Some(ref device) = self.ctx {
            let d_concat = device.htod_sync_copy(&concat).unwrap();
            let d_prev_c = device.htod_sync_copy(prev_c).unwrap();

            let wf_flat = flatten_matrix(&self.wf);
            let wi_flat = flatten_matrix(&self.wi);
            let wc_flat = flatten_matrix(&self.wc);
            let wo_flat = flatten_matrix(&self.wo);

            let d_wf = device.htod_sync_copy(&wf_flat).unwrap();
            let d_wi = device.htod_sync_copy(&wi_flat).unwrap();
            let d_wc = device.htod_sync_copy(&wc_flat).unwrap();
            let d_wo = device.htod_sync_copy(&wo_flat).unwrap();
            let d_bf = device.htod_sync_copy(&self.bf).unwrap();
            let d_bi = device.htod_sync_copy(&self.bi).unwrap();
            let d_bc = device.htod_sync_copy(&self.bc).unwrap();
            let d_bo = device.htod_sync_copy(&self.bo).unwrap();

            let d_sum_f: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_sum_i: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_sum_c: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_sum_o: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();

            let d_h: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_c: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_fg: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_ig: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_c_tilde: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_og: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_tanh_c: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();

            let blocks = (self.hidden_size as u32).div_ceil(BLOCK_SIZE);
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            };

            let matvec_fn = device.get_func("rnn_kernels", "k_matvec_add").unwrap();
            unsafe {
                matvec_fn.clone().launch(cfg, (&d_sum_f, &d_wf, &d_concat, &d_bf, self.hidden_size as i32, concat_size as i32)).unwrap();
                matvec_fn.clone().launch(cfg, (&d_sum_i, &d_wi, &d_concat, &d_bi, self.hidden_size as i32, concat_size as i32)).unwrap();
                matvec_fn.clone().launch(cfg, (&d_sum_c, &d_wc, &d_concat, &d_bc, self.hidden_size as i32, concat_size as i32)).unwrap();
                matvec_fn.clone().launch(cfg, (&d_sum_o, &d_wo, &d_concat, &d_bo, self.hidden_size as i32, concat_size as i32)).unwrap();
            }

            // Step 1: Compute gates
            let lstm_gates_fn = device.get_func("rnn_kernels", "k_lstm_gates").unwrap();
            unsafe {
                lstm_gates_fn.launch(cfg, (
                    &d_fg, &d_ig, &d_c_tilde, &d_og,
                    &d_sum_f, &d_sum_i, &d_sum_c, &d_sum_o,
                    self.hidden_size as i32
                )).unwrap();
            }

            // Step 2: Compute cell and hidden state
            let lstm_state_fn = device.get_func("rnn_kernels", "k_lstm_state").unwrap();
            unsafe {
                lstm_state_fn.launch(cfg, (
                    &d_h, &d_c, &d_tanh_c,
                    &d_fg, &d_ig, &d_c_tilde, &d_og, &d_prev_c,
                    self.hidden_size as i32
                )).unwrap();
            }

            device.synchronize().unwrap();

            let h = device.dtoh_sync_copy(&d_h).unwrap();
            let c = device.dtoh_sync_copy(&d_c).unwrap();
            let fg = device.dtoh_sync_copy(&d_fg).unwrap();
            let ig = device.dtoh_sync_copy(&d_ig).unwrap();
            let c_tilde = device.dtoh_sync_copy(&d_c_tilde).unwrap();
            let og = device.dtoh_sync_copy(&d_og).unwrap();
            let tanh_c = device.dtoh_sync_copy(&d_tanh_c).unwrap();

            return (h, c, fg, ig, c_tilde, og, tanh_c);
        }

        // CPU fallback
        let mut h = vec![0.0; self.hidden_size];
        let mut c = vec![0.0; self.hidden_size];
        let mut fg = vec![0.0; self.hidden_size];
        let mut ig = vec![0.0; self.hidden_size];
        let mut c_tilde = vec![0.0; self.hidden_size];
        let mut og = vec![0.0; self.hidden_size];
        let mut tanh_c = vec![0.0; self.hidden_size];

        for k in 0..self.hidden_size {
            let mut sum_f = self.bf[k];
            let mut sum_i = self.bi[k];
            let mut sum_c = self.bc[k];
            let mut sum_o = self.bo[k];

            for j in 0..concat_size {
                sum_f += self.wf[k][j] * concat[j];
                sum_i += self.wi[k][j] * concat[j];
                sum_c += self.wc[k][j] * concat[j];
                sum_o += self.wo[k][j] * concat[j];
            }

            fg[k] = Activation::apply(sum_f, ActivationType::Sigmoid);
            ig[k] = Activation::apply(sum_i, ActivationType::Sigmoid);
            c_tilde[k] = sum_c.tanh();
            og[k] = Activation::apply(sum_o, ActivationType::Sigmoid);
            c[k] = fg[k] * prev_c[k] + ig[k] * c_tilde[k];
            tanh_c[k] = c[k].tanh();
            h[k] = og[k] * tanh_c[k];
        }

        (h, c, fg, ig, c_tilde, og, tanh_c)
    }

    pub fn backward(
        &mut self,
        d_h: &DArray,
        d_c: &DArray,
        _h: &DArray,
        _c: &DArray,
        fg: &DArray,
        ig: &DArray,
        c_tilde: &DArray,
        og: &DArray,
        tanh_c: &DArray,
        prev_h: &DArray,
        prev_c: &DArray,
        input: &DArray,
        clip_val: f64,
    ) -> (DArray, DArray, DArray) {
        let concat = concat_arrays(input, prev_h);
        let concat_size = concat.len();

        let mut d_og = vec![0.0; self.hidden_size];
        let mut d_c_total = vec![0.0; self.hidden_size];
        let mut d_fg = vec![0.0; self.hidden_size];
        let mut d_ig = vec![0.0; self.hidden_size];
        let mut d_c_tilde = vec![0.0; self.hidden_size];
        let mut d_input = vec![0.0; self.input_size];
        let mut d_prev_h = vec![0.0; self.hidden_size];
        let mut d_prev_c = vec![0.0; self.hidden_size];

        for k in 0..self.hidden_size {
            d_og[k] = clip_value(
                d_h[k] * tanh_c[k] * Activation::derivative(og[k], ActivationType::Sigmoid),
                clip_val,
            );
            d_c_total[k] = clip_value(
                d_h[k] * og[k] * (1.0 - tanh_c[k] * tanh_c[k]) + d_c[k],
                clip_val,
            );
            d_fg[k] = clip_value(
                d_c_total[k] * prev_c[k] * Activation::derivative(fg[k], ActivationType::Sigmoid),
                clip_val,
            );
            d_ig[k] = clip_value(
                d_c_total[k] * c_tilde[k] * Activation::derivative(ig[k], ActivationType::Sigmoid),
                clip_val,
            );
            d_c_tilde[k] = clip_value(
                d_c_total[k] * ig[k] * Activation::derivative(c_tilde[k], ActivationType::Tanh),
                clip_val,
            );
            d_prev_c[k] = d_c_total[k] * fg[k];
        }

        for k in 0..self.hidden_size {
            for j in 0..concat_size {
                self.d_wf[k][j] += d_fg[k] * concat[j];
                self.d_wi[k][j] += d_ig[k] * concat[j];
                self.d_wc[k][j] += d_c_tilde[k] * concat[j];
                self.d_wo[k][j] += d_og[k] * concat[j];

                if j < self.input_size {
                    d_input[j] += self.wf[k][j] * d_fg[k]
                        + self.wi[k][j] * d_ig[k]
                        + self.wc[k][j] * d_c_tilde[k]
                        + self.wo[k][j] * d_og[k];
                } else {
                    d_prev_h[j - self.input_size] += self.wf[k][j] * d_fg[k]
                        + self.wi[k][j] * d_ig[k]
                        + self.wc[k][j] * d_c_tilde[k]
                        + self.wo[k][j] * d_og[k];
                }
            }
            self.d_bf[k] += d_fg[k];
            self.d_bi[k] += d_ig[k];
            self.d_bc[k] += d_c_tilde[k];
            self.d_bo[k] += d_og[k];
        }

        (d_input, d_prev_h, d_prev_c)
    }

    pub fn apply_gradients(&mut self, lr: f64, clip_val: f64) {
        let concat_size = self.input_size + self.hidden_size;
        for k in 0..self.hidden_size {
            for j in 0..concat_size {
                self.wf[k][j] -= lr * clip_value(self.d_wf[k][j], clip_val);
                self.wi[k][j] -= lr * clip_value(self.d_wi[k][j], clip_val);
                self.wc[k][j] -= lr * clip_value(self.d_wc[k][j], clip_val);
                self.wo[k][j] -= lr * clip_value(self.d_wo[k][j], clip_val);
                self.d_wf[k][j] = 0.0;
                self.d_wi[k][j] = 0.0;
                self.d_wc[k][j] = 0.0;
                self.d_wo[k][j] = 0.0;
            }
            self.bf[k] -= lr * clip_value(self.d_bf[k], clip_val);
            self.bi[k] -= lr * clip_value(self.d_bi[k], clip_val);
            self.bc[k] -= lr * clip_value(self.d_bc[k], clip_val);
            self.bo[k] -= lr * clip_value(self.d_bo[k], clip_val);
            self.d_bf[k] = 0.0;
            self.d_bi[k] = 0.0;
            self.d_bc[k] = 0.0;
            self.d_bo[k] = 0.0;
        }
    }

    pub fn reset_gradients(&mut self) {
        let concat_size = self.input_size + self.hidden_size;
        self.d_wf = zero_matrix(self.hidden_size, concat_size);
        self.d_wi = zero_matrix(self.hidden_size, concat_size);
        self.d_wc = zero_matrix(self.hidden_size, concat_size);
        self.d_wo = zero_matrix(self.hidden_size, concat_size);
        self.d_bf = zero_array(self.hidden_size);
        self.d_bi = zero_array(self.hidden_size);
        self.d_bc = zero_array(self.hidden_size);
        self.d_bo = zero_array(self.hidden_size);
    }
}

pub struct GRUCell {
    pub input_size: usize,
    pub hidden_size: usize,
    pub activation: ActivationType,
    pub wz: TDArray2D,
    pub wr: TDArray2D,
    pub wh: TDArray2D,
    pub bz: DArray,
    pub br: DArray,
    pub bh: DArray,
    pub d_wz: TDArray2D,
    pub d_wr: TDArray2D,
    pub d_wh: TDArray2D,
    pub d_bz: DArray,
    pub d_br: DArray,
    pub d_bh: DArray,
    ctx: Option<Arc<CudaDevice>>,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize, activation: ActivationType) -> Self {
        let concat_size = input_size + hidden_size;
        let scale = (2.0 / concat_size as f64).sqrt();

        Self {
            input_size,
            hidden_size,
            activation,
            wz: init_matrix(hidden_size, concat_size, scale),
            wr: init_matrix(hidden_size, concat_size, scale),
            wh: init_matrix(hidden_size, concat_size, scale),
            bz: zero_array(hidden_size),
            br: zero_array(hidden_size),
            bh: zero_array(hidden_size),
            d_wz: zero_matrix(hidden_size, concat_size),
            d_wr: zero_matrix(hidden_size, concat_size),
            d_wh: zero_matrix(hidden_size, concat_size),
            d_bz: zero_array(hidden_size),
            d_br: zero_array(hidden_size),
            d_bh: zero_array(hidden_size),
            ctx: None,
        }
    }

    pub fn set_cuda_context(&mut self, device: Arc<CudaDevice>) {
        self.ctx = Some(device);
    }

    pub fn forward(&self, input: &DArray, prev_h: &DArray) -> (DArray, DArray, DArray, DArray) {
        let concat = concat_arrays(input, prev_h);
        let concat_size = self.input_size + self.hidden_size;

        if let Some(ref device) = self.ctx {
            let d_concat = device.htod_sync_copy(&concat).unwrap();
            let d_prev_h = device.htod_sync_copy(prev_h).unwrap();

            let wz_flat = flatten_matrix(&self.wz);
            let wr_flat = flatten_matrix(&self.wr);
            let wh_flat = flatten_matrix(&self.wh);

            let d_wz = device.htod_sync_copy(&wz_flat).unwrap();
            let d_wr = device.htod_sync_copy(&wr_flat).unwrap();
            let d_wh = device.htod_sync_copy(&wh_flat).unwrap();
            let d_bz = device.htod_sync_copy(&self.bz).unwrap();
            let d_br = device.htod_sync_copy(&self.br).unwrap();
            let d_bh = device.htod_sync_copy(&self.bh).unwrap();

            let d_sum_z: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_sum_r: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_sum_h: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();

            let d_h_out: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_z: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_r: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();
            let d_h_tilde: CudaSlice<f64> = device.alloc_zeros(self.hidden_size).unwrap();

            let blocks = (self.hidden_size as u32).div_ceil(BLOCK_SIZE);
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: 0,
            };

            let matvec_fn = device.get_func("rnn_kernels", "k_matvec_add").unwrap();
            unsafe {
                matvec_fn.clone().launch(cfg, (&d_sum_z, &d_wz, &d_concat, &d_bz, self.hidden_size as i32, concat_size as i32)).unwrap();
                matvec_fn.clone().launch(cfg, (&d_sum_r, &d_wr, &d_concat, &d_br, self.hidden_size as i32, concat_size as i32)).unwrap();
            }

            let gru_gates_fn = device.get_func("rnn_kernels", "k_gru_gates").unwrap();
            unsafe {
                gru_gates_fn.launch(cfg, (&d_z, &d_r, &d_sum_z, &d_sum_r, self.hidden_size as i32)).unwrap();
            }
            device.synchronize().unwrap();

            // Download R to build ConcatR on CPU (mixed approach like original)
            let r_host = device.dtoh_sync_copy(&d_r).unwrap();
            let mut concat_r = vec![0.0; concat_size];
            for k in 0..self.input_size {
                concat_r[k] = input[k];
            }
            for k in 0..self.hidden_size {
                concat_r[self.input_size + k] = r_host[k] * prev_h[k];
            }
            let d_concat_r = device.htod_sync_copy(&concat_r).unwrap();

            unsafe {
                matvec_fn.launch(cfg, (&d_sum_h, &d_wh, &d_concat_r, &d_bh, self.hidden_size as i32, concat_size as i32)).unwrap();
            }

            let gru_hidden_fn = device.get_func("rnn_kernels", "k_gru_hidden").unwrap();
            unsafe {
                gru_hidden_fn.launch(cfg, (&d_h_out, &d_h_tilde, &d_sum_h, &d_z, &d_prev_h, self.hidden_size as i32)).unwrap();
            }
            device.synchronize().unwrap();

            let h = device.dtoh_sync_copy(&d_h_out).unwrap();
            let z = device.dtoh_sync_copy(&d_z).unwrap();
            let h_tilde = device.dtoh_sync_copy(&d_h_tilde).unwrap();

            return (h, z, r_host, h_tilde);
        }

        // CPU fallback
        let mut h = vec![0.0; self.hidden_size];
        let mut z = vec![0.0; self.hidden_size];
        let mut r = vec![0.0; self.hidden_size];
        let mut h_tilde = vec![0.0; self.hidden_size];

        for k in 0..self.hidden_size {
            let mut sum_z = self.bz[k];
            let mut sum_r = self.br[k];
            for j in 0..concat_size {
                sum_z += self.wz[k][j] * concat[j];
                sum_r += self.wr[k][j] * concat[j];
            }
            z[k] = Activation::apply(sum_z, ActivationType::Sigmoid);
            r[k] = Activation::apply(sum_r, ActivationType::Sigmoid);
        }

        let mut concat_r = vec![0.0; concat_size];
        for k in 0..self.input_size {
            concat_r[k] = input[k];
        }
        for k in 0..self.hidden_size {
            concat_r[self.input_size + k] = r[k] * prev_h[k];
        }

        for k in 0..self.hidden_size {
            let mut sum_h = self.bh[k];
            for j in 0..concat_size {
                sum_h += self.wh[k][j] * concat_r[j];
            }
            h_tilde[k] = sum_h.tanh();
            h[k] = (1.0 - z[k]) * prev_h[k] + z[k] * h_tilde[k];
        }

        (h, z, r, h_tilde)
    }

    pub fn backward(
        &mut self,
        d_h: &DArray,
        _h: &DArray,
        z: &DArray,
        r: &DArray,
        h_tilde: &DArray,
        prev_h: &DArray,
        input: &DArray,
        clip_val: f64,
    ) -> (DArray, DArray) {
        let concat = concat_arrays(input, prev_h);
        let concat_size = concat.len();

        let mut concat_r = vec![0.0; concat_size];
        for k in 0..self.input_size {
            concat_r[k] = input[k];
        }
        for k in 0..self.hidden_size {
            concat_r[self.input_size + k] = r[k] * prev_h[k];
        }

        let mut d_z = vec![0.0; self.hidden_size];
        let mut d_r = vec![0.0; self.hidden_size];
        let mut d_h_tilde = vec![0.0; self.hidden_size];
        let mut d_input = vec![0.0; self.input_size];
        let mut d_prev_h = vec![0.0; self.hidden_size];

        for k in 0..self.hidden_size {
            d_prev_h[k] = d_h[k] * (1.0 - z[k]);
        }

        for k in 0..self.hidden_size {
            d_h_tilde[k] = clip_value(
                d_h[k] * z[k] * Activation::derivative(h_tilde[k], ActivationType::Tanh),
                clip_val,
            );
            d_z[k] = clip_value(
                d_h[k] * (h_tilde[k] - prev_h[k]) * Activation::derivative(z[k], ActivationType::Sigmoid),
                clip_val,
            );
        }

        for k in 0..self.hidden_size {
            for j in 0..concat_size {
                self.d_wh[k][j] += d_h_tilde[k] * concat_r[j];
                if j < self.input_size {
                    d_input[j] += self.wh[k][j] * d_h_tilde[k];
                } else {
                    d_r[j - self.input_size] += self.wh[k][j] * d_h_tilde[k] * prev_h[j - self.input_size];
                    d_prev_h[j - self.input_size] += self.wh[k][j] * d_h_tilde[k] * r[j - self.input_size];
                }
            }
            self.d_bh[k] += d_h_tilde[k];
        }

        for k in 0..self.hidden_size {
            d_r[k] = clip_value(d_r[k] * Activation::derivative(r[k], ActivationType::Sigmoid), clip_val);
        }

        for k in 0..self.hidden_size {
            for j in 0..concat_size {
                self.d_wz[k][j] += d_z[k] * concat[j];
                self.d_wr[k][j] += d_r[k] * concat[j];
                if j < self.input_size {
                    d_input[j] += self.wz[k][j] * d_z[k] + self.wr[k][j] * d_r[k];
                } else {
                    d_prev_h[j - self.input_size] += self.wz[k][j] * d_z[k] + self.wr[k][j] * d_r[k];
                }
            }
            self.d_bz[k] += d_z[k];
            self.d_br[k] += d_r[k];
        }

        (d_input, d_prev_h)
    }

    pub fn apply_gradients(&mut self, lr: f64, clip_val: f64) {
        let concat_size = self.input_size + self.hidden_size;
        for k in 0..self.hidden_size {
            for j in 0..concat_size {
                self.wz[k][j] -= lr * clip_value(self.d_wz[k][j], clip_val);
                self.wr[k][j] -= lr * clip_value(self.d_wr[k][j], clip_val);
                self.wh[k][j] -= lr * clip_value(self.d_wh[k][j], clip_val);
                self.d_wz[k][j] = 0.0;
                self.d_wr[k][j] = 0.0;
                self.d_wh[k][j] = 0.0;
            }
            self.bz[k] -= lr * clip_value(self.d_bz[k], clip_val);
            self.br[k] -= lr * clip_value(self.d_br[k], clip_val);
            self.bh[k] -= lr * clip_value(self.d_bh[k], clip_val);
            self.d_bz[k] = 0.0;
            self.d_br[k] = 0.0;
            self.d_bh[k] = 0.0;
        }
    }

    pub fn reset_gradients(&mut self) {
        let concat_size = self.input_size + self.hidden_size;
        self.d_wz = zero_matrix(self.hidden_size, concat_size);
        self.d_wr = zero_matrix(self.hidden_size, concat_size);
        self.d_wh = zero_matrix(self.hidden_size, concat_size);
        self.d_bz = zero_array(self.hidden_size);
        self.d_br = zero_array(self.hidden_size);
        self.d_bh = zero_array(self.hidden_size);
    }
}

pub struct OutputLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub activation: ActivationType,
    pub w: TDArray2D,
    pub b: DArray,
    pub d_w: TDArray2D,
    pub d_b: DArray,
}

impl OutputLayer {
    pub fn new(input_size: usize, output_size: usize, activation: ActivationType) -> Self {
        let scale = (2.0 / input_size as f64).sqrt();
        Self {
            input_size,
            output_size,
            activation,
            w: init_matrix(output_size, input_size, scale),
            b: zero_array(output_size),
            d_w: zero_matrix(output_size, input_size),
            d_b: zero_array(output_size),
        }
    }

    pub fn forward(&self, input: &DArray) -> (DArray, DArray) {
        let mut pre = vec![0.0; self.output_size];
        let mut output = vec![0.0; self.output_size];

        for i in 0..self.output_size {
            let mut sum = self.b[i];
            for j in 0..self.input_size {
                sum += self.w[i][j] * input[j];
            }
            pre[i] = sum;
        }

        if self.activation == ActivationType::Linear {
            for i in 0..self.output_size {
                output[i] = pre[i];
            }
        } else {
            for i in 0..self.output_size {
                output[i] = Activation::apply(pre[i], self.activation);
            }
        }

        (output, pre)
    }

    pub fn backward(
        &mut self,
        d_out: &DArray,
        output: &DArray,
        _pre: &DArray,
        input: &DArray,
        clip_val: f64,
    ) -> DArray {
        let mut d_pre = vec![0.0; self.output_size];
        let mut d_input = vec![0.0; self.input_size];

        for i in 0..self.output_size {
            d_pre[i] = clip_value(d_out[i] * Activation::derivative(output[i], self.activation), clip_val);
        }

        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.d_w[i][j] += d_pre[i] * input[j];
                d_input[j] += self.w[i][j] * d_pre[i];
            }
            self.d_b[i] += d_pre[i];
        }

        d_input
    }

    pub fn apply_gradients(&mut self, lr: f64, clip_val: f64) {
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.w[i][j] -= lr * clip_value(self.d_w[i][j], clip_val);
                self.d_w[i][j] = 0.0;
            }
            self.b[i] -= lr * clip_value(self.d_b[i], clip_val);
            self.d_b[i] = 0.0;
        }
    }

    pub fn reset_gradients(&mut self) {
        self.d_w = zero_matrix(self.output_size, self.input_size);
        self.d_b = zero_array(self.output_size);
    }
}

pub enum RNNCells {
    Simple(Vec<SimpleRNNCell>),
    LSTM(Vec<LSTMCell>),
    GRU(Vec<GRUCell>),
}

pub struct AdvancedRNN {
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub cell_type: CellType,
    pub activation: ActivationType,
    pub output_activation: ActivationType,
    pub loss_type: LossType,
    pub learning_rate: f64,
    pub gradient_clip: f64,
    pub bptt_steps: usize,
    pub cells: RNNCells,
    pub output_layer: OutputLayer,
    cuda_ctx: Option<Arc<CudaDevice>>,
}

impl AdvancedRNN {
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        cell_type: CellType,
        activation: ActivationType,
        output_activation: ActivationType,
        loss_type: LossType,
        learning_rate: f64,
        gradient_clip: f64,
        bptt_steps: usize,
    ) -> Self {
        let mut prev_size = input_size;

        let cells = match cell_type {
            CellType::SimpleRNN => {
                let cells: Vec<_> = hidden_sizes
                    .iter()
                    .map(|&hs| {
                        let cell = SimpleRNNCell::new(prev_size, hs, activation);
                        prev_size = hs;
                        cell
                    })
                    .collect();
                RNNCells::Simple(cells)
            }
            CellType::LSTM => {
                let cells: Vec<_> = hidden_sizes
                    .iter()
                    .map(|&hs| {
                        let cell = LSTMCell::new(prev_size, hs, activation);
                        prev_size = hs;
                        cell
                    })
                    .collect();
                RNNCells::LSTM(cells)
            }
            CellType::GRU => {
                let cells: Vec<_> = hidden_sizes
                    .iter()
                    .map(|&hs| {
                        let cell = GRUCell::new(prev_size, hs, activation);
                        prev_size = hs;
                        cell
                    })
                    .collect();
                RNNCells::GRU(cells)
            }
        };

        let output_layer = OutputLayer::new(prev_size, output_size, output_activation);

        Self {
            input_size,
            output_size,
            hidden_sizes,
            cell_type,
            activation,
            output_activation,
            loss_type,
            learning_rate,
            gradient_clip,
            bptt_steps,
            cells,
            output_layer,
            cuda_ctx: None,
        }
    }

    pub fn set_cuda_context(&mut self, device: Arc<CudaDevice>) {
        self.cuda_ctx = Some(device.clone());
        match &mut self.cells {
            RNNCells::LSTM(cells) => {
                for cell in cells {
                    cell.set_cuda_context(device.clone());
                }
            }
            RNNCells::GRU(cells) => {
                for cell in cells {
                    cell.set_cuda_context(device.clone());
                }
            }
            _ => {}
        }
    }

    fn init_hidden_states(&self) -> TDArray3D {
        self.hidden_sizes
            .iter()
            .map(|&hs| vec![zero_array(hs), zero_array(hs)])
            .collect()
    }

    pub fn forward_sequence(
        &self,
        inputs: &TDArray2D,
        caches: &mut Vec<TimeStepCache>,
        states: &mut TDArray3D,
    ) -> TDArray2D {
        let mut result = Vec::with_capacity(inputs.len());
        let mut new_states = self.init_hidden_states();

        for t in 0..inputs.len() {
            let mut x = inputs[t].clone();
            caches[t].input = x.clone();
            caches[t].layers = vec![LayerCache::default(); self.hidden_sizes.len()];

            for layer in 0..self.hidden_sizes.len() {
                caches[t].layers[layer].input = x.clone();
                
                match &self.cells {
                    RNNCells::Simple(cells) => {
                        let (h, pre_h) = cells[layer].forward(&x, &states[layer][0]);
                        new_states[layer][0] = h.clone();
                        caches[t].layers[layer].h = h.clone();
                        caches[t].layers[layer].pre_h = pre_h;
                        x = h;
                    }
                    RNNCells::LSTM(cells) => {
                        let (h, c, fg, ig, c_tilde, og, tanh_c) =
                            cells[layer].forward(&x, &states[layer][0], &states[layer][1]);
                        new_states[layer][0] = h.clone();
                        new_states[layer][1] = c.clone();
                        caches[t].layers[layer].h = h.clone();
                        caches[t].layers[layer].c = c;
                        caches[t].layers[layer].f = fg;
                        caches[t].layers[layer].i = ig;
                        caches[t].layers[layer].c_tilde = c_tilde;
                        caches[t].layers[layer].o = og;
                        caches[t].layers[layer].tanh_c = tanh_c;
                        x = h;
                    }
                    RNNCells::GRU(cells) => {
                        let (h, z, r, h_tilde) = cells[layer].forward(&x, &states[layer][0]);
                        new_states[layer][0] = h.clone();
                        caches[t].layers[layer].h = h.clone();
                        caches[t].layers[layer].z = z;
                        caches[t].layers[layer].r = r;
                        caches[t].layers[layer].h_tilde = h_tilde;
                        x = h;
                    }
                }
            }

            let (out_val, out_pre) = self.output_layer.forward(&x);
            caches[t].out_val = out_val.clone();
            caches[t].out_pre = out_pre;
            result.push(out_val);

            *states = new_states.clone();
        }

        result
    }

    pub fn backward_sequence(
        &mut self,
        targets: &TDArray2D,
        caches: &Vec<TimeStepCache>,
        _states: &TDArray3D,
    ) -> f64 {
        let t_len = targets.len();
        let bptt_limit = if self.bptt_steps > 0 { self.bptt_steps } else { t_len };

        let mut total_loss = 0.0;

        let mut d_states_h: Vec<DArray> = self.hidden_sizes.iter().map(|&hs| zero_array(hs)).collect();
        let mut d_states_c: Vec<DArray> = self.hidden_sizes.iter().map(|&hs| zero_array(hs)).collect();

        let start = t_len.saturating_sub(bptt_limit);

        for t in (start..t_len).rev() {
            total_loss += Loss::compute(&caches[t].out_val, &targets[t], self.loss_type);
            let grad = Loss::gradient(&caches[t].out_val, &targets[t], self.loss_type);

            let last_layer = self.hidden_sizes.len() - 1;
            let mut d_h = self.output_layer.backward(
                &grad,
                &caches[t].out_val,
                &caches[t].out_pre,
                &caches[t].layers[last_layer].h,
                self.gradient_clip,
            );

            for layer in (0..self.hidden_sizes.len()).rev() {
                let mut d_out = vec![0.0; self.hidden_sizes[layer]];
                for k in 0..self.hidden_sizes[layer] {
                    d_out[k] = d_h[k] + d_states_h[layer][k];
                }

                let prev_h = if t > 0 {
                    caches[t - 1].layers[layer].h.clone()
                } else {
                    zero_array(self.hidden_sizes[layer])
                };

                let layer_cache = &caches[t].layers[layer];

                match &mut self.cells {
                    RNNCells::Simple(cells) => {
                        let (d_input, d_prev_h) = cells[layer].backward(
                            &d_out,
                            &layer_cache.h,
                            &layer_cache.pre_h,
                            &prev_h,
                            &layer_cache.input,
                            self.gradient_clip,
                        );
                        d_states_h[layer] = d_prev_h;
                        d_h = d_input;
                    }
                    RNNCells::LSTM(cells) => {
                        let prev_c = if t > 0 {
                            caches[t - 1].layers[layer].c.clone()
                        } else {
                            zero_array(self.hidden_sizes[layer])
                        };

                        let d_c = d_states_c[layer].clone();

                        let (d_input, d_prev_h, d_prev_c) = cells[layer].backward(
                            &d_out,
                            &d_c,
                            &layer_cache.h,
                            &layer_cache.c,
                            &layer_cache.f,
                            &layer_cache.i,
                            &layer_cache.c_tilde,
                            &layer_cache.o,
                            &layer_cache.tanh_c,
                            &prev_h,
                            &prev_c,
                            &layer_cache.input,
                            self.gradient_clip,
                        );
                        d_states_h[layer] = d_prev_h;
                        d_states_c[layer] = d_prev_c;
                        d_h = d_input;
                    }
                    RNNCells::GRU(cells) => {
                        let (d_input, d_prev_h) = cells[layer].backward(
                            &d_out,
                            &layer_cache.h,
                            &layer_cache.z,
                            &layer_cache.r,
                            &layer_cache.h_tilde,
                            &prev_h,
                            &layer_cache.input,
                            self.gradient_clip,
                        );
                        d_states_h[layer] = d_prev_h;
                        d_h = d_input;
                    }
                }
            }
        }

        total_loss / t_len as f64
    }

    pub fn train_sequence(&mut self, inputs: &TDArray2D, targets: &TDArray2D) -> f64 {
        self.reset_gradients();
        let mut caches: Vec<TimeStepCache> = (0..inputs.len()).map(|_| TimeStepCache::default()).collect();
        let mut states = self.init_hidden_states();
        self.forward_sequence(inputs, &mut caches, &mut states);
        let loss = self.backward_sequence(targets, &caches, &states);
        self.apply_gradients();
        loss
    }

    pub fn predict(&self, inputs: &TDArray2D) -> TDArray2D {
        let mut caches: Vec<TimeStepCache> = (0..inputs.len()).map(|_| TimeStepCache::default()).collect();
        let mut states = self.init_hidden_states();
        self.forward_sequence(inputs, &mut caches, &mut states)
    }

    pub fn compute_loss(&self, inputs: &TDArray2D, targets: &TDArray2D) -> f64 {
        let outputs = self.predict(inputs);
        let mut result = 0.0;
        for t in 0..outputs.len() {
            result += Loss::compute(&outputs[t], &targets[t], self.loss_type);
        }
        result / outputs.len() as f64
    }

    fn reset_gradients(&mut self) {
        match &mut self.cells {
            RNNCells::Simple(cells) => {
                for cell in cells {
                    cell.reset_gradients();
                }
            }
            RNNCells::LSTM(cells) => {
                for cell in cells {
                    cell.reset_gradients();
                }
            }
            RNNCells::GRU(cells) => {
                for cell in cells {
                    cell.reset_gradients();
                }
            }
        }
        self.output_layer.reset_gradients();
    }

    fn apply_gradients(&mut self) {
        match &mut self.cells {
            RNNCells::Simple(cells) => {
                for cell in cells {
                    cell.apply_gradients(self.learning_rate, self.gradient_clip);
                }
            }
            RNNCells::LSTM(cells) => {
                for cell in cells {
                    cell.apply_gradients(self.learning_rate, self.gradient_clip);
                }
            }
            RNNCells::GRU(cells) => {
                for cell in cells {
                    cell.apply_gradients(self.learning_rate, self.gradient_clip);
                }
            }
        }
        self.output_layer.apply_gradients(self.learning_rate, self.gradient_clip);
    }

    pub fn save_model_to_json(&self, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;

        writeln!(file, "{{")?;
        writeln!(file, "  \"input_size\": {},", self.input_size)?;
        writeln!(file, "  \"output_size\": {},", self.output_size)?;
        writeln!(file, "  \"hidden_sizes\": [")?;
        for (i, hs) in self.hidden_sizes.iter().enumerate() {
            if i > 0 {
                writeln!(file, ",")?;
            }
            write!(file, "    {}", hs)?;
        }
        writeln!(file, "\n  ],")?;

        writeln!(file, "  \"cell_type\": \"{}\",", self.cell_type)?;
        writeln!(file, "  \"activation\": \"{}\",", self.activation)?;
        writeln!(file, "  \"output_activation\": \"{}\",", self.output_activation)?;
        writeln!(file, "  \"loss_type\": \"{}\",", self.loss_type)?;
        writeln!(file, "  \"learning_rate\": {:.17},", self.learning_rate)?;
        writeln!(file, "  \"gradient_clip\": {:.17},", self.gradient_clip)?;
        writeln!(file, "  \"bptt_steps\": {},", self.bptt_steps)?;
        writeln!(file, "  \"dropout_rate\": 0,")?;

        writeln!(file, "  \"cells\": [")?;

        match &self.cells {
            RNNCells::Simple(cells) => {
                for (i, cell) in cells.iter().enumerate() {
                    if i > 0 {
                        writeln!(file, ",")?;
                    }
                    writeln!(file, "    {{")?;
                    writeln!(file, "      \"Wih\": {},", array_2d_to_json(&cell.wih))?;
                    writeln!(file, "      \"Whh\": {},", array_2d_to_json(&cell.whh))?;
                    writeln!(file, "      \"bh\": {}", array_1d_to_json(&cell.bh))?;
                    write!(file, "    }}")?;
                }
            }
            RNNCells::LSTM(cells) => {
                for (i, cell) in cells.iter().enumerate() {
                    if i > 0 {
                        writeln!(file, ",")?;
                    }
                    writeln!(file, "    {{")?;
                    writeln!(file, "      \"Wf\": {},", array_2d_to_json(&cell.wf))?;
                    writeln!(file, "      \"Wi\": {},", array_2d_to_json(&cell.wi))?;
                    writeln!(file, "      \"Wc\": {},", array_2d_to_json(&cell.wc))?;
                    writeln!(file, "      \"Wo\": {},", array_2d_to_json(&cell.wo))?;
                    writeln!(file, "      \"Bf\": {},", array_1d_to_json(&cell.bf))?;
                    writeln!(file, "      \"Bi\": {},", array_1d_to_json(&cell.bi))?;
                    writeln!(file, "      \"Bc\": {},", array_1d_to_json(&cell.bc))?;
                    writeln!(file, "      \"Bo\": {}", array_1d_to_json(&cell.bo))?;
                    write!(file, "    }}")?;
                }
            }
            RNNCells::GRU(cells) => {
                for (i, cell) in cells.iter().enumerate() {
                    if i > 0 {
                        writeln!(file, ",")?;
                    }
                    writeln!(file, "    {{")?;
                    writeln!(file, "      \"Wz\": {},", array_2d_to_json(&cell.wz))?;
                    writeln!(file, "      \"Wr\": {},", array_2d_to_json(&cell.wr))?;
                    writeln!(file, "      \"Wh\": {},", array_2d_to_json(&cell.wh))?;
                    writeln!(file, "      \"Bz\": {},", array_1d_to_json(&cell.bz))?;
                    writeln!(file, "      \"Br\": {},", array_1d_to_json(&cell.br))?;
                    writeln!(file, "      \"Bh\": {}", array_1d_to_json(&cell.bh))?;
                    write!(file, "    }}")?;
                }
            }
        }

        writeln!(file, "\n  ],")?;

        writeln!(file, "  \"output_layer\": {{")?;
        writeln!(file, "    \"W\": {},", array_2d_to_json(&self.output_layer.w))?;
        writeln!(file, "    \"B\": {}", array_1d_to_json(&self.output_layer.b))?;
        writeln!(file, "  }}")?;
        writeln!(file, "}}")?;

        println!("Model saved to JSON: {}", filename);
        Ok(())
    }

    pub fn load_model_from_json(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(filename)?;
        
        let input_size: usize = extract_json_value(&content, "input_size")
            .ok_or("Could not parse input_size")?
            .parse()?;
        let output_size: usize = extract_json_value(&content, "output_size")
            .ok_or("Could not parse output_size")?
            .parse()?;
        
        let cell_type_str = extract_json_value(&content, "cell_type").unwrap_or_else(|| "lstm".to_string());
        let cell_type: CellType = cell_type_str.parse()?;
        
        let hidden_str = extract_json_value(&content, "hidden_sizes").unwrap_or_else(|| "[32]".to_string());
        let hidden_sizes = parse_int_array(&hidden_str);
        
        let activation_str = extract_json_value(&content, "activation")
            .or_else(|| extract_json_value(&content, "hidden_activation"))
            .unwrap_or_else(|| "tanh".to_string());
        let activation: ActivationType = activation_str.parse()?;
        
        let output_act_str = extract_json_value(&content, "output_activation")
            .unwrap_or_else(|| "linear".to_string());
        let output_activation: ActivationType = output_act_str.parse()?;
        
        let loss_str = extract_json_value(&content, "loss_type").unwrap_or_else(|| "mse".to_string());
        let loss_type: LossType = loss_str.parse()?;
        
        let learning_rate: f64 = extract_json_value(&content, "learning_rate")
            .unwrap_or_else(|| "0.01".to_string())
            .parse()?;
        
        let gradient_clip: f64 = extract_json_value(&content, "gradient_clip")
            .unwrap_or_else(|| "5.0".to_string())
            .parse()?;
        
        let bptt_steps: usize = extract_json_value(&content, "bptt_steps")
            .unwrap_or_else(|| "0".to_string())
            .parse()?;

        let model = Self::new(
            input_size,
            hidden_sizes.iter().map(|&x| x as usize).collect(),
            output_size,
            cell_type,
            activation,
            output_activation,
            loss_type,
            learning_rate,
            gradient_clip,
            bptt_steps,
        );

        println!("Model loaded from JSON: {}", filename);
        Ok(model)
    }
}

fn array_1d_to_json(arr: &DArray) -> String {
    let parts: Vec<String> = arr.iter().map(|v| format!("{:.17}", v)).collect();
    format!("[{}]", parts.join(","))
}

fn array_2d_to_json(arr: &TDArray2D) -> String {
    let parts: Vec<String> = arr.iter().map(array_1d_to_json).collect();
    format!("[{}]", parts.join(","))
}

fn extract_json_value(json: &str, key: &str) -> Option<String> {
    let search_key = format!("\"{}\"", key);
    let key_pos = json.find(&search_key)?;
    
    let colon_pos = json[key_pos..].find(':')? + key_pos;
    let start_pos = colon_pos + 1;
    let rest = json[start_pos..].trim_start();
    
    if rest.starts_with('"') {
        let end_quote = rest[1..].find('"')? + 1;
        return Some(rest[1..end_quote].to_string());
    }
    
    if rest.starts_with('[') {
        let mut bracket_count = 1;
        let mut end_pos = 1;
        for c in rest[1..].chars() {
            if c == '[' {
                bracket_count += 1;
            } else if c == ']' {
                bracket_count -= 1;
            }
            if bracket_count == 0 {
                break;
            }
            end_pos += 1;
        }
        return Some(rest[..=end_pos].to_string());
    }
    
    let end_pos = rest.find([',', '}', ']'])?;
    Some(rest[..end_pos].trim().to_string())
}

fn parse_int_array(s: &str) -> Vec<i32> {
    let trimmed = s.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(',')
        .filter_map(|token| token.trim().parse().ok())
        .collect()
}

#[allow(dead_code)]
fn parse_double_array(s: &str) -> DArray {
    s.split(',')
        .filter_map(|token| token.trim().parse().ok())
        .collect()
}

fn load_data_from_csv(filename: &str) -> Result<(TDArray2D, TDArray2D), Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        
        let tokens: Vec<f64> = line
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        
        if tokens.len() >= 2 {
            let split_point = tokens.len() / 2;
            inputs.push(tokens[..split_point].to_vec());
            targets.push(tokens[split_point..].to_vec());
        }
    }
    
    Ok((inputs, targets))
}

#[derive(Parser)]
#[command(name = "rnn")]
#[command(about = "RNN (CUDA Accelerated with cudarc)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Create {
        #[arg(long)]
        input: usize,
        #[arg(long, value_delimiter = ',')]
        hidden: Vec<usize>,
        #[arg(long)]
        output: usize,
        #[arg(long)]
        save: String,
        #[arg(long, default_value = "lstm")]
        cell: String,
        #[arg(long, default_value = "0.01")]
        lr: f64,
        #[arg(long, default_value = "tanh")]
        hidden_act: String,
        #[arg(long, default_value = "linear")]
        output_act: String,
        #[arg(long, default_value = "mse")]
        loss: String,
        #[arg(long, default_value = "5.0")]
        clip: f64,
        #[arg(long, default_value = "0")]
        bptt: usize,
    },
    Train {
        #[arg(long)]
        model: String,
        #[arg(long)]
        data: String,
        #[arg(long)]
        save: String,
        #[arg(long, default_value = "100")]
        epochs: usize,
        #[arg(long, default_value = "1")]
        batch: usize,
        #[arg(long)]
        lr: Option<f64>,
        #[arg(long)]
        seq_len: Option<usize>,
        #[arg(long, default_value = "false")]
        verbose: bool,
    },
    Predict {
        #[arg(long)]
        model: String,
        #[arg(long, value_delimiter = ',')]
        input: Vec<f64>,
    },
    Info {
        #[arg(long)]
        model: String,
    },
    Help,
}

fn print_usage() {
    println!("RNN (CUDA Accelerated with cudarc)\n");
    println!("Commands:");
    println!("  create   Create a new RNN model and save to JSON");
    println!("  train    Train an existing model with data from JSON");
    println!("  predict  Make predictions with a trained model from JSON");
    println!("  info     Display model information from JSON");
    println!("  help     Show this help message\n");
    println!("Create Options:");
    println!("  --input=N              Input layer size (required)");
    println!("  --hidden=N,N,...       Hidden layer sizes (required)");
    println!("  --output=N             Output layer size (required)");
    println!("  --save=FILE.json       Save model to JSON file (required)");
    println!("  --cell=TYPE            simplernn|lstm|gru (default: lstm)");
    println!("  --lr=VALUE             Learning rate (default: 0.01)");
    println!("  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: tanh)");
    println!("  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)");
    println!("  --loss=TYPE            mse|crossentropy (default: mse)");
    println!("  --clip=VALUE           Gradient clipping (default: 5.0)");
    println!("  --bptt=N               BPTT steps (default: 0 = full)\n");
    println!("Train Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!("  --data=FILE.csv        Training data CSV file (required)");
    println!("  --save=FILE.json       Save trained model to JSON (required)");
    println!("  --epochs=N             Number of training epochs (default: 100)");
    println!("  --batch=N              Batch size (default: 1)");
    println!("  --lr=VALUE             Override learning rate");
    println!("  --seq-len=N            Sequence length (default: auto-detect)\n");
    println!("Predict Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!("  --input=v1,v2,...      Input values as CSV (required)\n");
    println!("Info Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)\n");
    println!("Options:");
    println!("  --input       Input size (create) or input values (predict)");
    println!("  --hidden      Hidden layer sizes (comma-separated)");
    println!("  --output      Output size");
    println!("  --cell        Cell type: simplernn, lstm, gru (default: lstm)");
    println!("  --hidden-act  Hidden activation: sigmoid, tanh, relu, linear (default: tanh)");
    println!("  --output-act  Output activation: sigmoid, tanh, relu, linear (default: linear)");
    println!("  --loss        Loss function: mse, crossentropy (default: mse)");
    println!("  --lr          Learning rate (default: 0.01)");
    println!("  --clip        Gradient clipping value (default: 5.0)");
    println!("  --bptt        BPTT steps (default: 0 = full sequence)");
    println!("  --epochs      Training epochs (default: 100)");
    println!("  --batch       Batch size (default: 1)");
    println!("  --model       Model file path");
    println!("  --data        Data file path");
    println!("  --save        Save file path");
    println!("  --verbose     Verbose output");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let cuda_ctx = match CudaContext::new() {
        Ok(ctx) => {
            println!("Using CUDA device: {}", ctx.device.name()?);
            Some(ctx)
        }
        Err(e) => {
            eprintln!("Warning: CUDA initialization failed ({}), falling back to CPU", e);
            None
        }
    };

    match cli.command {
        Commands::Create {
            input,
            hidden,
            output,
            save,
            cell,
            lr,
            hidden_act,
            output_act,
            loss,
            clip,
            bptt,
        } => {
            let cell_type: CellType = cell.parse()?;
            let hidden_activation: ActivationType = hidden_act.parse()?;
            let output_activation: ActivationType = output_act.parse()?;
            let loss_type: LossType = loss.parse()?;

            let mut model = AdvancedRNN::new(
                input,
                hidden.clone(),
                output,
                cell_type,
                hidden_activation,
                output_activation,
                loss_type,
                lr,
                clip,
                bptt,
            );

            if let Some(ref ctx) = cuda_ctx {
                model.set_cuda_context(ctx.device.clone());
            }

            println!("Created RNN model:");
            println!("  Input size: {}", input);
            println!("  Hidden sizes: {:?}", hidden);
            println!("  Output size: {}", output);
            println!("  Cell type: {}", cell_type);
            println!("  Hidden activation: {}", hidden_activation);
            println!("  Output activation: {}", output_activation);
            println!("  Loss function: {}", loss_type);
            println!("  Learning rate: {:.6}", lr);
            println!("  Gradient clip: {:.2}", clip);
            println!("  BPTT steps: {}", bptt);

            model.save_model_to_json(&save)?;
            println!("Model saved to: {}", save);
        }

        Commands::Train {
            model,
            data,
            save,
            epochs,
            batch: _,
            lr,
            seq_len: _,
            verbose,
        } => {
            println!("Loading model from JSON: {}", model);
            let mut rnn_model = AdvancedRNN::load_model_from_json(&model)?;
            
            if let Some(new_lr) = lr {
                rnn_model.learning_rate = new_lr;
            }

            if let Some(ref ctx) = cuda_ctx {
                rnn_model.set_cuda_context(ctx.device.clone());
            }

            println!("Model loaded successfully.");

            println!("Loading training data from: {}", data);
            let (inputs, targets) = load_data_from_csv(&data)?;

            if inputs.is_empty() {
                return Err("No data loaded from CSV file".into());
            }

            println!("Loaded {} timesteps of training data", inputs.len());
            println!("Starting training for {} epochs...", epochs);

            for epoch in 1..=epochs {
                let train_loss = rnn_model.train_sequence(&inputs, &targets);

                if !train_loss.is_nan() && !train_loss.is_infinite()
                    && (verbose || (epoch % 10 == 0) || (epoch == epochs)) {
                        println!(
                            "Epoch {:4}/{} - Loss: {:.6}",
                            epoch, epochs, train_loss
                        );
                    }
            }

            println!("Training completed.");
            println!("Saving trained model to: {}", save);
            rnn_model.save_model_to_json(&save)?;
        }

        Commands::Predict { model, input } => {
            let mut rnn_model = AdvancedRNN::load_model_from_json(&model)?;

            if let Some(ref ctx) = cuda_ctx {
                rnn_model.set_cuda_context(ctx.device.clone());
            }

            let inputs = vec![input.clone()];
            let predictions = rnn_model.predict(&inputs);

            print!("Input: ");
            for (i, v) in input.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.4}", v);
            }
            println!();

            if !predictions.is_empty() && !predictions.last().unwrap().is_empty() {
                print!("Output: ");
                let last_pred = predictions.last().unwrap();
                for (i, v) in last_pred.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{:.6}", v);
                }
                println!();

                if last_pred.len() > 1 {
                    let max_idx = last_pred
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap();
                    println!("Max index: {}", max_idx);
                }
            }
        }

        Commands::Info { model } => {
            println!("Loading model from JSON: {}", model);
            let content = std::fs::read_to_string(&model)?;

            println!("Model Information:");
            println!(
                "  Input size: {}",
                extract_json_value(&content, "input_size").unwrap_or_default()
            );
            println!(
                "  Output size: {}",
                extract_json_value(&content, "output_size").unwrap_or_default()
            );
            println!(
                "  Hidden sizes: {}",
                extract_json_value(&content, "hidden_sizes").unwrap_or_default()
            );
            println!(
                "  Cell type: {}",
                extract_json_value(&content, "cell_type").unwrap_or_default()
            );
            println!(
                "  Activation: {}",
                extract_json_value(&content, "activation").unwrap_or_default()
            );
            println!(
                "  Output activation: {}",
                extract_json_value(&content, "output_activation").unwrap_or_default()
            );
            println!(
                "  Loss type: {}",
                extract_json_value(&content, "loss_type").unwrap_or_default()
            );
            println!(
                "  Learning rate: {}",
                extract_json_value(&content, "learning_rate").unwrap_or_default()
            );
            println!(
                "  Gradient clip: {}",
                extract_json_value(&content, "gradient_clip").unwrap_or_default()
            );
            println!(
                "  BPTT steps: {}",
                extract_json_value(&content, "bptt_steps").unwrap_or_default()
            );
        }

        Commands::Help => {
            print_usage();
        }
    }

    Ok(())
}
