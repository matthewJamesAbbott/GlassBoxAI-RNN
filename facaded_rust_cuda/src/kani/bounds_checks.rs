/*
 * Requirement 1: Strict Bound Checks
 *
 * Prove that all collection indexing (arrays, slices, vectors) is mathematically
 * incapable of out-of-bounds access under any symbolic input.
 */

use crate::{
    ActivationType, Loss, LossType,
    zero_array, zero_matrix, concat_arrays, flatten_matrix,
    SimpleRNNCell, LSTMCell, GRUCell, OutputLayer,
};

#[kani::proof]
#[kani::unwind(10)]
fn verify_zero_array_bounds() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 64);
    
    let arr = zero_array(size);
    
    kani::assert(arr.len() == size, "zero_array length matches requested size");
    
    let idx: usize = kani::any();
    kani::assume(idx < size);
    let _val = arr[idx];
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_zero_matrix_bounds() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 8);
    kani::assume(cols > 0 && cols <= 8);
    
    let mat = zero_matrix(rows, cols);
    
    kani::assert(mat.len() == rows, "zero_matrix row count correct");
    
    let row_idx: usize = kani::any();
    let col_idx: usize = kani::any();
    kani::assume(row_idx < rows);
    kani::assume(col_idx < cols);
    
    kani::assert(mat[row_idx].len() == cols, "each row has correct column count");
    let _val = mat[row_idx][col_idx];
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_concat_arrays_bounds() {
    let size_a: usize = kani::any();
    let size_b: usize = kani::any();
    kani::assume(size_a > 0 && size_a <= 8);
    kani::assume(size_b > 0 && size_b <= 8);
    
    let a = zero_array(size_a);
    let b = zero_array(size_b);
    let result = concat_arrays(&a, &b);
    
    kani::assert(result.len() == size_a + size_b, "concat result has correct length");
    
    let idx: usize = kani::any();
    kani::assume(idx < size_a + size_b);
    let _val = result[idx];
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_flatten_matrix_bounds() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 4);
    kani::assume(cols > 0 && cols <= 4);
    
    let mat = zero_matrix(rows, cols);
    let flat = flatten_matrix(&mat);
    
    kani::assert(flat.len() == rows * cols, "flattened matrix has correct length");
    
    let idx: usize = kani::any();
    kani::assume(idx < rows * cols);
    let _val = flat[idx];
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_simple_rnn_cell_forward_bounds() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 4);
    kani::assume(hidden_size > 0 && hidden_size <= 4);
    
    let cell = SimpleRNNCell::new(input_size, hidden_size, ActivationType::Tanh);
    let input = zero_array(input_size);
    let prev_h = zero_array(hidden_size);
    
    let (h, pre_h) = cell.forward(&input, &prev_h);
    
    kani::assert(h.len() == hidden_size, "hidden state has correct size");
    kani::assert(pre_h.len() == hidden_size, "pre-activation has correct size");
    
    let idx: usize = kani::any();
    kani::assume(idx < hidden_size);
    let _h_val = h[idx];
    let _pre_val = pre_h[idx];
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_output_layer_forward_bounds() {
    let input_size: usize = kani::any();
    let output_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 4);
    kani::assume(output_size > 0 && output_size <= 4);
    
    let layer = OutputLayer::new(input_size, output_size, ActivationType::Linear);
    let input = zero_array(input_size);
    
    let (output, pre) = layer.forward(&input);
    
    kani::assert(output.len() == output_size, "output has correct size");
    kani::assert(pre.len() == output_size, "pre-activation has correct size");
    
    let idx: usize = kani::any();
    kani::assume(idx < output_size);
    let _out_val = output[idx];
    let _pre_val = pre[idx];
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_loss_compute_bounds() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);
    
    let pred = zero_array(size);
    let target = zero_array(size);
    
    let _loss_mse = Loss::compute(&pred, &target, LossType::MSE);
    let _loss_ce = Loss::compute(&pred, &target, LossType::CrossEntropy);
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_loss_gradient_bounds() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);
    
    let pred = zero_array(size);
    let target = zero_array(size);
    
    let grad_mse = Loss::gradient(&pred, &target, LossType::MSE);
    let grad_ce = Loss::gradient(&pred, &target, LossType::CrossEntropy);
    
    kani::assert(grad_mse.len() == size, "MSE gradient has correct size");
    kani::assert(grad_ce.len() == size, "CrossEntropy gradient has correct size");
}

#[kani::proof]
#[kani::unwind(20)]
fn verify_lstm_cell_forward_bounds() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 3);
    kani::assume(hidden_size > 0 && hidden_size <= 3);
    
    let cell = LSTMCell::new(input_size, hidden_size, ActivationType::Tanh);
    let input = zero_array(input_size);
    let prev_h = zero_array(hidden_size);
    let prev_c = zero_array(hidden_size);
    
    let (h, c, fg, ig, c_tilde, og, tanh_c) = cell.forward(&input, &prev_h, &prev_c);
    
    kani::assert(h.len() == hidden_size, "h has correct size");
    kani::assert(c.len() == hidden_size, "c has correct size");
    kani::assert(fg.len() == hidden_size, "forget gate has correct size");
    kani::assert(ig.len() == hidden_size, "input gate has correct size");
    kani::assert(c_tilde.len() == hidden_size, "cell candidate has correct size");
    kani::assert(og.len() == hidden_size, "output gate has correct size");
    kani::assert(tanh_c.len() == hidden_size, "tanh_c has correct size");
}

#[kani::proof]
#[kani::unwind(20)]
fn verify_gru_cell_forward_bounds() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 3);
    kani::assume(hidden_size > 0 && hidden_size <= 3);
    
    let cell = GRUCell::new(input_size, hidden_size, ActivationType::Tanh);
    let input = zero_array(input_size);
    let prev_h = zero_array(hidden_size);
    
    let (h, z, r, h_tilde) = cell.forward(&input, &prev_h);
    
    kani::assert(h.len() == hidden_size, "h has correct size");
    kani::assert(z.len() == hidden_size, "update gate has correct size");
    kani::assert(r.len() == hidden_size, "reset gate has correct size");
    kani::assert(h_tilde.len() == hidden_size, "hidden candidate has correct size");
}
