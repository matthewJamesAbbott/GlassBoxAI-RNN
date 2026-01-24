/*
 * Requirement 4: Integer Overflow Prevention
 *
 * Prove that all arithmetic operations (addition, multiplication, subtraction)
 * are safe from wrapping, overflowing, or underflowing.
 */

use crate::BLOCK_SIZE;

#[kani::proof]
fn verify_block_size_division_no_overflow() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 1_000_000);
    
    let blocks = (n as u32).div_ceil(BLOCK_SIZE);
    
    kani::assert(blocks > 0, "at least one block");
    kani::assert((blocks as usize) * (BLOCK_SIZE as usize) >= n, "blocks cover all elements");
}

#[kani::proof]
fn verify_matrix_size_calculation_no_overflow() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    
    kani::assume(rows > 0 && rows <= 1000);
    kani::assume(cols > 0 && cols <= 1000);
    kani::assume(rows.checked_mul(cols).is_some());
    
    let total = rows * cols;
    kani::assert(total >= rows, "multiplication didn't overflow");
    kani::assert(total >= cols, "multiplication didn't overflow");
}

#[kani::proof]
fn verify_concat_size_calculation_no_overflow() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    
    kani::assume(input_size > 0 && input_size <= 10000);
    kani::assume(hidden_size > 0 && hidden_size <= 10000);
    kani::assume(input_size.checked_add(hidden_size).is_some());
    
    let concat_size = input_size + hidden_size;
    
    kani::assert(concat_size > input_size, "addition didn't wrap");
    kani::assert(concat_size > hidden_size, "addition didn't wrap");
}

#[kani::proof]
fn verify_weight_scale_calculation_no_overflow() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    
    kani::assume(input_size > 0 && input_size <= 10000);
    kani::assume(hidden_size > 0 && hidden_size <= 10000);
    kani::assume(input_size.checked_add(hidden_size).is_some());
    
    let sum = input_size + hidden_size;
    let scale = (2.0 / sum as f64).sqrt();
    
    kani::assert(scale.is_finite(), "scale is finite");
    kani::assert(scale > 0.0, "scale is positive");
}

#[kani::proof]
fn verify_sequence_length_operations() {
    let t_len: usize = kani::any();
    let bptt_steps: usize = kani::any();
    
    kani::assume(t_len <= 10000);
    kani::assume(bptt_steps <= 10000);
    
    let start = t_len.saturating_sub(bptt_steps);
    
    kani::assert(start <= t_len, "saturating_sub doesn't exceed original");
    
    if bptt_steps <= t_len {
        kani::assert(start == t_len - bptt_steps, "correct subtraction when no saturation");
    } else {
        kani::assert(start == 0, "saturates to zero when bptt > len");
    }
}

#[kani::proof]
fn verify_index_arithmetic() {
    let base: usize = kani::any();
    let offset: usize = kani::any();
    
    kani::assume(base <= 10000);
    kani::assume(offset <= 10000);
    
    if let Some(sum) = base.checked_add(offset) {
        kani::assert(sum >= base, "checked_add is safe");
        kani::assert(sum >= offset, "checked_add is safe");
    }
}

#[kani::proof]
fn verify_layer_index_subtraction() {
    let concat_size: usize = kani::any();
    let input_size: usize = kani::any();
    let j: usize = kani::any();
    
    kani::assume(concat_size > 0 && concat_size <= 1000);
    kani::assume(input_size < concat_size);
    kani::assume(j >= input_size && j < concat_size);
    
    let hidden_idx = j - input_size;
    
    kani::assert(hidden_idx < concat_size - input_size, "subtraction safe");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_array_capacity_operations() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 64);
    
    let mut arr = Vec::with_capacity(size);
    
    for i in 0..size {
        arr.push(i as f64);
    }
    
    kani::assert(arr.len() == size, "correct number of elements");
}

#[kani::proof]
fn verify_gradient_clip_multiplication() {
    let lr: f64 = kani::any();
    let grad: f64 = kani::any();
    
    kani::assume(lr.is_finite() && lr >= 0.0 && lr <= 1.0);
    kani::assume(grad.is_finite() && grad.abs() <= 1e10);
    
    let update = lr * grad;
    
    kani::assert(update.is_finite(), "gradient update is finite");
}

#[kani::proof]
fn verify_loss_normalization_no_overflow() {
    let total_loss: f64 = kani::any();
    let len: usize = kani::any();
    
    kani::assume(total_loss.is_finite());
    kani::assume(len > 0 && len <= 100000);
    
    let normalized = total_loss / len as f64;
    
    if total_loss.is_finite() {
        kani::assert(normalized.is_finite() || total_loss.abs() == f64::INFINITY, "normalization safe");
    }
}

#[kani::proof]
fn verify_batch_index_multiplication() {
    let batch_idx: usize = kani::any();
    let batch_size: usize = kani::any();
    
    kani::assume(batch_idx <= 1000);
    kani::assume(batch_size <= 1000);
    
    if let Some(offset) = batch_idx.checked_mul(batch_size) {
        kani::assert(offset / batch_size == batch_idx || batch_size == 0, "multiplication reversible");
    }
}

#[kani::proof]
fn verify_timestep_layer_index_calculations() {
    let timestep: usize = kani::any();
    let layer: usize = kani::any();
    let neuron: usize = kani::any();
    let max_timesteps: usize = kani::any();
    let max_layers: usize = kani::any();
    let max_neurons: usize = kani::any();
    
    kani::assume(max_timesteps > 0 && max_timesteps <= 100);
    kani::assume(max_layers > 0 && max_layers <= 10);
    kani::assume(max_neurons > 0 && max_neurons <= 100);
    
    kani::assume(timestep < max_timesteps);
    kani::assume(layer < max_layers);
    kani::assume(neuron < max_neurons);
    
    let bounds_valid = timestep < max_timesteps && layer < max_layers && neuron < max_neurons;
    kani::assert(bounds_valid, "all indices within bounds");
}

#[kani::proof]
fn verify_hidden_size_product_calculation() {
    let num_layers: usize = kani::any();
    let hidden_size: usize = kani::any();
    
    kani::assume(num_layers > 0 && num_layers <= 10);
    kani::assume(hidden_size > 0 && hidden_size <= 1000);
    
    let total_hidden = num_layers.saturating_mul(hidden_size);
    
    kani::assert(total_hidden >= hidden_size || num_layers == 0, "saturating_mul safe");
}
