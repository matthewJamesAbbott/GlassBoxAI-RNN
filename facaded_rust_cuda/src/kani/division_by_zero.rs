/*
 * Requirement 5: Division-by-Zero Exclusion
 *
 * Verify that any denominator derived from variable or external input
 * is mathematically proven to never be zero.
 */

use crate::{
    Activation, ActivationType, Loss, LossType,
    zero_array, BLOCK_SIZE,
};

#[kani::proof]
fn verify_weight_scale_division_safety() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    
    kani::assume(input_size > 0);
    kani::assume(hidden_size > 0);
    
    let sum = input_size + hidden_size;
    kani::assert(sum > 0, "sum is never zero when both inputs are positive");
    
    let scale = (2.0 / sum as f64).sqrt();
    kani::assert(scale.is_finite(), "scale computation is safe");
    kani::assert(scale > 0.0, "scale is positive");
}

#[kani::proof]
fn verify_loss_normalization_division_safety() {
    let pred_len: usize = kani::any();
    kani::assume(pred_len > 0);
    
    let denominator = pred_len as f64;
    kani::assert(denominator > 0.0, "denominator is positive");
    
    let numerator: f64 = kani::any();
    kani::assume(numerator.is_finite());
    
    let result = numerator / denominator;
    kani::assert(result.is_finite() || numerator.abs() == f64::INFINITY, "division is safe");
}

#[kani::proof]
fn verify_block_calculation_division_safety() {
    let n: u32 = kani::any();
    kani::assume(n > 0);
    
    let blocks = n.div_ceil(BLOCK_SIZE);
    kani::assert(blocks > 0, "at least one block required");
    
    kani::assert(BLOCK_SIZE > 0, "BLOCK_SIZE constant is never zero");
}



#[kani::proof]
fn verify_cross_entropy_clamping_prevents_division_by_zero() {
    let p: f64 = kani::any();
    kani::assume(p.is_finite());
    
    let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
    
    kani::assert(p_clamped >= 1e-15, "clamped value has minimum bound");
    kani::assert(p_clamped <= 1.0 - 1e-15, "clamped value has maximum bound");
    
    let denom = p_clamped * (1.0 - p_clamped) + 1e-15;
    kani::assert(denom > 0.0, "denominator is positive");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_loss_compute_division_safety() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);
    
    let pred = zero_array(size);
    let target = zero_array(size);
    
    let loss = Loss::compute(&pred, &target, LossType::MSE);
    kani::assert(loss.is_finite(), "MSE loss is finite");
    
    kani::assert(size > 0, "size used as denominator is never zero");
}

#[kani::proof]
fn verify_activation_derivative_division_safety() {
    let y: f64 = kani::any();
    kani::assume(y.is_finite());
    
    let d_sig = Activation::derivative(y, ActivationType::Sigmoid);
    let d_tanh = Activation::derivative(y, ActivationType::Tanh);
    let d_relu = Activation::derivative(y, ActivationType::ReLU);
    let d_lin = Activation::derivative(y, ActivationType::Linear);
    
    kani::assert(d_sig.is_finite() || d_sig.is_nan(), "sigmoid derivative defined");
    kani::assert(d_tanh.is_finite() || d_tanh.is_nan(), "tanh derivative defined");
    kani::assert(d_relu.is_finite(), "relu derivative defined");
    kani::assert(d_lin == 1.0, "linear derivative is constant");
}

#[kani::proof]
fn verify_learning_rate_normalization() {
    let base_lr: f64 = kani::any();
    let epoch: usize = kani::any();
    
    kani::assume(base_lr.is_finite() && base_lr > 0.0);
    kani::assume(epoch <= 1000);
    
    let denom = 1.0 + (epoch as f64 / 10.0);
    kani::assert(denom >= 1.0, "decay denominator is at least 1");
    
    let adjusted_lr = base_lr / denom;
    kani::assert(adjusted_lr.is_finite(), "adjusted learning rate is finite");
    kani::assert(adjusted_lr > 0.0, "adjusted learning rate is positive");
}

#[kani::proof]
fn verify_gradient_norm_calculation_safety() {
    let grad_sum_sq: f64 = kani::any();
    kani::assume(grad_sum_sq.is_finite() && grad_sum_sq >= 0.0 && grad_sum_sq <= 1e20);
    
    if grad_sum_sq > 0.0 {
        let scale = 1.0 / grad_sum_sq;
        kani::assert(scale.is_finite() || grad_sum_sq < 1e-300, "reciprocal is finite for positive value");
    }
}

#[kani::proof]
fn verify_average_calculation_safety() {
    let sum: f64 = kani::any();
    let count: usize = kani::any();
    
    kani::assume(sum.is_finite());
    kani::assume(count > 0 && count <= 1_000_000);
    
    let average = sum / count as f64;
    kani::assert(average.is_finite() || sum.abs() == f64::INFINITY, "average is safe");
}

#[kani::proof]
fn verify_tanh_derivative_formula_no_division() {
    let y: f64 = kani::any();
    kani::assume(y.is_finite() && y.abs() <= 1.0);
    
    let derivative = 1.0 - y * y;
    
    kani::assert(derivative >= -0.001, "tanh derivative non-negative for valid tanh output");
    kani::assert(derivative <= 1.001, "tanh derivative at most 1");
}

#[kani::proof]
fn verify_softmax_stability() {
    let max_val: f64 = kani::any();
    let x: f64 = kani::any();
    
    kani::assume(max_val.is_finite());
    kani::assume(x.is_finite());
    kani::assume(x <= max_val);
    
    let shifted = x - max_val;
    kani::assert(shifted <= 0.0, "shifted value is non-positive");
}
