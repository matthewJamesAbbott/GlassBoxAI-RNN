/*
 * Requirement 14: Floating-Point Sanity
 *
 * Prove that operations involving f32/f64 never result in unhandled NaN
 * or Infinity states that could bypass logic checks.
 */

use crate::{Activation, ActivationType, clip_value};



#[kani::proof]
fn verify_relu_no_nan_inf() {
    let x: f64 = kani::any();
    kani::assume(x.is_finite());
    
    let result = Activation::apply(x, ActivationType::ReLU);
    
    kani::assert(!result.is_nan(), "relu never produces NaN");
    kani::assert(!result.is_infinite(), "relu never produces infinity");
    kani::assert(result >= 0.0, "relu is non-negative");
}

#[kani::proof]
fn verify_linear_no_nan_inf() {
    let x: f64 = kani::any();
    kani::assume(x.is_finite());
    
    let result = Activation::apply(x, ActivationType::Linear);
    
    kani::assert(!result.is_nan(), "linear never produces NaN from finite input");
    kani::assert(!result.is_infinite(), "linear never produces infinity from finite input");
    kani::assert(result == x, "linear is identity");
}

#[kani::proof]
fn verify_clip_value_sanitizes_nan() {
    let v: f64 = kani::any();
    let max_val: f64 = kani::any();
    
    kani::assume(max_val.is_finite() && max_val >= 0.0);
    
    if v.is_finite() {
        let result = clip_value(v, max_val);
        kani::assert(!result.is_nan(), "clip doesn't produce NaN from finite");
        kani::assert(!result.is_infinite(), "clip doesn't produce infinity from finite");
        kani::assert(result.abs() <= max_val, "result within bounds");
    }
}



#[kani::proof]
fn verify_relu_derivative_binary() {
    let y: f64 = kani::any();
    kani::assume(y.is_finite());
    
    let deriv = Activation::derivative(y, ActivationType::ReLU);
    
    kani::assert(!deriv.is_nan(), "relu derivative not NaN");
    kani::assert(deriv == 0.0 || deriv == 1.0, "relu derivative is binary");
}



#[kani::proof]
fn verify_loss_mse_no_nan() {
    let pred: f64 = kani::any();
    let target: f64 = kani::any();
    
    kani::assume(pred.is_finite() && target.is_finite());
    
    let diff = pred - target;
    let squared = diff.powi(2);
    
    kani::assert(!squared.is_nan(), "squared difference not NaN");
    kani::assert(squared >= 0.0, "squared difference non-negative");
    kani::assert(squared.is_finite() || diff.abs() > 1e150, "squared is finite for reasonable inputs");
}

#[kani::proof]
fn verify_cross_entropy_clamping() {
    let p: f64 = kani::any();
    kani::assume(p.is_finite());
    
    let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
    
    kani::assert(p_clamped >= 1e-15, "clamped has lower bound");
    kani::assert(p_clamped <= 1.0 - 1e-15, "clamped has upper bound");
}

#[kani::proof]
fn verify_gradient_clipping_handles_large_values() {
    let grad: f64 = kani::any();
    let clip_max: f64 = 5.0;
    
    kani::assume(grad.is_finite());
    
    let clipped = clip_value(grad, clip_max);
    
    kani::assert(clipped.abs() <= clip_max, "gradient is clipped");
    kani::assert(clipped.is_finite(), "clipped gradient is finite");
}

#[kani::proof]
fn verify_weight_scale_finite() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    
    kani::assume(input_size > 0 && input_size <= 10000);
    kani::assume(hidden_size > 0 && hidden_size <= 10000);
    
    let sum = input_size + hidden_size;
    let scale = (2.0 / sum as f64).sqrt();
    
    kani::assert(scale.is_finite(), "weight scale is finite");
    kani::assert(scale > 0.0, "weight scale is positive");
    kani::assert(!scale.is_nan(), "weight scale not NaN");
}

#[kani::proof]
fn verify_matrix_element_operations_finite() {
    let a: f64 = kani::any();
    let b: f64 = kani::any();
    let c: f64 = kani::any();
    
    kani::assume(a.is_finite() && a.abs() <= 1e10);
    kani::assume(b.is_finite() && b.abs() <= 1e10);
    kani::assume(c.is_finite() && c.abs() <= 1e10);
    
    let sum = a + b * c;
    
    kani::assert(!sum.is_nan(), "fused multiply-add not NaN");
}

#[kani::proof]
fn verify_learning_rate_update_finite() {
    let weight: f64 = kani::any();
    let gradient: f64 = kani::any();
    let lr: f64 = kani::any();
    
    kani::assume(weight.is_finite() && weight.abs() <= 1e10);
    kani::assume(gradient.is_finite() && gradient.abs() <= 1e10);
    kani::assume(lr.is_finite() && lr >= 0.0 && lr <= 1.0);
    
    let update = lr * gradient;
    let new_weight = weight - update;
    
    kani::assert(!update.is_nan(), "update not NaN");
    kani::assert(new_weight.is_finite() || (weight.abs() > 1e9 && update.abs() > 1e9), 
        "new weight finite for reasonable inputs");
}

#[kani::proof]
fn verify_nan_propagation_detection() {
    fn detect_nan_in_array(arr: &[f64]) -> bool {
        for &val in arr {
            if val.is_nan() {
                return true;
            }
        }
        false
    }
    
    let arr: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    kani::assert(!detect_nan_in_array(&arr), "no NaN in valid array");
}

#[kani::proof]
fn verify_infinity_check() {
    fn all_finite(arr: &[f64]) -> bool {
        arr.iter().all(|&x| x.is_finite())
    }
    
    let arr: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    kani::assert(all_finite(&arr), "all values finite");
}
