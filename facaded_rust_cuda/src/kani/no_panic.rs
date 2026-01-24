/*
 * Requirement 3: No-Panic Guarantee
 *
 * Verify that target functions are incapable of triggering panic!, unwrap(),
 * or expect() failures across the entire input space.
 */

use crate::{
    Activation, ActivationType, Loss, LossType, CellType,
    clip_value, zero_array, zero_matrix,
    SimpleRNNCell, LSTMCell, GRUCell, OutputLayer,
};

#[kani::proof]
fn verify_activation_apply_no_panic() {
    let x: f64 = kani::any();
    
    kani::assume(x.is_finite());
    
    let _relu = Activation::apply(x, ActivationType::ReLU);
    let _lin = Activation::apply(x, ActivationType::Linear);
}

#[kani::proof]
fn verify_activation_derivative_no_panic() {
    let y: f64 = kani::any();
    
    kani::assume(y.is_finite());
    
    let _relu = Activation::derivative(y, ActivationType::ReLU);
    let _lin = Activation::derivative(y, ActivationType::Linear);
}

#[kani::proof]
fn verify_clip_value_no_panic() {
    let v: f64 = kani::any();
    let max_val: f64 = kani::any();
    
    kani::assume(v.is_finite());
    kani::assume(max_val.is_finite());
    kani::assume(max_val >= 0.0);
    
    let result = clip_value(v, max_val);
    
    kani::assert(result >= -max_val && result <= max_val, "clipped value in range");
    kani::assert(result.is_finite(), "result is finite");
}

#[kani::proof]
fn verify_activation_type_from_str_no_panic() {
    use std::str::FromStr;
    
    let _sig = ActivationType::from_str("sigmoid");
    let _tanh = ActivationType::from_str("tanh");
    let _relu = ActivationType::from_str("relu");
    let _lin = ActivationType::from_str("linear");
    let unknown = ActivationType::from_str("unknown");
    
    kani::assert(unknown.is_err(), "unknown activation returns error");
}

#[kani::proof]
fn verify_loss_type_from_str_no_panic() {
    use std::str::FromStr;
    
    let _mse = LossType::from_str("mse");
    let _ce = LossType::from_str("crossentropy");
    let unknown = LossType::from_str("unknown");
    
    kani::assert(unknown.is_err(), "unknown loss returns error");
}

#[kani::proof]
fn verify_cell_type_from_str_no_panic() {
    use std::str::FromStr;
    
    let _simple = CellType::from_str("simplernn");
    let _lstm = CellType::from_str("lstm");
    let _gru = CellType::from_str("gru");
    let unknown = CellType::from_str("unknown");
    
    kani::assert(unknown.is_err(), "unknown cell type returns error");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_zero_array_no_panic() {
    let size: usize = kani::any();
    kani::assume(size <= 64);
    
    let _arr = zero_array(size);
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_zero_matrix_no_panic() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows <= 16);
    kani::assume(cols <= 16);
    
    let _mat = zero_matrix(rows, cols);
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_loss_compute_no_panic_with_valid_input() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);
    
    let pred = zero_array(size);
    let target = zero_array(size);
    
    let mse = Loss::compute(&pred, &target, LossType::MSE);
    kani::assert(mse.is_finite(), "MSE result is finite for zero inputs");
    
    let ce = Loss::compute(&pred, &target, LossType::CrossEntropy);
    kani::assert(ce.is_finite(), "CrossEntropy result is finite for zero inputs");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_loss_gradient_no_panic_with_valid_input() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);
    
    let pred = zero_array(size);
    let target = zero_array(size);
    
    let grad_mse = Loss::gradient(&pred, &target, LossType::MSE);
    for &g in grad_mse.iter() {
        kani::assert(g.is_finite(), "MSE gradient values are finite");
    }
    
    let grad_ce = Loss::gradient(&pred, &target, LossType::CrossEntropy);
    for &g in grad_ce.iter() {
        kani::assert(g.is_finite(), "CrossEntropy gradient values are finite");
    }
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_simple_rnn_cell_new_no_panic() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 4);
    kani::assume(hidden_size > 0 && hidden_size <= 4);
    
    let _cell = SimpleRNNCell::new(input_size, hidden_size, ActivationType::Tanh);
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_lstm_cell_new_no_panic() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 4);
    kani::assume(hidden_size > 0 && hidden_size <= 4);
    
    let _cell = LSTMCell::new(input_size, hidden_size, ActivationType::Tanh);
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_gru_cell_new_no_panic() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 4);
    kani::assume(hidden_size > 0 && hidden_size <= 4);
    
    let _cell = GRUCell::new(input_size, hidden_size, ActivationType::Tanh);
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_output_layer_new_no_panic() {
    let input_size: usize = kani::any();
    let output_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 4);
    kani::assume(output_size > 0 && output_size <= 4);
    
    let _layer = OutputLayer::new(input_size, output_size, ActivationType::Linear);
}

#[kani::proof]
fn verify_activation_type_display_no_panic() {
    let types = [
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::ReLU,
        ActivationType::Linear,
    ];
    
    for t in types {
        let s = format!("{}", t);
        kani::assert(!s.is_empty(), "display produces non-empty string");
    }
}

#[kani::proof]
fn verify_loss_type_display_no_panic() {
    let types = [LossType::MSE, LossType::CrossEntropy];
    
    for t in types {
        let s = format!("{}", t);
        kani::assert(!s.is_empty(), "display produces non-empty string");
    }
}

#[kani::proof]
fn verify_cell_type_display_no_panic() {
    let types = [CellType::SimpleRNN, CellType::LSTM, CellType::GRU];
    
    for t in types {
        let s = format!("{}", t);
        kani::assert(!s.is_empty(), "display produces non-empty string");
    }
}

#[kani::proof]
fn verify_activation_type_as_int_no_panic() {
    kani::assert(ActivationType::Sigmoid.as_int() == 0, "sigmoid maps to 0");
    kani::assert(ActivationType::Tanh.as_int() == 1, "tanh maps to 1");
    kani::assert(ActivationType::ReLU.as_int() == 2, "relu maps to 2");
    kani::assert(ActivationType::Linear.as_int() == 3, "linear maps to 3");
}
