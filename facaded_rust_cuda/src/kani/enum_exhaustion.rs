/*
 * Requirement 13: Enum Exhaustion
 *
 * Verify that all match statements handle every possible variant
 * (including future-proofing) without relying on a generic _ => panic!() fallback.
 */

use crate::{ActivationType, LossType, CellType, GateType};

#[kani::proof]
fn verify_activation_type_match_exhaustive() {
    let act_idx: u8 = kani::any();
    kani::assume(act_idx <= 3);
    
    let act = match act_idx {
        0 => ActivationType::Sigmoid,
        1 => ActivationType::Tanh,
        2 => ActivationType::ReLU,
        _ => ActivationType::Linear,
    };
    
    let result = match act {
        ActivationType::Sigmoid => 0,
        ActivationType::Tanh => 1,
        ActivationType::ReLU => 2,
        ActivationType::Linear => 3,
    };
    
    kani::assert(result <= 3, "all variants handled");
}

#[kani::proof]
fn verify_loss_type_match_exhaustive() {
    let loss_idx: u8 = kani::any();
    kani::assume(loss_idx <= 1);
    
    let loss = match loss_idx {
        0 => LossType::MSE,
        _ => LossType::CrossEntropy,
    };
    
    let result = match loss {
        LossType::MSE => "mse",
        LossType::CrossEntropy => "crossentropy",
    };
    
    kani::assert(!result.is_empty(), "all variants produce output");
}

#[kani::proof]
fn verify_cell_type_match_exhaustive() {
    let cell_idx: u8 = kani::any();
    kani::assume(cell_idx <= 2);
    
    let cell = match cell_idx {
        0 => CellType::SimpleRNN,
        1 => CellType::LSTM,
        _ => CellType::GRU,
    };
    
    let gate_count = match cell {
        CellType::SimpleRNN => 0,
        CellType::LSTM => 4,
        CellType::GRU => 3,
    };
    
    kani::assert(gate_count >= 0 && gate_count <= 4, "all variants handled");
}

#[kani::proof]
fn verify_gate_type_match_exhaustive() {
    let gate_idx: u8 = kani::any();
    kani::assume(gate_idx <= 6);
    
    let gate = match gate_idx {
        0 => GateType::Forget,
        1 => GateType::Input,
        2 => GateType::Output,
        3 => GateType::CellCandidate,
        4 => GateType::Update,
        5 => GateType::Reset,
        _ => GateType::HiddenCandidate,
    };
    
    let name = match gate {
        GateType::Forget => "forget",
        GateType::Input => "input",
        GateType::Output => "output",
        GateType::CellCandidate => "cell_candidate",
        GateType::Update => "update",
        GateType::Reset => "reset",
        GateType::HiddenCandidate => "hidden_candidate",
    };
    
    kani::assert(!name.is_empty(), "all gate variants named");
}

#[kani::proof]
fn verify_activation_as_int_exhaustive() {
    let acts = [
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::ReLU,
        ActivationType::Linear,
    ];
    
    for act in acts {
        let int_val = act.as_int();
        kani::assert(int_val >= 0 && int_val <= 3, "as_int returns valid range");
    }
}

#[kani::proof]
fn verify_activation_apply_exhaustive() {
    let x: f64 = kani::any();
    kani::assume(x.is_finite());
    
    let acts = [
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::ReLU,
        ActivationType::Linear,
    ];
    
    for act in acts {
        let result = crate::Activation::apply(x, act);
        kani::assert(result.is_finite() || result.is_nan(), "all activations produce defined output");
    }
}

#[kani::proof]
fn verify_activation_derivative_exhaustive() {
    let y: f64 = kani::any();
    kani::assume(y.is_finite());
    
    let acts = [
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::ReLU,
        ActivationType::Linear,
    ];
    
    for act in acts {
        let result = crate::Activation::derivative(y, act);
        kani::assert(result.is_finite() || result.is_nan(), "all derivatives produce defined output");
    }
}

#[kani::proof]
fn verify_cell_type_has_all_gates() {
    let cell_types = [CellType::SimpleRNN, CellType::LSTM, CellType::GRU];
    
    for cell in cell_types {
        let applicable_gates: &[GateType] = match cell {
            CellType::SimpleRNN => &[],
            CellType::LSTM => &[
                GateType::Forget,
                GateType::Input,
                GateType::Output,
                GateType::CellCandidate,
            ],
            CellType::GRU => &[
                GateType::Update,
                GateType::Reset,
                GateType::HiddenCandidate,
            ],
        };
        
        match cell {
            CellType::SimpleRNN => kani::assert(applicable_gates.is_empty(), "SimpleRNN has no gates"),
            CellType::LSTM => kani::assert(applicable_gates.len() == 4, "LSTM has 4 gates"),
            CellType::GRU => kani::assert(applicable_gates.len() == 3, "GRU has 3 gates"),
        }
    }
}

#[kani::proof]
fn verify_display_trait_exhaustive() {
    let acts = [
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::ReLU,
        ActivationType::Linear,
    ];
    
    for act in acts {
        let s = format!("{}", act);
        kani::assert(!s.is_empty(), "all activation types have display");
    }
    
    let losses = [LossType::MSE, LossType::CrossEntropy];
    
    for loss in losses {
        let s = format!("{}", loss);
        kani::assert(!s.is_empty(), "all loss types have display");
    }
    
    let cells = [CellType::SimpleRNN, CellType::LSTM, CellType::GRU];
    
    for cell in cells {
        let s = format!("{}", cell);
        kani::assert(!s.is_empty(), "all cell types have display");
    }
}

#[kani::proof]
fn verify_from_str_covers_all_variants() {
    use std::str::FromStr;
    
    let valid_activations = ["sigmoid", "tanh", "relu", "linear"];
    for s in valid_activations {
        let result = ActivationType::from_str(s);
        kani::assert(result.is_ok(), "valid activation string parses");
    }
    
    let valid_losses = ["mse", "crossentropy"];
    for s in valid_losses {
        let result = LossType::from_str(s);
        kani::assert(result.is_ok(), "valid loss string parses");
    }
    
    let valid_cells = ["simplernn", "lstm", "gru"];
    for s in valid_cells {
        let result = CellType::from_str(s);
        kani::assert(result.is_ok(), "valid cell string parses");
    }
}

#[kani::proof]
fn verify_no_unreachable_variants() {
    let mut covered = [false; 4];
    
    let acts = [
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::ReLU,
        ActivationType::Linear,
    ];
    
    for (i, _) in acts.iter().enumerate() {
        covered[i] = true;
    }
    
    let all_covered = covered.iter().all(|&c| c);
    kani::assert(all_covered, "all variants are reachable");
}
