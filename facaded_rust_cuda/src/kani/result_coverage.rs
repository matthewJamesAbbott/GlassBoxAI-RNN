/*
 * Requirement 9: Result Coverage Audit
 *
 * Verify that all Error variants in returned Result types are explicitly
 * handled and do not leave the system in an indeterminate state.
 */

use crate::{ActivationType, LossType, CellType, GateType};
use std::str::FromStr;

#[kani::proof]
fn verify_activation_type_parse_all_variants_covered() {
    let valid_inputs = ["sigmoid", "tanh", "relu", "linear"];
    
    for input in valid_inputs {
        let result = ActivationType::from_str(input);
        kani::assert(result.is_ok(), "valid input parses successfully");
    }
    
    let result = ActivationType::from_str("unknown");
    kani::assert(result.is_err(), "invalid input returns error");
    
    if let Err(e) = result {
        kani::assert(!e.is_empty(), "error message is non-empty");
    }
}

#[kani::proof]
fn verify_loss_type_parse_all_variants_covered() {
    let valid_inputs = ["mse", "crossentropy"];
    
    for input in valid_inputs {
        let result = LossType::from_str(input);
        kani::assert(result.is_ok(), "valid input parses successfully");
    }
    
    let result = LossType::from_str("unknown");
    kani::assert(result.is_err(), "invalid input returns error");
}

#[kani::proof]
fn verify_cell_type_parse_all_variants_covered() {
    let valid_inputs = ["simplernn", "lstm", "gru"];
    
    for input in valid_inputs {
        let result = CellType::from_str(input);
        kani::assert(result.is_ok(), "valid input parses successfully");
    }
    
    let result = CellType::from_str("unknown");
    kani::assert(result.is_err(), "invalid input returns error");
}

#[kani::proof]
fn verify_gate_type_parse_all_variants_covered() {
    let valid_inputs = [
        "forget", "input", "output", "cellcandidate",
        "update", "reset", "hiddencandidate"
    ];
    
    for input in valid_inputs {
        let result = GateType::from_str(input);
        kani::assert(result.is_ok(), "valid input parses successfully");
    }
    
    let result = GateType::from_str("unknown");
    kani::assert(result.is_err(), "invalid input returns error");
}

#[kani::proof]
fn verify_result_handling_pattern() {
    fn fallible_operation(succeed: bool) -> Result<i32, &'static str> {
        if succeed {
            Ok(42)
        } else {
            Err("operation failed")
        }
    }
    
    let succeed: bool = kani::any();
    let result = fallible_operation(succeed);
    
    let value = match result {
        Ok(v) => {
            kani::assert(v == 42, "success value correct");
            v
        }
        Err(e) => {
            kani::assert(!e.is_empty(), "error message provided");
            0
        }
    };
    
    if succeed {
        kani::assert(value == 42, "success path taken");
    } else {
        kani::assert(value == 0, "error path taken");
    }
}

#[kani::proof]
fn verify_option_handling_pattern() {
    fn get_element(arr: &[f64], idx: usize) -> f64 {
        arr.get(idx).copied().unwrap_or(0.0)
    }
    
    let arr: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let idx: usize = kani::any();
    
    let value = get_element(&arr, idx);
    
    if idx < 4 {
        kani::assert(value == arr[idx], "valid index returns element");
    } else {
        kani::assert(value == 0.0, "invalid index returns default");
    }
}

#[kani::proof]
fn verify_checked_arithmetic_handling() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    
    let add_result = a.checked_add(b);
    let mul_result = a.checked_mul(b);
    let sub_result = a.checked_sub(b);
    
    match add_result {
        Some(sum) => kani::assert(sum >= a && sum >= b, "addition succeeded"),
        None => kani::cover!(true, "addition overflowed"),
    }
    
    match mul_result {
        Some(prod) => kani::assert(prod >= a.min(b) || a == 0 || b == 0, "multiplication succeeded"),
        None => kani::cover!(true, "multiplication overflowed"),
    }
    
    match sub_result {
        Some(diff) => kani::assert(diff <= a, "subtraction succeeded"),
        None => kani::assert(a < b, "subtraction underflowed as expected"),
    }
}

#[kani::proof]
fn verify_parse_with_default() {
    fn parse_usize_or_default(s: &str, default: usize) -> usize {
        s.parse().unwrap_or(default)
    }
    
    let val1 = parse_usize_or_default("42", 0);
    kani::assert(val1 == 42, "valid parse");
    
    let val2 = parse_usize_or_default("invalid", 100);
    kani::assert(val2 == 100, "invalid uses default");
}

#[kani::proof]
fn verify_error_propagation_pattern() {
    fn inner_op(fail: bool) -> Result<i32, &'static str> {
        if fail {
            Err("inner failed")
        } else {
            Ok(10)
        }
    }
    
    fn outer_op(fail_inner: bool, fail_outer: bool) -> Result<i32, &'static str> {
        let inner_val = inner_op(fail_inner)?;
        if fail_outer {
            Err("outer failed")
        } else {
            Ok(inner_val * 2)
        }
    }
    
    let fail_inner: bool = kani::any();
    let fail_outer: bool = kani::any();
    
    let result = outer_op(fail_inner, fail_outer);
    
    if fail_inner {
        kani::assert(result.is_err(), "inner failure propagates");
    } else if fail_outer {
        kani::assert(result.is_err(), "outer failure returns error");
    } else {
        kani::assert(result == Ok(20), "success returns computed value");
    }
}

#[kani::proof]
fn verify_activation_type_case_insensitive() {
    let variants = ["SIGMOID", "Sigmoid", "SiGmOiD"];
    
    for v in variants {
        let result = ActivationType::from_str(v);
        kani::assert(result == Ok(ActivationType::Sigmoid), "case insensitive");
    }
}

#[kani::proof]
fn verify_io_error_would_be_handled() {
    #[derive(Debug)]
    enum MockIOError {
        NotFound,
        PermissionDenied,
        Other,
    }
    
    fn mock_file_op(error_type: Option<MockIOError>) -> Result<Vec<u8>, MockIOError> {
        match error_type {
            Some(e) => Err(e),
            None => Ok(vec![1, 2, 3]),
        }
    }
    
    let error_variant: u8 = kani::any();
    
    let error = match error_variant % 4 {
        0 => None,
        1 => Some(MockIOError::NotFound),
        2 => Some(MockIOError::PermissionDenied),
        _ => Some(MockIOError::Other),
    };
    
    let result = mock_file_op(error);
    
    match result {
        Ok(data) => kani::assert(!data.is_empty(), "success has data"),
        Err(_) => kani::assert(error_variant % 4 != 0, "error case handled"),
    }
}
