# Kani Verification Test Suite - CISA Secure by Design

This directory contains formal verification harnesses using the [Kani Rust Verifier](https://github.com/model-checking/kani) to prove memory safety, arithmetic integrity, and security properties for the facaded RNN CUDA implementation.

## Overview

The verification suite implements 15 CISA "Secure by Design" security requirements:

| # | Requirement | File | Description |
|---|-------------|------|-------------|
| 1 | Strict Bound Checks | `bounds_checks.rs` | Proves all collection indexing is incapable of out-of-bounds access |
| 2 | Pointer Validity Proofs | `pointer_validity.rs` | Verifies raw pointer dereferences are valid, aligned, and point to initialized memory |
| 3 | No-Panic Guarantee | `no_panic.rs` | Verifies functions cannot trigger panic!, unwrap(), or expect() failures |
| 4 | Integer Overflow Prevention | `integer_overflow.rs` | Proves arithmetic operations are safe from wrapping/overflow/underflow |
| 5 | Division-by-Zero Exclusion | `division_by_zero.rs` | Verifies denominators are never zero |
| 6 | Global State Consistency | `concurrency.rs` | Proves concurrent access maintains invariants |
| 7 | Deadlock-Free Logic | `concurrency.rs` | Verifies locking follows strict hierarchy |
| 8 | Input Sanitization Bounds | `input_sanitization.rs` | Proves loops/recursion have formal upper bounds |
| 9 | Result Coverage Audit | `result_coverage.rs` | Verifies all Error variants are explicitly handled |
| 10 | Memory Leak/Leakage Proofs | `memory_leaks.rs` | Proves all allocated memory is freed or reachable |
| 11 | Constant-Time Execution | `constant_time.rs` | Verifies branching doesn't depend on secrets |
| 12 | State Machine Integrity | `state_machine.rs` | Proves no invalid state transitions |
| 13 | Enum Exhaustion | `enum_exhaustion.rs` | Verifies all match statements handle every variant |
| 14 | Floating-Point Sanity | `floating_point.rs` | Proves no unhandled NaN or Infinity states |
| 15 | Resource Limit Compliance | `resource_limits.rs` | Verifies allocations never exceed security budget |

## Prerequisites

Install Kani Rust Verifier:

```bash
cargo install --locked kani-verifier
kani setup
```

## Running Verification

### Run All Proofs

```bash
cargo kani
```

### Run Specific Proof Module

```bash
cargo kani --harness verify_sigmoid_no_nan_inf
```

### Run All Proofs in a Specific File

```bash
cargo kani --tests -p facaded_rnn_cuda --harness "verify_*_bounds*"
```

### Run with Verbose Output

```bash
cargo kani --verbose
```

## Verification Harness Structure

Each harness follows this pattern:

```rust
#[kani::proof]
#[kani::unwind(N)]  // Optional: set loop unwinding bound
fn verify_property_name() {
    // 1. Create symbolic inputs
    let x: usize = kani::any();
    
    // 2. Constrain inputs to valid ranges
    kani::assume(x > 0 && x <= 100);
    
    // 3. Execute code under test
    let result = function_under_test(x);
    
    // 4. Assert properties
    kani::assert!(result.is_valid(), "property must hold");
}
```

## Key Kani Primitives Used

- `kani::any()` - Creates symbolic/non-deterministic value
- `kani::assume(condition)` - Constrains symbolic inputs
- `kani::assert!(condition, message)` - Verifies property
- `kani::cover!(condition, message)` - Checks reachability
- `#[kani::unwind(N)]` - Sets loop unwinding bound
- `#[kani::proof]` - Marks function as verification harness

## Coverage Report

Generate HTML coverage report:

```bash
cargo kani --coverage -Z line-coverage
```

## Interpreting Results

### Success
```
VERIFICATION:- SUCCESSFUL
```

### Failure
```
VERIFICATION:- FAILED
Check 1: verify_bounds.assertion.1
         - Status: FAILURE
         - Description: "index out of bounds"
```

## Notes

1. **CUDA Code**: The CUDA kernel code and cudarc interactions are not directly verified by Kani. The harnesses verify the CPU-side Rust code.

2. **Unwinding Bounds**: Some harnesses use `#[kani::unwind(N)]` to limit loop iterations for tractable verification.

3. **Symbolic Execution**: Kani uses bounded model checking - all possible values within constraints are explored.

4. **Memory**: Verification can be memory-intensive for complex proofs. Adjust unwind bounds if needed.

## Adding New Harnesses

1. Add harness to appropriate module file
2. Use `#[kani::proof]` attribute
3. Create symbolic inputs with `kani::any()`
4. Add constraints with `kani::assume()`
5. Assert properties with `kani::assert!()`
6. Run `cargo kani --harness your_harness_name`

## License

MIT License - Same as parent project
