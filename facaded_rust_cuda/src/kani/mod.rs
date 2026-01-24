/*
 * Kani Verification Test Suite - CISA Secure by Design
 *
 * This module contains formal verification harnesses using Kani Rust Verifier
 * to prove memory safety, arithmetic integrity, and security properties.
 */

#[cfg(kani)]
mod bounds_checks;
#[cfg(kani)]
mod pointer_validity;
#[cfg(kani)]
mod no_panic;
#[cfg(kani)]
mod integer_overflow;
#[cfg(kani)]
mod division_by_zero;
#[cfg(kani)]
mod concurrency;
#[cfg(kani)]
mod input_sanitization;
#[cfg(kani)]
mod result_coverage;
#[cfg(kani)]
mod memory_leaks;
#[cfg(kani)]
mod constant_time;
#[cfg(kani)]
mod state_machine;
#[cfg(kani)]
mod enum_exhaustion;
#[cfg(kani)]
mod floating_point;
#[cfg(kani)]
mod resource_limits;
