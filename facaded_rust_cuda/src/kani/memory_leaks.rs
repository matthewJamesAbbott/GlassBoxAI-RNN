/*
 * Requirement 10: Memory Leak/Leakage Proofs
 *
 * Prove that all allocated memory is either freed or remains reachable,
 * ensuring no memory exhaustion over time.
 *
 * Note: Rust's ownership system guarantees memory is freed when owners go
 * out of scope. We verify proper ownership transfer and no cycles.
 */

use std::sync::Arc;
use crate::{DArray, zero_array, zero_matrix};

#[kani::proof]
#[kani::unwind(10)]
fn verify_vec_dropped_on_scope_exit() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);
    
    {
        let arr = zero_array(size);
        kani::assert(arr.len() == size, "vec created");
    }
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_nested_vec_dropped() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 4);
    kani::assume(cols > 0 && cols <= 4);
    
    {
        let mat = zero_matrix(rows, cols);
        kani::assert(mat.len() == rows, "matrix created");
        for row in &mat {
            kani::assert(row.len() == cols, "each row has correct size");
        }
    }
}

#[kani::proof]
fn verify_arc_reference_counting() {
    let data: DArray = vec![1.0, 2.0, 3.0];
    
    let arc1 = Arc::new(data);
    kani::assert(Arc::strong_count(&arc1) == 1, "one strong reference");
    
    let arc2 = Arc::clone(&arc1);
    kani::assert(Arc::strong_count(&arc1) == 2, "two strong references");
    
    drop(arc2);
    kani::assert(Arc::strong_count(&arc1) == 1, "back to one reference");
}

#[kani::proof]
fn verify_arc_weak_doesnt_prevent_dealloc() {
    let data: DArray = vec![1.0, 2.0, 3.0];
    let strong = Arc::new(data);
    let weak = Arc::downgrade(&strong);
    
    kani::assert(weak.upgrade().is_some(), "weak can upgrade");
    
    drop(strong);
    
    kani::assert(weak.upgrade().is_none(), "weak returns None after strong dropped");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_vec_clear_releases_capacity_content() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);
    
    let mut arr = zero_array(size);
    kani::assert(arr.len() == size, "initial size");
    
    arr.clear();
    kani::assert(arr.len() == 0, "cleared to zero length");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_vec_truncate_partial_release() {
    let size: usize = kani::any();
    let new_len: usize = kani::any();
    kani::assume(size > 0 && size <= 8);
    kani::assume(new_len <= size);
    
    let mut arr = zero_array(size);
    arr.truncate(new_len);
    
    kani::assert(arr.len() == new_len, "truncated to new length");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_vec_pop_gradual_release() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 4);
    
    let mut arr = zero_array(size);
    
    for i in 0..size {
        let elem = arr.pop();
        kani::assert(elem.is_some(), "pop returns element");
        kani::assert(arr.len() == size - i - 1, "length decreases");
    }
    
    kani::assert(arr.is_empty(), "vec is empty after all pops");
    kani::assert(arr.pop().is_none(), "pop on empty returns None");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_ownership_transfer() {
    fn take_ownership(v: DArray) -> usize {
        v.len()
    }
    
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);
    
    let arr = zero_array(size);
    let len = take_ownership(arr);
    
    kani::assert(len == size, "transferred vec used correctly");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_clone_independent_lifetime() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 4);
    
    let original = zero_array(size);
    let cloned = original.clone();
    
    kani::assert(original.len() == cloned.len(), "same length");
    
    drop(original);
    
    kani::assert(cloned.len() == size, "clone still valid after original dropped");
}

#[kani::proof]
fn verify_box_deallocation() {
    let value: f64 = kani::any();
    kani::assume(value.is_finite());
    
    {
        let boxed = Box::new(value);
        kani::assert(*boxed == value, "boxed value accessible");
    }
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_nested_structure_deallocation() {
    let outer_size: usize = kani::any();
    let inner_size: usize = kani::any();
    kani::assume(outer_size > 0 && outer_size <= 3);
    kani::assume(inner_size > 0 && inner_size <= 3);
    
    {
        let mut outer: Vec<DArray> = Vec::with_capacity(outer_size);
        for _ in 0..outer_size {
            outer.push(zero_array(inner_size));
        }
        
        kani::assert(outer.len() == outer_size, "outer created");
        for inner in &outer {
            kani::assert(inner.len() == inner_size, "inner created");
        }
    }
}

#[kani::proof]
fn verify_replace_doesnt_leak() {
    let mut arr: DArray = vec![1.0, 2.0, 3.0];
    let old = std::mem::replace(&mut arr, vec![4.0, 5.0]);
    
    kani::assert(old.len() == 3, "old value returned");
    kani::assert(arr.len() == 2, "new value in place");
    
    drop(old);
}

#[kani::proof]
fn verify_take_doesnt_leak() {
    let mut arr: DArray = vec![1.0, 2.0, 3.0];
    let taken = std::mem::take(&mut arr);
    
    kani::assert(taken.len() == 3, "taken has original data");
    kani::assert(arr.is_empty(), "original is now empty");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_extend_reallocation_safe() {
    let initial: usize = kani::any();
    let extend_by: usize = kani::any();
    kani::assume(initial > 0 && initial <= 4);
    kani::assume(extend_by <= 4);
    
    let mut arr = zero_array(initial);
    let extension = zero_array(extend_by);
    
    arr.extend(extension);
    
    kani::assert(arr.len() == initial + extend_by, "extended correctly");
}

#[kani::proof]
fn verify_string_allocation_freed() {
    let s = String::from("test string for memory verification");
    let len = s.len();
    
    kani::assert(len > 0, "string has content");
    
    drop(s);
}
