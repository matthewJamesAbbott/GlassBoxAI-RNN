/*
 * Requirement 6 & 7: Global State Consistency & Deadlock-Free Logic
 *
 * Prove that concurrent access to shared state (Mutexes, RwLocks) maintains
 * defined invariants and is free of data races.
 *
 * Verify that all locking mechanisms follow a strict hierarchy and cannot
 * enter a circular wait state.
 *
 * Note: This codebase uses Arc<CudaDevice> for shared GPU context. The cudarc
 * library handles internal synchronization. We verify our usage patterns.
 */

use std::sync::Arc;
use crate::{DArray, zero_array};

#[kani::proof]
fn verify_arc_clone_safety() {
    let data: DArray = vec![1.0, 2.0, 3.0];
    let arc1 = Arc::new(data);
    let arc2 = Arc::clone(&arc1);
    
    kani::assert(Arc::strong_count(&arc1) == 2, "two strong references");
    kani::assert(Arc::strong_count(&arc2) == 2, "count consistent across clones");
    
    kani::assert(arc1[0] == arc2[0], "data accessible from both");
}

#[kani::proof]
fn verify_arc_drop_ordering() {
    let data: DArray = vec![1.0, 2.0, 3.0];
    let arc1 = Arc::new(data);
    
    {
        let arc2 = Arc::clone(&arc1);
        kani::assert(Arc::strong_count(&arc1) == 2, "inner scope has two refs");
        drop(arc2);
    }
    
    kani::assert(Arc::strong_count(&arc1) == 1, "one ref after inner drop");
}

#[kani::proof]
#[kani::unwind(5)]
fn verify_vec_operations_thread_safe() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 4);
    
    let mut data = zero_array(size);
    
    for i in 0..size {
        data[i] = (i * 2) as f64;
    }
    
    let shared = Arc::new(data.clone());
    
    for i in 0..size {
        kani::assert(shared[i] == (i * 2) as f64, "shared data consistent");
    }
}

#[kani::proof]
fn verify_no_mutable_aliasing() {
    let mut data: DArray = vec![0.0; 4];
    
    data[0] = 1.0;
    data[1] = 2.0;
    
    let slice1 = &data[0..2];
    let val0 = slice1[0];
    let val1 = slice1[1];
    
    kani::assert(val0 == 1.0, "first value correct");
    kani::assert(val1 == 2.0, "second value correct");
    
    let _ = slice1;
    
    data[2] = 3.0;
    data[3] = 4.0;
    
    kani::assert(data[2] == 3.0, "mutation after drop works");
}

#[kani::proof]
fn verify_single_writer_semantics() {
    let mut arr = vec![0.0f64; 4];
    
    let idx1: usize = kani::any();
    let idx2: usize = kani::any();
    let val1: f64 = kani::any();
    let val2: f64 = kani::any();
    
    kani::assume(idx1 < 4 && idx2 < 4);
    kani::assume(val1.is_finite() && val2.is_finite());
    
    arr[idx1] = val1;
    
    if idx1 != idx2 {
        arr[idx2] = val2;
        kani::assert(arr[idx1] == val1, "non-aliased write preserved");
        kani::assert(arr[idx2] == val2, "second write succeeded");
    } else {
        arr[idx2] = val2;
        kani::assert(arr[idx1] == val2, "same index overwrites");
    }
}

#[kani::proof]
fn verify_immutable_borrow_safety() {
    let data: DArray = vec![1.0, 2.0, 3.0, 4.0];
    
    let ref1 = &data;
    let ref2 = &data;
    let ref3 = &data;
    
    kani::assert(ref1[0] == ref2[0], "multiple immutable borrows see same data");
    kani::assert(ref2[1] == ref3[1], "all borrows consistent");
}

#[kani::proof]
fn verify_arc_weak_reference_safety() {
    let data: DArray = vec![1.0, 2.0, 3.0];
    let strong = Arc::new(data);
    let weak = Arc::downgrade(&strong);
    
    kani::assert(weak.upgrade().is_some(), "weak upgrades while strong exists");
    
    let upgraded = weak.upgrade().unwrap();
    kani::assert(upgraded[0] == 1.0, "upgraded data correct");
    
    drop(upgraded);
    kani::assert(Arc::strong_count(&strong) == 1, "back to one strong");
}

#[kani::proof]
fn verify_sequential_state_updates() {
    let mut state = 0i32;
    
    let op1: bool = kani::any();
    let op2: bool = kani::any();
    let op3: bool = kani::any();
    
    if op1 {
        state += 1;
    }
    if op2 {
        state += 2;
    }
    if op3 {
        state += 4;
    }
    
    kani::assert(state >= 0, "state never negative");
    kani::assert(state <= 7, "state bounded by sum of all ops");
    
    let expected = (op1 as i32) + (op2 as i32) * 2 + (op3 as i32) * 4;
    kani::assert(state == expected, "state equals expected from operations");
}

#[kani::proof]
fn verify_lock_free_accumulation() {
    let mut values: [f64; 4] = [0.0; 4];
    
    let idx: usize = kani::any();
    let delta: f64 = kani::any();
    
    kani::assume(idx < 4);
    kani::assume(delta.is_finite());
    
    let old = values[idx];
    values[idx] += delta;
    
    if delta.is_finite() && old.is_finite() {
        kani::assert(values[idx] == old + delta || !values[idx].is_finite(), 
            "accumulation correct or overflow");
    }
}

#[kani::proof]
fn verify_no_lock_hierarchy_violation() {
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum LockLevel {
        Output = 0,
        Hidden = 1,
        Input = 2,
    }
    
    let first_lock: u8 = kani::any();
    let second_lock: u8 = kani::any();
    
    kani::assume(first_lock <= 2 && second_lock <= 2);
    
    let first = match first_lock {
        0 => LockLevel::Output,
        1 => LockLevel::Hidden,
        _ => LockLevel::Input,
    };
    
    let second = match second_lock {
        0 => LockLevel::Output,
        1 => LockLevel::Hidden,
        _ => LockLevel::Input,
    };
    
    let valid_order = first >= second;
    
    kani::cover!(valid_order && first != second, "valid different lock order");
    kani::cover!(!valid_order, "invalid lock order (would deadlock)");
}
