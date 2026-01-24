/*
 * Requirement 2: Pointer Validity Proofs
 *
 * For any unsafe blocks, verify that all raw pointer dereferences are valid,
 * aligned, and point to initialized memory.
 *
 * Note: This codebase primarily uses safe Rust with cudarc handling unsafe
 * CUDA operations internally. We verify the safety of our interactions with
 * the CUDA abstraction layer.
 */

use crate::{
    DArray, TDArray2D, zero_array, zero_matrix, concat_arrays, flatten_matrix,
};

#[kani::proof]
#[kani::unwind(10)]
fn verify_vec_get_returns_valid_reference() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 16);
    
    let arr = zero_array(size);
    
    let idx: usize = kani::any();
    
    if idx < size {
        let opt = arr.get(idx);
        kani::assert(opt.is_some(), "get within bounds returns Some");
    } else {
        let opt = arr.get(idx);
        kani::assert(opt.is_none(), "get out of bounds returns None");
    }
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_slice_pointer_validity() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 16);
    
    let arr = zero_array(size);
    
    let start: usize = kani::any();
    let end: usize = kani::any();
    kani::assume(start <= end);
    kani::assume(end <= size);
    
    let slice = &arr[start..end];
    kani::assert(slice.len() == end - start, "slice has correct length");
    
    if start < end {
        let _first = slice[0];
        let _last = slice[slice.len() - 1];
    }
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_matrix_nested_access_validity() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 4);
    kani::assume(cols > 0 && cols <= 4);
    
    let mat = zero_matrix(rows, cols);
    
    let row_idx: usize = kani::any();
    let col_idx: usize = kani::any();
    
    if row_idx < rows && col_idx < cols {
        let row_ref = mat.get(row_idx);
        kani::assert(row_ref.is_some(), "valid row access");
        let col_ref = row_ref.unwrap().get(col_idx);
        kani::assert(col_ref.is_some(), "valid column access");
    }
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_iter_produces_valid_references() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);
    
    let arr = zero_array(size);
    
    let mut count = 0usize;
    for val in arr.iter() {
        kani::assert(*val == 0.0, "all values are initialized to zero");
        count += 1;
    }
    
    kani::assert(count == size, "iterator visits all elements");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_concat_preserves_data_integrity() {
    let size_a: usize = kani::any();
    let size_b: usize = kani::any();
    kani::assume(size_a > 0 && size_a <= 4);
    kani::assume(size_b > 0 && size_b <= 4);
    
    let mut a: DArray = Vec::with_capacity(size_a);
    for i in 0..size_a {
        a.push(i as f64);
    }
    
    let mut b: DArray = Vec::with_capacity(size_b);
    for i in 0..size_b {
        b.push((i + 100) as f64);
    }
    
    let result = concat_arrays(&a, &b);
    
    for i in 0..size_a {
        kani::assert(result[i] == i as f64, "first array preserved");
    }
    for i in 0..size_b {
        kani::assert(result[size_a + i] == (i + 100) as f64, "second array preserved");
    }
}

#[kani::proof]
#[kani::unwind(20)]
fn verify_flatten_matrix_data_integrity() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows > 0 && rows <= 3);
    kani::assume(cols > 0 && cols <= 3);
    
    let mut mat: TDArray2D = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row: DArray = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push((i * cols + j) as f64);
        }
        mat.push(row);
    }
    
    let flat = flatten_matrix(&mat);
    
    for i in 0..rows {
        for j in 0..cols {
            let flat_idx = i * cols + j;
            kani::assert(flat[flat_idx] == (i * cols + j) as f64, "flattened data integrity");
        }
    }
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_mutable_access_validity() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);
    
    let mut arr = zero_array(size);
    
    let idx: usize = kani::any();
    kani::assume(idx < size);
    
    let new_val: f64 = kani::any();
    kani::assume(new_val.is_finite());
    
    arr[idx] = new_val;
    kani::assert(arr[idx] == new_val, "mutable write persisted");
}

#[kani::proof]
#[kani::unwind(10)]
fn verify_vec_push_maintains_validity() {
    let initial_cap: usize = kani::any();
    kani::assume(initial_cap > 0 && initial_cap <= 8);
    
    let mut arr: DArray = Vec::with_capacity(initial_cap);
    
    let push_count: usize = kani::any();
    kani::assume(push_count <= 8);
    
    for i in 0..push_count {
        arr.push(i as f64);
        kani::assert(arr.len() == i + 1, "length grows correctly");
    }
    
    for i in 0..push_count {
        kani::assert(arr[i] == i as f64, "all pushed values accessible");
    }
}
