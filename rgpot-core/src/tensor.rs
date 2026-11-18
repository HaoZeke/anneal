// MIT License
// Copyright 2023--present rgpot developers

//! DLPack tensor helpers for creating, freeing, and validating tensors.
//!
//! This module provides the bridge between rgpot's C API and the DLPack tensor
//! exchange format. Two categories of tensors are supported:
//!
//! - **Borrowed (non-owning)**: wraps an existing raw pointer. The deleter
//!   frees only the `DLManagedTensorVersioned` metadata, not the data.
//! - **Owned**: wraps a `Vec<T>`. The deleter frees both metadata and data.
//!
//! All exported `extern "C"` functions are collected by cbindgen into `rgpot.h`.

use std::os::raw::{c_int, c_void};

use dlpk::sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensorVersioned, DLPackVersion,
    DLTensor, DLPACK_FLAG_BITMASK_IS_COPIED,
};

// ---------------------------------------------------------------------------
// Internal: row-major stride computation
// ---------------------------------------------------------------------------

fn compute_row_major_strides(shape: &[i64]) -> Vec<i64> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1i64; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// DLDataType constants
// ---------------------------------------------------------------------------

fn dtype_f64() -> DLDataType {
    DLDataType {
        code: DLDataTypeCode::kDLFloat,
        bits: 64,
        lanes: 1,
    }
}

fn dtype_i32() -> DLDataType {
    DLDataType {
        code: DLDataTypeCode::kDLInt,
        bits: 32,
        lanes: 1,
    }
}

fn cpu_device() -> DLDevice {
    DLDevice {
        device_type: DLDeviceType::kDLCPU,
        device_id: 0,
    }
}

fn dlpack_version() -> DLPackVersion {
    DLPackVersion { major: 1, minor: 0 }
}

// ---------------------------------------------------------------------------
// Borrowed (non-owning) tensors
// ---------------------------------------------------------------------------

/// Metadata context for a borrowed tensor.  Owns shape and strides arrays
/// but does NOT own the data buffer.
struct BorrowedTensorContext {
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn borrowed_deleter(ptr: *mut DLManagedTensorVersioned) {
    if ptr.is_null() {
        return;
    }
    let ctx = unsafe { (*ptr).manager_ctx.cast::<BorrowedTensorContext>() };
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx) });
    }
    drop(unsafe { Box::from_raw(ptr) });
}

/// Create a non-owning `DLManagedTensorVersioned` that borrows `data`.
///
/// # Safety
/// `data` must remain valid for the lifetime of the returned tensor.
unsafe fn create_borrowed_tensor(
    data: *mut c_void,
    dtype: DLDataType,
    shape_vec: Vec<i64>,
) -> *mut DLManagedTensorVersioned {
    let ndim = shape_vec.len() as i32;
    let strides_vec = compute_row_major_strides(&shape_vec);

    let mut ctx = Box::new(BorrowedTensorContext {
        shape: shape_vec,
        strides: strides_vec,
    });

    let dl_tensor = DLTensor {
        data,
        device: cpu_device(),
        ndim,
        dtype,
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let managed = Box::new(DLManagedTensorVersioned {
        version: dlpack_version(),
        manager_ctx: Box::into_raw(ctx).cast(),
        deleter: Some(borrowed_deleter),
        flags: 0,
        dl_tensor,
    });

    Box::into_raw(managed)
}

// ---------------------------------------------------------------------------
// Owned tensors
// ---------------------------------------------------------------------------

struct OwnedF64TensorContext {
    _data: Vec<f64>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn owned_f64_deleter(ptr: *mut DLManagedTensorVersioned) {
    if ptr.is_null() {
        return;
    }
    let ctx = unsafe { (*ptr).manager_ctx.cast::<OwnedF64TensorContext>() };
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx) });
    }
    drop(unsafe { Box::from_raw(ptr) });
}

#[allow(dead_code)]
struct OwnedI32TensorContext {
    _data: Vec<i32>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

#[allow(dead_code)]
unsafe extern "C" fn owned_i32_deleter(ptr: *mut DLManagedTensorVersioned) {
    if ptr.is_null() {
        return;
    }
    let ctx = unsafe { (*ptr).manager_ctx.cast::<OwnedI32TensorContext>() };
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx) });
    }
    drop(unsafe { Box::from_raw(ptr) });
}

/// Create an owning f64 DLPack tensor.  The `Vec<f64>` is kept alive inside
/// the manager context and freed when the deleter runs.
pub(crate) fn create_owned_f64_tensor(
    mut data: Vec<f64>,
    shape_vec: Vec<i64>,
) -> *mut DLManagedTensorVersioned {
    let ndim = shape_vec.len() as i32;
    let strides_vec = compute_row_major_strides(&shape_vec);
    let data_ptr = data.as_mut_ptr();

    let mut ctx = Box::new(OwnedF64TensorContext {
        _data: data,
        shape: shape_vec,
        strides: strides_vec,
    });

    let dl_tensor = DLTensor {
        data: data_ptr.cast(),
        device: cpu_device(),
        ndim,
        dtype: dtype_f64(),
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let managed = Box::new(DLManagedTensorVersioned {
        version: dlpack_version(),
        manager_ctx: Box::into_raw(ctx).cast(),
        deleter: Some(owned_f64_deleter),
        flags: DLPACK_FLAG_BITMASK_IS_COPIED,
        dl_tensor,
    });

    Box::into_raw(managed)
}

/// Create an owning i32 DLPack tensor.
#[allow(dead_code)]
pub(crate) fn create_owned_i32_tensor(
    mut data: Vec<i32>,
    shape_vec: Vec<i64>,
) -> *mut DLManagedTensorVersioned {
    let ndim = shape_vec.len() as i32;
    let strides_vec = compute_row_major_strides(&shape_vec);
    let data_ptr = data.as_mut_ptr();

    let mut ctx = Box::new(OwnedI32TensorContext {
        _data: data,
        shape: shape_vec,
        strides: strides_vec,
    });

    let dl_tensor = DLTensor {
        data: data_ptr.cast(),
        device: cpu_device(),
        ndim,
        dtype: dtype_i32(),
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    };

    let managed = Box::new(DLManagedTensorVersioned {
        version: dlpack_version(),
        manager_ctx: Box::into_raw(ctx).cast(),
        deleter: Some(owned_i32_deleter),
        flags: DLPACK_FLAG_BITMASK_IS_COPIED,
        dl_tensor,
    });

    Box::into_raw(managed)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate a positions tensor: f64, 2-D, shape [n, 3].  Returns n_atoms.
#[allow(dead_code)]
pub(crate) fn validate_positions(
    tensor: *const DLManagedTensorVersioned,
) -> Result<usize, String> {
    if tensor.is_null() {
        return Err("positions tensor is NULL".into());
    }
    let t = unsafe { &(*tensor).dl_tensor };
    if t.dtype != dtype_f64() {
        return Err(format!("positions: expected f64, got {:?}", t.dtype));
    }
    if t.ndim != 2 {
        return Err(format!("positions: expected ndim=2, got {}", t.ndim));
    }
    let shape = unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) };
    if shape[1] != 3 {
        return Err(format!("positions: expected shape[1]=3, got {}", shape[1]));
    }
    Ok(shape[0] as usize)
}

/// Validate an atomic numbers tensor: i32, 1-D, shape [n].
#[allow(dead_code)]
pub(crate) fn validate_atomic_numbers(
    tensor: *const DLManagedTensorVersioned,
    expected_n: usize,
) -> Result<(), String> {
    if tensor.is_null() {
        return Err("atomic_numbers tensor is NULL".into());
    }
    let t = unsafe { &(*tensor).dl_tensor };
    if t.dtype != dtype_i32() {
        return Err(format!("atomic_numbers: expected i32, got {:?}", t.dtype));
    }
    if t.ndim != 1 {
        return Err(format!(
            "atomic_numbers: expected ndim=1, got {}",
            t.ndim
        ));
    }
    let shape = unsafe { std::slice::from_raw_parts(t.shape, 1) };
    if shape[0] as usize != expected_n {
        return Err(format!(
            "atomic_numbers: expected len={}, got {}",
            expected_n, shape[0]
        ));
    }
    Ok(())
}

/// Validate a box matrix tensor: f64, 2-D, shape [3, 3].
#[allow(dead_code)]
pub(crate) fn validate_box_matrix(
    tensor: *const DLManagedTensorVersioned,
) -> Result<(), String> {
    if tensor.is_null() {
        return Err("box_matrix tensor is NULL".into());
    }
    let t = unsafe { &(*tensor).dl_tensor };
    if t.dtype != dtype_f64() {
        return Err(format!("box_matrix: expected f64, got {:?}", t.dtype));
    }
    if t.ndim != 2 {
        return Err(format!("box_matrix: expected ndim=2, got {}", t.ndim));
    }
    let shape = unsafe { std::slice::from_raw_parts(t.shape, 2) };
    if shape[0] != 3 || shape[1] != 3 {
        return Err(format!(
            "box_matrix: expected shape=[3,3], got [{},{}]",
            shape[0], shape[1]
        ));
    }
    Ok(())
}

/// Resolve the data pointer of an f64 CPU tensor holding exactly
/// `expected_len` elements, after checking every property the pointer
/// arithmetic relies on: non-null tensor and data, f64 dtype with one lane,
/// CPU device, 1-D or 2-D shape with non-negative extents whose checked
/// product equals `expected_len`, and C-contiguous strides when present.
pub(crate) fn checked_f64_data(
    tensor: *const DLManagedTensorVersioned,
    label: &str,
    expected_len: usize,
) -> Result<*mut f64, String> {
    checked_data::<f64>(tensor, label, expected_len, dtype_f64(), "f64")
}

/// [`checked_f64_data`] for an i32 tensor.
pub(crate) fn checked_i32_data(
    tensor: *const DLManagedTensorVersioned,
    label: &str,
    expected_len: usize,
) -> Result<*mut i32, String> {
    checked_data::<i32>(tensor, label, expected_len, dtype_i32(), "i32")
}

fn checked_data<T>(
    tensor: *const DLManagedTensorVersioned,
    label: &str,
    expected_len: usize,
    dtype: DLDataType,
    dtype_name: &str,
) -> Result<*mut T, String> {
    if tensor.is_null() {
        return Err(format!("{label}: tensor is NULL"));
    }
    let t = unsafe { &(*tensor).dl_tensor };
    if t.data.is_null() {
        return Err(format!("{label}: data pointer is NULL"));
    }
    if t.dtype != dtype {
        return Err(format!("{label}: expected {dtype_name}, got {:?}", t.dtype));
    }
    if t.device.device_type != DLDeviceType::kDLCPU {
        return Err(format!("{label}: expected a CPU tensor"));
    }
    if t.ndim != 1 && t.ndim != 2 {
        return Err(format!("{label}: expected ndim 1 or 2, got {}", t.ndim));
    }
    if t.shape.is_null() {
        return Err(format!("{label}: shape pointer is NULL"));
    }
    let ndim = t.ndim as usize;
    let shape = unsafe { std::slice::from_raw_parts(t.shape, ndim) };
    let mut count: usize = 1;
    for &extent in shape {
        let extent = usize::try_from(extent)
            .map_err(|_| format!("{label}: negative extent {extent}"))?;
        count = count
            .checked_mul(extent)
            .ok_or_else(|| format!("{label}: element count overflows"))?;
    }
    if count != expected_len {
        return Err(format!(
            "{label}: expected {expected_len} elements, got {count}"
        ));
    }
    if !t.strides.is_null() {
        let strides = unsafe { std::slice::from_raw_parts(t.strides, ndim) };
        let contiguous = if ndim == 1 {
            strides[0] == 1 || shape[0] <= 1
        } else {
            (strides[1] == 1 || shape[1] <= 1) && (strides[0] == shape[1] || shape[0] <= 1)
        };
        if !contiguous {
            return Err(format!("{label}: tensor is not C-contiguous"));
        }
    }
    let base = t.data.cast::<u8>();
    if t.byte_offset > isize::MAX as u64 {
        return Err(format!("{label}: byte_offset out of range"));
    }
    let ptr = unsafe { base.add(t.byte_offset as usize) }.cast::<T>();
    let align = std::mem::align_of::<T>();
    if !(ptr as usize).is_multiple_of(align) {
        return Err(format!("{label}: data is not {align}-byte aligned"));
    }
    Ok(ptr)
}

/// Copy a validated i32 tensor into a `Vec`; see [`checked_f64_data`].
pub(crate) fn copy_i32_tensor(
    tensor: *const DLManagedTensorVersioned,
    label: &str,
    expected_len: usize,
) -> Result<Vec<i32>, String> {
    let ptr = checked_i32_data(tensor, label, expected_len)?;
    Ok(unsafe { std::slice::from_raw_parts(ptr, expected_len) }.to_vec())
}

/// Copy a validated f64 tensor into a `Vec`; see [`checked_f64_data`].
/// With `require_finite`, a NaN or infinite element is an error.
pub(crate) fn copy_f64_tensor(
    tensor: *const DLManagedTensorVersioned,
    label: &str,
    expected_len: usize,
    require_finite: bool,
) -> Result<Vec<f64>, String> {
    let ptr = checked_f64_data(tensor, label, expected_len)?;
    let values = unsafe { std::slice::from_raw_parts(ptr, expected_len) }.to_vec();
    if require_finite {
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            return Err(format!("{label}: non-finite value at element {i}"));
        }
    }
    Ok(values)
}

// ---------------------------------------------------------------------------
// C-exported tensor functions
// ---------------------------------------------------------------------------

/// Create a non-owning 2-D f64 tensor on CPU wrapping an existing buffer.
///
/// The returned tensor borrows `data` — the caller must keep `data` alive
/// for the lifetime of the tensor.  Call `rgpot_tensor_free` when done.
///
/// # Safety
/// `data` must point to at least `rows * cols` contiguous `f64` values.
#[no_mangle]
pub unsafe extern "C" fn rgpot_tensor_cpu_f64_2d(
    data: *mut f64,
    rows: i64,
    cols: i64,
) -> *mut DLManagedTensorVersioned {
    create_borrowed_tensor(data.cast(), dtype_f64(), vec![rows, cols])
}

/// Create a non-owning 1-D i32 tensor on CPU wrapping an existing buffer.
///
/// # Safety
/// `data` must point to at least `len` contiguous `c_int` values.
#[no_mangle]
pub unsafe extern "C" fn rgpot_tensor_cpu_i32_1d(
    data: *mut c_int,
    len: i64,
) -> *mut DLManagedTensorVersioned {
    create_borrowed_tensor(data.cast(), dtype_i32(), vec![len])
}

/// Create a non-owning 2-D f64 tensor on CPU for a 3x3 matrix.
///
/// Convenience wrapper — equivalent to `rgpot_tensor_cpu_f64_2d(data, 3, 3)`.
///
/// # Safety
/// `data` must point to at least 9 contiguous `f64` values.
#[no_mangle]
pub unsafe extern "C" fn rgpot_tensor_cpu_f64_matrix3(
    data: *mut f64,
) -> *mut DLManagedTensorVersioned {
    create_borrowed_tensor(data.cast(), dtype_f64(), vec![3, 3])
}

/// Create an **owning** 2-D f64 tensor on CPU by copying data.
///
/// The returned tensor owns a copy of the data — the caller may free the
/// original buffer after this call.  Call `rgpot_tensor_free` when done.
///
/// Returns NULL, with a `rgpot_last_error()` message, when `data` is NULL,
/// `rows` or `cols` is negative, or `rows * cols` overflows.
///
/// # Safety
/// `data` must point to at least `rows * cols` contiguous `f64` values.
#[no_mangle]
pub unsafe extern "C" fn rgpot_tensor_owned_cpu_f64_2d(
    data: *const f64,
    rows: i64,
    cols: i64,
) -> *mut DLManagedTensorVersioned {
    use crate::status::set_last_error;
    if data.is_null() {
        set_last_error("rgpot_tensor_owned_cpu_f64_2d: data is NULL");
        return std::ptr::null_mut();
    }
    let len = match (usize::try_from(rows), usize::try_from(cols)) {
        (Ok(r), Ok(c)) => r.checked_mul(c).filter(|&n| n <= isize::MAX as usize / 8),
        _ => None,
    };
    let Some(len) = len else {
        set_last_error(&format!(
            "rgpot_tensor_owned_cpu_f64_2d: invalid shape [{rows}, {cols}]"
        ));
        return std::ptr::null_mut();
    };
    let vec = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
    create_owned_f64_tensor(vec, vec![rows, cols])
}

/// Free a DLPack tensor by invoking its deleter.
///
/// If `tensor` is `NULL`, this is a no-op.
///
/// # Safety
/// `tensor` must have been obtained from one of the `rgpot_tensor_*` creation
/// functions, or be a valid `DLManagedTensorVersioned` with a deleter.
#[no_mangle]
pub unsafe extern "C" fn rgpot_tensor_free(tensor: *mut DLManagedTensorVersioned) {
    if tensor.is_null() {
        return;
    }
    if let Some(deleter) = unsafe { (*tensor).deleter } {
        unsafe { deleter(tensor) };
    }
}

/// Get the device of a DLPack tensor.
///
/// For a NULL `tensor`, returns `{kDLCPU, -1}`; no real device has id -1.
///
/// # Safety
/// `tensor` must be NULL or a valid `DLManagedTensorVersioned*`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_tensor_device(
    tensor: *const DLManagedTensorVersioned,
) -> DLDevice {
    if tensor.is_null() {
        return DLDevice {
            device_type: DLDeviceType::kDLCPU,
            device_id: -1,
        };
    }
    unsafe { (*tensor).dl_tensor.device }
}

/// Get the raw data pointer of a DLPack tensor.
///
/// Returns NULL for a NULL `tensor`.
///
/// # Safety
/// `tensor` must be NULL or a valid `DLManagedTensorVersioned*`.
#[no_mangle]
pub unsafe extern "C" fn rgpot_tensor_data(
    tensor: *const DLManagedTensorVersioned,
) -> *const c_void {
    if tensor.is_null() {
        return std::ptr::null();
    }
    unsafe { (*tensor).dl_tensor.data as *const c_void }
}

/// Get the shape array and number of dimensions of a DLPack tensor.
///
/// Writes the number of dimensions to `*ndim_out` and returns a pointer to
/// the shape array (length `*ndim_out`).  For a NULL `tensor`, writes 0 to
/// `*ndim_out` and returns NULL; for a NULL `ndim_out`, returns NULL.
///
/// # Safety
/// `tensor` and `ndim_out` must each be NULL or a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn rgpot_tensor_shape(
    tensor: *const DLManagedTensorVersioned,
    ndim_out: *mut i32,
) -> *const i64 {
    if ndim_out.is_null() {
        return std::ptr::null();
    }
    if tensor.is_null() {
        unsafe { *ndim_out = 0 };
        return std::ptr::null();
    }
    let t = unsafe { &(*tensor) };
    unsafe { *ndim_out = t.dl_tensor.ndim };
    t.dl_tensor.shape as *const i64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_f64_2d_has_correct_metadata() {
        let mut data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = unsafe { rgpot_tensor_cpu_f64_2d(data.as_mut_ptr(), 2, 3) };
        assert!(!tensor.is_null());

        let t = unsafe { &(*tensor) };
        assert_eq!(t.dl_tensor.device, cpu_device());
        assert_eq!(t.dl_tensor.dtype, dtype_f64());
        assert_eq!(t.dl_tensor.ndim, 2);
        let shape = unsafe { std::slice::from_raw_parts(t.dl_tensor.shape, 2) };
        assert_eq!(shape, &[2, 3]);
        let strides = unsafe { std::slice::from_raw_parts(t.dl_tensor.strides, 2) };
        assert_eq!(strides, &[3, 1]);
        assert_eq!(t.dl_tensor.byte_offset, 0);

        // Data pointer should point to our array.
        let data_ptr = t.dl_tensor.data as *const f64;
        assert_eq!(data_ptr, data.as_ptr());

        unsafe { rgpot_tensor_free(tensor) };
        // data array is still valid (borrowed, not freed).
        assert_eq!(data[0], 1.0);
    }

    #[test]
    fn borrowed_i32_1d_has_correct_metadata() {
        let mut data = [6_i32, 1, 8];
        let tensor = unsafe { rgpot_tensor_cpu_i32_1d(data.as_mut_ptr(), 3) };
        assert!(!tensor.is_null());

        let t = unsafe { &(*tensor) };
        assert_eq!(t.dl_tensor.dtype, dtype_i32());
        assert_eq!(t.dl_tensor.ndim, 1);
        let shape = unsafe { std::slice::from_raw_parts(t.dl_tensor.shape, 1) };
        assert_eq!(shape, &[3]);

        unsafe { rgpot_tensor_free(tensor) };
        assert_eq!(data[2], 8);
    }

    #[test]
    fn borrowed_matrix3_is_3x3() {
        let mut data = [0.0_f64; 9];
        let tensor = unsafe { rgpot_tensor_cpu_f64_matrix3(data.as_mut_ptr()) };
        let t = unsafe { &(*tensor) };
        assert_eq!(t.dl_tensor.ndim, 2);
        let shape = unsafe { std::slice::from_raw_parts(t.dl_tensor.shape, 2) };
        assert_eq!(shape, &[3, 3]);
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn owned_f64_tensor_survives_source_drop() {
        let tensor = {
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            create_owned_f64_tensor(data, vec![2, 3])
        };
        // Source vec is dropped, but the tensor still has valid data.
        let t = unsafe { &(*tensor) };
        let slice =
            unsafe { std::slice::from_raw_parts(t.dl_tensor.data as *const f64, 6) };
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(t.flags & DLPACK_FLAG_BITMASK_IS_COPIED != 0);
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn owned_i32_tensor_survives_source_drop() {
        let tensor = {
            let data = vec![1_i32, 8, 6];
            create_owned_i32_tensor(data, vec![3])
        };
        let t = unsafe { &(*tensor) };
        let slice =
            unsafe { std::slice::from_raw_parts(t.dl_tensor.data as *const i32, 3) };
        assert_eq!(slice, &[1, 8, 6]);
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn owned_cpu_f64_2d_copies_data() {
        let source = [10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0];
        let tensor = unsafe { rgpot_tensor_owned_cpu_f64_2d(source.as_ptr(), 2, 3) };
        let t = unsafe { &(*tensor) };
        // Data pointer should NOT be source.as_ptr() (it was copied).
        assert_ne!(t.dl_tensor.data as *const f64, source.as_ptr());
        let slice =
            unsafe { std::slice::from_raw_parts(t.dl_tensor.data as *const f64, 6) };
        assert_eq!(slice, &source);
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn owned_cpu_f64_2d_rejects_bad_arguments() {
        let source = [1.0_f64; 6];
        let null = unsafe { rgpot_tensor_owned_cpu_f64_2d(std::ptr::null(), 2, 3) };
        assert!(null.is_null());
        let msg = unsafe { std::ffi::CStr::from_ptr(crate::status::rgpot_last_error()) };
        assert!(msg.to_string_lossy().contains("data is NULL"), "{msg:?}");
        let negative = unsafe { rgpot_tensor_owned_cpu_f64_2d(source.as_ptr(), -2, -3) };
        assert!(negative.is_null());
        let overflow = unsafe { rgpot_tensor_owned_cpu_f64_2d(source.as_ptr(), i64::MAX, 2) };
        assert!(overflow.is_null());
    }

    #[test]
    fn accessors_tolerate_null() {
        let dev = unsafe { rgpot_tensor_device(std::ptr::null()) };
        assert_eq!(dev.device_type, DLDeviceType::kDLCPU);
        assert_eq!(dev.device_id, -1);
        assert!(unsafe { rgpot_tensor_data(std::ptr::null()) }.is_null());
        let mut ndim: i32 = 7;
        let shape = unsafe { rgpot_tensor_shape(std::ptr::null(), &mut ndim) };
        assert!(shape.is_null());
        assert_eq!(ndim, 0);
        let mut data = [0.0_f64; 6];
        let tensor = unsafe { rgpot_tensor_cpu_f64_2d(data.as_mut_ptr(), 2, 3) };
        assert!(unsafe { rgpot_tensor_shape(tensor, std::ptr::null_mut()) }.is_null());
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn free_null_is_noop() {
        unsafe { rgpot_tensor_free(std::ptr::null_mut()) };
    }

    #[test]
    fn device_accessor_returns_cpu() {
        let mut data = [0.0_f64; 6];
        let tensor = unsafe { rgpot_tensor_cpu_f64_2d(data.as_mut_ptr(), 2, 3) };
        let dev = unsafe { rgpot_tensor_device(tensor) };
        assert_eq!(dev.device_type, DLDeviceType::kDLCPU);
        assert_eq!(dev.device_id, 0);
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn data_accessor_returns_correct_pointer() {
        let mut data = [42.0_f64; 3];
        let tensor = unsafe { rgpot_tensor_cpu_f64_2d(data.as_mut_ptr(), 1, 3) };
        let ptr = unsafe { rgpot_tensor_data(tensor) };
        assert_eq!(ptr as *const f64, data.as_ptr());
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn shape_accessor_returns_correct_values() {
        let mut data = [0.0_f64; 12];
        let tensor = unsafe { rgpot_tensor_cpu_f64_2d(data.as_mut_ptr(), 4, 3) };
        let mut ndim: i32 = 0;
        let shape_ptr = unsafe { rgpot_tensor_shape(tensor, &mut ndim) };
        assert_eq!(ndim, 2);
        let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndim as usize) };
        assert_eq!(shape, &[4, 3]);
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn validate_positions_ok() {
        let mut data = [0.0_f64; 6];
        let tensor = unsafe { rgpot_tensor_cpu_f64_2d(data.as_mut_ptr(), 2, 3) };
        assert_eq!(validate_positions(tensor), Ok(2));
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn validate_positions_null() {
        assert!(validate_positions(std::ptr::null()).is_err());
    }

    #[test]
    fn validate_positions_wrong_shape() {
        let mut data = [0.0_f64; 8];
        let tensor = unsafe { rgpot_tensor_cpu_f64_2d(data.as_mut_ptr(), 2, 4) };
        assert!(validate_positions(tensor).is_err());
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn validate_positions_wrong_dtype() {
        let mut data = [0_i32; 6];
        let tensor = unsafe { rgpot_tensor_cpu_i32_1d(data.as_mut_ptr(), 6) };
        assert!(validate_positions(tensor).is_err());
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn validate_atomic_numbers_ok() {
        let mut data = [1_i32, 8];
        let tensor = unsafe { rgpot_tensor_cpu_i32_1d(data.as_mut_ptr(), 2) };
        assert!(validate_atomic_numbers(tensor, 2).is_ok());
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn validate_atomic_numbers_wrong_count() {
        let mut data = [1_i32, 8];
        let tensor = unsafe { rgpot_tensor_cpu_i32_1d(data.as_mut_ptr(), 2) };
        assert!(validate_atomic_numbers(tensor, 3).is_err());
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn validate_box_matrix_ok() {
        let mut data = [0.0_f64; 9];
        let tensor = unsafe { rgpot_tensor_cpu_f64_matrix3(data.as_mut_ptr()) };
        assert!(validate_box_matrix(tensor).is_ok());
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn validate_box_matrix_wrong_shape() {
        let mut data = [0.0_f64; 6];
        let tensor = unsafe { rgpot_tensor_cpu_f64_2d(data.as_mut_ptr(), 2, 3) };
        assert!(validate_box_matrix(tensor).is_err());
        unsafe { rgpot_tensor_free(tensor) };
    }

    #[test]
    fn row_major_strides_1d() {
        assert_eq!(compute_row_major_strides(&[5]), vec![1]);
    }

    #[test]
    fn row_major_strides_2d() {
        assert_eq!(compute_row_major_strides(&[4, 3]), vec![3, 1]);
    }

    #[test]
    fn row_major_strides_3d() {
        assert_eq!(compute_row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn row_major_strides_empty() {
        assert_eq!(compute_row_major_strides(&[]), Vec::<i64>::new());
    }
}
