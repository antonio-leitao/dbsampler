use cblas_sys::{cblas_sdot, cblas_sgemm, CblasNoTrans, CblasRowMajor, CblasTrans};

/// Computes ||row_i||^2 for each row.
pub fn row_norms_sq(rows: usize, cols: usize, data: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    debug_assert!(data.len() >= rows * cols);
    unsafe {
        for i in 0..rows {
            let ptr = data.as_ptr().add(i * cols);
            out[i] = cblas_sdot(cols as i32, ptr, 1, ptr, 1);
        }
    }
    out
}

/// Computes the raw dot products between queries X and dataset A.
/// S = X * A^T
/// S is of size (K x N), stored row-major.
#[inline]
pub fn compute_xat_dot_products(
    k: usize,
    n: usize,
    d: usize,
    x: &[f32],
    a: &[f32],
    s_buf: &mut [f32],
) {
    debug_assert!(x.len() >= k * d);
    debug_assert!(a.len() >= n * d);
    debug_assert!(s_buf.len() >= k * n);
    unsafe {
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            k as i32,
            n as i32,
            d as i32,
            1.0,
            x.as_ptr(),
            d as i32,
            a.as_ptr(),
            d as i32,
            0.0,
            s_buf.as_mut_ptr(),
            n as i32,
        );
    }
}

/// Precomputes half-norms: 0.5 * ||a_j||^2 for each row.
pub fn half_row_norms_sq(norms: &[f32]) -> Vec<f32> {
    norms.iter().map(|&v| 0.5 * v).collect()
}
