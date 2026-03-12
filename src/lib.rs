extern crate blas_src;
use pyo3::prelude::*;
use std::collections::HashMap;
mod linalg;
use numpy::PyReadonlyArray2;
use rand::Rng;
// use rayon::prelude::*;
mod blas_ops;

/// Decision Boundary Sampling.
///
/// Parameters
/// ----------
/// data : np.ndarray, shape (n, d)
///     The dataset, float64. Converted internally to float32.
/// y : list[int] or np.ndarray
///     Class labels per data point, length n.
/// n_points : int, default 1000
///     Number of random query points (k) to generate.
/// max_iter : int, default 100
///     Maximum number of iterations.
/// tol : float, default 1e-6
///     Convergence threshold on mean squared displacement.
/// distill : bool, default True
///     If True, deduplicate points sharing the same boundary.
///
/// Returns
/// -------
/// DbsOutput
///     Object with fields: x, pairs, n_points, inertia, iterations, converged.
#[pyfunction(name = "dbs")]
#[pyo3(signature = (data, y, n_points=1000, max_iter=100, tol=1e-6, sparse=true))]
fn dbs_py(
    py: Python<'_>,
    data: PyReadonlyArray2<f64>,
    y: Vec<usize>,
    n_points: usize,
    max_iter: usize,
    tol: f64,
    sparse: bool,
) -> PyResult<Vec<Vec<f32>>> {
    let arr = data.as_array();
    let n = arr.nrows();
    let d = arr.ncols();

    if y.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "y has length {} but data has {} rows",
            y.len(),
            n
        )));
    }

    // Convert f64 → f32 and ensure contiguous row-major layout.
    // .as_standard_layout() guarantees C-contiguous even if the input is
    // Fortran-ordered or has weird strides. mapv then gives us an owned f32 array.
    let a_contiguous = arr.as_standard_layout();
    let a_flat: Vec<f32> = a_contiguous.iter().map(|&v| v as f32).collect();
    let tol_f32 = tol as f32;

    // Run the algorithm entirely GIL-free.
    // All data is owned (Vec), so the closure is Send and safe to run
    // on any thread. When we add rayon later, the thread pool spawned
    // inside will also be GIL-free.
    let result =
        py.detach(move || dbs_core(n_points, n, d, &a_flat, &y, max_iter, tol_f32, sparse));

    let x_py: Vec<Vec<f32>> = result
        .x_matrix
        .chunks(d)
        .map(|chunk| chunk.to_vec())
        .collect();

    Ok(x_py)
}

#[pymodule]
fn dbsampler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dbs_py, m)?)?;
    Ok(())
}

/// The result of one iteration: inertia + the neighbor pair per query point.
pub struct IterationResult {
    /// Mean squared displacement across all query points.
    pub inertia: f32,
    /// For each query point, the two nearest neighbors of different classes,
    /// stored as (min_idx, max_idx) so that pair order is canonical.
    pub pairs: Vec<[usize; 2]>,
}

/// Final output of the DBS algorithm.
pub struct DbsResult {
    /// Converged (and optionally deduplicated) query points, flat (m × d).
    pub x_matrix: Vec<f32>,
    /// Canonical neighbor pairs per point.
    pub pairs: Vec<[usize; 2]>,
    /// Number of points (may be < k after dedup).
    pub m: usize,
    /// Final inertia (mean squared displacement) at convergence.
    pub final_inertia: f32,
    /// Number of iterations actually run.
    pub iterations: usize,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
}

/// Tile threshold: if k * n * 4 bytes > this, use chunked iteration.
/// 32 MB is a conservative L3 estimate that works on most machines.
const TILE_THRESHOLD_BYTES: usize = 32 * 1024 * 1024;

/// Column chunk size for the tiled path.
const CHUNK_SIZE: usize = 1024;

/// Main entry point for the Decision Boundary Sampling algorithm.
///
/// - `k`: number of random query points to generate
/// - `n`: number of data points (rows in A)
/// - `d`: dimensionality (columns in A)
/// - `a_matrix`: data matrix, row-major (n × d)
/// - `labels`: class label per data point (length n)
/// - `max_iterations`: hard cap on iterations
/// - `tol`: convergence threshold on inertia (mean squared displacement)
/// - `distill`: if true, deduplicate points sharing the same boundary
pub fn dbs_core(
    k: usize,
    n: usize,
    d: usize,
    a_matrix: &[f32],
    labels: &[usize],
    max_iterations: usize,
    tol: f32,
    distill: bool,
) -> DbsResult {
    // 1. Precompute squared norms of A rows
    let a_norms = blas_ops::row_norms_sq(n, d, a_matrix);

    // 2. Generate random starting points in the bounding box of A
    let mut x_matrix = generate_random_points(k, d, a_matrix);

    // 3. Decide iteration strategy based on S matrix size
    let s_bytes = k * n * std::mem::size_of::<f32>();
    let use_tiled = s_bytes > TILE_THRESHOLD_BYTES;

    // Pre-allocate S buffer only for the non-tiled path
    let mut s_matrix = if use_tiled {
        Vec::new()
    } else {
        vec![0.0f32; k * n]
    };

    // 4. Iterate until convergence or max_iterations
    let mut last_result = IterationResult {
        inertia: f32::INFINITY,
        pairs: Vec::new(),
    };
    let mut converged = false;
    let mut iters_run = 0;

    for iter in 0..max_iterations {
        let result = if use_tiled {
            chunked_iteration(
                k,
                n,
                d,
                &mut x_matrix,
                a_matrix,
                &a_norms,
                labels,
                CHUNK_SIZE,
            )
        } else {
            full_iteration(
                k,
                n,
                d,
                &mut x_matrix,
                a_matrix,
                &a_norms,
                labels,
                &mut s_matrix,
            )
        };

        iters_run = iter + 1;
        let inertia = result.inertia;
        last_result = result;

        if inertia < tol {
            converged = true;
            break;
        }
    }

    // 5. Optionally distill (deduplicate)
    if distill {
        let (deduped_x, deduped_pairs) = dedup_points(&x_matrix, d, &last_result.pairs);
        let m = deduped_pairs.len();
        DbsResult {
            x_matrix: deduped_x,
            pairs: deduped_pairs,
            m,
            final_inertia: last_result.inertia,
            iterations: iters_run,
            converged,
        }
    } else {
        DbsResult {
            x_matrix,
            pairs: last_result.pairs,
            m: k,
            final_inertia: last_result.inertia,
            iterations: iters_run,
            converged,
        }
    }
}

fn generate_random_points(k: usize, d: usize, a_matrix: &[f32]) -> Vec<f32> {
    let mut rng = rand::thread_rng();

    let mut mins = vec![f32::INFINITY; d];
    let mut maxs = vec![f32::NEG_INFINITY; d];
    for row in a_matrix.chunks_exact(d) {
        for (dim, &val) in row.iter().enumerate() {
            if val < mins[dim] {
                mins[dim] = val;
            }
            if val > maxs[dim] {
                maxs[dim] = val;
            }
        }
    }

    let mut x = vec![0.0f32; k * d];
    for point in x.chunks_exact_mut(d) {
        for (dim, val) in point.iter_mut().enumerate() {
            *val = rng.gen_range(mins[dim]..=maxs[dim]);
        }
    }
    x
}

// ─────────────────────────────────────────────────────────────
//  Shared helpers
// ─────────────────────────────────────────────────────────────

/// Canonicalizes a neighbor pair so (3,7) and (7,3) map to the same key.
#[inline]
fn canonical_pair(a: usize, b: usize) -> [usize; 2] {
    if a <= b {
        [a, b]
    } else {
        [b, a]
    }
}

/// Packs a canonical pair into a single u64 for hashing.
#[inline]
fn pack_pair(pair: [usize; 2]) -> u64 {
    (pair[0] as u64) << 32 | pair[1] as u64
}

/// Finds the two nearest neighbors of different classes for a single query,
/// given an iterator of (global_index, score) pairs.
///
/// Returns (idx_1, idx_2, score_1, score_2) or None if fewer than 2 classes.
#[inline]
fn find_two_nearest(
    scores: impl Iterator<Item = (usize, f32)>,
    labels: &[usize],
) -> Option<(usize, usize, f32, f32)> {
    let mut best_score_1 = f32::NEG_INFINITY;
    let mut best_score_2 = f32::NEG_INFINITY;
    let mut idx_1 = 0usize;
    let mut idx_2 = 0usize;
    let mut class_1 = usize::MAX;

    for (j, score) in scores {
        if score > best_score_1 {
            if labels[j] != class_1 {
                best_score_2 = best_score_1;
                idx_2 = idx_1;
            }
            best_score_1 = score;
            idx_1 = j;
            class_1 = labels[j];
        } else if score > best_score_2 && labels[j] != class_1 {
            best_score_2 = score;
            idx_2 = j;
        }
    }

    if best_score_2 == f32::NEG_INFINITY {
        None
    } else {
        Some((idx_1, idx_2, best_score_1, best_score_2))
    }
}

/// Projects x onto the bisecting hyperplane of a[idx_1] and a[idx_2].
/// Returns the squared displacement.
#[inline]
fn project_onto_bisector(
    x_row: &mut [f32],
    a_matrix: &[f32],
    a_norms: &[f32],
    d: usize,
    idx_1: usize,
    idx_2: usize,
    dot_x1: f32,
    dot_x2: f32,
) -> f32 {
    let dot_12 = blas_ops::dot_rows(a_matrix, d, idx_1, idx_2);

    let norm_1 = a_norms[idx_1];
    let norm_2 = a_norms[idx_2];

    let num = dot_x2 - dot_x1 - 0.5 * (norm_2 - norm_1);
    let den = norm_1 + norm_2 - 2.0 * dot_12;
    let s = num / (den + 1e-12);

    blas_ops::axpy_row(x_row, s, a_matrix, d, idx_1); // x += s * a1
    blas_ops::axpy_row(x_row, -s, a_matrix, d, idx_2); // x -= s * a2

    s * s * den
}

// ─────────────────────────────────────────────────────────────
//  Deduplication
// ─────────────────────────────────────────────────────────────

/// Deduplicates query points that converged to the same decision boundary.
///
/// Two points are duplicates if they share the same canonical neighbor pair.
/// Keeps the first occurrence of each pair.
/// Returns the compacted X matrix and unique pairs.
pub fn dedup_points(
    x_matrix: &[f32],
    d: usize,
    pairs: &[[usize; 2]],
) -> (Vec<f32>, Vec<[usize; 2]>) {
    let mut seen = HashMap::new();
    let mut unique_x = Vec::new();
    let mut unique_pairs = Vec::new();

    for (i, pair) in pairs.iter().enumerate() {
        let key = pack_pair(*pair);
        if seen.insert(key, i).is_none() {
            let start = i * d;
            unique_x.extend_from_slice(&x_matrix[start..start + d]);
            unique_pairs.push(*pair);
        }
    }

    (unique_x, unique_pairs)
}

// ─────────────────────────────────────────────────────────────
//  Full (non-tiled) iteration
// ─────────────────────────────────────────────────────────────

pub fn full_iteration(
    k: usize,
    n: usize,
    d: usize,
    x_matrix: &mut [f32],
    a_matrix: &[f32],
    a_norms: &[f32],
    labels: &[usize],
    s_matrix: &mut [f32],
) -> IterationResult {
    let half_norms = blas_ops::half_row_norms_sq(a_norms);
    let mut pairs = Vec::with_capacity(k);
    let mut total_sq_movement = 0.0f32;

    blas_ops::compute_xat_dot_products(k, n, d, x_matrix, a_matrix, s_matrix);

    for (x_row, s_row) in x_matrix.chunks_exact_mut(d).zip(s_matrix.chunks_exact(n)) {
        let scores = (0..n).map(|j| (j, s_row[j] - half_norms[j]));

        let Some((idx_1, idx_2, _, _)) = find_two_nearest(scores, labels) else {
            pairs.push([0, 0]);
            continue;
        };

        let sq_disp = project_onto_bisector(
            x_row,
            a_matrix,
            a_norms,
            d,
            idx_1,
            idx_2,
            s_row[idx_1],
            s_row[idx_2],
        );

        total_sq_movement += sq_disp;
        pairs.push(canonical_pair(idx_1, idx_2));
    }

    IterationResult {
        inertia: total_sq_movement / k as f32,
        pairs,
    }
}

// ─────────────────────────────────────────────────────────────
//  Tiled (chunked) iteration
// ─────────────────────────────────────────────────────────────

pub fn chunked_iteration(
    k: usize,
    n: usize,
    d: usize,
    x_matrix: &mut [f32],
    a_matrix: &[f32],
    a_norms: &[f32],
    labels: &[usize],
    chunk_size: usize,
) -> IterationResult {
    let half_norms = blas_ops::half_row_norms_sq(a_norms);

    // Per-query accumulators for the tiled search
    let mut best_score_1 = vec![f32::NEG_INFINITY; k];
    let mut best_score_2 = vec![f32::NEG_INFINITY; k];
    let mut idx_1 = vec![0usize; k];
    let mut idx_2 = vec![0usize; k];
    let mut class_1 = vec![usize::MAX; k];

    let mut s_partial = vec![0.0f32; k * chunk_size];

    // ── Phase 1: Tiled search ──
    let mut j_start = 0;
    while j_start < n {
        let j_end = (j_start + chunk_size).min(n);
        let cur_chunk = j_end - j_start;

        let a_chunk = &a_matrix[j_start * d..j_end * d];
        blas_ops::compute_xat_dot_products(
            k,
            cur_chunk,
            d,
            x_matrix,
            a_chunk,
            &mut s_partial[..k * cur_chunk],
        );

        for i in 0..k {
            let s_row = &s_partial[i * cur_chunk..(i + 1) * cur_chunk];
            for local_j in 0..cur_chunk {
                let global_j = j_start + local_j;
                let score = s_row[local_j] - half_norms[global_j];

                if score > best_score_1[i] {
                    if labels[global_j] != class_1[i] {
                        best_score_2[i] = best_score_1[i];
                        idx_2[i] = idx_1[i];
                    }
                    best_score_1[i] = score;
                    idx_1[i] = global_j;
                    class_1[i] = labels[global_j];
                } else if score > best_score_2[i] && labels[global_j] != class_1[i] {
                    best_score_2[i] = score;
                    idx_2[i] = global_j;
                }
            }
        }

        j_start = j_end;
    }

    // ── Phase 2: Projection ──
    let mut total_sq_movement = 0.0f32;
    let mut pairs = Vec::with_capacity(k);

    for (i, x_row) in x_matrix.chunks_exact_mut(d).enumerate() {
        if best_score_2[i] == f32::NEG_INFINITY {
            pairs.push([0, 0]);
            continue;
        }

        let dot_x1 = best_score_1[i] + half_norms[idx_1[i]];
        let dot_x2 = best_score_2[i] + half_norms[idx_2[i]];

        let sq_disp = project_onto_bisector(
            x_row, a_matrix, a_norms, d, idx_1[i], idx_2[i], dot_x1, dot_x2,
        );

        total_sq_movement += sq_disp;
        pairs.push(canonical_pair(idx_1[i], idx_2[i]));
    }

    IterationResult {
        inertia: total_sq_movement / k as f32,
        pairs,
    }
}
