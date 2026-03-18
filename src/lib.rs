extern crate blas_src;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

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
/// sparse : bool, default True
///     If True, deduplicate points sharing the same boundary.
/// parallel : bool, default True
///     If True, use rayon parallelism for per-point projection.
/// seed : int or None, default None
///     Random seed for reproducibility. If None, uses entropy.
///
/// Returns
/// -------
/// list[list[float]]
///     Converged boundary points, each of length d.
#[pyfunction(name = "dbs")]
#[pyo3(signature = (data, y, n_points=1000, max_iter=100, tol=1e-6, sparse=true, parallel=true, seed=None))]
fn dbs_py(
    py: Python<'_>,
    data: PyReadonlyArray2<f64>,
    y: Vec<usize>,
    n_points: usize,
    max_iter: usize,
    tol: f64,
    sparse: bool,
    parallel: bool,
    seed: Option<u64>,
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
    let a_contiguous = arr.as_standard_layout();
    let a_flat: Vec<f32> = a_contiguous.iter().map(|&v| v as f32).collect();
    let tol_f32 = tol as f32;

    // Run the algorithm entirely GIL-free.
    let result = py.detach(move || {
        dbs_core(
            n_points, n, d, &a_flat, &y, max_iter, tol_f32, sparse, parallel, seed,
        )
    });

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

// ─────────────────────────────────────────────────────────────
//  Core types
// ─────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────
//  Main entry point
// ─────────────────────────────────────────────────────────────

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
/// - `parallel`: if true, use rayon for per-point work
/// - `seed`: optional random seed for reproducibility
pub fn dbs_core(
    k: usize,
    n: usize,
    d: usize,
    a_matrix: &[f32],
    labels: &[usize],
    max_iterations: usize,
    tol: f32,
    distill: bool,
    parallel: bool,
    seed: Option<u64>,
) -> DbsResult {
    // 1. Precompute norms (once)
    let a_norms = blas_ops::row_norms_sq(n, d, a_matrix);
    let half_norms = blas_ops::half_row_norms_sq(&a_norms);

    // 2. Generate random starting points in the bounding box of A
    let mut x_matrix = generate_random_points(k, d, a_matrix, seed);

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
                &half_norms,
                labels,
                CHUNK_SIZE,
                parallel,
            )
        } else {
            full_iteration(
                k,
                n,
                d,
                &mut x_matrix,
                a_matrix,
                &a_norms,
                &half_norms,
                labels,
                &mut s_matrix,
                parallel,
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

// ─────────────────────────────────────────────────────────────
//  Random initialization
// ─────────────────────────────────────────────────────────────

fn generate_random_points(k: usize, d: usize, a_matrix: &[f32], seed: Option<u64>) -> Vec<f32> {
    let mut rng: Box<dyn rand::RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

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
///
/// NOTE: `dot_x1` and `dot_x2` must be the *raw* dot products x·a[idx],
/// i.e. the sgemm output before any half-norm subtraction.
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

/// Per-query-point work shared by both iteration paths:
/// find the two nearest neighbors of different classes, then project.
///
/// `s_row` contains the *raw* dot products from sgemm (length n).
#[inline]
fn process_query_point(
    x_row: &mut [f32],
    s_row: &[f32],
    a_matrix: &[f32],
    a_norms: &[f32],
    half_norms: &[f32],
    labels: &[usize],
    d: usize,
    n: usize,
) -> (f32, [usize; 2]) {
    let scores = (0..n).map(|j| (j, s_row[j] - half_norms[j]));

    let Some((idx_1, idx_2, _, _)) = find_two_nearest(scores, labels) else {
        return (0.0, [0, 0]);
    };

    // s_row[idx] is the raw dot product x·a[idx] (sgemm output)
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

    (sq_disp, canonical_pair(idx_1, idx_2))
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
    half_norms: &[f32],
    labels: &[usize],
    s_matrix: &mut [f32],
    parallel: bool,
) -> IterationResult {
    // Bulk matrix multiply: S = X * A^T (BLAS handles its own threading)
    blas_ops::compute_xat_dot_products(k, n, d, x_matrix, a_matrix, s_matrix);

    // Per-point: find nearest pair + project onto bisector
    let results: Vec<(f32, [usize; 2])> = if parallel {
        x_matrix
            .par_chunks_exact_mut(d)
            .zip(s_matrix.par_chunks_exact(n))
            .map(|(x_row, s_row)| {
                process_query_point(x_row, s_row, a_matrix, a_norms, half_norms, labels, d, n)
            })
            .collect()
    } else {
        x_matrix
            .chunks_exact_mut(d)
            .zip(s_matrix.chunks_exact(n))
            .map(|(x_row, s_row)| {
                process_query_point(x_row, s_row, a_matrix, a_norms, half_norms, labels, d, n)
            })
            .collect()
    };

    let mut total_sq_movement = 0.0f32;
    let mut pairs = Vec::with_capacity(k);
    for (sq, pair) in results {
        total_sq_movement += sq;
        pairs.push(pair);
    }

    IterationResult {
        inertia: total_sq_movement / k as f32,
        pairs,
    }
}

// ─────────────────────────────────────────────────────────────
//  Tiled (chunked) iteration
// ─────────────────────────────────────────────────────────────

/// Per-query accumulator state for the tiled nearest-neighbor search.
struct QueryState {
    best_score_1: f32,
    best_score_2: f32,
    idx_1: usize,
    idx_2: usize,
    class_1: usize,
}

impl Default for QueryState {
    fn default() -> Self {
        Self {
            best_score_1: f32::NEG_INFINITY,
            best_score_2: f32::NEG_INFINITY,
            idx_1: 0,
            idx_2: 0,
            class_1: usize::MAX,
        }
    }
}

impl QueryState {
    /// Update this accumulator with one candidate from the current chunk.
    #[inline]
    fn update(&mut self, global_j: usize, score: f32, label: usize) {
        if score > self.best_score_1 {
            if label != self.class_1 {
                self.best_score_2 = self.best_score_1;
                self.idx_2 = self.idx_1;
            }
            self.best_score_1 = score;
            self.idx_1 = global_j;
            self.class_1 = label;
        } else if score > self.best_score_2 && label != self.class_1 {
            self.best_score_2 = score;
            self.idx_2 = global_j;
        }
    }
}

pub fn chunked_iteration(
    k: usize,
    n: usize,
    d: usize,
    x_matrix: &mut [f32],
    a_matrix: &[f32],
    a_norms: &[f32],
    half_norms: &[f32],
    labels: &[usize],
    chunk_size: usize,
    parallel: bool,
) -> IterationResult {
    let mut states: Vec<QueryState> = (0..k).map(|_| QueryState::default()).collect();
    let mut s_partial = vec![0.0f32; k * chunk_size];

    // ── Phase 1: Tiled nearest-neighbor search ──
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

        if parallel {
            states.par_iter_mut().enumerate().for_each(|(i, state)| {
                let s_row = &s_partial[i * cur_chunk..(i + 1) * cur_chunk];
                for local_j in 0..cur_chunk {
                    let global_j = j_start + local_j;
                    let score = s_row[local_j] - half_norms[global_j];
                    state.update(global_j, score, labels[global_j]);
                }
            });
        } else {
            for i in 0..k {
                let s_row = &s_partial[i * cur_chunk..(i + 1) * cur_chunk];
                for local_j in 0..cur_chunk {
                    let global_j = j_start + local_j;
                    let score = s_row[local_j] - half_norms[global_j];
                    states[i].update(global_j, score, labels[global_j]);
                }
            }
        }

        j_start = j_end;
    }

    // ── Phase 2: Projection ──
    // Build per-point inputs from the accumulated states, then project.
    // We need (idx_1, idx_2, raw_dot_x1, raw_dot_x2) per query point.
    // Raw dot = score + half_norm (undoing the subtraction from phase 1).
    let phase2_inputs: Vec<_> = states
        .iter()
        .map(|st| {
            (
                st.idx_1,
                st.idx_2,
                st.best_score_1 + half_norms[st.idx_1],
                st.best_score_2 + half_norms[st.idx_2],
                st.best_score_2 > f32::NEG_INFINITY,
            )
        })
        .collect();

    let results: Vec<(f32, [usize; 2])> = if parallel {
        x_matrix
            .par_chunks_exact_mut(d)
            .zip(phase2_inputs.par_iter())
            .map(|(x_row, &(i1, i2, dot_x1, dot_x2, valid))| {
                if !valid {
                    return (0.0, [0, 0]);
                }
                let sq = project_onto_bisector(x_row, a_matrix, a_norms, d, i1, i2, dot_x1, dot_x2);
                (sq, canonical_pair(i1, i2))
            })
            .collect()
    } else {
        x_matrix
            .chunks_exact_mut(d)
            .zip(phase2_inputs.iter())
            .map(|(x_row, &(i1, i2, dot_x1, dot_x2, valid))| {
                if !valid {
                    return (0.0, [0, 0]);
                }
                let sq = project_onto_bisector(x_row, a_matrix, a_norms, d, i1, i2, dot_x1, dot_x2);
                (sq, canonical_pair(i1, i2))
            })
            .collect()
    };

    let mut total_sq_movement = 0.0f32;
    let mut pairs = Vec::with_capacity(k);
    for (sq, pair) in results {
        total_sq_movement += sq;
        pairs.push(pair);
    }

    IterationResult {
        inertia: total_sq_movement / k as f32,
        pairs,
    }
}
