extern crate blas_src;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;
use std::ffi::CString;

mod batching;
mod blas_ops;
mod validation;

pub use batching::{dbs_batched_core, BatchOptions, BatchedResult};

#[cfg(test)]
mod tests;

/// Decision Boundary Sampling.
///
/// Parameters
/// ----------
/// data : np.ndarray, shape (n, d)
///     The dataset, float32 or float64. Computed internally in float32.
/// y : list[int] or np.ndarray
///     Class labels per data point, length n.
/// n_points : int, default 1000
///     Number of boundary points to request.
/// max_iter : int, default 100
///     Maximum number of iterations.
/// tol : float, default 1e-6
///     Per-point convergence threshold on squared displacement.
/// sparse : bool, default True
///     If True, sample in batches until n_points distinct neighboring data
///     pairs are found or max_batches is reached.
/// parallel : bool, default True
///     If True, use rayon parallelism for per-point projection.
/// seed : int or None, default None
///     Random seed for reproducibility. If None, uses entropy.
/// batch_size : int, default 256
///     Maximum number of query points in each sparse batch.
/// max_batches : int, default 20
///     Hard limit on sparse batches.
///
/// Returns
/// -------
/// list[list[float]]
///     Sampled boundary points, each of length d. With sparse=True, the
///     result can be shorter than n_points when max_batches is reached.
#[pyfunction(name = "dbs")]
#[pyo3(signature = (data, y, n_points=1000, max_iter=100, tol=1e-6, sparse=true, parallel=true, seed=None, *, batch_size=256, max_batches=20))]
fn dbs_py(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    y: Vec<i64>,
    n_points: usize,
    max_iter: usize,
    tol: f64,
    sparse: bool,
    parallel: bool,
    seed: Option<u64>,
    batch_size: usize,
    max_batches: usize,
) -> PyResult<Vec<Vec<f32>>> {
    if n_points == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_points must be greater than zero",
        ));
    }
    if sparse && batch_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "batch_size must be greater than zero",
        ));
    }
    if sparse && max_batches == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_batches must be greater than zero",
        ));
    }

    let validation_points = if sparse { batch_size } else { n_points };
    let prepared = if let Ok(array) = data.extract::<PyReadonlyArray2<f32>>() {
        validation::prepare(array.as_array(), &y, validation_points, max_iter, tol)
    } else if let Ok(array) = data.extract::<PyReadonlyArray2<f64>>() {
        validation::prepare(array.as_array(), &y, validation_points, max_iter, tol)
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "data must be a two-dimensional NumPy array with dtype float32 or float64",
        ));
    }
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let n = prepared.n;
    let d = prepared.d;
    let center = prepared.center;
    let scale = prepared.scale;
    let tolerance = prepared.tolerance;

    // Run the algorithm entirely GIL-free.
    let points = py
        .detach(move || {
            if sparse {
                let options = BatchOptions {
                    target_points: n_points,
                    batch_size,
                    max_batches,
                    max_iterations: max_iter,
                    tolerance,
                    parallel,
                    seed,
                    adaptive: true,
                };
                dbs_batched_core(n, d, &prepared.data, &prepared.labels, &options)
                    .map(|result| result.x_matrix)
            } else {
                Ok(dbs_core(
                    n_points,
                    n,
                    d,
                    &prepared.data,
                    &prepared.labels,
                    max_iter,
                    tolerance,
                    false,
                    parallel,
                    seed,
                )
                .x_matrix)
            }
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let found = points.len() / d;
    if sparse && found < n_points {
        let message = CString::new(format!(
            "requested {n_points} distinct boundary points, but found {found} before reaching max_batches={max_batches}"
        ))
        .expect("warning message contains no null bytes");
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
            message.as_c_str(),
            1,
        )?;
    }

    Ok(denormalize_points(&points, d, scale, &center))
}

fn denormalize_points(points: &[f32], d: usize, scale: f64, center: &[f64]) -> Vec<Vec<f32>> {
    points
        .chunks(d)
        .map(|point| {
            point
                .iter()
                .enumerate()
                .map(|(dim, &value)| (value as f64 * scale + center[dim]) as f32)
                .collect()
        })
        .collect()
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
    /// Whether each query point has met the per-point stopping condition.
    pub finished: Vec<bool>,
}

/// Final output of the DBS algorithm.
pub struct DbsResult {
    /// Final (and optionally deduplicated) query points, flat (m × d).
    pub x_matrix: Vec<f32>,
    /// Canonical neighbor pairs per point.
    pub pairs: Vec<[usize; 2]>,
    /// Number of points (may be < k after dedup).
    pub m: usize,
    /// Mean squared displacement in the final iteration.
    pub final_inertia: f32,
    /// Number of iterations actually run.
    pub iterations: usize,
    /// Whether every point converged within tolerance.
    pub converged: bool,
}

pub(crate) struct ConvergenceResult {
    pub x_matrix: Vec<f32>,
    pub pairs: Vec<[usize; 2]>,
    pub finished: Vec<bool>,
    pub point_iterations: Vec<usize>,
    pub final_inertia: f32,
    pub iterations: usize,
    pub converged: bool,
}

/// Tile threshold: if k * n * 4 bytes > this, use chunked iteration.
/// 32 MB is a conservative L3 estimate that works on most machines.
const TILE_THRESHOLD_BYTES: usize = 32 * 1024 * 1024;

/// Maximum number of scores kept by the tiled path.
const SCORE_TILE_ELEMENTS: usize = TILE_THRESHOLD_BYTES / std::mem::size_of::<f32>();

/// Preferred data-side tile. This keeps score scans cache-friendly while
/// avoiding very small matrix multiplications.
const DATA_TILE_SIZE: usize = 2048;

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
/// - `tol`: per-point threshold on squared displacement
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
    // 1. Precompute norms and generate the initial points
    let a_norms = blas_ops::row_norms_sq(n, d, a_matrix);
    let half_norms = blas_ops::half_row_norms_sq(&a_norms);
    let x_matrix = generate_random_points(k, d, a_matrix, seed);
    let result = converge_points(
        x_matrix,
        n,
        d,
        a_matrix,
        &half_norms,
        labels,
        max_iterations,
        tol,
        parallel,
        false,
    );

    // 2. Optionally distill (deduplicate)
    if distill {
        let (deduped_x, deduped_pairs) =
            dedup_points(&result.x_matrix, d, &result.pairs, &result.finished);
        let m = deduped_pairs.len();
        DbsResult {
            x_matrix: deduped_x,
            pairs: deduped_pairs,
            m,
            final_inertia: result.final_inertia,
            iterations: result.iterations,
            converged: result.converged,
        }
    } else {
        DbsResult {
            x_matrix: result.x_matrix,
            pairs: result.pairs,
            m: k,
            final_inertia: result.final_inertia,
            iterations: result.iterations,
            converged: result.converged,
        }
    }
}

pub(crate) fn converge_points(
    mut x_matrix: Vec<f32>,
    n: usize,
    d: usize,
    a_matrix: &[f32],
    half_norms: &[f32],
    labels: &[usize],
    max_iterations: usize,
    tol: f32,
    parallel: bool,
    track_point_iterations: bool,
) -> ConvergenceResult {
    let k = x_matrix.len() / d;
    let score_elements = k.saturating_mul(n);
    let use_tiled = score_elements > SCORE_TILE_ELEMENTS;

    let mut s_matrix = if use_tiled {
        Vec::new()
    } else {
        vec![0.0f32; k * n]
    };

    let mut last_result = IterationResult {
        inertia: f32::INFINITY,
        pairs: vec![[usize::MAX, usize::MAX]; k],
        finished: vec![false; k],
    };
    let mut converged = false;
    let mut iters_run = 0;
    let mut point_iterations = if track_point_iterations {
        vec![0usize; k]
    } else {
        Vec::new()
    };

    for iter in 0..max_iterations {
        let result = if use_tiled {
            chunked_iteration(
                k,
                n,
                d,
                &mut x_matrix,
                a_matrix,
                half_norms,
                labels,
                SCORE_TILE_ELEMENTS,
                parallel,
                &last_result.pairs,
                &last_result.finished,
                tol,
            )
        } else {
            full_iteration(
                k,
                n,
                d,
                &mut x_matrix,
                a_matrix,
                half_norms,
                labels,
                &mut s_matrix,
                parallel,
                &last_result.pairs,
                &last_result.finished,
                tol,
            )
        };

        iters_run = iter + 1;
        if track_point_iterations {
            for (index, &point_finished) in result.finished.iter().enumerate() {
                if point_finished && !last_result.finished[index] {
                    point_iterations[index] = iters_run;
                }
            }
        }
        converged = result.finished.iter().all(|&finished| finished);
        last_result = result;

        if converged {
            break;
        }
    }

    if track_point_iterations {
        for point_iteration in &mut point_iterations {
            if *point_iteration == 0 {
                *point_iteration = iters_run;
            }
        }
    }

    ConvergenceResult {
        x_matrix,
        pairs: last_result.pairs,
        finished: last_result.finished,
        point_iterations,
        final_inertia: last_result.inertia,
        iterations: iters_run,
        converged,
    }
}

// ─────────────────────────────────────────────────────────────
//  Random initialization
// ─────────────────────────────────────────────────────────────

fn generate_random_points(k: usize, d: usize, a_matrix: &[f32], seed: Option<u64>) -> Vec<f32> {
    let (mins, maxs) = data_bounds(d, a_matrix);
    match seed {
        Some(seed) => {
            let mut rng = StdRng::seed_from_u64(seed);
            generate_random_points_from_bounds(k, d, &mins, &maxs, &mut rng)
        }
        None => {
            let mut rng = rand::thread_rng();
            generate_random_points_from_bounds(k, d, &mins, &maxs, &mut rng)
        }
    }
}

pub(crate) fn data_bounds(d: usize, a_matrix: &[f32]) -> (Vec<f32>, Vec<f32>) {
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

    (mins, maxs)
}

pub(crate) fn generate_random_points_from_bounds<R: Rng + ?Sized>(
    k: usize,
    d: usize,
    mins: &[f32],
    maxs: &[f32],
    rng: &mut R,
) -> Vec<f32> {
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
    d: usize,
    idx_1: usize,
    idx_2: usize,
) -> f32 {
    let a_1 = &a_matrix[idx_1 * d..(idx_1 + 1) * d];
    let a_2 = &a_matrix[idx_2 * d..(idx_2 + 1) * d];
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;

    for dim in 0..d {
        let difference = a_2[dim] as f64 - a_1[dim] as f64;
        let midpoint = a_1[dim] as f64 + 0.5 * difference;
        numerator += (x_row[dim] as f64 - midpoint) * difference;
        denominator += difference * difference;
    }

    debug_assert!(denominator > 0.0);
    if denominator == 0.0 {
        return f32::INFINITY;
    }

    let scale = numerator / denominator;
    for dim in 0..d {
        let difference = a_2[dim] as f64 - a_1[dim] as f64;
        x_row[dim] = (x_row[dim] as f64 - scale * difference) as f32;
    }

    (scale * scale * denominator) as f32
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
    half_norms: &[f32],
    labels: &[usize],
    d: usize,
    n: usize,
    previous_pair: [usize; 2],
    tolerance: f32,
) -> (f32, [usize; 2], bool) {
    let scores = (0..n).map(|j| (j, s_row[j] - half_norms[j]));

    let Some((idx_1, idx_2, _, _)) = find_two_nearest(scores, labels) else {
        return (f32::INFINITY, [usize::MAX, usize::MAX], false);
    };
    let pair = canonical_pair(idx_1, idx_2);

    // The previous iteration already put x on this pair's bisector.
    if pair == previous_pair {
        return (0.0, pair, true);
    }

    let sq_disp = project_onto_bisector(x_row, a_matrix, d, idx_1, idx_2);

    (sq_disp, pair, sq_disp <= tolerance)
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
    finished: &[bool],
) -> (Vec<f32>, Vec<[usize; 2]>) {
    let mut seen = HashSet::new();
    let mut unique_x = Vec::new();
    let mut unique_pairs = Vec::new();

    for (i, pair) in pairs.iter().enumerate() {
        if finished[i] && seen.insert(*pair) {
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
    half_norms: &[f32],
    labels: &[usize],
    s_matrix: &mut [f32],
    parallel: bool,
    previous_pairs: &[[usize; 2]],
    finished: &[bool],
    tolerance: f32,
) -> IterationResult {
    // Bulk matrix multiply: S = X * A^T (BLAS handles its own threading)
    blas_ops::compute_xat_dot_products(k, n, d, x_matrix, a_matrix, s_matrix);

    // Per-point: find nearest pair + project onto bisector
    let results: Vec<(f32, [usize; 2], bool)> = if parallel {
        x_matrix
            .par_chunks_exact_mut(d)
            .zip(s_matrix.par_chunks_exact(n))
            .enumerate()
            .map(|(index, (x_row, s_row))| {
                if finished[index] {
                    return (0.0, previous_pairs[index], true);
                }
                process_query_point(
                    x_row,
                    s_row,
                    a_matrix,
                    half_norms,
                    labels,
                    d,
                    n,
                    previous_pairs[index],
                    tolerance,
                )
            })
            .collect()
    } else {
        x_matrix
            .chunks_exact_mut(d)
            .zip(s_matrix.chunks_exact(n))
            .enumerate()
            .map(|(index, (x_row, s_row))| {
                if finished[index] {
                    return (0.0, previous_pairs[index], true);
                }
                process_query_point(
                    x_row,
                    s_row,
                    a_matrix,
                    half_norms,
                    labels,
                    d,
                    n,
                    previous_pairs[index],
                    tolerance,
                )
            })
            .collect()
    };

    let mut total_sq_movement = 0.0f32;
    let mut pairs = Vec::with_capacity(k);
    let mut next_finished = Vec::with_capacity(k);
    for (sq, pair, point_finished) in results {
        total_sq_movement += sq;
        pairs.push(pair);
        next_finished.push(point_finished);
    }

    IterationResult {
        inertia: total_sq_movement / k as f32,
        pairs,
        finished: next_finished,
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

/// Chooses a two-dimensional score tile no larger than `max_scores`.
/// When possible, the smaller matrix dimension is kept whole so BLAS still
/// receives large, efficient matrix multiplications.
fn score_tile_shape(k: usize, n: usize, max_scores: usize) -> (usize, usize) {
    let max_scores = max_scores.max(1);
    if k <= max_scores / n {
        return (k, n);
    }

    let data_tile = if n >= k {
        DATA_TILE_SIZE.min(n).min((max_scores / k).max(1))
    } else if n <= max_scores {
        n
    } else {
        (max_scores as f64).sqrt().floor().max(1.0) as usize
    };
    let query_tile = (max_scores / data_tile).max(1).min(k);
    (query_tile, data_tile)
}

pub fn chunked_iteration(
    k: usize,
    n: usize,
    d: usize,
    x_matrix: &mut [f32],
    a_matrix: &[f32],
    half_norms: &[f32],
    labels: &[usize],
    max_scores: usize,
    parallel: bool,
    previous_pairs: &[[usize; 2]],
    finished: &[bool],
    tolerance: f32,
) -> IterationResult {
    let (query_tile, data_tile) = score_tile_shape(k, n, max_scores);
    let mut s_partial = vec![0.0f32; query_tile * data_tile];
    let mut total_sq_movement = 0.0f32;
    let mut pairs = Vec::with_capacity(k);
    let mut next_finished = Vec::with_capacity(k);

    // Each query tile completes its nearest-neighbor search and projection
    // before the next tile starts. The score buffer is reused throughout.
    let mut i_start = 0;
    while i_start < k {
        let i_end = (i_start + query_tile).min(k);
        let cur_queries = i_end - i_start;
        let x_tile = &mut x_matrix[i_start * d..i_end * d];
        let mut states: Vec<QueryState> = (0..cur_queries).map(|_| QueryState::default()).collect();

        let mut j_start = 0;
        while j_start < n {
            let j_end = (j_start + data_tile).min(n);
            let cur_data = j_end - j_start;
            let a_tile = &a_matrix[j_start * d..j_end * d];
            blas_ops::compute_xat_dot_products(
                cur_queries,
                cur_data,
                d,
                x_tile,
                a_tile,
                &mut s_partial[..cur_queries * cur_data],
            );

            if parallel {
                states
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(local_i, state)| {
                        if finished[i_start + local_i] {
                            return;
                        }
                        let s_row = &s_partial[local_i * cur_data..(local_i + 1) * cur_data];
                        for (local_j, &dot) in s_row.iter().enumerate() {
                            let global_j = j_start + local_j;
                            state.update(global_j, dot - half_norms[global_j], labels[global_j]);
                        }
                    });
            } else {
                for (local_i, state) in states.iter_mut().enumerate() {
                    if finished[i_start + local_i] {
                        continue;
                    }
                    let s_row = &s_partial[local_i * cur_data..(local_i + 1) * cur_data];
                    for (local_j, &dot) in s_row.iter().enumerate() {
                        let global_j = j_start + local_j;
                        state.update(global_j, dot - half_norms[global_j], labels[global_j]);
                    }
                }
            }

            j_start = j_end;
        }

        let results: Vec<(f32, [usize; 2], bool)> = if parallel {
            x_tile
                .par_chunks_exact_mut(d)
                .zip(states.par_iter())
                .enumerate()
                .map(|(local_i, (x_row, state))| {
                    let index = i_start + local_i;
                    if finished[index] {
                        return (0.0, previous_pairs[index], true);
                    }
                    if state.best_score_2 == f32::NEG_INFINITY {
                        return (f32::INFINITY, [usize::MAX, usize::MAX], false);
                    }
                    let pair = canonical_pair(state.idx_1, state.idx_2);
                    if pair == previous_pairs[index] {
                        return (0.0, pair, true);
                    }
                    let sq = project_onto_bisector(x_row, a_matrix, d, state.idx_1, state.idx_2);
                    (sq, pair, sq <= tolerance)
                })
                .collect()
        } else {
            x_tile
                .chunks_exact_mut(d)
                .zip(states.iter())
                .enumerate()
                .map(|(local_i, (x_row, state))| {
                    let index = i_start + local_i;
                    if finished[index] {
                        return (0.0, previous_pairs[index], true);
                    }
                    if state.best_score_2 == f32::NEG_INFINITY {
                        return (f32::INFINITY, [usize::MAX, usize::MAX], false);
                    }
                    let pair = canonical_pair(state.idx_1, state.idx_2);
                    if pair == previous_pairs[index] {
                        return (0.0, pair, true);
                    }
                    let sq = project_onto_bisector(x_row, a_matrix, d, state.idx_1, state.idx_2);
                    (sq, pair, sq <= tolerance)
                })
                .collect()
        };

        for (sq, pair, point_finished) in results {
            total_sq_movement += sq;
            pairs.push(pair);
            next_finished.push(point_finished);
        }
        i_start = i_end;
    }

    IterationResult {
        inertia: total_sq_movement / k as f32,
        pairs,
        finished: next_finished,
    }
}
