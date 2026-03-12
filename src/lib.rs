use pyo3::prelude::*;
mod blas_ops;
mod linalg;
use anyhow::{anyhow, Result};
use hashbrown::HashSet;
use nohash_hasher::IntSet;
use rand::Rng;
use rayon::prelude::*;

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct Neighbours {
    first: usize,
    second: usize,
}

impl Neighbours {
    pub fn new(a: usize, b: usize) -> Neighbours {
        // Ensure the smaller value is stored in `first`
        if a < b {
            Neighbours {
                first: a,
                second: b,
            }
        } else {
            Neighbours {
                first: b,
                second: a,
            }
        }
    }
}

struct Point {
    coords: Vec<f64>,
    norm: f64,
    neighbors: Neighbours,
    skip: bool,
}

fn argsort(data: &[f64]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
    indices
}

fn generate_uniform_points(
    n: usize,
    dimensions: usize,
    min_values: Vec<f64>,
    max_values: Vec<f64>,
) -> Result<Vec<Point>> {
    if min_values.len() != max_values.len() || max_values.len() != dimensions {
        return Err(anyhow!("Dimensions are inconsistent"));
    }
    let mut rng = rand::thread_rng();
    let cover = (0..n)
        .map(|_| {
            let coords: Vec<f64> = (0..dimensions)
                .map(|dim| rng.gen_range(min_values[dim]..max_values[dim]))
                .collect();
            let norm = linalg::ddot(&coords, &coords);
            Point {
                coords,
                norm,
                neighbors: Neighbours::new(0, 0),
                skip: false,
            }
        })
        .collect();
    Ok(cover)
}

pub fn dataset_dimensions_and_extremes(
    dataset: &[Vec<f64>],
) -> Result<(usize, Vec<f64>, Vec<f64>)> {
    let mut min_values: Vec<f64>;
    let mut max_values: Vec<f64>;
    let dimensions = match dataset.first() {
        Some(first_point) => {
            max_values = first_point.clone();
            min_values = first_point.clone();
            first_point.len()
        }
        None => return Err(anyhow!("Dataset is empty")),
    };

    for point in dataset.iter().skip(1) {
        for (dim, &value) in point.iter().enumerate() {
            min_values[dim] = min_values[dim].min(value);
            max_values[dim] = max_values[dim].max(value);
        }
    }

    Ok((dimensions, min_values, max_values))
}

fn closest_neighbours(
    point: &Point,
    data: &Vec<Vec<f64>>,
    norms: &Vec<f64>,
    labels: &Vec<usize>,
    n_classes: usize,
) -> Neighbours {
    let mut distances: Vec<f64> = vec![f64::MAX; n_classes];
    let mut indexes: Vec<usize> = vec![0; n_classes];
    for ((index, datapoint), (datanorm, datalabel)) in
        data.iter().enumerate().zip(norms.iter().zip(labels.iter()))
    {
        let distance =
            linalg::euclidean_distance(&point.coords, &datapoint, &point.norm, &datanorm);
        if let Some(prev_distance) = distances.get_mut(*datalabel) {
            if *prev_distance > distance {
                *prev_distance = distance;
                indexes[*datalabel] = index;
            }
        };
    }
    let close_classes = argsort(&distances);
    Neighbours::new(indexes[close_classes[0]], indexes[close_classes[1]])
}

fn count_unique_elements(vec: Vec<usize>) -> usize {
    let unique_elements: IntSet<usize> = vec.into_iter().collect();
    unique_elements.len()
}

fn core_loop(
    data: &Vec<Vec<f64>>,
    y: &Vec<usize>,
    n_classes: usize,
    norms: &Vec<f64>,
    cover: &mut Vec<Point>,
) {
    // NEEDS BETTER STOPPING CONDITION
    for _ in 0..10 {
        // go through each point,norm and label
        cover
            .iter_mut()
            .filter(|point| !point.skip)
            .for_each(|point| {
                // get closest neighbours of different classes
                let neighbors = closest_neighbours(point, &data, &norms, &y, n_classes);
                if point.neighbors == neighbors {
                    point.skip = true;
                } else {
                    point.neighbors = neighbors.clone();
                    linalg::reject(
                        &data[neighbors.first],
                        &data[neighbors.second],
                        &mut point.coords,
                    )
                }
            });

        if cover.iter().all(|point| point.skip) {
            break;
        }
    }
}
fn core_loop_parallel(
    data: &Vec<Vec<f64>>,
    y: &Vec<usize>,
    n_classes: usize,
    norms: &Vec<f64>,
    cover: &mut Vec<Point>,
) {
    // NEEDS BETTER STOPPING CONDITION
    for _ in 0..10 {
        // go through each point,norm and label
        cover
            .par_iter_mut()
            .filter(|point| !point.skip)
            .for_each(|point| {
                // get closest neighbours of different classes
                let neighbors = closest_neighbours(point, &data, &norms, &y, n_classes);
                if point.neighbors == neighbors {
                    point.skip = true;
                } else {
                    point.neighbors = neighbors.clone();
                    linalg::reject(
                        &data[neighbors.first],
                        &data[neighbors.second],
                        &mut point.coords,
                    )
                }
            });

        if cover.iter().all(|point| point.skip) {
            break;
        }
    }
}

fn distill(cover: &mut Vec<Point>) {
    let mut set: HashSet<Neighbours> = HashSet::new();
    cover.retain(|point| {
        if set.contains(&point.neighbors) {
            return false;
        }
        set.insert(point.neighbors.clone());
        true
    });
}

#[pyfunction]
#[pyo3(signature = (data,y, n_points=1000, parallel=false,sparse=false))]
fn dbs(
    data: Vec<Vec<f64>>,
    y: Vec<usize>,
    n_points: usize,
    parallel: bool,
    sparse: bool,
) -> PyResult<Vec<Vec<f64>>> {
    //get number of classes
    //TODO: make sure it starts at zero and is a range
    let n_classes = count_unique_elements(y.clone());
    // Get the norms
    let norms: Vec<f64> = data
        .iter()
        .map(|point| linalg::ddot(point, point))
        .collect();
    // Get  the max and min values
    let (dimensions, min_values, max_values) = dataset_dimensions_and_extremes(&data).unwrap();
    // Generate cover
    let mut cover = generate_uniform_points(n_points, dimensions, min_values, max_values).unwrap();
    if parallel {
        core_loop_parallel(&data, &y, n_classes, &norms, &mut cover);
    } else {
        core_loop(&data, &y, n_classes, &norms, &mut cover);
    }
    if sparse {
        distill(&mut cover);
    }
    Ok(cover.into_iter().map(|point| point.coords).collect())
}

/// A Python module implemented in Rust.
#[pymodule]
fn dbsampler(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dbs, m)?)?;
    Ok(())
}

fn generate_random_points(k: usize, d: usize, a_matrix: &[f32]) -> Vec<f32> {
    let mut rng = rand::thread_rng();

    // Find per-dimension min/max
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

    // Generate k points uniformly in the bounding box
    let mut x = vec![0.0f32; k * d];
    for point in x.chunks_exact_mut(d) {
        for (dim, val) in point.iter_mut().enumerate() {
            *val = rng.gen_range(mins[dim]..=maxs[dim]);
        }
    }
    x
}
pub fn dbs_core(k: usize, n: usize, d: usize, a_matrix: &[f32], labels: &[usize]) {
    // 1. Precompute ||a_j||^2
    let a_norms = blas_ops::row_norms_sq(n, d, a_matrix);
    let mut x_matrix = generate_random_points(k, d, &a_matrix);
    let mut s_matrix = vec![0.0; k * n];
    let inertia = full_iteration(
        k,
        n,
        d,
        &mut x_matrix,
        &a_matrix,
        &a_norms,
        labels,
        &mut s_matrix,
    );
}

/// Computes the projection of each point in X onto the bisecting hyperplane
/// of its two nearest neighbors of different classes in A.
///
/// Returns the Mean Squared Displacement across all k query points.
pub fn full_iteration(
    k: usize,
    n: usize,
    d: usize,
    x_matrix: &mut [f32],
    a_matrix: &[f32],
    a_norms: &[f32],
    labels: &[usize],
    s_matrix: &mut [f32],
) -> f32 {
    let mut total_sq_movement = 0.0f32;

    // Precompute half-norms once (avoids k*n redundant multiplies by 0.5)
    let half_norms = blas_ops::half_row_norms_sq(a_norms);

    // S = X * A^T  (one batched GEMM)
    blas_ops::compute_xat_dot_products(k, n, d, x_matrix, a_matrix, s_matrix);

    // Process each query point
    for (x_row, s_row) in x_matrix.chunks_exact_mut(d).zip(s_matrix.chunks_exact(n)) {
        let mut best_score_1 = f32::NEG_INFINITY;
        let mut best_score_2 = f32::NEG_INFINITY;
        let mut idx_1 = 0usize;
        let mut idx_2 = 0usize;
        let mut class_1 = usize::MAX;

        // --- STEP A: Find two nearest neighbors of different classes ---
        for j in 0..n {
            let score = s_row[j] - half_norms[j];
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
            continue;
        }

        // --- STEP B: dot(a1, a2) via BLAS ---
        let dot_12 = blas_ops::dot_rows(a_matrix, d, idx_1, idx_2);

        // --- STEP C: Scalar projection factor ---
        let dot_x1 = s_row[idx_1];
        let dot_x2 = s_row[idx_2];
        let norm_1 = a_norms[idx_1];
        let norm_2 = a_norms[idx_2];

        let num = dot_x2 - dot_x1 - 0.5 * (norm_2 - norm_1);
        let den = norm_1 + norm_2 - 2.0 * dot_12;
        let s = num / (den + 1e-12);

        total_sq_movement += s * s * den;

        // --- STEP D: x -= s*(a2 - a1) via two SAXPY calls ---
        blas_ops::axpy_row(x_row, s, a_matrix, d, idx_1); // x += s * a1
        blas_ops::axpy_row(x_row, -s, a_matrix, d, idx_2); // x -= s * a2
    }

    total_sq_movement / (k as f32)
}

/// Tiled iteration that processes A in column-chunks to keep S_partial in cache.
///
/// Instead of materializing the full (k × n) dot-product matrix S, we compute
/// S_partial = X * A_chunk^T for each chunk, immediately scan it to update
/// running nearest-neighbor candidates, then discard it. This avoids a
/// multi-GB DRAM round-trip for large n.
///
/// Returns the Mean Squared Displacement across all k query points.
pub fn chunked_iteration(
    k: usize,
    n: usize,
    d: usize,
    x_matrix: &mut [f32],
    a_matrix: &[f32],
    a_norms: &[f32],
    labels: &[usize],
    chunk_size: usize,
) -> f32 {
    let half_norms = blas_ops::half_row_norms_sq(a_norms);

    // Per-query running state for the two-nearest-of-different-classes search
    let mut best_score_1 = vec![f32::NEG_INFINITY; k];
    let mut best_score_2 = vec![f32::NEG_INFINITY; k];
    let mut idx_1 = vec![0usize; k];
    let mut idx_2 = vec![0usize; k];
    let mut class_1 = vec![usize::MAX; k];

    // Reusable buffer — this is the whole point: k * chunk_size fits in cache
    let mut s_partial = vec![0.0f32; k * chunk_size];

    // ========== PHASE 1: Tiled search ==========
    let mut j_start = 0;
    while j_start < n {
        let j_end = (j_start + chunk_size).min(n);
        let cur_chunk = j_end - j_start;

        // S_partial = X * A[j_start..j_end]^T
        // A rows are contiguous in memory so slicing works directly
        let a_chunk = &a_matrix[j_start * d..j_end * d];
        blas_ops::compute_xat_dot_products(
            k,
            cur_chunk,
            d,
            x_matrix,
            a_chunk,
            &mut s_partial[..k * cur_chunk],
        );

        // Scan partial scores and update running best-2 per query
        for i in 0..k {
            let s_row = &s_partial[i * cur_chunk..(i + 1) * cur_chunk];
            for local_j in 0..cur_chunk {
                let global_j = j_start + local_j;
                let score = s_row[local_j] - half_norms[global_j];

                if score > best_score_1[i] {
                    if labels[global_j] != class_1[i] {
                        // New best is a different class — demote old best to second
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

    // ========== PHASE 2: Projection ==========
    let mut total_sq_movement = 0.0f32;

    for (i, x_row) in x_matrix.chunks_exact_mut(d).enumerate() {
        if best_score_2[i] == f32::NEG_INFINITY {
            continue;
        }

        // Recover raw dot products from stored scores:
        //   score_j = (x · a_j) - half_norm_j
        //   => x · a_j = score_j + half_norm_j
        let dot_x1 = best_score_1[i] + half_norms[idx_1[i]];
        let dot_x2 = best_score_2[i] + half_norms[idx_2[i]];

        let dot_12 = blas_ops::dot_rows(a_matrix, d, idx_1[i], idx_2[i]);

        let norm_1 = a_norms[idx_1[i]];
        let norm_2 = a_norms[idx_2[i]];

        let num = dot_x2 - dot_x1 - 0.5 * (norm_2 - norm_1);
        let den = norm_1 + norm_2 - 2.0 * dot_12;
        let s = num / (den + 1e-12);

        total_sq_movement += s * s * den;

        // x -= s * (a2 - a1) via two SAXPY calls
        blas_ops::axpy_row(x_row, s, a_matrix, d, idx_1[i]); // x += s * a1
        blas_ops::axpy_row(x_row, -s, a_matrix, d, idx_2[i]); // x -= s * a2
    }

    total_sq_movement / (k as f32)
}
