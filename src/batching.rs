use crate::{blas_ops, converge_points, data_bounds, generate_random_points_from_bounds};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Beta, Distribution};
use std::collections::{HashMap, HashSet};

const PROJECTION_DIMS: usize = 4;
const COARSE_BINS: usize = 4;
const FINE_BINS: usize = 8;
const CANDIDATES_PER_POINT: usize = 4;
const ADAPTIVE_YIELD_THRESHOLD: f64 = 0.75;
const LOCAL_CANDIDATE_FRACTION: usize = 4;

#[derive(Clone)]
pub struct BatchOptions {
    pub target_points: usize,
    pub batch_size: usize,
    pub max_batches: usize,
    pub max_iterations: usize,
    pub tolerance: f32,
    pub parallel: bool,
    pub seed: Option<u64>,
    /// Used by benchmarks to compare adaptive and uniform batches.
    pub adaptive: bool,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            target_points: 1000,
            batch_size: 256,
            max_batches: 20,
            max_iterations: 100,
            tolerance: 1e-6,
            parallel: true,
            seed: None,
            adaptive: true,
        }
    }
}

pub struct BatchedResult {
    pub x_matrix: Vec<f32>,
    pub pairs: Vec<[usize; 2]>,
    pub m: usize,
    pub batches: usize,
    pub queries: usize,
    pub score_evaluations: u64,
}

#[derive(Clone, Copy)]
struct CellKeys {
    coarse: u64,
    fine: u64,
}

#[derive(Default)]
struct Posterior {
    successes: u32,
    failures: u32,
    cost_sum: u64,
    observations: u32,
}

impl Posterior {
    fn observe(&mut self, success: bool, cost: usize) {
        if success {
            self.successes = self.successes.saturating_add(1);
        } else {
            self.failures = self.failures.saturating_add(1);
        }
        self.cost_sum = self.cost_sum.saturating_add(cost as u64);
        self.observations = self.observations.saturating_add(1);
    }

    fn sample_utility<R: Rng + ?Sized>(&self, default_cost: f64, rng: &mut R) -> f64 {
        let novelty = Beta::new(1.0 + self.successes as f64, 1.0 + self.failures as f64)
            .expect("positive beta parameters")
            .sample(rng);
        let cost = if self.observations == 0 {
            default_cost
        } else {
            self.cost_sum as f64 / self.observations as f64
        };
        novelty / cost.max(1.0)
    }
}

struct NoveltyModel {
    d: usize,
    weights: Vec<f32>,
    projection_mins: [f32; PROJECTION_DIMS],
    projection_maxs: [f32; PROJECTION_DIMS],
    coarse: HashMap<u64, Posterior>,
    fine: HashMap<u64, Posterior>,
    successful_starts: Vec<f32>,
    total_cost: u64,
    observations: u64,
}

impl NoveltyModel {
    fn new<R: Rng + ?Sized>(d: usize, mins: &[f32], maxs: &[f32], rng: &mut R) -> Self {
        let mut weights = Vec::with_capacity(PROJECTION_DIMS * d);
        let mut projection_mins = [0.0f32; PROJECTION_DIMS];
        let mut projection_maxs = [0.0f32; PROJECTION_DIMS];

        for projection in 0..PROJECTION_DIMS {
            for dim in 0..d {
                let weight = if rng.gen::<bool>() { 1.0 } else { -1.0 };
                weights.push(weight);
                if weight > 0.0 {
                    projection_mins[projection] += mins[dim];
                    projection_maxs[projection] += maxs[dim];
                } else {
                    projection_mins[projection] -= maxs[dim];
                    projection_maxs[projection] -= mins[dim];
                }
            }
        }

        Self {
            d,
            weights,
            projection_mins,
            projection_maxs,
            coarse: HashMap::new(),
            fine: HashMap::new(),
            successful_starts: Vec::new(),
            total_cost: 0,
            observations: 0,
        }
    }

    fn keys(&self, point: &[f32]) -> CellKeys {
        let mut projected = [0.0f32; PROJECTION_DIMS];
        for (projection, value) in projected.iter_mut().enumerate() {
            let weights = &self.weights[projection * self.d..(projection + 1) * self.d];
            *value = point
                .iter()
                .zip(weights.iter())
                .map(|(&coordinate, &weight)| coordinate * weight)
                .sum();
            let width = self.projection_maxs[projection] - self.projection_mins[projection];
            *value = if width > 0.0 {
                ((*value - self.projection_mins[projection]) / width).clamp(0.0, 1.0)
            } else {
                0.5
            };
        }

        CellKeys {
            coarse: cell_key(&projected, COARSE_BINS),
            fine: cell_key(&projected, FINE_BINS),
        }
    }

    fn observe(&mut self, keys: CellKeys, point: &[f32], success: bool, cost: usize) {
        self.coarse
            .entry(keys.coarse)
            .or_default()
            .observe(success, cost);
        self.fine
            .entry(keys.fine)
            .or_default()
            .observe(success, cost);
        if success {
            self.successful_starts.extend_from_slice(point);
        }
        self.total_cost = self.total_cost.saturating_add(cost as u64);
        self.observations = self.observations.saturating_add(1);
    }

    fn select_batch<R: Rng + ?Sized>(
        &self,
        batch_size: usize,
        mins: &[f32],
        maxs: &[f32],
        rng: &mut R,
    ) -> Result<(Vec<f32>, Vec<CellKeys>), String> {
        let pool_size = batch_size
            .checked_mul(CANDIDATES_PER_POINT)
            .ok_or_else(|| "batch_size is too large".to_string())?;
        pool_size
            .checked_mul(self.d)
            .ok_or_else(|| "candidate pool is too large".to_string())?;
        let mut candidates = generate_random_points_from_bounds(pool_size, self.d, mins, maxs, rng);
        if !self.successful_starts.is_empty() {
            let successful_points = self.successful_starts.len() / self.d;
            let local_points = pool_size / LOCAL_CANDIDATE_FRACTION;
            let local_start = pool_size - local_points;
            let radius = 0.2 / (self.d as f32).sqrt();
            for point in candidates[local_start * self.d..].chunks_exact_mut(self.d) {
                let source = rng.gen_range(0..successful_points);
                let source = &self.successful_starts[source * self.d..(source + 1) * self.d];
                for dim in 0..self.d {
                    let width = maxs[dim] - mins[dim];
                    point[dim] = (source[dim] + rng.gen_range(-radius..=radius) * width)
                        .clamp(mins[dim], maxs[dim]);
                }
            }
        }
        let default_cost = if self.observations == 0 {
            1.0
        } else {
            self.total_cost as f64 / self.observations as f64
        };

        let mut coarse_scores = HashMap::new();
        let mut fine_scores = HashMap::new();
        let mut ranked = Vec::with_capacity(pool_size);
        for (index, point) in candidates.chunks_exact(self.d).enumerate() {
            let keys = self.keys(point);
            let coarse_score = *coarse_scores.entry(keys.coarse).or_insert_with(|| {
                self.coarse
                    .get(&keys.coarse)
                    .unwrap_or(&Posterior::default())
                    .sample_utility(default_cost, rng)
            });
            let fine_score = *fine_scores.entry(keys.fine).or_insert_with(|| {
                self.fine
                    .get(&keys.fine)
                    .unwrap_or(&Posterior::default())
                    .sample_utility(default_cost, rng)
            });
            ranked.push((
                0.35 * coarse_score + 0.65 * fine_score,
                rng.gen::<f64>(),
                index,
                keys,
            ));
        }
        ranked.sort_unstable_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| right.1.total_cmp(&left.1))
        });

        let mut selected = Vec::with_capacity(batch_size);
        let mut deferred = Vec::new();
        let mut selected_cells = HashSet::with_capacity(batch_size);
        for candidate in ranked {
            if selected_cells.insert(candidate.3.fine) {
                selected.push(candidate);
                if selected.len() == batch_size {
                    break;
                }
            } else {
                deferred.push(candidate);
            }
        }
        if selected.len() < batch_size {
            selected.extend(deferred.into_iter().take(batch_size - selected.len()));
        }

        let mut points = Vec::with_capacity(batch_size * self.d);
        let mut keys = Vec::with_capacity(batch_size);
        for (_, _, index, cell_keys) in selected {
            points.extend_from_slice(&candidates[index * self.d..(index + 1) * self.d]);
            keys.push(cell_keys);
        }
        Ok((points, keys))
    }
}

fn cell_key(projected: &[f32; PROJECTION_DIMS], bins: usize) -> u64 {
    projected.iter().fold(0u64, |key, &value| {
        let bin = ((value * bins as f32) as usize).min(bins - 1);
        key * bins as u64 + bin as u64
    })
}

pub fn dbs_batched_core(
    n: usize,
    d: usize,
    a_matrix: &[f32],
    labels: &[usize],
    options: &BatchOptions,
) -> Result<BatchedResult, String> {
    let data_len = n
        .checked_mul(d)
        .ok_or_else(|| "data dimensions are too large".to_string())?;
    if n == 0 || d == 0 || a_matrix.len() != data_len {
        return Err("invalid data dimensions".to_string());
    }
    if labels.len() != n {
        return Err("labels must match the number of data points".to_string());
    }
    if options.target_points == 0 {
        return Err("target_points must be greater than zero".to_string());
    }
    if options.batch_size == 0 {
        return Err("batch_size must be greater than zero".to_string());
    }
    if options.max_batches == 0 {
        return Err("max_batches must be greater than zero".to_string());
    }
    if options.max_iterations == 0 {
        return Err("max_iterations must be greater than zero".to_string());
    }
    if !options.tolerance.is_finite() || options.tolerance < 0.0 {
        return Err("tolerance must be finite and non-negative".to_string());
    }
    if n > i32::MAX as usize || d > i32::MAX as usize || options.batch_size > i32::MAX as usize {
        return Err("data dimensions exceed the supported BLAS integer range".to_string());
    }
    options
        .target_points
        .checked_mul(d)
        .and_then(|_| options.batch_size.checked_mul(d))
        .and_then(|_| options.batch_size.checked_mul(n))
        .and_then(|_| options.batch_size.checked_mul(CANDIDATES_PER_POINT))
        .and_then(|pool_size| pool_size.checked_mul(d))
        .ok_or_else(|| "batch dimensions are too large".to_string())?;

    let (mins, maxs) = data_bounds(d, a_matrix);
    let norms = blas_ops::row_norms_sq(n, d, a_matrix);
    let half_norms = blas_ops::half_row_norms_sq(&norms);
    let base_seed = options.seed.unwrap_or_else(|| rand::thread_rng().gen());
    let mut rng = StdRng::seed_from_u64(base_seed);
    let mut model_rng = StdRng::seed_from_u64(base_seed ^ 0x9e3779b97f4a7c15);
    let mut model = options
        .adaptive
        .then(|| NoveltyModel::new(d, &mins, &maxs, &mut model_rng));

    let mut seen = HashSet::new();
    let mut unique_points = Vec::with_capacity(options.target_points * d);
    let mut unique_pairs = Vec::with_capacity(options.target_points);
    let mut queries = 0usize;
    let mut score_evaluations = 0u64;
    let mut batches = 0usize;
    let mut last_yield = 1.0f64;

    while unique_pairs.len() < options.target_points && batches < options.max_batches {
        let batch_size = options
            .batch_size
            .min(options.target_points - unique_pairs.len());
        let (starting_points, keys) = if let Some(model) = &model {
            if batches == 0 || last_yield >= ADAPTIVE_YIELD_THRESHOLD {
                let points =
                    generate_random_points_from_bounds(batch_size, d, &mins, &maxs, &mut rng);
                let keys = points
                    .chunks_exact(d)
                    .map(|point| model.keys(point))
                    .collect();
                (points, keys)
            } else {
                model.select_batch(batch_size, &mins, &maxs, &mut rng)?
            }
        } else {
            (
                generate_random_points_from_bounds(batch_size, d, &mins, &maxs, &mut rng),
                vec![CellKeys { coarse: 0, fine: 0 }; batch_size],
            )
        };

        let model_starts = model.as_ref().map(|_| starting_points.clone());
        let result = converge_points(
            starting_points,
            n,
            d,
            a_matrix,
            &half_norms,
            labels,
            options.max_iterations,
            options.tolerance,
            options.parallel,
            true,
        );
        queries = queries.saturating_add(batch_size);
        score_evaluations = score_evaluations.saturating_add(
            (n as u64)
                .saturating_mul(batch_size as u64)
                .saturating_mul(result.iterations as u64),
        );
        batches += 1;
        let pairs_before = unique_pairs.len();

        for (index, &cell_keys) in keys.iter().enumerate() {
            let cost = result.point_iterations[index].max(1);
            let success = result.finished[index] && seen.insert(result.pairs[index]);
            if let Some(model) = &mut model {
                let starts = model_starts.as_ref().expect("adaptive starts");
                model.observe(
                    cell_keys,
                    &starts[index * d..(index + 1) * d],
                    success,
                    cost,
                );
            }
            if success && unique_pairs.len() < options.target_points {
                unique_points.extend_from_slice(&result.x_matrix[index * d..(index + 1) * d]);
                unique_pairs.push(result.pairs[index]);
            }
        }
        last_yield = (unique_pairs.len() - pairs_before) as f64 / batch_size as f64;
    }

    let m = unique_pairs.len();
    Ok(BatchedResult {
        x_matrix: unique_points,
        pairs: unique_pairs,
        m,
        batches,
        queries,
        score_evaluations,
    })
}
