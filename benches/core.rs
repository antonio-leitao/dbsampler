//! Core end-to-end benchmarks plus a direct comparison with the old tile shape.
//! Run with `cargo bench --bench core`.

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use dbsampler::{chunked_iteration, dbs_batched_core, dbs_core, BatchOptions};
use std::time::Duration;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    k: usize,
    n: usize,
    d: usize,
}

const CASES: [Case; 5] = [
    Case {
        name: "below_tile_threshold",
        k: 1_000,
        n: 8_000,
        d: 32,
    },
    Case {
        name: "above_tile_threshold",
        k: 1_000,
        n: 9_000,
        d: 32,
    },
    Case {
        name: "many_data_points",
        k: 512,
        n: 20_000,
        d: 32,
    },
    Case {
        name: "many_query_points",
        k: 20_000,
        n: 512,
        d: 32,
    },
    Case {
        name: "high_dimension",
        k: 512,
        n: 2_000,
        d: 256,
    },
];

const TILED_CASES: [Case; 3] = [CASES[1], CASES[2], CASES[3]];
const SCORE_TILE_ELEMENTS: usize = 32 * 1024 * 1024 / std::mem::size_of::<f32>();

fn make_input(n: usize, d: usize) -> (Vec<f32>, Vec<usize>) {
    let mut state = 0x4d595df4d0f33173u64;
    let mut data = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        data.push((state as u32 as f32 / u32::MAX as f32) - 0.5);
    }
    let labels = (0..n).map(|index| index % 4).collect();
    (data, labels)
}

fn core_benchmarks(criterion: &mut Criterion) {
    for parallel in [false, true] {
        let group_name = if parallel {
            "dbs_core/parallel"
        } else {
            "dbs_core/sequential"
        };
        let mut group = criterion.benchmark_group(group_name);
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(2));

        for case in CASES {
            let (data, labels) = make_input(case.n, case.d);
            let probe = dbs_core(
                case.k,
                case.n,
                case.d,
                &data,
                &labels,
                4,
                0.0,
                false,
                parallel,
                Some(42),
            );
            eprintln!(
                "{}: iterations={}, converged={}, final_inertia={:.3e}",
                case.name, probe.iterations, probe.converged, probe.final_inertia
            );
            group.throughput(Throughput::Elements(
                (case.k * case.n * probe.iterations) as u64,
            ));
            group.bench_with_input(
                BenchmarkId::new(case.name, format!("k{}-n{}-d{}", case.k, case.n, case.d)),
                &case,
                |bencher, case| {
                    bencher.iter(|| {
                        black_box(dbs_core(
                            case.k,
                            case.n,
                            case.d,
                            black_box(&data),
                            black_box(&labels),
                            4,
                            0.0,
                            false,
                            parallel,
                            Some(42),
                        ))
                    });
                },
            );
        }
        group.finish();
    }
}

fn half_norms(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    data.chunks_exact(d)
        .take(n)
        .map(|row| 0.5 * row.iter().map(|value| value * value).sum::<f32>())
        .collect()
}

fn tiled_iteration_benchmarks(criterion: &mut Criterion) {
    for parallel in [false, true] {
        let group_name = if parallel {
            "tiled_iteration/parallel"
        } else {
            "tiled_iteration/sequential"
        };
        let mut group = criterion.benchmark_group(group_name);
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(300));
        group.measurement_time(Duration::from_secs(1));

        for case in TILED_CASES {
            let (data, labels) = make_input(case.n, case.d);
            let (initial_points, _) = make_input(case.k, case.d);
            let norms = half_norms(&data, case.n, case.d);
            let previous_pairs = vec![[usize::MAX, usize::MAX]; case.k];
            let finished = vec![false; case.k];
            let old_score_capacity = if case.n >= case.k {
                case.k * case.n.min(1024)
            } else {
                case.k * case.n
            };

            group.throughput(Throughput::Elements((case.k * case.n) as u64));
            for (tile_name, score_capacity) in [
                ("old_shape", old_score_capacity),
                ("safe_2d", SCORE_TILE_ELEMENTS),
            ] {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}/{}", case.name, tile_name),
                        format!("k{}-n{}-d{}", case.k, case.n, case.d),
                    ),
                    &case,
                    |bencher, case| {
                        bencher.iter_batched(
                            || initial_points.clone(),
                            |mut points| {
                                black_box(chunked_iteration(
                                    case.k,
                                    case.n,
                                    case.d,
                                    &mut points,
                                    &data,
                                    &norms,
                                    &labels,
                                    score_capacity,
                                    parallel,
                                    &previous_pairs,
                                    &finished,
                                    0.0,
                                ))
                            },
                            BatchSize::LargeInput,
                        );
                    },
                );
            }
        }
        group.finish();
    }
}

fn clustered_boundaries(repeats: usize, sites: usize, d: usize) -> (Vec<f32>, Vec<usize>) {
    let mut data = Vec::with_capacity(repeats * sites * d);
    let mut labels = Vec::with_capacity(repeats * sites);
    for site in 0..sites {
        let position = if site == 0 {
            -0.5
        } else {
            0.2 + 0.3 * site as f32 / (sites - 1) as f32
        };
        for _ in 0..repeats {
            data.push(position);
            data.resize(data.len() + d - 1, 0.0);
            labels.push(site % 2);
        }
    }
    (data, labels)
}

fn batching_benchmarks(criterion: &mut Criterion) {
    let d = 8;
    let (data, labels) = clustered_boundaries(128, 64, d);
    let n = labels.len();
    let mut group = criterion.benchmark_group("batching/clustered_boundaries");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for adaptive in [false, true] {
        let options = BatchOptions {
            target_points: 40,
            batch_size: 16,
            max_batches: 20,
            max_iterations: 10,
            tolerance: 0.0,
            parallel: true,
            seed: Some(42),
            adaptive,
        };
        let probe = dbs_batched_core(n, d, &data, &labels, &options).unwrap();
        let mut total_queries = 0usize;
        let mut total_score_evaluations = 0u64;
        for seed in 0..16 {
            let mut seeded_options = options.clone();
            seeded_options.seed = Some(seed);
            let seeded = dbs_batched_core(n, d, &data, &labels, &seeded_options).unwrap();
            total_queries += seeded.queries;
            total_score_evaluations += seeded.score_evaluations;
        }
        eprintln!(
            "{}: unique={}, batches={}, queries={}, score_evaluations={}, mean_queries_16_seeds={:.1}, mean_scores_16_seeds={:.0}",
            if adaptive { "adaptive" } else { "uniform" },
            probe.m,
            probe.batches,
            probe.queries,
            probe.score_evaluations,
            total_queries as f64 / 16.0,
            total_score_evaluations as f64 / 16.0,
        );
        let name = if adaptive { "adaptive" } else { "uniform" };
        group.bench_function(name, |bencher| {
            bencher.iter(|| {
                black_box(
                    dbs_batched_core(n, d, black_box(&data), black_box(&labels), &options).unwrap(),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    core_benchmarks,
    tiled_iteration_benchmarks,
    batching_benchmarks
);
criterion_main!(benches);
