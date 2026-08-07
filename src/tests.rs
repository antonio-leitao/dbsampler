use super::*;
use numpy::ndarray::{array, s};

fn assert_close(left: f32, right: f32, tolerance: f32) {
    assert!(
        (left - right).abs() <= tolerance,
        "{left} is not within {tolerance} of {right}"
    );
}

#[test]
fn projection_is_stable_far_from_origin() {
    let data = [1_000_000.0f32, 1_000_001.0];
    let mut point = [1_000_000.0f32 + 0.125];

    let movement = project_onto_bisector(&mut point, &data, 1, 0, 1);

    assert!(movement.is_finite());
    assert_eq!(point[0], 1_000_000.5);
}

#[test]
fn two_sites_converge_to_their_midpoint() {
    let result = dbs_core(
        32,
        2,
        1,
        &[-0.5, 0.5],
        &[0, 1],
        10,
        0.0,
        false,
        false,
        Some(7),
    );

    assert!(result.converged);
    assert!(result.x_matrix.iter().all(|&value| value == 0.0));
}

#[test]
fn normalization_preserves_translation_and_scale() {
    let labels = [-3, 8];
    let original = array![[10_000.0f64], [10_001.0]];
    let transformed = array![[80_003.0f64], [80_010.0]];

    let first = validation::prepare(original.view(), &labels, 10, 10, 1e-6).unwrap();
    let second = validation::prepare(transformed.view(), &labels, 10, 10, 49e-6).unwrap();

    assert_eq!(first.data, second.data);
    assert_close(first.tolerance, second.tolerance, f32::EPSILON);

    let result = dbs_core(
        8,
        first.n,
        first.d,
        &first.data,
        &first.labels,
        10,
        first.tolerance,
        false,
        false,
        Some(2),
    );
    for value in result.x_matrix {
        let restored = value as f64 * first.scale + first.center[0];
        assert!((restored - 10_000.5).abs() < 1e-6);
    }
}

#[test]
fn outputs_lie_on_a_cross_class_boundary() {
    let data = [-1.0f32, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
    let labels = [0usize, 0, 1, 1];
    let result = dbs_core(256, 4, 2, &data, &labels, 10, 1e-10, false, true, Some(42));

    assert!(result.converged);
    for point in result.x_matrix.chunks_exact(2) {
        let left = point[0].mul_add(point[0] + 2.0, 1.0);
        let right = point[0].mul_add(point[0] - 2.0, 1.0);
        assert_close(left, right, 1e-5);
    }
}

#[test]
fn serial_and_parallel_paths_match() {
    let data = [-1.0f32, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
    let labels = [0usize, 0, 1, 1];

    let serial = dbs_core(512, 4, 2, &data, &labels, 10, 1e-8, false, false, Some(123));
    let parallel = dbs_core(512, 4, 2, &data, &labels, 10, 1e-8, false, true, Some(123));

    assert_eq!(serial.x_matrix, parallel.x_matrix);
    assert_eq!(serial.pairs, parallel.pairs);
}

#[test]
fn full_and_chunked_iterations_match() {
    let data = [-1.0f32, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
    let labels = [0usize, 0, 1, 1];
    let norms = blas_ops::row_norms_sq(4, 2, &data);
    let half_norms = blas_ops::half_row_norms_sq(&norms);
    let previous = vec![[usize::MAX, usize::MAX]; 3];
    let finished = vec![false; 3];
    let mut full_x = vec![-0.75, -0.25, 0.2, 0.8, 0.9, -0.6];
    let mut chunked_x = full_x.clone();
    let mut scores = vec![0.0; 12];

    let full = full_iteration(
        3,
        4,
        2,
        &mut full_x,
        &data,
        &half_norms,
        &labels,
        &mut scores,
        false,
        &previous,
        &finished,
        0.0,
    );
    let chunked = chunked_iteration(
        3,
        4,
        2,
        &mut chunked_x,
        &data,
        &half_norms,
        &labels,
        2,
        false,
        &previous,
        &finished,
        0.0,
    );

    assert_eq!(full.pairs, chunked.pairs);
    assert_eq!(full.finished, chunked.finished);
    for (&left, &right) in full_x.iter().zip(chunked_x.iter()) {
        assert_close(left, right, 1e-6);
    }
}

#[test]
fn sparse_deduplication_keeps_only_finished_unique_pairs() {
    let points = [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let pairs = [[0, 1], [0, 1], [1, 2], [0, 1]];
    let finished = [true, true, false, true];

    let (unique_points, unique_pairs) = dedup_points(&points, 2, &pairs, &finished);

    assert_eq!(unique_pairs, vec![[0, 1]]);
    assert_eq!(unique_points, vec![0.0, 0.0]);
}

#[test]
fn validation_accepts_strided_arrays_and_rejects_invalid_inputs() {
    let data = array![[0.0f64, 1.0], [2.0, 3.0], [4.0, 5.0]];
    let reversed = data.slice(s![..;-1, ..]);
    let prepared = validation::prepare(reversed, &[0, 1, 1], 8, 10, 1e-6).unwrap();
    assert_eq!(prepared.data.len(), 6);

    assert!(validation::prepare(data.view(), &[0, 0, 0], 8, 10, 1e-6).is_err());

    let non_finite = array![[0.0f64, f64::NAN], [1.0, 1.0]];
    assert!(validation::prepare(non_finite.view(), &[0, 1], 8, 10, 1e-6).is_err());

    let duplicates = array![[0.0f64, 0.0], [0.0, 0.0], [1.0, 1.0]];
    assert!(validation::prepare(duplicates.view(), &[0, 1, 1], 8, 10, 1e-6).is_err());
}
