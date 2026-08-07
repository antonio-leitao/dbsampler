use numpy::ndarray::ArrayView2;
use std::collections::HashMap;

pub struct PreparedData {
    pub data: Vec<f32>,
    pub labels: Vec<usize>,
    pub center: Vec<f64>,
    pub scale: f64,
    pub tolerance: f32,
    pub n: usize,
    pub d: usize,
}

pub fn prepare<T>(
    array: ArrayView2<'_, T>,
    labels: &[i64],
    n_points: usize,
    max_iter: usize,
    tolerance: f64,
) -> Result<PreparedData, String>
where
    T: Copy + Into<f64>,
{
    let n = array.nrows();
    let d = array.ncols();

    if n == 0 {
        return Err("data must contain at least one row".to_string());
    }
    if d == 0 {
        return Err("data must contain at least one column".to_string());
    }
    if labels.len() != n {
        return Err(format!(
            "y has length {} but data has {} rows",
            labels.len(),
            n
        ));
    }
    if n_points == 0 {
        return Err("n_points must be greater than zero".to_string());
    }
    if max_iter == 0 {
        return Err("max_iter must be greater than zero".to_string());
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err("tol must be finite and non-negative".to_string());
    }
    if n > i32::MAX as usize || d > i32::MAX as usize || n_points > i32::MAX as usize {
        return Err("data dimensions exceed the supported BLAS integer range".to_string());
    }
    n.checked_mul(d)
        .and_then(|_| n_points.checked_mul(d))
        .and_then(|_| n_points.checked_mul(n))
        .ok_or_else(|| "data dimensions are too large".to_string())?;

    let first_label = labels[0];
    if !labels.iter().skip(1).any(|&label| label != first_label) {
        return Err("y must contain at least two different classes".to_string());
    }

    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for row in array.rows() {
        for (dim, &value) in row.iter().enumerate() {
            let value = value.into();
            if !value.is_finite() {
                return Err("data must contain only finite values".to_string());
            }
            mins[dim] = mins[dim].min(value);
            maxs[dim] = maxs[dim].max(value);
        }
    }

    let center: Vec<f64> = mins
        .iter()
        .zip(maxs.iter())
        .map(|(&min, &max)| 0.5 * min + 0.5 * max)
        .collect();
    let scale = mins
        .iter()
        .zip(maxs.iter())
        .map(|(&min, &max)| max - min)
        .fold(0.0f64, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return Err("data must contain at least two distinct points".to_string());
    }

    // Keep the core input contiguous and close to unit scale. This preserves
    // Euclidean geometry while avoiding cancellation in float32 operations.
    let mut data = Vec::with_capacity(n * d);
    for row in array.rows() {
        for (dim, &value) in row.iter().enumerate() {
            data.push(((value.into() - center[dim]) / scale) as f32);
        }
    }

    let mut label_map = HashMap::new();
    let mut compact_labels = Vec::with_capacity(n);
    for &label in labels {
        let next = label_map.len();
        compact_labels.push(*label_map.entry(label).or_insert(next));
    }

    validate_cross_class_duplicates(&data, d, &compact_labels)?;

    let scaled_tolerance = if tolerance == 0.0 {
        0.0
    } else {
        let ratio = tolerance.sqrt() / scale;
        ratio.mul_add(ratio, 0.0).min(f32::MAX as f64) as f32
    };

    Ok(PreparedData {
        data,
        labels: compact_labels,
        center,
        scale,
        tolerance: scaled_tolerance,
        n,
        d,
    })
}

fn validate_cross_class_duplicates(data: &[f32], d: usize, labels: &[usize]) -> Result<(), String> {
    let mut rows = HashMap::with_capacity(labels.len());

    for (index, row) in data.chunks_exact(d).enumerate() {
        let mut key = hash_row(row);
        loop {
            match rows.get(&key) {
                Some(&other) => {
                    let other_row = &data[other * d..(other + 1) * d];
                    if row == other_row {
                        if labels[index] != labels[other] {
                            return Err(
                                "points from different classes are identical after normalization"
                                    .to_string(),
                            );
                        }
                        break;
                    }
                    key = key.wrapping_add(0x9e3779b97f4a7c15);
                }
                None => {
                    rows.insert(key, index);
                    break;
                }
            }
        }
    }

    Ok(())
}

fn hash_row(row: &[f32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &value in row {
        let bits = if value == 0.0 { 0 } else { value.to_bits() };
        hash ^= bits as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
