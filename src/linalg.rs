use std::cmp;
// Shift slices in place and add 8 elements at a time.
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let n = cmp::min(x.len(), y.len());
    let (mut x, mut y) = (&x[..n], &y[..n]);

    let mut sum = 0.0;
    while x.len() >= 8 {
        sum += x[0] * y[0]
            + x[1] * y[1]
            + x[2] * y[2]
            + x[3] * y[3]
            + x[4] * y[4]
            + x[5] * y[5]
            + x[6] * y[6]
            + x[7] * y[7];
        x = &x[8..];
        y = &y[8..];
    }

    // Take care of any left over elements (if len is not divisible by 8).
    x.iter()
        .zip(y.iter())
        .fold(sum, |sum, (&ex, &ey)| sum + (ex * ey))
}

pub fn euclidean_distance(x: &[f64], y: &[f64], norm_x: &f64, norm_y: &f64) -> f64 {
    norm_x + norm_y - 2.0 * ddot(x, y)
}

pub fn reject(u: &[f64], v: &[f64], point: &mut [f64]) {
    let center: Vec<f64> = u.iter().zip(v.iter()).map(|(a, b)| (a + b) / 2.0).collect();
    let b: Vec<f64> = u.iter().zip(v.iter()).map(|(a, b)| a - b).collect();

    //point - center vector
    point
        .iter_mut()
        .zip(center.iter())
        .for_each(|(a, b)| *a -= b);

    //project point-center to u-v
    let bb = ddot(&b, &b);
    let projection = ddot(&b, point) / bb;

    point
        .iter_mut()
        .zip(b.iter().zip(center.iter()))
        .for_each(|(x, (y, z))| *x = *x - projection * y + z)
}
