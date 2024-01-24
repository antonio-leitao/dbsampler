use rblas::Dot;

pub fn euclidean_distance(x: &[f64], y: &[f64], norm_x: &f64, norm_y: &f64) -> f64 {
    norm_x + norm_y - 2.0 * Dot::dot(x, y)
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
    let bb = Dot::dot(&b, &b);
    let projection = Dot::dot(&b, point) / bb;

    point
        .iter_mut()
        .zip(b.iter().zip(center.iter()))
        .for_each(|(x, (y, z))| *x = *x - projection * y + z)
}
