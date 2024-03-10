// For each pair of points x,y
// get centerpoint c and hyperplane of u = x-y
// find nearest neighbours of c (n1,n2)
// Add nearest neighbours to edge list. remove pair from evaluation list
//
// if pair is not x,y then
// for each:
// find closesst x or y
// get v = n-x
// project into I - vvT -uuT
// check if new point satisfies conditions
// continue
//
use crate::linalg;
use crate::{dataset_dimensions_and_extremes, Neighbours};
use hashbrown::HashSet;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;

fn sample_point(dimensions: usize, min_values: &[f64], max_values: &[f64]) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let coords: Vec<f64> = (0..dimensions)
        .map(|dim| rng.gen_range(min_values[dim]..max_values[dim]))
        .collect();
    coords
}

fn nearest_neighbours(point: &[f64], points: &[Vec<f64>]) -> Neighbours {
    assert!(!points.is_empty(), "Vector of points must not be empty.");

    let mut distances = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let distance = linalg::euclidean(point, p);
        distances.push((i, distance));
    }

    distances.sort_by(|(_, dist1), (_, dist2)| dist1.partial_cmp(dist2).unwrap_or(Ordering::Equal));

    // Get the indices of the two nearest neighbors
    let (u, _) = distances[0];
    let (v, _) = distances[1];

    Neighbours::new(u, v)
}

pub fn core_loop(data: Vec<Vec<f64>>, n_points: usize) -> Vec<(usize, usize)> {
    // Get  the max and min values
    let (dimensions, min_values, max_values) = dataset_dimensions_and_extremes(&data).unwrap();
    let edges: HashSet<Neighbours> = (0..n_points)
        .map(|_| {
            //generate a sample point
            let point = sample_point(dimensions, &min_values, &max_values);
            //find nearest neighbour
            nearest_neighbours(&point, &data)
        })
        .collect();
    edges
        .into_iter()
        .map(|neigh| (neigh.first, neigh.second))
        .collect()
}

pub fn core_loop_parallel(data: Vec<Vec<f64>>, n_points: usize) -> Vec<(usize, usize)> {
    // Get  the max and min values
    let (dimensions, min_values, max_values) = dataset_dimensions_and_extremes(&data).unwrap();
    let edges: HashSet<Neighbours> = (0..n_points)
        .into_par_iter()
        .map(|_| {
            //generate a sample point
            let point = sample_point(dimensions, &min_values, &max_values);
            //find nearest neighbour
            nearest_neighbours(&point, &data)
        })
        .collect();

    edges
        .into_iter()
        .map(|neigh| (neigh.first, neigh.second))
        .collect()
}
