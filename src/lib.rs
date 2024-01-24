extern crate rblas;
use pyo3::prelude::*;
mod linalg;
use anyhow::{anyhow, Result};
use hashbrown::HashSet;
use nohash_hasher::IntSet;
use rand::Rng;
use rayon::prelude::*;
use rblas::Dot;

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
struct Neighbours {
    first: usize,
    second: usize,
}

impl Neighbours {
    fn new(a: usize, b: usize) -> Neighbours {
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
            let coords = (0..dimensions)
                .map(|dim| rng.gen_range(min_values[dim]..max_values[dim]))
                .collect();
            let norm = Dot::dot(&coords, &coords);
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

fn dataset_dimensions_and_extremes(dataset: &[Vec<f64>]) -> Result<(usize, Vec<f64>, Vec<f64>)> {
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
    let norms: Vec<f64> = data.iter().map(|point| Dot::dot(point, point)).collect();
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

/// A Python module implemented in Rust.
#[pymodule]
fn dbsampler(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dbs, m)?)?;
    Ok(())
}
