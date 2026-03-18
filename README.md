<p align="center">
  <img src="images/logo.png" />
</p>

### Decision Boundary Sampler

_Sample the decision boundary of classification problems
Blazingly fast and theoretically sound
Built with Rust_

[![Pepy Total Downlods](https://img.shields.io/pepy/dt/dbsampler?style=for-the-badge&logo=python&labelColor=white&color=blue)](https://pepy.tech/project/dbsampler)

### Contents

- [Installation](#installation)
  - [Compiling from source](#compilation-from-source)
- [Usage](#usage)
  - [Sparse](#sparse)
- [How does it work](#how-does-it-work)
- [Performance](#performance)
- [Citing](#citing)

<p align="center">
  <img src="images/linear.png" width="49%" />
  <img src="images/concentric.png" width="49%" />
</p>

DBSampler is a package to sample points on the decision boundary of classification problems (binary or multiclass). It is theoretically exact and efficient for very high dimensions. The guarantees:

- Returns a sample of points uniformly distributed on the decision boundary.
- Number of points is user defined. More points for a denser sample, fewer for a faster run.
- The points are guaranteed to come from the edges of the condensed Voronoi Diagram (more below).

## Installation

Pre-built packages for MacOS, Windows and Linux systems are available on [PyPI](https://pypi.org/project/dbsampler/) and can be installed with:

```
pip install dbsampler
```

On uncommon architectures, you may need to first
[install Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) before running `pip install dbsampler`.

### Compilation from source

In order to compile from source you will need [Rust/Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) and [maturin](https://github.com/PyO3/maturin) for the Python bindings.
Maturin is best used within a Python virtual environment:

```bash
# activate your desired virtual environment first, then:
pip install maturin
git clone https://github.com/antonio-leitao/dbsampler.git
cd dbsampler
# build and install the package:
maturin develop --release
```

## Usage

```python
import dbsampler
import numpy as np

cover = dbsampler.dbs(
    data=X,
    y=y,
    n_points=1000,
    max_iter=100,
    tol=1e-6,
    sparse=True,
    parallel=True,
    seed=42,
)
```

**Parameters:**

- `data`: numpy array of shape `(n, d)` with the points of every class (`float64`, converted internally to `float32`).
- `y`: 1-dimensional array or list of integer class labels, length `n`.
- `n_points`: number of points to sample from the decision boundary. More points give a denser sample but increase runtime. Default `1000`.
- `max_iter`: maximum number of projection iterations. Default `100`.
- `tol`: convergence threshold on the mean squared displacement. The algorithm stops early when inertia drops below this value. Default `1e-6`.
- `sparse`: if `True` (default), removes points that converge to the same Voronoi edge, keeping only the first occurrence.
- `parallel`: if `True` (default), uses [rayon](https://github.com/rayon-rs/rayon) to parallelize the per-point nearest-neighbor search and projection steps across CPU cores. The BLAS matrix multiplications are multithreaded independently of this flag.
- `seed`: optional integer seed for reproducible results. When `None` (default), initialization is random. Set to a fixed value (e.g. `seed=42`) for deterministic runs.

**Returns:**

- `cover`: list of lists, each of length `d` — the sampled boundary points (as `float32` values). With `sparse=True` the number of returned points may be less than `n_points`.

### Sparse

Passing `sparse=True` removes cover points that fall on the same Voronoi edge, keeping only the first occurrence.
This can drastically reduce the number of points while maintaining a uniform and complete cover of the decision boundary.
Below is the example of `5000` points sampled (left) and the same points with `sparse=True`.

<p align="center">
  <img src="images/dense.png" width="49%" />
  <img src="images/sparse.png" width="49%" />
</p>

## How does it work?

For an in-depth explanation check our [paper](https://openreview.net/forum?id=I44kJPuvqPD). The algorithm aims to uniformly sample points from the edges of Voronoi cells belonging to points of different classes. The union of these edges forms the decision boundary that maximizes the distance between classes.

<p align="center">
  <img src="images/voronoi.png" width="60%" />
</p>

It starts by building an initial uniform sample of the space containing `n_points`. It then iteratively projects each point onto the bisecting hyperplane between its two nearest neighbors of different classes.

<p align="center">
  <img src="images/voronoiboudary.png" width="60%" />
</p>

**Sketch of proof of convergence.** At each iteration:

1. If both nearest neighbours have adjacent Voronoi cells then, after projection, the point lies on the decision boundary (by construction).
2. Otherwise there must exist a point from class A (or not A) that becomes the new nearest neighbour (by definition of Voronoi cells).

<p align="center">
  <img src="images/linear_0.png" width="24%" />
  <img src="images/linear_1.png" width="24%" />
  <img src="images/linear_2.png" width="24%" />
  <img src="images/linear.png" width="24%" />
</p>

## Performance

DBSampler is written in Rust with BLAS-accelerated linear algebra (via Accelerate on macOS, OpenBLAS on Linux/Windows). The core dot-product and matrix-multiply operations use `cblas_sgemm` and `cblas_sdot`, and the algorithm automatically switches to a tiled (chunked) iteration strategy when the score matrix would exceed 32 MB, keeping memory usage bounded for large datasets.

With `parallel=True`, the per-point nearest-neighbor search and bisector projection are distributed across CPU cores using rayon. The BLAS matrix multiplications are multithreaded independently. The algorithm also uses convergence-based stopping — it terminates early once the mean squared displacement drops below the tolerance, avoiding unnecessary iterations.

Pre-built binaries are available for Windows, macOS and most Linux distributions.

<p align="center">
  <img src="images/performance.png" width="80%" />
</p>

## Citing

If you use DBSampler in your work or parts of the algorithm please consider citing:

```
@inproceedings{petri2020on,
               title={On The Topological Expressive Power of Neural Networks},
               author={Giovanni Petri and Ant{\'o}nio Leit{\~a}o},
               booktitle={NeurIPS 2020 Workshop on Topological Data Analysis and Beyond},
               year={2020},
               url={https://openreview.net/forum?id=I44kJPuvqPD}
}
```

In the paper above you can find the pseudocode of the algorithm along with the proof of convergence.

## License

DBSampler is distributed under the [3-clause BSD license](LICENSE.md).
