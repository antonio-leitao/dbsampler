# Decision Bounday Sampler (DBS)



[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger) [![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/Antonio-Leitao/dbsampler/blob/main/LICENSE) 

DBSampler is a package to sample points in the decision boundary of  classification problems (binary or multiclass). It is theorically exact and efficient for very high dimensions. The guarentees:

  - Returns a sample of points uniformly distributed in the decision boundary.
  - Number of points is user defined. More points for a denser sample, less for a faster run.
  - The points are guarenteed to come from the edges of the condensed Voronoi Diagram (more below).

## Installation
Dependencies:
  - Numpy
  - Scipy
  - Sklearn

DBSampler is available on PyPI,

```sh
pip install dbsampler
```

## Usage
```python
import dbsampler
cover = dbsampler.DBS(X=X,y=y,n_points=1000,n_epochs=5, distribution='uniform')
```
**Parameters:**
-  ``X``: numpy array of shape (samples,features) contatining the points of every class.
 -  ``y``: 1-dimensional numpy array with labels of each points. Array must be flattened.
 -  ``n_points``: This determines the number of points sampled from the decision boundary. More points equates for a denser sample but slows the algorithm. Default is 1000.
 -  ``n_epochs``: This determines the number of epochs to be used. It is an iterative algorithm but it is very fast to converge. Default is 5. Currently working on a proof for an upper bound on the number of necessary iterations. 
 -  ``distribution``: Initial point distribution, it is also the distribution of    the points in the decision boundary. Currently supports only _uniform_         (default) and _normal_
 -  ``metric``: metric used to compute the nearest neighbours. Currently only      supports euclidean
 
**Returns:**
 -  ``cover``: numpy array (n_points, n_features) of points in the decision boundary.

## How does it work?
For an in-depth explanation look at this [post](https://antonio-leitao.netlify.app/post/aprox_decision/) or at our [paper](https://openreview.net/forum?id=I44kJPuvqPD). The algorithm aims at sampling uniformly points from the edges of Voronoi Cells belonging to points of different classes. The union of these edges is the decision boundary that maximizes the distance between classes.
 
 (image)
 
 It starts by building an initial uniform sample of the space containing ``n_points``. It then iterativelly "pushes" each point to the hyperplane orthogonal to the one between its closest neighbors of different classes.
 
 (image)
 
Sketch of proof of convergence. At each iteration in ``n_epochs``:
 1. If both nearest neighbours have adjacent Voronoi Cells then, after projection the point is in the decision boundary (by construction).
 2. Else then there must exist a point form class A (or not A) that is the new nearest neighbour (by definition of Voronoi Cells).
 
## Performance
The bottleneck of the algorithm is the calculation of a orthogonal hyperplane for each point at each iteration. For low dimensions (<200) we use the ``null space`` of a matrix. For higher dimensions we approximate it using ``QR-Decomposition``.

## Citation
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
In the paper above you can find the pseudocode of the algorithm along with the proof of convergence. A complete paper about the method is coming soon.
