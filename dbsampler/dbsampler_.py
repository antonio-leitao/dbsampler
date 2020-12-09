import numpy as np
from scipy.linalg import null_space
from sklearn.metrics import pairwise_distances

def DBS(X,y,n_points=1000, n_epochs=5):
    y=y.astype(int)
    n = X.shape[-1]
    cover = make_slate(X,n_points = n_points)

    for epoch in range(n_epochs):

        dist = pairwise_distances(X=cover, Y=X)
        first = np.argmin(dist,axis=1)
        first_label= y[first]
        mask=[np.argwhere(y==label) for label in np.unique(y)]
        for i in range(n_points):
            dist[i,mask[first_label[i]]]=np.nan
        second = np.nanargmin(dist,axis=1)

        vectors = (X[first]-X[second])/np.linalg.norm(X[first]-X[second])
        centers = np.mean([X[first],X[second]],axis=0)

        new_cover = np.array([reproject(point,center,vector,n=n) for point,center,vector in zip(cover, centers, vectors)])
        #new_cover = np.array([project(point,center,vector,n=n) for point,center,vector in zip(cover, centers, vectors)])
        cover = new_cover  
    return cover
    
def project(point,center,vector,n):
    A = np.zeros((n,n))
    A[0,:]= vector
    ns = null_space(A)
    v = np.sum([np.dot(ns[:,i].T,point-center)*ns[:,i] for i in range(ns.shape[1])],axis=0)
    return center+v.flatten()

def reproject(point,center,vector,n):
    A = np.zeros((n,n))
    A[0,:]= vector
    Q, _ = np.linalg.qr(A.T)
    ns = Q[:,n-1:]
    v = np.sum([np.dot(ns[:,i].T,point-center)*ns[:,i] for i in range(ns.shape[1])],axis=0)
    return center+v.flatten()
    
def make_slate(X,n_points=6000):
    slate = np.random.uniform(low=np.min(X,axis=0), high = np.max(X,axis=0),size=(n_points,X.shape[1]))
    return slate  
