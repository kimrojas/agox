import timeit
from typing import Tuple
import numpy
import jax.numpy as np
from jax import jit, vmap, grad
from numba import jit as njit


def timer(func, *args, **kwargs):
    n = 1000
    start_time = timeit.default_timer()
    for _ in range(n):
        func(*args, **kwargs)
    end_time = timeit.default_timer()
    return (end_time - start_time) / n


@jit
def sqeuclidean(x: np.array, y: np.array) -> float:
    return np.sum((x - y) ** 2)

def sqeuclidean2(x: np.array, y: np.array) -> float:
    return (x - y) ** 2

@jit
def cdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return vmap(vmap(sqeuclidean, (0, None), 0), (None, 0), 1)(x, y)

@jit
def cdist2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(vmap(sqeuclidean2, in_axes=(None, 0), out_axes=(1))(x, y), axis=2)

@jit
def dist(cell_replica: np.ndarray, p0: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = p.at[:].add(cell_replica)
    d = cdist(p0, p)
    p = p.at[:].add(-cell_replica)    
    return d
    
@jit
def dist_vectorized(cell_replicas: np.ndarray, p0: np.ndarray, p: np.ndarray) -> np.ndarray:
    return vmap(dist, (0, None, None), 2)(cell_replicas, p0, p)

@jit
def distance_matrix(positions: np.ndarray, cell: np.ndarray, repeats: np.ndarray) -> np.ndarray:
    p0 = np.array(positions)
    p = p0.copy()
    cells = repeats @ cell
    all_dists = dist_vectorized(cells, p0, p)
    return np.sqrt(np.min(all_dists, axis=2))

@jit
def all_distance_vectors(positions: np.ndarray, cell: np.ndarray, repeats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.array(positions)
    p0 = p.copy()
    cells = repeats @ cell
    all_dists = dist_vectorized(cells, p0, p)
    return np.sqrt(all_dists), cells
    


@jit
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr = arr.at[...,i].set(a)
    return arr.reshape(-1, la)

def get_repeats(cell: np.ndarray, rcut: float, pbc: Tuple) -> np.ndarray:
    max_repeat = 10.
    xs = np.min(np.array([pbc[0]*max_repeat, rcut//np.linalg.norm(cell[:,0])+1]))
    ys = np.min(np.array([pbc[1]*max_repeat, rcut//np.linalg.norm(cell[:,1])+1]))
    zs = np.min(np.array([pbc[2]*max_repeat, rcut//np.linalg.norm(cell[:,2])+1]))
    return cartesian_product(np.arange(-xs, xs+1), np.arange(-ys, ys+1), np.arange(-zs, zs+1))


########## FOR GRADIENTS ##########
@jit
def diff(ri: np.array, rj: np.array) -> np.ndarray:
    return (ri - rj)

@jit
def diff_vectorized(ri: np.array, r: np.array) -> np.ndarray:
    return vmap(diff, (None, 0), 0)(ri, r)


########## STATIC ##########
no = np.array([
    [0.,0.,0.],
])

x = np.array([
    [1., 0., 0.],
    [0., 0., 0.],
    [-1., 0., 0.],   
])

y = np.array([
    [0., 1., 0.],
    [0., 0., 0.],
    [0., -1., 0.],   
])

z = np.array([
    [0., 0., 1.],
    [0., 0., 0.],
    [0., 0., -1.],   
])

xy = np.array([
    [0., 0., 0.],
    [1., 0., 0.],
    [-1., 0., 0.],
    [0., 1., 0.],
    [0., -1., 0.],
    [1., 1., 0.],
    [1., -1., 0.],
    [-1., 1., 0.],
    [-1., -1., 0.],    
])


xz = np.array([
    [0., 0., 0.],
    [1., 0., 0.],
    [-1., 0., 0.],
    [0., 0., 1.],
    [0., 0., -1.],
    [1., 0., 1.],
    [1., 0., -1.],
    [-1., 0., 1.],
    [-1., 0., -1.],    
])



yz = np.array([
    [0., 0., 0.],
    [0., 1., 0.],
    [0., -1., 0.],
    [0., 0., 1.],
    [0., 0., -1.],
    [0., 1., 1.],
    [0., 1., -1.],
    [0., -1., 1.],
    [0., -1., -1.],    
])


xyz = np.array([
    [-1., -1., -1.],
    [-1., -1., 0.],
    [-1., -1., 1.],
    [-1., 0., -1.],
    [-1., 0., 0.],
    [-1., 0., 1.],
    [-1., 1., -1.],
    [-1., 1., 0.],
    [-1., 1., 1.],
    [0., -1., -1.],
    [0., -1., 0.],
    [0., -1., 1.],
    [0., 0., -1.],
    [0., 0., 0.],
    [0., 0., 1.],
    [0., 1., -1.],
    [0., 1., 0.],
    [0., 1., 1.],
    [1., -1., -1.],
    [1., -1., 0.],
    [1., -1., 1.],
    [1., 0., -1.],
    [1., 0., 0.],
    [1., 0., 1.],
    [1., 1., -1.],
    [1., 1., 0.],
    [1., 1., 1.],
])


periodicity = {
    'no':no,
    'x':x,
    'y':y,
    'z':z,
    'xy':xy,
    'xz':xz,
    'yz':yz,
    'xyz':xyz
}
