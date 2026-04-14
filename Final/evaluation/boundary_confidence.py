import numpy as np
from scipy.ndimage import label, find_objects
from scipy.signal import convolve2d
from typing import List

def get_shrub_list(grid: np.ndarray) -> List:
    island_kernel = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ])

    labelled_grid, _ = label(grid, structure=island_kernel)
    slices = find_objects(labelled_grid)

    shrub_list = list()
    for group_id, bounding_box in enumerate(slices, start=1):
        individual_shrub = grid[bounding_box]
        group_mask = labelled_grid[bounding_box] == group_id
        individual_shrub = np.where(group_mask, individual_shrub, 0)
        shrub_list.append(individual_shrub)

    return shrub_list

def get_shrub_conf_universal(shrub: np.ndarray, mean: int|np.ndarray=None, var: int|np.ndarray=None, eps: float=np.e) -> np.ndarray:
    """
    Applies a 1-padded convolution to estimate confidence of shrub existence around an individual shrub

    Args:
        shrub - M x N array containing a single shrub (as defined by the island kernel)
        mean - precalculated mean of appropriate dimension
        var - precalculated variance of appropriate dimension
        eps - increase epsilon to reducing diagonal edge weighting

    Returns:
        weighted_shrub - (M+1) x (N+1) array containing confidence estimates in (0,1]
    """
    transform_diag = -np.sqrt(1/(2+eps**2))
    transform_adj = 1/(1+eps)
    transform_kernel = (1/np.sum(np.abs([1, 4*transform_adj, 4*transform_diag]))) * np.array([
        [transform_diag, transform_adj, transform_diag],
        [transform_adj,1,transform_adj],
        [transform_diag, transform_adj, transform_diag],
    ])

    weighted_shrub = convolve2d(np.pad(shrub, 1), transform_kernel, mode="same") 
    if mean is None:
        mean = weighted_shrub.mean()

    if var is None:
        var = weighted_shrub.var()

    weighted_shrub = 1 - np.e**(-(weighted_shrub-mean)**2/(2*var)) # standard normal weighting. denser shrub regions will produce a greater contrast

    return weighted_shrub

def get_weighted_shrubs(grid: np.ndarray) -> np.ndarray:
    """
    Aggregator function for finding and weighting shrubs from arbitrary tiling

    Args:
        grid - M x N binary array with shrubs as 1 and non-shrub as 0
    
    Returns:
        weighted_shrubs - (M+1) x (N+1) array with confidence values in (0,1]
    """
    shrub_list = get_shrub_list(grid)
    weighted_shrubs = list()
    for shrub in shrub_list:
        weighted_shrub = get_shrub_conf_universal(shrub)
        weighted_shrubs.append(weighted_shrub)

    return weighted_shrubs