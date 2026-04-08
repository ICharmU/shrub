import numpy as np
from typing import List

def binarize_labels(mask: np.ndarray):
    """
    Converts multiclass labels in binary labels.
    0 represents a non-shrub location. All other values represent shrub locations.

    Args:
        mask - M x N array of multiclass labels

    Returns:
        binarized - M x N binary array of shrub/non-shrub labels

    Ex.
    labeled_img = np.array([
        [1,2,3],
        [3,1,0]
    ])
    binarize_labels(labeled_img)

    Output:
    array([[1, 1, 1],
           [1, 1, 0]])
    """
    binarized = (mask != 0).astype(int)
    return binarized

def contains_shrub_attributes(*attrs: List[bool]):
    """
    Determines if a shrub is infeasible.

    Args:
        *attrs - list of boolean attributes that a shrub must satisfy

    Ex.
    has_stem = True
    has_leaves = False
    other_attr = False
    attrs = [has_stem, has_leaves, other_attr]
    contains_shrub_attributes(attrs)

    Output: True
    """
    return any(attrs)


