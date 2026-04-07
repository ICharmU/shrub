import numpy as np
from typing import Iterable

class ShrubObject:
    pass

def shape_compatibility(func):
    """
    Decorator to verify label and prediction inputs have the same shape.
    """
    def check_compatibility(label_shrub, pred_shrub, *args):
        label_mask, pred_mask = label_shrub.get_mask().shape, pred_shrub.get_mask().shape
        compatible = label_mask == pred_mask
        if not compatible:
            raise Exception(f"Shape error: label_shrub has shape {label_mask}, but pred_shrub has shape {pred_mask}")
        else:
            return func(label_shrub, pred_shrub, *args)
    return check_compatibility

class ShrubObject:
    def __init__(self, mask):
        self.mask = mask

    def get_mask(self):
        return self.mask
    
    def find_center(self, shrub_value: int|np.ndarray):
        """
        Finds the center of shrub along each direction.

        Args:
            shrub_value: int for grayscale images
                         np array for more than one color

        Ex. (2d binary)
        pred_mask_2d = np.array([
            [0,1,0,0],
            [1,1,1,1],
            [0,1,0,0]
        ])
        
        s2 = ShrubObject(np.expand_dims(pred_mask_2d, axis=-1)) # need to expand last dim for grayscale. last dim is assumed to be a color scale.
        s2.find_center(1)

        Output: 
        array([1.        , 1.33333333])

        Ex. (2d rgb)
        pred_mask_2d_rgb = np.array([
            [[0,0,0],[255,255,255],[0,0,0],[0,0,0]],
            [[255,255,255],[255,255,255],[255,255,255],[255,255,255]],
            [[0,0,0],[255,255,255],[0,0,0],[0,0,0]]
        ])
        
        shrub_value = np.array([255,255,255])
        s2_rgb = ShrubObject(pred_mask_2d_rgb)
        s2_rgb.find_center(shrub_value)

        Output: 
        array([1.        , 1.33333333])

        Ex. (3d binary)
        pred_mask_3d = np.array([
            [[0,0,0,0],
            [0,1,0,0],
            [0,0,0,0]],
            [[0,1,0,0],
            [1,1,1,1],
            [0,1,0,0]],
            [[0,0,0,0],
            [0,1,0,0],
            [0,0,0,0]]
        ])

        s3 = ShrubObject(np.expand_dims(pred_mask_3d, axis=-1))
        s3.find_center(1)

        Output: 
        array([1.  , 1.  , 1.25])
        """
        indices = np.argwhere(np.all(self.mask == shrub_value, axis=-1))
        center = indices.T.mean(axis=-1)

        return center
    
    def get_count_type(self, shrub_value: int|np.ndarray = 1):
        """
        Counts the number of shrub-related pixels in the mask.

        Args:
            shrub_value - shrub pixel/voxel color tile. 
                          grayscale must be its own color dimension.
        """
        Xel_count = (self.mask == shrub_value).sum()
        other_count = (self.mask != shrub_value).sum()
        return Xel_count, other_count

    @shape_compatibility
    def get_count_difference(label_shrub, pred_shrub: ShrubObject, shrub_value: int|np.ndarray): 
        """
        Finds shrub and non-shrub count differences. Assumes classical sets are provided, not fuzzy sets, for the pixel/voxel matching.

        Args:
            label_shrub (self) - current shrub object to take difference relative to
            pred_shrub - ShrubObject to take difference against
            shrub_value - int or np array representing shrub color scale value
        
        Returns:
            n_diff - a list of the form [difference in shrub counts, difference in non-shrub counts]

        Ex.
        pred_mask_2d = np.array([
            [0,1,0,0],
            [1,1,1,1],
            [0,1,0,0]
        ])

        inv_pred_mask_2d = 1 - pred_mask_2d

        s2 = ShrubObject(np.expand_dims(pred_mask_2d, axis=-1))
        label_s2 = ShrubObject(np.expand_dims(inv_pred_mask_2d, axis=-1))
        label_s2.get_count_difference(s2, 1)

        Output: 
        [np.int64(0), np.int64(0)]
        """
        label_counts = label_shrub.get_count_type(shrub_value)
        pred_counts = pred_shrub.get_count_type(shrub_value)
        n_diff = [lc - pc for lc, pc in zip(label_counts, pred_counts)]
        return n_diff
    
    @shape_compatibility
    def get_count_correctness(label_shrub, pred_shrub: ShrubObject, shrub_value: int|np.ndarray):
        """
        Returns the number of correct and incorrect predictions for shrub and non-shrub classifications.

        Args:
            label_shrub (self) - current shrub object to treat as label
            pred_shrub - ShrubObject to compare against
            shrub_value - int or np array representing shrub color scale value

        Returns:
            tp - number of true positives
            fn - number of false negatives
            tn - number of true negatives
            fp - number of false positives

        Ex.
        pred_mask_2d = np.array([
            [0,1,1,0],
            [1,1,1,1],
            [0,1,0,0]
        ])

        inv_pred_mask_2d = 1 - pred_mask_2d

        s2 = ShrubObject(np.expand_dims(pred_mask_2d, axis=-1))
        label_s2 = ShrubObject(np.expand_dims(inv_pred_mask_2d, axis=-1))
        label_s2.get_count_correctness(s2, 1)

        Output: 
        (np.int64(0), np.int64(5), np.int64(0), np.int64(7))
        """
        # actually shrub
        tp = ((label_shrub.get_mask() == shrub_value) & (pred_shrub.get_mask() == shrub_value)).sum()
        fn = ((label_shrub.get_mask() == shrub_value) & (pred_shrub.get_mask() != shrub_value)).sum()
        # actually not shrub
        tn = ((label_shrub.get_mask() != shrub_value) & (pred_shrub.get_mask() != shrub_value)).sum()
        fp = ((label_shrub.get_mask() != shrub_value) & (pred_shrub.get_mask() == shrub_value)).sum()

        return tp, fn, tn, fp
    
    @shape_compatibility
    def get_accuracy_summary(label_shrub, pred_shrub: ShrubObject, shrub_value: int|np.ndarray, beta: float|Iterable = None):
        """
        Calculates balanced error summary statistics.

        Args:
            label_shrub (self) - current shrub object to treat as label
            pred_shrub - ShrubObject to compare against
            shrub_value - int or np array representing shrub color scale value

        Returns:
            summary - a dictionary of the below form
                      {
                        "overall": ...,
                        "precision": ...,
                        "recall": ...,
                        "fpr": ...,
                        "f1": ...,
                        (opt.) "f_beta": <not included if beta not provided> | <scalar if beta is a scalar> | <list of scores if beta is iterable>
                      }

        Ex.
        pred_mask_2d = np.array([
            [0,1,1,0],
            [1,1,1,1],
            [0,1,0,0]
        ])

        inv_pred_mask_2d = np.array([
            [0,1,0,0],
            [1,1,1,1],
            [0,1,0,0]
        ])

        s2 = ShrubObject(np.expand_dims(pred_mask_2d, axis=-1))
        label_s2 = ShrubObject(np.expand_dims(inv_pred_mask_2d, axis=-1))
        label_s2.get_accuracy_summary(s2, 1, [0, 0.5, 1, 1.5, 2])

        Output: 
        {'overall': np.float64(0.5),
        'precision': np.float64(0.8571428571428571),
        'recall': np.float64(1.0),
        'fpr': np.float64(0.16666666666666666),
        'f1': np.float64(0.923076923076923),
        'f_beta': [np.float64(0.8571428571428571),
        np.float64(0.8823529411764707),
        np.float64(0.923076923076923),
        np.float64(0.951219512195122),
        np.float64(0.9677419354838709)]}
        """
        summary = dict()
        tp, fn, tn, fp = label_shrub.get_count_correctness(pred_shrub, shrub_value)

        summary["overall"] = ShrubObject.get_accuracy(tp, fn, tn, fp)
        summary["precision"] = ShrubObject.get_precision(tp, fp)
        summary["recall"] = ShrubObject.get_recall(tp, fn)
        summary["fpr"] = ShrubObject.get_fpr(tn, fp)
        summary["f1"] = ShrubObject.get_f1(summary["precision"], summary["recall"])
        if beta:
            if hasattr(beta, '__iter__'):
                summary["f_beta"] = list()
                for B in beta:
                    summary["f_beta"].append(ShrubObject.get_fbeta(summary["precision"], summary["recall"], B))
            else:
                summary["f_beta"] = ShrubObject.get_fbeta(summary["precision"], summary["recall"], beta)

        return summary

    
    @staticmethod
    def get_accuracy(tp: int, fn: int, tn: int, fp: int):
        return (tp+fn) / (tp+fn+tn+fp)

    @staticmethod
    def get_precision(tp: int, fp: int):
        return tp / (tp+fp)
    
    @staticmethod
    def get_recall(tp: int, fn: int):
        return tp / (tp+fn)
    
    @staticmethod
    def get_fpr(tn: int, fp: int):  
        return fp / (fp+tn)
    
    @staticmethod
    def get_f1(precision: float, recall: float):
        return ShrubObject.get_fbeta(precision, recall, 1)

    @staticmethod
    def get_fbeta(precision: float, recall: float, beta: float):
        if beta**2 * precision + recall == 0:
            return None
        
        return ((1+beta**2) * precision * recall) / (beta**2 * precision + recall)