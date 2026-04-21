import numpy as np

def probabilities_to_predictions(probabilities, threshold=0.5):
    """
    Convert soft probability predictions to binary mask.
    
    Parameters:
    -----------
    probabilities : array
        Probability map [0, 1] from neural network
    threshold : float
        Threshold for binarization (default 0.5)
    
    Returns:
    --------
    binary_mask : array
        Binary prediction mask {0, 1}
    """
    binary_mask = (probabilities > threshold).astype(int)
    return binary_mask


def extract_shrub_features(binary_mask):
    """
    Extract geometric features from binary shrub mask.
    
    Parameters:
    -----------
    binary_mask : array
        Binary prediction mask
    
    Returns:
    --------
    features : dict
        Dictionary with keys: height, width, area, aspect_ratio
    """
    # Get bounding box and basic stats
    _, h, w = binary_mask.shape
    height = h
    width = w
    area = np.sum(binary_mask)
    
    # Calculate aspect ratio (safeguard against w=0)
    aspect_ratio = height / width if width > 0 else 0
    
    return {
        'height': height,
        'width': width,
        # 'area': area,
        'aspect_ratio': aspect_ratio
    }

def get_predict_mask(probs):
    preds = np.round(probs)
    return preds

def get_shrub_locs(predict_mask):
    locs = np.argwhere(predict_mask == 1)
    return locs