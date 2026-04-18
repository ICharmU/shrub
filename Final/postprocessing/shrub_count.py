from scipy import ndimage
from skimage import morphology, feature
from skimage.segmentation import watershed
from skimage.feature import blob_log
import cv2
import numpy as np

def apply_bias_correction(ensemble_count, shrub_features, model, scaler):
    """
    Apply Ridge regression bias correction to ensemble count.
    
    Parameters:
    -----------
    ensemble_count : int
        Initial count from ensemble
    shrub_features : dict
        Features dict from extract_shrub_features()
    model : sklearn.linear_model.Ridge
        Trained Ridge regression model
    scaler : sklearn.preprocessing.StandardScaler
        Fitted StandardScaler for feature normalization
    
    Returns:
    --------
    corrected_count : float
        Bias-corrected shrub count
    """
    # Build feature vector [ensemble_count, height, width, area, aspect_ratio]
    features = np.array([[
        ensemble_count,
        shrub_features['height'],
        shrub_features['width'],
        shrub_features['area'],
        shrub_features['aspect_ratio']
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    corrected_count = model.predict(features_scaled)[0]
    
    return corrected_count

def count_shrubs_ccl(binary_mask):
    """Connected Component Labeling"""
    labeled_array, num_features = ndimage.label(binary_mask)
    return num_features, labeled_array

def count_shrubs_watershed(binary_mask):
    """Watershed segmentation"""
    # Distance transform
    dist_transform = ndimage.distance_transform_edt(binary_mask)
    # Find local maxima as markers
    local_maxima = ndimage.maximum_filter(dist_transform, size=7) == dist_transform
    markers = ndimage.label(local_maxima)[0]
    # Watershed
    segmented = watershed(-dist_transform, markers, mask=binary_mask) if np.any(binary_mask) else np.zeros_like(binary_mask)
    num_objects = len(np.unique(segmented)) - 1
    return num_objects, segmented

def count_shrubs_morphological(binary_mask, kernel_size=5):
    """Morphological opening to separate shrubs"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    labeled_array, num_features = ndimage.label(opened)
    return num_features, labeled_array

def count_shrubs_blob_log(binary_mask, min_sigma=5, max_sigma=30):
    """Laplacian of Gaussian blob detection"""
    if np.any(binary_mask) == 0:
        return 0, binary_mask
    try:
        blobs = blob_log(binary_mask, min_sigma=min_sigma, max_sigma=max_sigma, threshold=0.1)
        return len(blobs), binary_mask
    except:
        return 0, binary_mask

def count_shrubs_dbscan(binary_mask, eps=10, min_samples=5):
    """DBSCAN clustering on object pixels"""
    if np.any(binary_mask) == 0:
        return 0, binary_mask
    points = np.column_stack(np.where(binary_mask > 0))
    if len(points) < min_samples:
        return 1, binary_mask
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return len(set(clustering.labels_)), binary_mask

def ensemble_predict_shrub_count(binary_mask):
    """
    Simple ensemble: Average predictions from 5 algorithms with outlier rejection.
    
    Parameters:
        binary_mask: Binary segmentation mask (0s and 1s)
        verbose: Print detailed breakdown
    
    Returns:
        final_count: Final ensemble shrub count
        predictions_dict: Individual algorithm predictions
    """
    ccl_count, _ = count_shrubs_ccl(binary_mask)
    watershed_count, _ = count_shrubs_watershed(binary_mask)
    morph_count, _ = count_shrubs_morphological(binary_mask)
    blob_count = count_shrubs_blob_log(binary_mask)[0]
    dbscan_count = count_shrubs_dbscan(binary_mask)[0]
    
    predictions = {
        'CCL': ccl_count,
        'Watershed': watershed_count,
        'Morphological': morph_count,
        'LoG': blob_count,
        'DBSCAN': dbscan_count
    }
    
    pred_values = np.array(list(predictions.values()))
    mean_pred = np.mean(pred_values)
    std_pred = np.std(pred_values)
    median_pred = np.median(pred_values)

    # heuristic
    lower_bound = mean_pred - 1 * std_pred
    upper_bound = mean_pred + 1 * std_pred
    kept_predictions = pred_values[(pred_values >= lower_bound) & (pred_values <= upper_bound)]
    
    if len(kept_predictions) > 0:
        final_estimate = np.mean(kept_predictions)
    else:
        final_estimate = median_pred
    
    final_count = int(np.round(final_estimate))

    return final_count, predictions

def ensemble_predict_shrub_count(binary_mask):
    """
    Ensemble shrub counting using 5 parallel algorithms.
    
    Algorithms:
    -----------
    1. Connected Component Labeling (CCL)
    2. Watershed Segmentation
    3. Morphological Opening
    4. Laplacian of Gaussian (LoG)
    5. DBSCAN clustering
    
    Uses outlier rejection: keeps predictions within mean ± 1σ
    
    Parameters:
    -----------
    binary_mask : array
        Binary prediction mask {0, 1}
    verbose : bool
        Print algorithm breakdown (default False)
    
    Returns:
    --------
    ensemble_count : int
        Final ensemble prediction (median of kept predictions)
    """
    predictions = []
    
    # Algorithm 1: Connected Component Labeling
    labeled, num_components = ndimage.label(binary_mask)
    predictions.append(num_components)
    
    # Algorithm 2: Watershed Segmentation
    distance = ndimage.distance_transform_edt(binary_mask)
    local_maxima = ndimage.maximum_filter(distance, size=3) == distance
    markers = ndimage.label(local_maxima)[0]
    watershed_result = watershed(-distance, markers=markers, mask=binary_mask)
    watershed_count = len(np.unique(watershed_result)) - 1  # Exclude background
    predictions.append(watershed_count)
    
    # Algorithm 3: Morphological Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    morph_labeled, morph_count = ndimage.label(morphed)
    predictions.append(morph_count)
    
    # Algorithm 4: Laplacian of Gaussian
    try:
        blobs = feature.blob_log(binary_mask.astype(float), max_sigma=30, threshold=0.1)
        log_count = len(blobs)
        predictions.append(log_count)
    except:
        predictions.append(0)

    
    # Algorithm 5: DBSCAN
    try:
        from sklearn.cluster import DBSCAN
        white_pixels = np.argwhere(binary_mask > 0)
        if len(white_pixels) > 0:
            dbscan = DBSCAN(eps=1.5, min_samples=1).fit(white_pixels)
            dbscan_count = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
            predictions.append(dbscan_count)
        else:
            predictions.append(0)
    except:
        predictions.append(0)

    predictions_arr = np.array(predictions)
    mean_pred = predictions_arr.mean()
    std_pred = predictions_arr.std()
    
    kept_predictions = predictions_arr[
        (predictions_arr >= mean_pred - std_pred) & 
        (predictions_arr <= mean_pred + std_pred)
    ]
    
    ensemble_count = int(np.mean(kept_predictions))
    
    return ensemble_count


# correction model isn't feasible without labels.
# testing with binary masks containing circles saw improvements from a correction of 1-2 (i.e. increase prediction by this much)
def shrub_counting_pipeline(
    binary_mask,
    apply_correction=True,
    model=None,
    scaler=None,
    verbose=True
):
    """
    Complete shrub counting pipeline.
    
    Pipeline stages:
    ----------------
    1. Convert probabilities to binary predictions
    2. Extract geometric features from mask
    3. Run 5-algorithm ensemble for counting
    4. (Optional) Apply Ridge regression bias correction
    
    Parameters:
    -----------
    probabilities : array
        Probability map [0, 1] from neural network
    prob_threshold : float
        Threshold for probability→prediction conversion (default 0.5)
    apply_correction : bool
        Whether to apply bias correction (default True)
    model : sklearn.linear_model.Ridge or None
        Trained Ridge model for correction.
    scaler : sklearn.preprocessing.StandardScaler or None
        Fitted scaler for feature normalization.
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - binary_mask: Binary prediction mask
        - ensemble_count: Count from 5-algorithm ensemble
        - corrected_count: Count after bias correction (if applied)
        - final_count: Final prediction (int)
        - features: Dictionary of geometric features
        - method_used: 'ensemble_only' or 'ensemble+correction'
    """
    shrub_features = extract_shrub_features(binary_mask)
    ensemble_count = ensemble_predict_shrub_count(binary_mask, verbose=verbose)
    
    corrected_count = None
    method_used = 'ensemble_only'
    
    if apply_correction:
        if model is not None and scaler is not None:
            corrected_count = apply_bias_correction(
                ensemble_count, shrub_features, model, scaler
            )
            corrected_count_int = int(np.round(corrected_count))
            method_used = 'ensemble+correction'
            final_count = corrected_count_int
        else:
            final_count = ensemble_count
    else:
        final_count = ensemble_count

    
    return {
        'binary_mask': binary_mask,
        'ensemble_count': ensemble_count,
        'corrected_count': corrected_count,
        'final_count': final_count,
        'features': shrub_features,
        'method_used': method_used
    }
