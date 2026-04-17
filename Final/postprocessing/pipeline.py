from scipy import ndimage
from skimage import morphology, feature
from skimage.segmentation import watershed
import cv2
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
    h, w = binary_mask.shape
    height = h
    width = w
    area = np.sum(binary_mask)
    
    # Calculate aspect ratio (safeguard against w=0)
    aspect_ratio = height / width if width > 0 else 1.0
    
    return {
        'height': height,
        'width': width,
        'area': area,
        'aspect_ratio': aspect_ratio
    }


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


def ensemble_predict_shrub_count(binary_mask, verbose=False):
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
    if verbose:
        print(f"  CCL                 :   {num_components}")
    
    # Algorithm 2: Watershed Segmentation
    distance = ndimage.distance_transform_edt(binary_mask)
    local_maxima = ndimage.maximum_filter(distance, size=3) == distance
    markers = ndimage.label(local_maxima)[0]
    watershed_result = watershed(-distance, markers=markers, mask=binary_mask)
    watershed_count = len(np.unique(watershed_result)) - 1  # Exclude background
    predictions.append(watershed_count)
    if verbose:
        print(f"  Watershed           :   {watershed_count}")
    
    # Algorithm 3: Morphological Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    morph_labeled, morph_count = ndimage.label(morphed)
    predictions.append(morph_count)
    if verbose:
        print(f"  Morphological       :   {morph_count}")
    
    # Algorithm 4: Laplacian of Gaussian
    try:
        blobs = feature.blob_log(binary_mask.astype(float), max_sigma=30, threshold=0.1)
        log_count = len(blobs)
        predictions.append(log_count)
        if verbose:
            print(f"  LoG                 :   {log_count}")
    except:
        predictions.append(0)
        if verbose:
            print(f"  LoG                 :   0")
    
    # Algorithm 5: DBSCAN
    try:
        from sklearn.cluster import DBSCAN
        white_pixels = np.argwhere(binary_mask > 0)
        if len(white_pixels) > 0:
            dbscan = DBSCAN(eps=1.5, min_samples=1).fit(white_pixels)
            dbscan_count = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
            predictions.append(dbscan_count)
            if verbose:
                print(f"  DBSCAN              :   {dbscan_count}")
        else:
            predictions.append(0)
            if verbose:
                print(f"  DBSCAN              :   0")
    except:
        predictions.append(0)
        if verbose:
            print(f"  DBSCAN              :   0")
    
    # Outlier rejection: keep predictions within mean ± 1σ
    predictions_arr = np.array(predictions)
    mean_pred = predictions_arr.mean()
    std_pred = predictions_arr.std()
    
    kept_predictions = predictions_arr[
        (predictions_arr >= mean_pred - std_pred) & 
        (predictions_arr <= mean_pred + std_pred)
    ]
    
    ensemble_count = int(np.mean(kept_predictions))
    
    if verbose:
        print(f"  Mean: {mean_pred:.1f}, Std: {std_pred:.1f}")
        print(f"  Ensemble (after outlier rejection): {ensemble_count}")
    
    return ensemble_count

def shrub_counting_pipeline(
    probabilities,
    prob_threshold=0.5,
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
        Trained Ridge model for correction. If None, uses global 'final_model'
    scaler : sklearn.preprocessing.StandardScaler or None
        Fitted scaler for feature normalization. If None, uses global 'scaler'
    verbose : bool
        Print diagnostic output (default True)
    
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
    
    if verbose:
        print("\n" + "="*60)
        print("SHRUB COUNTING PIPELINE")
        print("="*60)
    
    # Stage 1: Probabilities → Predictions
    if verbose:
        print("\n[Stage 1] Converting probabilities to binary predictions...")
    binary_mask = probabilities_to_predictions(probabilities, threshold=prob_threshold)
    if verbose:
        print(f"  ✓ Binary mask shape: {binary_mask.shape}")
        print(f"  ✓ Positive pixels: {np.sum(binary_mask)}")
    
    # Stage 2: Extract features
    if verbose:
        print("\n[Stage 2] Extracting geometric features...")
    shrub_features = extract_shrub_features(binary_mask)
    if verbose:
        print(f"  ✓ Height: {shrub_features['height']}")
        print(f"  ✓ Width: {shrub_features['width']}")
        print(f"  ✓ Area: {shrub_features['area']}")
        print(f"  ✓ Aspect Ratio: {shrub_features['aspect_ratio']:.3f}")
    
    # Stage 3: Ensemble counting
    if verbose:
        print("\n[Stage 3] Running 5-algorithm ensemble...")
    ensemble_count = ensemble_predict_shrub_count(binary_mask, verbose=verbose)
    if verbose:
        print(f"  ✓ Ensemble prediction: {ensemble_count} shrubs")
    
    # Stage 4: Optional bias correction
    corrected_count = None
    method_used = 'ensemble_only'
    
    if apply_correction:
        # Use provided model/scaler or fall back to globals
        corr_model = model if model is not None else globals().get('final_model')
        corr_scaler = scaler if scaler is not None else globals().get('scaler')
        
        if corr_model is not None and corr_scaler is not None:
            if verbose:
                print("\n[Stage 4] Applying bias correction (Ridge regression)...")
            corrected_count = apply_bias_correction(
                ensemble_count, shrub_features, corr_model, corr_scaler
            )
            corrected_count_int = int(np.round(corrected_count))
            if verbose:
                print(f"  ✓ Corrected prediction: {corrected_count:.2f} → {corrected_count_int} shrubs")
                print(f"  ✓ Adjustment: {corrected_count_int - ensemble_count:+d}")
            method_used = 'ensemble+correction'
            final_count = corrected_count_int
        else:
            if verbose:
                print("\n⚠ Warning: apply_correction=True but model/scaler not found")
                print("  Using ensemble-only prediction")
            final_count = ensemble_count
    else:
        final_count = ensemble_count
    
    if verbose:
        print("\n" + "="*60)
        print(f"FINAL PREDICTION: {final_count} shrubs ({method_used})")
        print("="*60 + "\n")
    
    return {
        # 'binary_mask': binary_mask,
        'ensemble_count': ensemble_count,
        'corrected_count': corrected_count,
        'final_count': final_count,
        'features': shrub_features,
        'method_used': method_used
    }
