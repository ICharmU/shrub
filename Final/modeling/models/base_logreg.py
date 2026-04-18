import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from Final.modeling.feature_importance import *

def train_log_reg(X_train, y_train):
    model = LogisticRegression(max_iter=250)
    model.fit(X_train, y_train)

    return model

def predict_log_reg(model, X_test):
    preds = model.predict(X_test)
    return np.round(preds)

def accuracy_log_reg(model, X_test, y_test, is_binary=True):
    """
    is_binary = False for multiclass weighting
    """
    preds = predict_log_reg(model, X_test)

    metrics = dict()
    metrics["overall"] = accuracy_score(y_test, preds)
    
    average = "binary" if is_binary else "weighted"
    metrics["f1"] = f1_score(y_test, preds, average=average)
    metrics["recall"] = recall_score(y_test, preds, average=average)
    metrics["precision"] = precision_score(y_test, preds, average=average)

    return metrics

# feature importance general pipeline structure
def pipeline_logreg(X_train, y_train, n_mrmr_features=20, is_binary=True):
    """
    Feature extraction pipeline that zeros out unused weights instead of slicing features.
    
    Each pixel (with all its channels) is treated as a feature.
    For shape (batch, channels, H, W): each pixel produces channels features
    Total features = channels * H * W
    
    This function performs:
    1. Feature Permutation analysis using Captum
    2. MRMR feature selection
    3. Consensus feature combination (union of non-zero importance + top 10% MRMR)
    4. Model training on ALL features
    5. Zero out coefficients for non-selected features
    
    Args:
        X_train: Training feature matrix (batch, channels, H, W) or (n_samples, n_features)
        y_train: Training labels (n_samples,)
        n_mrmr_features: Number of features to select via MRMR (default: 20)
        is_binary: Whether this is binary classification (default: True)
    
    Returns:
        results_dict: Dictionary containing:
            - 'model': Trained logistic regression model (with non-selected weights zeroed)
            - 'selected_feature_indices': Indices of selected features
            - 'n_original_features': Original number of features
            - 'n_selected_features': Number of selected features
            - 'reduction_percentage': Feature reduction as percentage
            - 'train_metrics': Training metrics on all data
            - 'consensus_df': DataFrame with consensus feature details
            - 'X_shape': Original input shape for reference
    """
    X_shape = X_train.shape
    if len(X_train.shape) == 4:
        batch_size = X_train.shape[0]
        X_train_flat = X_train.reshape(batch_size, -1)
    else:
        X_train_flat = X_train.copy()
    
    n_original_features = X_train_flat.shape[1]
    
    n_mrmr_features_actual = min(n_mrmr_features, n_original_features)
    
    perm_features, _ = feature_permutation_pipeline(
        X_train_flat, y_train,
        model_func=None,
        num_features=None,  # Get all with non-zero importance
        perturbation_type="logistic"
    )
    
    n_perm_features = len(perm_features)

    mrmr_features, _ = mrmr_pipeline(
        X_train_flat, y_train,
        num_features=n_mrmr_features_actual,
        task_type="classif"
    )
    
    consensus_features = combine_feature_importance_methods(
        perm_features,
        mrmr_features,
        dataset_name="Training Data"
    )
    
    if len(consensus_features) == 0:
        if len(perm_features) > 0:
            consensus_features = perm_features.copy()
        else:
            consensus_features = perm_features 
    
    n_selected_features = len(consensus_features)
    reduction_pct = (1 - n_selected_features / n_original_features) * 100
    
    model = train_log_reg(X_train_flat, y_train)
    train_metrics = accuracy_log_reg(model, X_train_flat, y_train, is_binary=is_binary)

    if len(consensus_features) > 0:
        selected_feature_names = consensus_features['feature'].tolist()
        selected_indices = np.array([int(f.split('_')[1]) for f in selected_feature_names])
        
        all_indices = np.arange(n_original_features)
        non_selected_mask = ~np.isin(all_indices, selected_indices)
        non_selected_indices = np.where(non_selected_mask)[0]
        
        if hasattr(model, 'coef_'):
            model.coef_[:, non_selected_indices] = 0 # zero out non-important features
        else:
            print("Warning: Model does not have coef_ attribute")
    else:
        print("No features selected for zeroing (using all features)")
        selected_indices = np.arange(n_original_features)
        non_selected_indices = np.array([])

    results_dict = {
        'model': model,
        'selected_feature_indices': selected_indices.tolist(),
        'non_selected_feature_indices': non_selected_indices.tolist(),
        'n_original_features': n_original_features,
        'n_selected_features': n_selected_features,
        'reduction_percentage': reduction_pct,
        'train_metrics': train_metrics,
        'consensus_df': consensus_features,
        'X_shape': X_shape,
        'permutation_features': perm_features['feature'].tolist(),
        'mrmr_features': mrmr_features,
    }
    
    return results_dict