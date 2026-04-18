from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
from captum.attr import FeaturePermutation
from mrmr import mrmr_classif, mrmr_regression

def feature_permutation_pipeline(X, y, model_func, num_features=None, perturbation_type="logistic", model_=None):
    """
    Compute feature importance using Captum's FeaturePermutation.
    
    Args:
        X: Input features (flattened for logistic regression or image tensors for CNN)
        y: Target labels
        model_func: Model function (model_logistic_regression or model_resnet18)
        num_features: Number of features to rank (None = all features)
        perturbation_type: Type of model ("logistic" or "cnn")
        
    Returns:
        feature_importance_df: DataFrame with feature rankings and importance scores
        attributions: Raw attribution values from Captum
    """
    
    if perturbation_type == "logistic": # logreg
        # imports are a bit messy here.
        from Final.modeling.models.base_logreg import *
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
        logreg = train_log_reg(X_train, y_train)
        
        def forward_func(inputs):
            preds = logreg.predict(inputs.numpy())
            preds = np.round(preds)
            accuracy = (preds == y_test).astype(float)
            return torch.tensor(1.0 - accuracy, dtype=torch.float32)
        
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
    else:  # cnn
        # imports are a bit messy here.
        from Final.modeling.models.base_cnn import *
        from Final.modeling.models.unet import *
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
        model = model_(X_train, y_train)
        
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        def forward_func(inputs):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_eval = model[0] if isinstance(model, tuple) else model
            model_eval.eval()
            model_eval = model_eval.to(device)
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model_eval(inputs)
                preds = outputs.argmax(dim=1)
            # Return loss (1 - accuracy) for each sample
            accuracy = (preds.cpu() == y_test).astype(float)
            return torch.tensor(1.0 - accuracy, dtype=torch.float32)
    
    feature_perm = FeaturePermutation(forward_func)
    attributions = feature_perm.attribute(X_test_tensor, perturbations_per_eval=1, show_progress=False)
    attr_flat = attributions.numpy().reshape(attributions.shape[0], -1)
    feature_importance = np.abs(attr_flat).mean(axis=0)
    
    feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_score': feature_importance
    }).sort_values('importance_score', ascending=False).reset_index(drop=True)
    
    if num_features:
        feature_importance_df = feature_importance_df.head(num_features)
    
    return feature_importance_df, attributions

def mrmr_pipeline(X, y, num_features=None, task_type="classif"):
    """
    Compute feature importance using MRMR (Minimum Redundancy Maximum Relevance).
    
    Args:
        X: Input features (numpy array or pandas DataFrame)
        y: Target labels (numpy array or pandas Series)
        num_features: Number of top features to select (None = use default)
        task_type: "classif" for classification or "regression" for regression
        
    Returns:
        selected_features: List of selected feature names (ranked)
        feature_importance_df: DataFrame with feature rankings
    """

    if isinstance(X, np.ndarray):
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        feature_names = X_df.columns.tolist()
    
    if isinstance(y, np.ndarray):
        y_series = pd.Series(y, name="target")
    else:
        y_series = y.copy()
    
    if num_features is None:
        num_features = len(feature_names)

    if task_type == "classif":
        selected_features = mrmr_classif(X=X_df, y=y_series, K=num_features, n_jobs=1)
    else:
        selected_features = mrmr_regression(X=X_df, y=y_series, K=num_features, n_jobs=1)
    
    feature_importance_df = pd.DataFrame({
        'feature': selected_features,
        'rank': range(1, len(selected_features) + 1)
    })
    
    return selected_features, feature_importance_df

def combine_feature_importance_methods(feature_importance_df, mrmr_features, p_feat_kept=0.01, dataset_name="Dataset"):
    """
    Combine results from MRMR and Feature Permutation to get consensus features.
    Returns features that have EITHER:
    1. Non-zero permutation importance, OR
    2. In the top 10% of MRMR features (by count)
       - BUT remove features at the boundary rank that have zero importance
    
    Args:
        feature_importance_df: DataFrame from feature_permutation_pipeline with columns ['feature', 'importance_score']
        mrmr_features: List of MRMR selected feature names (ordered by rank)
        p_feat_kept: proportion of features kept. should be in range (0,1)
        dataset_name: Name of the dataset for reporting
        
    Returns:
        consensus_df: DataFrame with consensus features ranked
    """
    
    perm_features_valid = feature_importance_df[feature_importance_df['importance_score'] > 0].copy()
    perm_features_valid['perm_rank'] = range(1, len(perm_features_valid) + 1)
    
    non_zero_features = set(perm_features_valid['feature'].tolist())
    
    # keep at a minimum 1% of features
    top_pct_count = max(1, int(np.ceil(len(mrmr_features) * p_feat_kept)))
    top_mrmr = set(mrmr_features[:top_pct_count])
    
    # Find the worst (highest) MRMR rank among top 1%; mrmr rankings are 1 indexed
    worst_rank_in_top = top_pct_count 
    
    # features with non-zero importance or in top 1% of mrmr ranking
    consensus_features = non_zero_features.union(top_mrmr)
    
    # do not include features that are in the worst mrmr ranking
    features_with_worst_rank = set()
    for i, feature in enumerate(mrmr_features):
        if i + 1 == worst_rank_in_top:  # Convert to 1-indexed rank
            features_with_worst_rank.add(feature)
    
    # remove features with worst rank and zero importance. features will not be useful
    for f in features_with_worst_rank:
        if f not in non_zero_features:
            consensus_features.discard(f)
    
    if len(consensus_features) == 0:
        print(f"\nWarning: No features found.")
        return pd.DataFrame()
    
    consensus_data = []
    for feature in consensus_features:
        perm_data = perm_features_valid[perm_features_valid['feature'] == feature]
        if len(perm_data) > 0:
            perm_rank = perm_data['perm_rank'].values[0]
            perm_score = perm_data['importance_score'].values[0]
        else:
            perm_rank = len(perm_features_valid) + 1
            perm_score = 0
        
        if feature in mrmr_features:
            mrmr_rank = mrmr_features.index(feature) + 1
        else:
            mrmr_rank = len(mrmr_features) + 1
        
        consensus_data.append({
            'feature': feature,
            'perm_importance': perm_score,
            'perm_rank': perm_rank,
            'mrmr_rank': mrmr_rank,
            'avg_rank': (perm_rank + mrmr_rank) / 2
        })
    
    consensus_df = pd.DataFrame(consensus_data)
    consensus_df = consensus_df.sort_values('avg_rank').reset_index(drop=True)
    
    return consensus_df