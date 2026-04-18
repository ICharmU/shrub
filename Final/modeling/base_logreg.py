from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def train_log_reg(X_train, y_train):
    """
    log reg assumes flattened inputs
    """
    model = LogisticRegression(max_iter=250)
    model.fit(X_train, y_train)

    return model

def log_reg_predict(model, X_test):
    """Get predictions from logistic regression model."""
    preds = model.predict(X_test)
    return np.round(preds)

def log_reg_accuracy(model, X_test, y_test, is_binary=True):
    """
    Binary log-reg
    """
    preds = log_reg_predict(model, X_test)

    metrics = dict()
    metrics["overall"] = accuracy_score(y_test, preds)
    
    average = "binary" if is_binary else "weighted"
    metrics["f1"] = f1_score(y_test, preds, average=average)
    metrics["recall"] = recall_score(y_test, preds, average=average)
    metrics["precision"] = precision_score(y_test, preds, average=average)

    return metrics