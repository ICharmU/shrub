import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Final.modeling.helpers import create_dataset, get_all_targets
from Final.modeling.helpers import train_epoch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

class SimpleCNN(nn.Module):
    """
    Simple CNN supporting both image-level and per-pixel (segmentation) tasks.
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 2)
    """
    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        self.conv_out = nn.Conv2d(128, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.conv_out(x)
        return x
        
def train_simple_cnn(model, X, y, n_classes=2, n_epochs=100, optimizer=optim.Adam):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset, train_loader = create_dataset(X_train, y_train, is_tensor=True)
    test_dataset, test_loader = create_dataset(X_test, y_test, is_tensor=True, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleCNN(num_classes=n_classes).to(device)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    return model, train_loss, train_loader, test_loader

def eval_simple_cnn(model, eval_loader, device):
    """
    eval_loader can be train_loader or test_loader
    """
    model.eval()
    with torch.no_grad():
        all_probs = list()
        all_preds = list()
        targets = list()
        for batch_X, batch_y in eval_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).astype(float)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            targets.append(batch_y.cpu().numpy())
        
        all_probs = np.concatenate(all_probs).flatten()
        all_preds = np.concatenate(all_preds).flatten()
        targets = np.concatenate(targets).flatten()

        dice = 2 * (all_preds * targets).sum() / (all_preds.sum() + targets.sum() + 1e-8)
        
    return all_preds, targets, all_probs, dice

def metrics_simple_cnn(model, train_loader, test_loader, binary=True):
    average = "binary" if binary else "weighted"
    model.eval()

    with torch.no_grad():
        train_preds = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            preds = logits.argmax(dim=1)
            train_preds.append(preds.cpu().numpy())
        train_preds = np.concatenate(train_preds)
        
        test_preds = []
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            preds = logits.argmax(dim=1)
            test_preds.append(preds.cpu().numpy())
        test_preds = np.concatenate(test_preds)
        
    y_train, y_test = get_all_targets(train_loader), get_all_targets(test_loader)
    
    train_metrics = {
        "overall": accuracy_score(y_train, train_preds),
        "f1": f1_score(y_train, train_preds, average=average),
        "recall": recall_score(y_train, train_preds, average=average),
        "precision": precision_score(y_train, train_preds, average=average)
    }
    test_metrics = {
        "overall": accuracy_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds, average=average),
        "recall": recall_score(y_test, test_preds, average=average),
        "precision": precision_score(y_test, test_preds, average=average)
    }

    return train_metrics, test_metrics

def create_cnn_wrapper(model_class, model_name="CNN", **init_kwargs):
    """
    Creates a wrapper function for CNN models that standardizes training/evaluation.
    
    Args:
        model_class: PyTorch model class (e.g., SimpleCNN, UNet)
        model_name: String name for logging
        **init_kwargs: Arguments to pass to model_class.__init__()
    
    Returns:
        wrapper_fn: Function with signature (X_train, y_train, X_test=None, y_test=None, **train_kwargs)
                   that returns (train_metrics, test_metrics or None)
    
    Example:
        cnn_fn = create_cnn_wrapper(SimpleCNN, model_name="SimpleCNN", num_classes=2)
        train_metrics, test_metrics = cnn_fn(X_train, y_train, X_test, y_test, epochs=100)
    """

    
    def wrapper_fn(X_train, y_train, X_test=None, y_test=None, 
                   optimizer = None,
                   epochs=50, batch_size=32, learning_rate=1e-3, **kwargs):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_class(**init_kwargs).to(device)
        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        criterion = nn.CrossEntropyLoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        train_dataset, train_loader = create_dataset(X_train_tensor, y_train_tensor, is_tensor=True)

        train_losses = []
        for epoch in range(epochs):
            avg_epoch_loss = train_epoch(model, train_loader, device, optimizer, criterion)
            train_losses.append(avg_epoch_loss)

        model.eval()
        with torch.no_grad():
            train_preds = model(torch.FloatTensor(X_train).to(device))
            train_preds = torch.argmax(train_preds, dim=1).cpu().numpy()
        
        train_metrics = {
            "accuracy": np.mean(train_preds == y_train),
            "loss": train_losses[-1],
            "epochs": epochs
        }
        
        test_metrics = None
        if X_test is not None and y_test is not None:
            test_dataset, test_loader = create_dataset(X_test, y_test)
            all_test_preds = list()
            n_correct = 0
            n_tot = 0
            with torch.no_grad():
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    test_preds = model(batch_X)
                    test_preds = torch.argmax(test_preds, dim=1).cpu().numpy()
                    all_test_preds.append(test_preds)

                    n_correct += np.sum(test_preds == batch_y.cpu().numpy())
                    n_tot += len(batch_y)

            test_preds = test_preds.flatten()
            test_metrics = {
                "accuracy": n_correct / n_tot if n_tot != 0 else 0,
            }
        else:
            test_preds = None
            test_metrics = None
        
        return train_preds, test_preds, train_metrics, test_metrics
    
    return wrapper_fn

def pipeline_cnn(X_train, y_train, X_test, y_test, in_channels=4, num_classes=2, epochs=10):
    """
    Train CNN for per-pixel binary predictions with feature importance.
    
    Args:
        X_train: Training features (batch, channels, H, W)
        y_train: Training labels (batch, H, W) - per-pixel binary labels
        X_test: Test features (batch, channels, H, W)
        y_test: Test labels (batch, H, W)
        in_channels: Number of input channels (default: 4)
        num_classes: Number of classes (default: 2)
        epochs: Number of training epochs (default: 10)
    
    Returns:
        results_dict: Dictionary containing:
            - 'model': Trained CNN model
            - 'train_preds': Training predictions (batch, H, W)
            - 'test_preds': Test predictions (batch, H, W)
            - 'train_accuracy': Per-pixel training accuracy
            - 'test_accuracy': Per-pixel test accuracy
            - 'train_metrics': Training metrics dict
            - 'test_metrics': Test metrics dict
    """
    X_shape = X_train.shape
    y_shape = y_train.shape
    batch_size, channels, height, width = X_shape
    n_pixels = height * width

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN(in_channels=channels, num_classes=num_classes)
    model.to(device)
    
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)
    
    train_dataset, train_loader = create_dataset(X_train, y_train)
    test_dataset, test_loader = create_dataset(X_test, y_test, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = list()
    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_loader, device, optimizer, criterion)
        train_losses.append(avg_loss)
        
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train_t)
        train_preds = torch.argmax(train_logits, dim=1).cpu().numpy()
    
    train_accuracy = np.mean(train_preds == y_train)
    train_metrics = {
        'accuracy': float(train_accuracy),
        'shape': train_preds.shape
    }

    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
    
    test_accuracy = np.mean(test_preds == y_test)
    test_metrics = {
        'accuracy': float(test_accuracy),
        'shape': test_preds.shape
    }
    
    results_dict = {
        'model': model,
        'train_preds': train_preds,
        'test_preds': test_preds,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
    }
    
    return results_dict