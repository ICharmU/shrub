from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, jaccard_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import pandas as pd
import os
from captum.attr import FeaturePermutation
import sys
from mrmr import mrmr_classif, mrmr_regression

####################
# BASELINES
####################

# LOG REG
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
    preds = model.predict(X_test)
    preds = np.round(preds) 

    metrics = dict()
    metrics["overall"] = accuracy_score(y_test, preds)
    
    average = "binary" if is_binary else "weighted"
    metrics["f1"] = f1_score(y_test, preds, average=average)
    metrics["recall"] = recall_score(y_test, preds, average=average)
    metrics["precision"] = precision_score(y_test, preds, average=average)

    return metrics

# RESNET 18 (not tuned/not intended for image segmentation)
def build_resnet18_classifier(num_classes=2):
    """
    Build ResNet18 classifier with replaced output head.
    Loads pretrained ResNet18 and freezes all weights except the final layer.
    """
    model = resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model

def model_resnet18(X, y):
    """
    Train and evaluate ResNet18 model following baseline workflow.
    Only trains the output layer while keeping backbone frozen.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_resnet18_classifier(num_classes=2)
    model = model.to(device)
    model.train()
    
    # Training - only optimize the output layer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    epochs = 100
    
    print(f"Training on {device}...")
    for epoch in range(epochs):
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")
    
    # Evaluation
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
    
    train_metrics = {
        "overall": accuracy_score(y_train, train_preds),
        "f1": f1_score(y_train, train_preds, average="binary"),
        "recall": recall_score(y_train, train_preds, average="binary"),
        "precision": precision_score(y_train, train_preds, average="binary")
    }
    test_metrics = {
        "overall": accuracy_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds, average="binary"),
        "recall": recall_score(y_test, test_preds, average="binary"),
        "precision": precision_score(y_test, test_preds, average="binary")
    }
    
    print(f"Train metrics:\n{train_metrics}")
    print(f"Test metrics:\n{test_metrics}")

# SIMPLE CNN
class SimpleCNN(nn.Module):
    """
    Simple CNN for binary classification on 32x32 RGB images.
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def model_simple_cnn(X, y):
    """
    Train and evaluate simple CNN model following baseline workflow.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleCNN(num_classes=2)
    model = model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100
    
    print(f"Training on {device}...")
    for epoch in range(epochs):
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")
    
    model.eval()
    train_loader_eval = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader_eval = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        train_preds = []
        for batch_X, batch_y in train_loader_eval:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            preds = logits.argmax(dim=1)
            train_preds.append(preds.cpu().numpy())
        train_preds = np.concatenate(train_preds)
        
        test_preds = []
        for batch_X, batch_y in test_loader_eval:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            preds = logits.argmax(dim=1)
            test_preds.append(preds.cpu().numpy())
        test_preds = np.concatenate(test_preds)
    
    train_metrics = {
        "overall": accuracy_score(y_train, train_preds),
        "f1": f1_score(y_train, train_preds, average="binary"),
        "recall": recall_score(y_train, train_preds, average="binary"),
        "precision": precision_score(y_train, train_preds, average="binary")
    }
    test_metrics = {
        "overall": accuracy_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds, average="binary"),
        "recall": recall_score(y_test, test_preds, average="binary"),
        "precision": precision_score(y_test, test_preds, average="binary")
    }

    return train_metrics, test_metrics

# SUPER SIMPLE CNN
class SimpleSegmentationNet(nn.Module):
    """Minimal 3-layer CNN - much simpler than UNet"""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.dec = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.dec(x)
        return x
    
def train_simple_cnn(X,y):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset, train_loader = create_dataset(X,y)
    criterion = DiceLoss()

    simple_model = SimpleSegmentationNet().to(device)
    simple_optimizer = optim.Adam(simple_model.parameters(), lr=5e-3)  # Higher LR for small model

    print("Training SIMPLE model (3-layer CNN)...")
    simple_losses = []
    simple_spatial_vars = []

    for epoch in range(30):
        epoch_loss = 0.0
        epoch_var = 0.0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            simple_optimizer.zero_grad()
            logits = simple_model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            simple_optimizer.step()
            
            probs = torch.sigmoid(logits)
            epoch_loss += loss.item()
            epoch_var += probs.std().item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        avg_var = epoch_var / batch_count
        simple_losses.append(avg_loss)
        simple_spatial_vars.append(avg_var)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Spatial_Var={avg_var:.6f}")

    print(f"\nSimple model results:")
    print(f"  Loss improvement: {simple_losses[0]:.4f} → {simple_losses[-1]:.4f}")
    print(f"  Final spatial variance: {simple_spatial_vars[-1]:.6f}")

    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_targets = get_all_targets(eval_loader)

    # Evaluate
    simple_model.eval()
    with torch.no_grad():
        simple_probs = []
        for batch_X, batch_y in eval_loader:
            batch_X = batch_X.to(device)
            logits = simple_model(batch_X)
            probs = torch.sigmoid(logits)
            simple_probs.append(probs.cpu().numpy())
        
        simple_probs = np.concatenate(simple_probs).flatten()
        simple_preds = (simple_probs > 0.5).astype(float)
        simple_dice = 2 * (simple_preds * all_targets).sum() / (simple_preds.sum() + all_targets.sum() + 1e-8)
        
        print(f"  Dice coefficient: {simple_dice:.4f}")
        print(f"  Prob range: [{simple_probs.min():.4f}, {simple_probs.max():.4f}]")
        print(f"  Prob std: {simple_probs.std():.6f}")

    if simple_dice > 0.5:
        print(f"\n✓ PROBLEM IS LEARNABLE! Simple model achieved Dice={simple_dice:.4f}")
        print(f"  →UNet might be overparameterized or poorly initialized for this task")
    else:
        print(f"\n✗ Problem is NOT learnable with simple model (Dice={simple_dice:.4f})")
        print(f"  → Issue is with PROBLEM SETUP, not the model architecture")

    return simple_probs, simple_probs, simple_dice

####################
# UNET
####################
def setup_unet_training(model, device, learning_rate=1e-3, weight_decay=1e-5, 
                       apply_sigmoid=True, scheduler_factor=0.5, scheduler_patience=10):
    """
    Setup loss function, optimizer, and learning rate scheduler for UNET training.
    
    Args:
        model: PyTorch UNET model
        device: 'cpu' or 'cuda'
        learning_rate: Adam learning rate
        weight_decay: L2 regularization strength
        apply_sigmoid: Whether loss should apply sigmoid (True for logits, False for probabilities)
        scheduler_factor: Multiplicative factor to reduce LR
        scheduler_patience: Epochs to wait before reducing LR
        
    Returns:
        loss_fn: DiceLoss instance
        optimizer: Adam optimizer
        scheduler: ReduceLROnPlateau scheduler
    """
    loss_fn = DiceLoss(smooth=1.0, apply_sigmoid=apply_sigmoid)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
    )
    return loss_fn, optimizer, scheduler

def train_unet_model(model, train_loader, eval_loader, device, epochs, 
                     optimizer, scheduler, loss_fn, model_name="UNET"):
    """
    Complete training loop for UNET model with validation.
    
    Args:
        model: UNET model
        train_loader: DataLoader for training data
        eval_loader: DataLoader for validation data
        device: 'cpu' or 'cuda'
        epochs: Number of training epochs
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        model_name: Name for printing progress
        
    Returns:
        train_losses: List of training losses per epoch
        eval_losses: List of validation losses per epoch
        best_loss: Best validation loss achieved
    """
    train_losses = []
    eval_losses = []
    best_loss = float('inf')
    
    print(f"Training {model_name} for {epochs} epochs...")
    print("-" * 70)
    
    for epoch in range(epochs):
        # Training
        avg_train_loss = train_epoch(model, train_loader, device, optimizer, loss_fn)
        train_losses.append(avg_train_loss)
        
        # Evaluation
        avg_eval_loss = evaluate_epoch(model, eval_loader, device, loss_fn)
        eval_losses.append(avg_eval_loss)
        
        # Update learning rate
        scheduler.step(avg_eval_loss)
        
        # Track best loss
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Train: {avg_train_loss:.6f} | Eval: {avg_eval_loss:.6f} | Best: {best_loss:.6f}")
    
    print("-" * 70)
    print(f"Training complete! Final loss: {eval_losses[-1]:.6f}, Best loss: {best_loss:.6f}")
    
    return train_losses, eval_losses, best_loss

def evaluate_unet_predictions(model, eval_loader, device, probability_threshold=0.5):
    """
    Generate predictions and probabilities on evaluation set.
    
    Args:
        model: UNET model in eval mode
        eval_loader: DataLoader for evaluation data
        device: 'cpu' or 'cuda'
        probability_threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        all_preds: Flattened binary predictions
        all_targets: Flattened ground truth targets
        all_probs: Flattened probability outputs
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in eval_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            preds = (probs > probability_threshold).float()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_probs = np.concatenate(all_probs).flatten()
    
    return all_preds, all_targets, all_probs

# CUSTOM UNET
class UNet(nn.Module):
    """
    U-Net architecture with encoder-decoder path and skip connections.
    Designed for image segmentation tasks with spatial preservation.

    Using this architecture as reference (https://github.com/milesial/pytorch-unet)
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        
        features = init_features
        
        # ENCODER - Downsampling path with skip connections
        self.encoder1 = self._conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._conv_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # BOTTLENECK
        self.bottleneck = self._conv_block(features * 8, features * 16)
        
        # DECODER - Upsampling path with skip connections
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._conv_block(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(features * 2, features)
        
        # FINAL OUTPUT
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
        # Initialize weights properly
        self._init_weights()
    
    def _conv_block(self, in_ch, out_ch):
        """Double convolution block - standard U-Net building block"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        """He initialization for Conv2d layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # ENCODER - Store skip connections
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool3(enc3)
        
        enc4 = self.encoder4(x)
        x = self.pool4(enc4)
        
        # BOTTLENECK
        x = self.bottleneck(x)
        
        # DECODER - Concatenate skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        
        # Final output layer
        x = self.final_conv(x)
        
        return x
    
# LIBRARY ALTERED UNET (worse performance on simple examples). 
# architecture is *nearly* identical to custom unit)
class LibraryUNetMultiBand(nn.Module):
    """
    Fixed Library UNet wrapper for multi-band input.
    - Accepts variable input channels (3 for RGB, 10-50 for future multi-spectral)
    - Uses library UNet architecture
    - Applies logit transformation to bypass model's internal sigmoid
    """
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()
        self.in_channels = in_channels
        self.unet = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=in_channels,  # ← Key change: Accept multiple input channels
            out_channels=out_channels,
            init_features=init_features,
            pretrained=False
        )
    
    def forward(self, x):
        """
        Forward pass with logit transformation.
        Library UNet applies sigmoid, so we invert it to get logits.
        """
        output = self.unet(x)  # Already sigmoid'd between [0, 1]
        
        # Logit transformation: inverse of sigmoid
        # logit(p) = log(p / (1-p))
        # Clamp to avoid log(0) and log(inf)
        output_clamped = torch.clamp(output, 1e-7, 1 - 1e-7)
        logits = torch.log(output_clamped / (1 - output_clamped))
        
        return logits

####################
# HELPERS
####################

def create_dataset(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32) / 255.0
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataset, train_loader

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, apply_sigmoid=True):
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
    
    def forward(self, logits, targets):
        # Only apply sigmoid if the input is logits (not already probabilities)
        if self.apply_sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = logits  # Input is already probabilities from library UNet
        
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = 2 * intersection / (union + self.smooth)
        return 1 - dice

def get_all_targets(eval_loader):
    with torch.no_grad():
        all_targets = []
        
        for batch_X, batch_y in eval_loader:
            all_targets.append(batch_y.cpu().numpy())
        
        all_targets = np.concatenate(all_targets).flatten()

    return all_targets

def train_epoch(model, train_loader, device, optimizer, loss_fn):
    """
    Train model for one epoch.
    
    Args:
        model: UNET model in training mode
        train_loader: DataLoader for training data
        device: 'cpu' or 'cuda'
        optimizer: Optimizer instance
        loss_fn: Loss function
        
    Returns:
        avg_loss: Average loss over all batches
    """
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
    
    return epoch_loss / batch_count if batch_count > 0 else 0.0

def evaluate_epoch(model, eval_loader, device, loss_fn):
    """
    Evaluate model on validation set for one epoch.
    
    Args:
        model: UNET model
        eval_loader: DataLoader for evaluation data
        device: 'cpu' or 'cuda'
        loss_fn: Loss function
        
    Returns:
        avg_loss: Average loss over all batches
    """
    model.eval()
    eval_loss = 0.0
    eval_count = 0
    
    with torch.no_grad():
        for batch_X, batch_y in eval_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            eval_loss += loss.item()
            eval_count += 1
    
    return eval_loss / eval_count if eval_count > 0 else 0.0

def calculate_unet_metrics(preds, targets, probs):
    """
    Calculate segmentation metrics.
    
    Args:
        preds: Binary predictions (0 or 1)
        targets: Ground truth binary labels
        probs: Probability outputs
        
    Returns:
        metrics: Dictionary with 'accuracy', 'iou', 'dice'
    """
    accuracy = np.mean(preds == targets)
    iou = jaccard_score(targets, preds, average='binary')
    dice = 2 * (preds * targets).sum() / (preds.sum() + targets.sum() + 1e-8)
    
    metrics = {
        'accuracy': accuracy,
        'iou': iou,
        'dice': dice,
        'prob_min': probs.min(),
        'prob_max': probs.max(),
        'prob_mean': probs.mean(),
        'prob_std': probs.std()
    }
    
    return metrics

def print_unet_metrics(model_name, metrics, loss=None):
    """
    Pretty print UNET evaluation metrics.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary from calculate_unet_metrics()
        loss: Optional best loss value
    """
    print("\n" + "="*70)
    print(f"{model_name.upper()} EVALUATION")
    print("="*70)
    
    print(f"\nPixel Accuracy: {metrics['accuracy']:.4f}")
    print(f"IoU (Jaccard):  {metrics['iou']:.4f}")
    print(f"Dice Coefficient: {metrics['dice']:.4f}")
    
    if loss is not None:
        print(f"Best Loss: {loss:.6f}")
    
    print(f"\nProbability distribution:")
    print(f"  Min: {metrics['prob_min']:.4f}, Max: {metrics['prob_max']:.4f}")
    print(f"  Mean: {metrics['prob_mean']:.4f}, Std: {metrics['prob_std']:.4f}")

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the folder containing original images.
            mask_dir (str): Path to the folder containing mask images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
        assert len(self.images) == len(self.masks), "Mismatch between number of images and masks!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        
        image = to_tensor(image)
        mask = to_tensor(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

####################
# FEATURE IMPORTANCE
####################

def feature_permutation_pipeline(X, y, model_func, num_features=None, perturbation_type="logistic"):
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
    
    # Train base model
    if perturbation_type == "logistic":
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
        logreg = train_log_reg(X_train, y_train)
        
        # Define forward function that returns error metric (negative accuracy for minimization)
        def forward_func(inputs):
            preds = logreg.predict(inputs.numpy())
            preds = np.round(preds)
            # Return loss (1 - accuracy) for each sample
            accuracy = (preds == y_test).astype(float)
            return torch.tensor(1.0 - accuracy, dtype=torch.float32)
        
        # Convert test data to tensor
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
    else:  # CNN/ResNet
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
        model = model_resnet18(X_train, y_train)
        
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
    
    # Create FeaturePermutation interpreter
    feature_perm = FeaturePermutation(forward_func)
    
    # Compute attributions
    print("Computing feature permutation importance...")
    attributions = feature_perm.attribute(X_test_tensor, perturbations_per_eval=1, show_progress=True)
    
    # Flatten and average attributions across samples
    attr_flat = attributions.numpy().reshape(attributions.shape[0], -1)
    feature_importance = np.abs(attr_flat).mean(axis=0)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_score': feature_importance
    }).sort_values('importance_score', ascending=False).reset_index(drop=True)
    
    # Limit to num_features if specified
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

    # Convert to DataFrame if needed
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
    
    # Determine number of features to select
    if num_features is None:
        num_features = max(1, min(20, len(feature_names) // 100))
    
    # Select features using MRMR
    print(f"Computing MRMR feature selection (selecting {num_features} features from {len(feature_names)})...")
    
    if task_type == "classif":
        selected_features = mrmr_classif(X=X_df, y=y_series, K=num_features, n_jobs=1)
    else:
        selected_features = mrmr_regression(X=X_df, y=y_series, K=num_features, n_jobs=1)
    
    # Create ranking DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': selected_features,
        'rank': range(1, len(selected_features) + 1)
    })
    
    return selected_features, feature_importance_df

def combine_feature_importance_methods(perm_df, mrmr_features, dataset_name="Dataset"):
    """
    Combine results from MRMR and Feature Permutation to get consensus features.
    Returns features that have EITHER:
    1. Non-zero permutation importance, OR
    2. In the top 10% of MRMR features (by count)
       - BUT remove features at the boundary rank that have zero importance
    
    Args:
        perm_df: DataFrame from feature_permutation_pipeline with columns ['feature', 'importance_score']
        mrmr_features: List of MRMR selected feature names (ordered by rank)
        dataset_name: Name of the dataset for reporting
        
    Returns:
        consensus_df: DataFrame with consensus features ranked
    """
    
    # Get features with non-zero importance from permutation
    perm_features_valid = perm_df[perm_df['importance_score'] > 0].copy()
    perm_features_valid['perm_rank'] = range(1, len(perm_features_valid) + 1)
    
    non_zero_features = set(perm_features_valid['feature'].tolist())
    
    # Get top 10% of MRMR features (by count)
    top_10_pct_count = max(1, int(np.ceil(len(mrmr_features) * 0.10)))
    top_10_mrmr = set(mrmr_features[:top_10_pct_count])
    
    # Find the worst (highest) MRMR rank among top 10%
    worst_rank_in_top10 = top_10_pct_count  # rank = position in list (1-indexed)
    
    # Union: features with non-zero importance OR in top 10% of MRMR
    consensus_features = non_zero_features.union(top_10_mrmr)
    
    # Find features with the worst rank in top 10%
    features_with_worst_rank = set()
    for i, feature in enumerate(mrmr_features):
        if i + 1 == worst_rank_in_top10:  # Convert to 1-indexed rank
            features_with_worst_rank.add(feature)
    
    # Remove those with worst rank that have zero importance
    for f in features_with_worst_rank:
        if f not in non_zero_features:
            consensus_features.discard(f)
    
    if len(consensus_features) == 0:
        print(f"\nWarning: No features found.")
        return pd.DataFrame()
    
    # Create consensus DataFrame with ranks from both methods
    consensus_data = []
    for feature in consensus_features:
        # Get permutation info
        perm_data = perm_features_valid[perm_features_valid['feature'] == feature]
        if len(perm_data) > 0:
            perm_rank = perm_data['perm_rank'].values[0]
            perm_score = perm_data['importance_score'].values[0]
        else:
            perm_rank = len(perm_features_valid) + 1
            perm_score = 0
        
        # Get MRMR rank
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
    
    print(f"\n✓ Selected {len(consensus_features)} consensus features:")
    print(f"  - {len(non_zero_features)} features with non-zero permutation importance")
    print(f"  - {len(top_10_mrmr)} features in top 10% of MRMR")
    if len(features_with_worst_rank) > 0:
        removed_from_boundary = len([f for f in features_with_worst_rank if f not in non_zero_features])
        print(f"  - Removed {removed_from_boundary} zero-importance features from boundary rank")
    print(f"  - Union: {len(consensus_features)} total features")
    
    return consensus_df

def extract_features_logreg(X_train, y_train, model_func, n_mrmr_features=20, perturbation_type="logistic"):
    """
    Complete feature extraction and model training pipeline.
    
    This function performs:
    1. Feature Permutation analysis using Captum
    2. MRMR feature selection
    3. Consensus feature combination (union of non-zero importance + top 10% MRMR)
    4. Model training on selected features
    
    Args:
        X_train: Training feature matrix (n_samples, n_features)
        y_train: Training labels (n_samples,)
        model_func: Model training function (e.g., model_logistic_regression)
        n_mrmr_features: Number of features to select via MRMR (default: 20)
        perturbation_type: "logistic" or "cnn" (default: "logistic")
    
    Returns:
        results_dict: Dictionary containing:
            - 'model': Trained model on selected features
            - 'selected_features': List of selected feature names
            - 'selected_indices': List of selected feature indices
            - 'n_original_features': Original number of features
            - 'n_selected_features': Number of selected features
            - 'reduction_percentage': Feature reduction as percentage
            - 'permutation_features': Features with non-zero permutation importance
            - 'mrmr_features': Features selected by MRMR
            - 'consensus_df': DataFrame with consensus feature details
    """
    
    print("=" * 80)
    print("FEATURE EXTRACTION AND MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    n_original_features = X_train.shape[1]
    print(f"\nStarting with {n_original_features} features")
    
    # ========================================================================
    # STEP 1: FEATURE PERMUTATION
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 1: Computing Feature Permutation Importance")
    print("-" * 80)
    
    perm_features, _ = feature_permutation_pipeline(
        X_train, y_train,
        model_func,
        num_features=None,  # Get all with non-zero importance
        perturbation_type=perturbation_type
    )
    
    n_perm_features = len(perm_features)
    print(f"✓ Found {n_perm_features} features with non-zero importance")
    
    # ========================================================================
    # STEP 2: MRMR FEATURE SELECTION
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 2: Running MRMR Feature Selection")
    print("-" * 80)
    
    mrmr_features, _ = mrmr_pipeline(
        X_train, y_train,
        num_features=n_mrmr_features,
        task_type="classif"
    )
    
    print(f"✓ Selected {len(mrmr_features)} features via MRMR")
    
    # ========================================================================
    # STEP 3: CONSENSUS FEATURE COMBINATION
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 3: Combining Methods via Consensus")
    print("-" * 80)
    
    consensus_features = combine_feature_importance_methods(
        perm_features,
        mrmr_features,
        dataset_name="Training Data"
    )
    
    n_selected_features = len(consensus_features)
    reduction_pct = (1 - n_selected_features / n_original_features) * 100
    
    print(f"✓ Consensus selected {n_selected_features} features ({reduction_pct:.2f}% reduction)")
    
    # ========================================================================
    # STEP 4: MODEL TRAINING
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 4: Training Model on Selected Features")
    print("-" * 80)
    
    # Get feature indices
    selected_feature_names = consensus_features['feature'].tolist()
    selected_indices = [int(f.split('_')[1]) for f in selected_feature_names]
    
    # Select features
    X_train_selected = X_train[:, selected_indices]
    
    # Train model
    print(f"Training model on {n_selected_features} selected features...")
    train_metrics, test_metrics = model_func(X_train_selected, y_train)
    
    print(f"✓ Model trained successfully")
    print(f"  - Train accuracy: {train_metrics['overall']:.4f}")
    print(f"  - Test accuracy: {test_metrics['overall']:.4f}")
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Original features: {n_original_features}")
    print(f"  - Selected features: {n_selected_features}")
    print(f"  - Reduction: {reduction_pct:.2f}%")
    print(f"  - Test accuracy: {test_metrics['overall']:.4f}")
    
    # Return results
    results_dict = {
        'model': model_func,  # Return function since model_func returns metrics, not model
        'selected_features': selected_feature_names,
        'selected_indices': selected_indices,
        'n_original_features': n_original_features,
        'n_selected_features': n_selected_features,
        'reduction_percentage': reduction_pct,
        'permutation_features': perm_features['feature'].tolist(),
        'mrmr_features': mrmr_features,
        'consensus_df': consensus_features,
        'test_metrics': test_metrics,
        'train_metrics': train_metrics
    }
    
    return results_dict

def create_cnn_wrapper(original_shape, selected_indices):
    """
    Factory function to create SimpleCNN wrapper that uses zero-masking.
    Keeps full image architecture but zeros out non-selected features.
    
    Args:
        original_shape: tuple (C, H, W) of original images
        selected_indices: array of selected feature indices
    """
    def model_simple_cnn_masked(X_selected, y):
        """
        SimpleCNN wrapper that reconstructs full images from selected features.
        Non-selected features are set to 0 (masking).
        X_selected: (N, n_selected_features) - values for selected pixels only
        """
        # Reconstruct full images with zero-masking
        n_samples = X_selected.shape[0]
        C, H, W = original_shape
        X_masked = np.zeros((n_samples, C, H, W), dtype=np.float32)
        
        # Fill in the selected features
        X_masked_flat = X_masked.reshape(n_samples, -1)
        X_masked_flat[:, selected_indices] = X_selected
        X_masked = X_masked_flat.reshape(n_samples, C, H, W)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_masked, y, train_size=0.8, random_state=42
        )
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = SimpleCNN(num_classes=2)
        model = model.to(device)
        model.train()
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 20
        
        print(f"  Training SimpleCNN on {device} with {len(selected_indices)} selected features (others zeroed)...")
        for epoch in range(epochs):
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_preds = []
            for batch_X, batch_y in torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False):
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                preds = logits.argmax(dim=1)
                train_preds.append(preds.cpu().numpy())
            train_preds = np.concatenate(train_preds)
            
            test_preds = []
            for batch_X, batch_y in torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False):
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                preds = logits.argmax(dim=1)
                test_preds.append(preds.cpu().numpy())
            test_preds = np.concatenate(test_preds)
        
        train_metrics = {
            "overall": accuracy_score(y_train, train_preds),
            "f1": f1_score(y_train, train_preds, average="binary", zero_division=0),
            "recall": recall_score(y_train, train_preds, average="binary", zero_division=0),
            "precision": precision_score(y_train, train_preds, average="binary", zero_division=0)
        }
        test_metrics = {
            "overall": accuracy_score(y_test, test_preds),
            "f1": f1_score(y_test, test_preds, average="binary", zero_division=0),
            "recall": recall_score(y_test, test_preds, average="binary", zero_division=0),
            "precision": precision_score(y_test, test_preds, average="binary", zero_division=0)
        }
        
        return train_metrics, test_metrics
    
    return model_simple_cnn_masked


def create_unet_wrapper(original_shape, selected_indices):
    """
    Factory function to create UNET wrapper that uses zero-masking.
    Keeps full image architecture but zeros out non-selected features.
    
    Args:
        original_shape: tuple (C, H, W) of original images
        selected_indices: array of selected feature indices
    """
    def model_unet_masked(X_selected, y):
        """
        UNET wrapper that reconstructs full images from selected features.
        Non-selected features are set to 0 (masking).
        X_selected: (N, n_selected_features) - values for selected pixels only
        """
        # Reconstruct full images with zero-masking
        n_samples = X_selected.shape[0]
        C, H, W = original_shape
        X_masked = np.zeros((n_samples, C, H, W), dtype=np.float32)
        
        # Fill in the selected features
        X_masked_flat = X_masked.reshape(n_samples, -1)
        X_masked_flat[:, selected_indices] = X_selected
        X_masked = X_masked_flat.reshape(n_samples, C, H, W)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_masked, y, train_size=0.8, random_state=42
        )
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = UNet(in_channels=1, out_channels=2)
        model = model.to(device)
        model.train()
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 15
        
        print(f"  Training UNET on {device} with {len(selected_indices)} selected features (others zeroed)...")
        for epoch in range(epochs):
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_X)
                
                # Global average pooling to convert segmentation output to classification
                logits_pooled = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
                loss = criterion(logits_pooled, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_preds = []
            for batch_X, batch_y in torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False):
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                logits_pooled = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
                preds = logits_pooled.argmax(dim=1)
                train_preds.append(preds.cpu().numpy())
            train_preds = np.concatenate(train_preds)
            
            test_preds = []
            for batch_X, batch_y in torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False):
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                logits_pooled = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
                preds = logits_pooled.argmax(dim=1)
                test_preds.append(preds.cpu().numpy())
            test_preds = np.concatenate(test_preds)
        
        train_metrics = {
            "overall": accuracy_score(y_train, train_preds),
            "f1": f1_score(y_train, train_preds, average="binary", zero_division=0),
            "recall": recall_score(y_train, train_preds, average="binary", zero_division=0),
            "precision": precision_score(y_train, train_preds, average="binary", zero_division=0)
        }
        test_metrics = {
            "overall": accuracy_score(y_test, test_preds),
            "f1": f1_score(y_test, test_preds, average="binary", zero_division=0),
            "recall": recall_score(y_test, test_preds, average="binary", zero_division=0),
            "precision": precision_score(y_test, test_preds, average="binary", zero_division=0)
        }
        
        return train_metrics, test_metrics
    
    return model_unet_masked