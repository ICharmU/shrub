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
class SimpleCNN(nn.Module):
    """
    Simple CNN supporting both image-level and per-pixel (segmentation) tasks.
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 2)
        output_type: 'image' for image-level classification or 'pixel' for per-pixel segmentation (default: 'image')
    """
    def __init__(self, in_channels=3, num_classes=2, output_type='pixel'):
        super(SimpleCNN, self).__init__()
        self.output_type = output_type
        self.num_classes = num_classes
        
        if output_type == 'pixel':
            # Per-pixel segmentation: preserve spatial dimensions
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()
            
            # Output layer: produce per-pixel predictions
            self.conv_out = nn.Conv2d(128, num_classes, kernel_size=1)
        else:
            # Image-level classification: downsample spatially
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
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
        if self.output_type == 'pixel':
            # Per-pixel path: no pooling, preserve spatial dimensions
            x = self.conv1(x)
            x = self.relu1(x)
            
            x = self.conv2(x)
            x = self.relu2(x)
            
            x = self.conv3(x)
            x = self.relu3(x)
            
            # Output: (batch, num_classes, H, W)
            x = self.conv_out(x)
            return x
        else:
            # Image-level path: downsample and flatten
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

def feature_permutation_pipeline(X, y, model_func, num_features=None, perturbation_type="logistic", model_=None, verbose=True):
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
        
    else:  # CNN
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
    
    # Create FeaturePermutation interpreter
    feature_perm = FeaturePermutation(forward_func)
    
    # Compute attributions
    print("Computing feature permutation importance...")
    attributions = feature_perm.attribute(X_test_tensor, perturbations_per_eval=1, show_progress=verbose)
    
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

def extract_features_logreg(X_train, y_train, n_mrmr_features=20, is_binary=True):
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
    
    print("=" * 80)
    print("FEATURE EXTRACTION WITH WEIGHT ZEROING")
    print("=" * 80)
    
    # Store original shape
    X_shape = X_train.shape
    
    # Flatten: (batch, channels, H, W) -> (batch, channels*H*W)
    # Each pixel (across all channels) is a feature
    if len(X_train.shape) == 4:
        batch_size = X_train.shape[0]
        X_train_flat = X_train.reshape(batch_size, -1)
        print(f"\nFlattened input from {X_shape} to {X_train_flat.shape}")
        print(f"  - Each pixel = {X_shape[1]} features (one per channel)")
        print(f"  - Total pixels = {X_shape[2]} x {X_shape[3]} = {X_shape[2]*X_shape[3]}")
        print(f"  - Total features = {X_shape[2]*X_shape[3]} pixels × {X_shape[1]} channels")
    else:
        X_train_flat = X_train.copy()
    
    n_original_features = X_train_flat.shape[1]
    print(f"Working with {n_original_features} features")
    
    # Cap n_mrmr_features to not exceed number of features
    n_mrmr_features_actual = min(n_mrmr_features, n_original_features)
    if n_mrmr_features_actual < n_mrmr_features:
        print(f"⚠ Capped MRMR selection from {n_mrmr_features} to {n_mrmr_features_actual} (limited by feature count)")
    
    # ========================================================================
    # STEP 1: FEATURE PERMUTATION
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 1: Computing Feature Permutation Importance")
    print("-" * 80)
    
    perm_features, _ = feature_permutation_pipeline(
        X_train_flat, y_train,
        model_func=None,
        num_features=None,  # Get all with non-zero importance
        perturbation_type="logistic"
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
        X_train_flat, y_train,
        num_features=n_mrmr_features_actual,
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
    
    # Handle case where no consensus features found
    if len(consensus_features) == 0:
        print("\n⚠ WARNING: No consensus features selected!")
        print("  Falling back to all features with non-zero importance...")
        if len(perm_features) > 0:
            consensus_features = perm_features.copy()
        else:
            print("  No features with non-zero importance either. Using all features.")
            consensus_features = perm_features  # Will be empty, but we'll handle below
    
    n_selected_features = len(consensus_features)
    reduction_pct = (1 - n_selected_features / n_original_features) * 100
    
    print(f"✓ Consensus selected {n_selected_features} features ({reduction_pct:.2f}% reduction)")
    
    # ========================================================================
    # STEP 4: TRAIN MODEL ON ALL FEATURES
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 4: Training Logistic Regression on ALL Features")
    print("-" * 80)
    
    model = train_log_reg(X_train_flat, y_train)
    train_metrics = log_reg_accuracy(model, X_train_flat, y_train, is_binary=is_binary)
    
    print(f"✓ Model trained on all {n_original_features} features")
    print(f"  - Train accuracy: {train_metrics['overall']:.4f}")
    
    # ========================================================================
    # STEP 5: ZERO OUT NON-SELECTED FEATURE WEIGHTS
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 5: Zeroing Out Non-Selected Feature Weights")
    print("-" * 80)
    
    if len(consensus_features) > 0:
        # Get selected feature indices
        selected_feature_names = consensus_features['feature'].tolist()
        selected_indices = np.array([int(f.split('_')[1]) for f in selected_feature_names])
        
        # Create mask for non-selected features
        all_indices = np.arange(n_original_features)
        non_selected_mask = ~np.isin(all_indices, selected_indices)
        non_selected_indices = np.where(non_selected_mask)[0]
        
        # Zero out coefficients for non-selected features
        if hasattr(model, 'coef_'):
            model.coef_[:, non_selected_indices] = 0
            print(f"✓ Zeroed out coefficients for {len(non_selected_indices)} non-selected features")
        else:
            print("⚠ Warning: Model does not have coef_ attribute")
    else:
        print("⚠ No features selected for zeroing (using all features)")
        selected_indices = np.arange(n_original_features)
        non_selected_indices = np.array([])
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Original shape: {X_shape}")
    print(f"  - Original features: {n_original_features}")
    print(f"  - Selected features: {n_selected_features}")
    print(f"  - Reduction: {reduction_pct:.2f}%")
    print(f"  - Train accuracy: {train_metrics['overall']:.4f}")
    print(f"  - Non-zero weights: {n_selected_features}")
    print(f"  - Zeroed weights: {len(non_selected_indices)}")
    
    # Return results
    results_dict = {
        'model': model,  # Modified model with zeroed weights
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
                   epochs=50, batch_size=32, learning_rate=1e-3, **kwargs):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{model_name}] Device: {device}, Epochs: {epochs}")
        
        # Initialize model
        model = model_class(**init_kwargs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Prepare train data
        train_tensor = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        
        # Training loop
        train_losses = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}")
        
        # Evaluate on train set
        model.eval()
        with torch.no_grad():
            train_preds = model(torch.FloatTensor(X_train).to(device))
            train_preds = torch.argmax(train_preds, dim=1).cpu().numpy()
        
        train_metrics = {
            "accuracy": np.mean(train_preds == y_train),
            "loss": train_losses[-1],
            "epochs": epochs
        }
        
        # Evaluate on test set if provided
        test_metrics = None
        if X_test is not None and y_test is not None:
            with torch.no_grad():
                test_preds = model(torch.FloatTensor(X_test).to(device))
                test_preds = torch.argmax(test_preds, dim=1).cpu().numpy()
            
            test_metrics = {
                "accuracy": np.mean(test_preds == y_test),
            }
        
        # print(f"[{model_name}] Train Accuracy: {train_metrics['accuracy']:.4f}")
        # if test_metrics:
        #     print(f"[{model_name}] Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        return train_preds, test_preds, train_metrics, test_metrics
    
    return wrapper_fn

def extract_features_cnn(X_train, y_train, X_test, y_test, in_channels=4, num_classes=2, epochs=10):
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
    
    print("=" * 80)
    print("CNN PER-PIXEL FEATURE EXTRACTION AND TRAINING")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: INITIALIZE MODEL AND DEVICE
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 1: Initializing Model")
    print("-" * 80)
    
    X_shape = X_train.shape
    y_shape = y_train.shape
    batch_size, channels, height, width = X_shape
    n_pixels = height * width
    
    print(f"Input shape: {X_shape}")
    print(f"Target shape: {y_shape}")
    print(f"  - Spatial dimensions: {height} × {width}")
    print(f"  - Channels: {channels}")
    print(f"  - Pixels (features per sample): {n_pixels}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN(in_channels=channels, num_classes=num_classes, output_type='pixel')
    model.to(device)
    
    print(f"✓ Model initialized on device: {device}")
    
    # ========================================================================
    # STEP 2: PREPARE DATA
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 2: Preparing Data")
    print("-" * 80)
    
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)
    
    print(f"✓ Training data: {X_train_t.shape}, {y_train_t.shape}")
    print(f"✓ Test data: {X_test_t.shape}, {y_test_t.shape}")
    
    # ========================================================================
    # STEP 3: TRAIN MODEL
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 3: Training CNN")
    print("-" * 80)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)  # (batch, num_classes, H, W)
        loss = criterion(logits, y_train_t)  # y shape: (batch, H, W)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    
    print(f"✓ Training complete")
    
    # ========================================================================
    # STEP 4: EVALUATE ON TRAINING DATA
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 4: Evaluating on Training Data")
    print("-" * 80)
    
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train_t)  # (batch, num_classes, H, W)
        train_preds = torch.argmax(train_logits, dim=1).cpu().numpy()  # (batch, H, W)
    
    train_accuracy = np.mean(train_preds == y_train)
    train_metrics = {
        'accuracy': float(train_accuracy),
        'shape': train_preds.shape
    }
    
    print(f"✓ Train accuracy: {train_accuracy:.4f}")
    print(f"✓ Prediction shape: {train_preds.shape}")
    
    # ========================================================================
    # STEP 5: EVALUATE ON TEST DATA
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 5: Evaluating on Test Data")
    print("-" * 80)
    
    with torch.no_grad():
        test_logits = model(X_test_t)  # (batch, num_classes, H, W)
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()  # (batch, H, W)
    
    test_accuracy = np.mean(test_preds == y_test)
    test_metrics = {
        'accuracy': float(test_accuracy),
        'shape': test_preds.shape
    }
    
    print(f"✓ Test accuracy: {test_accuracy:.4f}")
    print(f"✓ Prediction shape: {test_preds.shape}")
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Input shape: {X_shape}")
    print(f"  - Spatial dimensions: {height} × {width}")
    print(f"  - Train accuracy: {train_accuracy:.4f}")
    print(f"  - Test accuracy: {test_accuracy:.4f}")
    print(f"  - Output shape: {test_preds.shape}")
    
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

def extract_features_unet(X_train, y_train, X_test, y_test, in_channels=3, out_channels=1, 
                         epochs=20, init_features=32):
    """
    Train UNet for per-pixel segmentation with feature importance extraction.
    
    Args:
        X_train: Training features (batch, channels, H, W)
        y_train: Training labels (batch, H, W) or (batch, 1, H, W) - per-pixel binary labels
        X_test: Test features (batch, channels, H, W)
        y_test: Test labels (batch, H, W) or (batch, 1, H, W)
        in_channels: Number of input channels (default: 3)
        out_channels: Number of output channels (default: 1 for binary segmentation)
        epochs: Number of training epochs (default: 20)
        init_features: Initial feature maps in UNet (default: 32)
    
    Returns:
        results_dict: Dictionary containing:
            - 'model': Trained UNet model
            - 'train_preds': Training predictions (batch, H, W) or (batch, 1, H, W)
            - 'test_preds': Test predictions (batch, H, W) or (batch, 1, H, W)
            - 'train_accuracy': Per-pixel training accuracy
            - 'test_accuracy': Per-pixel test accuracy
            - 'train_metrics': Training metrics dict
            - 'test_metrics': Test metrics dict
            - 'train_losses': Loss values per epoch
            - 'test_losses': Loss values per epoch
    """
    
    print("=" * 80)
    print("UNET PER-PIXEL SEGMENTATION WITH FEATURE EXTRACTION")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: INITIALIZE MODEL AND DEVICE
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 1: Initializing UNet Model")
    print("-" * 80)
    
    X_shape = X_train.shape
    batch_size, channels, height, width = X_shape
    n_pixels = height * width
    
    # Handle y_train shape: if (batch, 1, H, W), squeeze to (batch, H, W)
    if len(y_train.shape) == 4 and y_train.shape[1] == 1:
        y_train_2d = y_train.squeeze(1)  # (batch, H, W)
        y_test_2d = y_test.squeeze(1)
    else:
        y_train_2d = y_train
        y_test_2d = y_test
    
    print(f"Input shape: {X_shape}")
    print(f"Target shape: {y_train_2d.shape}")
    print(f"  - Spatial dimensions: {height} × {width}")
    print(f"  - Input channels: {channels}")
    print(f"  - Output channels: {out_channels}")
    print(f"  - Pixels (features per sample): {n_pixels}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=channels, out_channels=out_channels, init_features=init_features)
    model.to(device)
    
    print(f"✓ UNet model initialized on device: {device}")
    
    # ========================================================================
    # STEP 2: PREPARE DATA
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 2: Preparing Data and DataLoaders")
    print("-" * 80)
    
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train_2d).float().to(device).unsqueeze(1)  # Add channel dim for BCE
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test_2d).float().to(device).unsqueeze(1)
    
    # Create datasets and loaders
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    batch_size_loader = min(32, len(X_train_t))
    train_loader = DataLoader(train_dataset, batch_size=batch_size_loader, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_loader, shuffle=False)
    
    print(f"✓ Training data: {X_train_t.shape}, {y_train_t.shape}")
    print(f"✓ Test data: {X_test_t.shape}, {y_test_t.shape}")
    print(f"✓ Batch size: {batch_size_loader}")
    
    # ========================================================================
    # STEP 3: SETUP TRAINING
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 3: Setting Up Loss, Optimizer, and Scheduler")
    print("-" * 80)
    
    criterion = nn.BCEWithLogitsLoss()  # For binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"✓ Loss function: BCEWithLogitsLoss")
    print(f"✓ Optimizer: Adam (lr=1e-3)")
    print(f"✓ Scheduler: ReduceLROnPlateau")
    
    # ========================================================================
    # STEP 4: TRAIN MODEL
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 4: Training UNet")
    print("-" * 80)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)  # (batch, out_channels, H, W)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        scheduler.step(test_loss)
        
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    print(f"✓ Training complete")
    
    # ========================================================================
    # STEP 5: EVALUATE ON TRAINING DATA
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 5: Evaluating on Training Data")
    print("-" * 80)
    
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train_t)  # (batch, out_channels, H, W)
        train_probs = torch.sigmoid(train_logits).squeeze(1)  # (batch, H, W)
        train_preds = (train_probs >= 0.5).long().cpu().numpy()  # (batch, H, W)
    
    train_accuracy = np.mean(train_preds == y_train_2d)
    train_metrics = {
        'accuracy': float(train_accuracy),
        'shape': train_preds.shape
    }
    
    print(f"✓ Train accuracy: {train_accuracy:.4f}")
    print(f"✓ Prediction shape: {train_preds.shape}")
    
    # ========================================================================
    # STEP 6: EVALUATE ON TEST DATA
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 6: Evaluating on Test Data")
    print("-" * 80)
    
    with torch.no_grad():
        test_logits = model(X_test_t)  # (batch, out_channels, H, W)
        test_probs = torch.sigmoid(test_logits).squeeze(1)  # (batch, H, W)
        test_preds = (test_probs >= 0.5).long().cpu().numpy()  # (batch, H, W)
    
    test_accuracy = np.mean(test_preds == y_test_2d)
    test_metrics = {
        'accuracy': float(test_accuracy),
        'shape': test_preds.shape
    }
    
    print(f"✓ Test accuracy: {test_accuracy:.4f}")
    print(f"✓ Prediction shape: {test_preds.shape}")
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Input shape: {X_shape}")
    print(f"  - Spatial dimensions: {height} × {width}")
    print(f"  - Final train loss: {train_losses[-1]:.4f}")
    print(f"  - Final test loss: {test_losses[-1]:.4f}")
    print(f"  - Train accuracy: {train_accuracy:.4f}")
    print(f"  - Test accuracy: {test_accuracy:.4f}")
    print(f"  - Output shape: {test_preds.shape}")
    
    results_dict = {
        'model': model,
        'train_preds': train_preds,
        'test_preds': test_preds,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        # 'train_losses': train_losses,
        # 'test_losses': test_losses,
    }
    
    return results_dict