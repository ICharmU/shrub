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
import os

####################
# BASELINES
####################

# LOG REG
def model_logistic_regression(X, y):
    def train_log_reg(X_train, y_train):
        """
        log reg assumes flattened inputs
        """
        model = LogisticRegression(max_iter=250)
        model.fit(X_train, y_train)

        return model
        
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


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    logreg = train_log_reg(X_train, y_train)

    train_metrics = log_reg_accuracy(logreg, X_train, y_train, is_binary=False)
    test_metrics = log_reg_accuracy(logreg, X_test, y_test, is_binary=False)
    print(f"Train metrics:\n{train_metrics}")
    print(f"Test metrics:\n{test_metrics}")

    return train_metrics, test_metrics

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
    
# LIBRARY ALTERED UNET (worse performance on simple examples. 
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
