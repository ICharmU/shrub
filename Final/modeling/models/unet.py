import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, jaccard_score
import torch.nn as nn
from Final.modeling.setup_helper import train_epoch, eval_epoch

class CustomUNet(nn.Module):
    """
    U-Net architecture with encoder-decoder path and skip connections.
    Designed for image segmentation tasks with spatial preservation.

    Using this architecture as reference (https://github.com/milesial/pytorch-unet)
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(CustomUNet, self).__init__()
        
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

class LibraryUNet(nn.Module):
    """
    in testing LibraryUNet was worse than the CustomUNet, but we should try both with real data.
    """
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()
        self.in_channels = in_channels
        self.unet = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=in_channels, 
            out_channels=out_channels,
            init_features=init_features,
            pretrained=False
        )
    
    def forward(self, x):
        """
        model outputs probabilities instead of logits
        """
        output = self.unet(x)
        output_clamped = torch.clamp(output, 1e-7, 1 - 1e-7)
        logits = torch.log(output_clamped / (1 - output_clamped))
        
        return logits

def setup_unet_training(model, device, learning_rate=1e-3, weight_decay=1e-5, 
                       apply_sigmoid=True, scheduler_factor=0.5, scheduler_patience=10):
    loss_fn = DiceLoss(smooth=1.0, apply_sigmoid=apply_sigmoid)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
    )
    return loss_fn, optimizer, scheduler


def train_unet_model(model, train_loader, device, epochs, 
                     loss_fn, optimizer, scheduler):
    train_losses = []
    eval_losses = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        avg_train_loss = train_epoch(model, train_loader, device, optimizer, loss_fn)
        train_losses.append(avg_train_loss)
        
        avg_eval_loss = eval_epoch(model, eval_loader, device, loss_fn)
        eval_losses.append(avg_eval_loss)
        
        scheduler.step(avg_eval_loss)
        
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss

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

def calculate_unet_metrics(preds, targets, probs):
    """
    Args:
        preds: Binary predictions (0 or 1)
        targets: Ground truth binary labels
        probs: Probability outputs
    """
    accuracy = np.mean(preds == targets)
    jaccard = jaccard_score(targets, preds, average='binary')
    dice = 2 * (preds * targets).sum() / (preds.sum() + targets.sum() + 1e-8)
    
    metrics = {
        'accuracy': accuracy,
        'jaccard': jaccard,
        'dice': dice,
        'prob_min': probs.min(),
        'prob_max': probs.max(),
        'prob_mean': probs.mean(),
        'prob_std': probs.std()
    }
    
    return metrics

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
        n_samples = X_selected.shape[0]
        C, H, W = original_shape
        X_masked = np.zeros((n_samples, C, H, W), dtype=np.float32)
        
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
        
        model = CustomUNet(in_channels=1, out_channels=2)
        model = model.to(device)
        model.train()
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 15
        
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

# feature importance general pipeline structure. feature importance implementation section specified
def pipeline_unet(X_train, y_train, X_test, y_test, in_channels=3, out_channels=1, 
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

    X_shape = X_train.shape
    batch_size, channels, height, width = X_shape
    n_pixels = height * width

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=channels, out_channels=out_channels, init_features=init_features).to(device)
    
    train_dataset, train_loader = create_dataset(X_train, y_train)
    test_dataset, test_loader = create_dataset(X_test, y_test, shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss()  # For binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    test_losses = []

    ########
    # FEATURE IMPORTANCE INTEGRATED DURING TRAINING
    ########
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, device, optimizer, criterion)
        train_losses.append(train_loss)
        
        test_loss = eval_epoch(model, train_loader, device, loss_fn)
        test_losses.append(test_loss)
        
        scheduler.step(test_loss)

    model.eval()
    with torch.no_grad():
        train_logits = list()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            curr_logits = model(batch_X)
            train_logits.append(curr_logits)
            
        train_logits = torch.cat(train_logits)  
        train_probs = torch.sigmoid(train_logits).squeeze(1) 
        train_preds = (train_probs >= 0.5).long().cpu().numpy() 
    
    train_accuracy = np.mean(train_preds == y_train)
    train_metrics = {
        'accuracy': float(train_accuracy),
        'shape': train_preds.shape
    }

    with torch.no_grad():
        test_logits = list()
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_preds = model(batch_X)
            test_logits.append(batch_preds)
            
        test_logits = torch.cat(test_logits)
        test_probs = torch.sigmoid(test_logits).squeeze(1) 
        test_preds = (test_probs >= 0.5).long().cpu().numpy()

    # might need to chunk. not sure how large arrays are going to be compared to memory
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
        'train_losses': train_losses,
        'test_losses': test_losses,
    }
    
    return results_dict