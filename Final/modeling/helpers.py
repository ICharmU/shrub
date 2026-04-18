from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import jaccard_score

def create_dataset(X, y, is_tensor=False, unsqueeze=True, shuffle=True):
    if is_tensor:
        X_tensor, y_tensor = X, y
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
    if unsqueeze:
        y_tensor = y_tensor.unsqueeze(1) 

    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=shuffle)

    return dataset, data_loader

def train_epoch(model, train_loader, device, optimizer, loss_fn):
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
    
    return epoch_loss / batch_count if batch_count > 0 else 0.0

def eval_epoch(model, eval_loader, device, loss_fn):
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

def get_all_targets(eval_loader):
    with torch.no_grad():
        all_targets = []
        
        for batch_X, batch_y in eval_loader:
            all_targets.append(batch_y.cpu().numpy())
        
        all_targets = np.concatenate(all_targets).flatten()

    return all_targets

