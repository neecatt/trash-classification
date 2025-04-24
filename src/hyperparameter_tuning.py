import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from src.model import TrashNetClassifier
from src.data_loader import get_dataloaders
from src import config


import logging
import time
from datetime import datetime
import os


def setup_tuning_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"hyperparameter_tuning_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def train_model_for_validation(model, train_loader, val_loader, lr, weight_decay, device, epochs=config.TUNING_EPOCHS):
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )
    
    best_val_acc = 0.0
    
    logging.info(f"Starting validation training with lr={lr}, weight_decay={weight_decay}")
    
    for epoch in range(epochs):

        model.train()
        running_loss, running_acc = 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx % 20 == 0:
                logging.info(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}")
                
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean()
            running_loss += loss.item()
            running_acc += acc.item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader)
        

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == labels).float().mean()
                val_loss += loss.item()
                val_acc += acc.item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        logging.info(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logging.info(f"  New best validation accuracy: {best_val_acc:.4f}")
    
    return best_val_acc

def run_hyperparameter_search():

    log_file = setup_tuning_logging(config.LOG_DIR)
    logging.info(f"Hyperparameter tuning logs will be saved to: {log_file}")
    
    device = torch.device(config.DEVICE)
    logging.info(f"Using device: {device}")
    

    logging.info("Loading datasets...")
    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.TUNING_BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS
    )
    

    learning_rates = [1e-5, 1e-4, 5e-4, 1e-3]
    weight_decays = [1e-5, 1e-4, 1e-3]
    

    num_trials = config.TUNING_TRIALS
    
    best_acc = 0.0
    best_config = {"lr": 0, "weight_decay": 0}
    
    logging.info("Starting hyperparameter search...")
    logging.info(f"Number of trials: {num_trials}")
    logging.info(f"Learning rates to try: {learning_rates}")
    logging.info(f"Weight decays to try: {weight_decays}")
    
    start_time = time.time()
    
    for trial in range(num_trials):
        trial_start = time.time()

        lr = random.choice(learning_rates)
        weight_decay = random.choice(weight_decays)
        
        logging.info(f"\nTrial {trial+1}/{num_trials}")
        logging.info(f"Testing lr={lr}, weight_decay={weight_decay}")
        

        model = TrashNetClassifier(num_classes=len(class_names))
        

        val_acc = train_model_for_validation(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=lr,
            weight_decay=weight_decay,
            device=device
        )
        
        trial_time = time.time() - trial_start
        logging.info(f"Trial {trial+1} completed in {trial_time:.2f}s")
        logging.info(f"Validation accuracy: {val_acc:.4f}")
        

        if val_acc > best_acc:
            best_acc = val_acc
            best_config = {"lr": lr, "weight_decay": weight_decay}
            logging.info(f"New best config found!")
    
    total_time = time.time() - start_time
    logging.info(f"\nHyperparameter search completed in {total_time:.2f}s")
    logging.info(f"Best config: lr={best_config['lr']}, weight_decay={best_config['weight_decay']}")
    logging.info(f"Best validation accuracy: {best_acc:.4f}")
    
    return best_config