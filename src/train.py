from datetime import datetime
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from src import config
import time
from torch.utils.tensorboard import SummaryWriter


def calculate_accuracy(y_pred, y_true):
    preds = torch.argmax(y_pred, dim=1)
    correct = (preds == y_true).sum().item()
    return correct / len(y_true)


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    batch_count = len(dataloader)

    logging.info(f"Training on {batch_count} batches")
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx % 10 == 0:
            logging.info(f"  Batch {batch_idx}/{batch_count}")

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.GRAD_CLIP_VALUE)

        optimizer.step()

        running_loss += loss.item()
        running_acc += acc

    return running_loss / len(dataloader), running_acc / len(dataloader)


def train_model(model, train_loader, val_loader, epochs=config.EPOCHS, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, device=config.DEVICE):

    log_file = setup_logging(config.LOG_DIR)
    logging.info(f"Training logs will be saved to: {log_file}")

    logging.info(f"Training configuration:")
    logging.info(f"  Epochs: {epochs}")
    logging.info(f"  Learning rate: {lr}")
    logging.info(f"  Weight decay: {weight_decay}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Batch size: {config.BATCH_SIZE}")
    logging.info(f"  Image size: {config.IMAGE_SIZE}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        verbose=True
    )

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    run_name = time.strftime("run_%Y%m%d-%H%M")
    log_dir = f"{config.LOG_DIR}/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)

    logging.info(f"Training on: {device.upper()}\n")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        logging.info(f"Epoch {epoch+1}/{epochs} started")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        logging.info("Validating...")
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start_time

        scheduler.step(val_acc)

        logging.info(
            f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
        logging.info(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        logging.info(
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            logging.info("Model saved!")

    writer.close()
    logging.info("Training complete. Best Val Acc: {:.2f}%".format(
        best_val_acc * 100))

    return best_val_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss, val_acc = 0.0, 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            val_loss += loss.item()
            val_acc += acc

    return val_loss / len(dataloader), val_acc / len(dataloader)
