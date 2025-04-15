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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc = 0.0, 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc

    return running_loss / len(dataloader), running_acc / len(dataloader)

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

def train_model(model, train_loader, val_loader, epochs=config.EPOCHS, lr=config.LEARNING_RATE, device=config.DEVICE):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    
 

    run_name = time.strftime("run_%Y%m%d-%H%M")
    log_dir = f"{config.LOG_DIR}/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Training on: {device.upper()}\n")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("Model saved!\n")

    writer.close()
    print("Training complete. Best Val Acc: {:.2f}%".format(best_val_acc * 100))
