import torch
from src.hyperparameter_tuning import run_hyperparameter_search
from src.model import TrashNetClassifier
from src.data_loader import get_dataloaders
from src.train import train_model
from src import config

if __name__ == "__main__":

    print("Starting hyperparameter search...")
    best_config = run_hyperparameter_search()
    

    print("\nTraining with best hyperparameters...")
    

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS
    )
    

    model = TrashNetClassifier(num_classes=len(class_names))
    

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        lr=best_config["lr"],
        weight_decay=best_config["weight_decay"],
        device=config.DEVICE
    )
    
    print("Training complete!")