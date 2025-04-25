import os
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src import config

def get_transforms(image_size=config.IMAGE_SIZE):
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(config.DATA_AUG_ROTATION),
        transforms.ColorJitter(
            brightness=config.DATA_AUG_COLOR_JITTER, 
            contrast=config.DATA_AUG_COLOR_JITTER, 
            saturation=config.DATA_AUG_COLOR_JITTER, 
            hue=config.DATA_AUG_COLOR_JITTER
        ),
        transforms.RandomAffine(degrees=0, translate=(config.DATA_AUG_TRANSLATE, config.DATA_AUG_TRANSLATE)),
        transforms.RandomResizedCrop(image_size, scale=config.DATA_AUG_SCALE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    return train_transforms, val_test_transforms

def get_dataloaders(data_dir, batch_size=config.BATCH_SIZE, image_size=config.IMAGE_SIZE, num_workers=config.NUM_WORKERS):
    train_transforms, val_test_transforms = get_transforms(image_size)

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    logging.info(f"Loading datasets from: {data_dir}")
    logging.info(f"Train directory: {train_dir}")
    logging.info(f"Validation directory: {val_dir}")
    logging.info(f"Test directory: {test_dir}")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    logging.info(f"Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names
