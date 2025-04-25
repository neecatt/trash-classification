import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.model import TrashNetClassifier

def evaluate_model(model_path, test_loader, class_names, device="cpu"):
    model = TrashNetClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    return cm
