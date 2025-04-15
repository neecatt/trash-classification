import torch
from torchvision import transforms
from PIL import Image
import os
from src.model import TrashNetClassifier
from src import config

def load_model(model_path, device):
    model = TrashNetClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # [1, C, H, W]

def predict_image(model, image_tensor, class_names, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_index = torch.argmax(probs, dim=1).item()
        pred_label = class_names[pred_index]
        confidence = probs[0][pred_index].item()
    return pred_label, confidence

def run_inference(image_path):
    device = config.DEVICE
    class_names = sorted(os.listdir(os.path.join(config.DATA_DIR, "train")))

    model = load_model(config.MODEL_SAVE_PATH, device)
    image_tensor = preprocess_image(image_path, config.IMAGE_SIZE)
    label, confidence = predict_image(model, image_tensor, class_names, device)

    print(f"Prediction: {label} ({confidence*100:.2f}%)")
    return label, confidence

