import streamlit as st
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from captum.attr import Saliency, GuidedBackprop, NoiseTunnel
from src.model import TrashNetClassifier
from src import config


@st.cache_resource
def load_model():
    model = TrashNetClassifier()
    model.load_state_dict(torch.load(
        config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval().to(config.DEVICE)
    return model


def compute_saliency_map(model, input_tensor, method):
    model.zero_grad()
    input_tensor = input_tensor.to(config.DEVICE)
    input_tensor.requires_grad_()

    # Get prediction
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()

    if method == "saliency":
        attr = Saliency(model)
        attributions = attr.attribute(input_tensor, target=pred_class)
    elif method == "smoothgrad":
        attr = NoiseTunnel(Saliency(model))
        attributions = attr.attribute(
            input_tensor, nt_type="smoothgrad", target=pred_class, nt_samples=20, stdevs=0.2)
    elif method == "guided":
        attr = GuidedBackprop(model)
        attributions = attr.attribute(input_tensor, target=pred_class)
    else:
        raise ValueError("Unsupported method")

    saliency = attributions.squeeze().abs().cpu().detach().numpy()
    saliency = np.max(saliency, axis=0)  # to grayscale

    return pred_class, confidence, saliency


def preprocess_image(uploaded_file):
    pil_image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return pil_image, transform(pil_image).unsqueeze(0)


def run_saliency(model, input_tensor):
    input_tensor = input_tensor.to(config.DEVICE)
    input_tensor.requires_grad_()

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()

    output[0, pred_class].backward()
    saliency = input_tensor.grad.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)  # convert to grayscale
    return pred_class, confidence, saliency


def get_saliency_figure(input_tensor, saliency_map):
    saliency_map -= saliency_map.min()
    saliency_map /= saliency_map.max() + 1e-10

    img_np = input_tensor.squeeze().detach().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # C,H,W → H,W,C
    img_np = (img_np * 0.5 + 0.5).clip(0, 1)

    saliency_rgb = np.stack([saliency_map]*3, axis=-1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(saliency_rgb, cmap="gray")
    axs[1].set_title("Saliency Map")
    axs[1].axis("off")

    fig.tight_layout()
    return fig


st.set_page_config(page_title="Saliency Demo", layout="centered")
st.title("🧠 Trash Classifier with Clean Saliency Visualization")
st.markdown(
    "Upload a trash image. The model will classify it and show pixel-level attention.")
method = st.radio("🧠 Select Explanation Method", [
                  "saliency", "smoothgrad", "guided"])

uploaded_file = st.file_uploader(
    "📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img, input_tensor = preprocess_image(uploaded_file)
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    with st.spinner(f"Computing {method} map..."):
        model = load_model()
        pred_class, confidence, saliency_map = compute_saliency_map(
            model, input_tensor, method)
        class_names = sorted(os.listdir(
            os.path.join(config.DATA_DIR, "train")))
        pred_label = class_names[pred_class]
        fig = get_saliency_figure(input_tensor, saliency_map)

    st.markdown(f"### 🧠 Prediction: **{pred_label}** ({confidence*100:.2f}%)")
    st.pyplot(fig)
