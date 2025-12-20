from fastai.vision.all import *
import gradio as gr
import json
import torch
import torchvision.transforms as T
from PIL import Image

# ===== Config =====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

# ===== Load classes =====
with open("classes.json", "r") as f:
    classes = json.load(f)

# ===== Build a minimal learner to host the model head =====
# We don’t use DataLoaders for inference; we just need the model architecture
# Set dls.c so fastai builds the correct classifier head
dls = DataLoaders.from_dsets([], [], bs=1)
dls.vocab = classes
dls.c = len(classes)

learn = vision_learner(
    dls,
    resnet50,
    pretrained=False,  # no backbone download on Spaces
    loss_func=CrossEntropyLossFlat(),
    metrics=accuracy
)

# ===== Load weights safely =====
# If your weights are at repo root:
state = torch.load("resnet50_weights.pth", map_location=DEVICE)
learn.model.load_state_dict(state)
learn.model.to(DEVICE)
learn.model.eval()

# ===== Preprocessing pipeline (ImageNet normalization) =====
# Matches what resnet50 expects and typical fastai training defaults
preproc = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ===== Prediction function (manual forward pass) =====
def classify_image(img: Image.Image):
    # Ensure RGB
    img = img.convert("RGB")
    x = preproc(img).unsqueeze(0).to(DEVICE)  # shape: [1,3,224,224]
    with torch.no_grad():
        logits = learn.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Return full dict; Gradio’s Label will show top-k
    return {classes[i]: float(probs[i]) for i in range(len(classes))}

# ===== Gradio interface =====
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Sports Image Classifier",
    description="Upload a sports image and get predictions across 100 classes."
)

if __name__ == "__main__":
    demo.launch()
