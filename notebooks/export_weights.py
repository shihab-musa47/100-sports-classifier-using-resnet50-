from fastai.vision.all import *
import torch
from torchvision import models

# Load your fastai learner (locally only)
learn = load_learner("resnet50_py310.pkl", cpu=True)

# Create a clean torchvision resnet50
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(learn.dls.vocab))

# Copy weights correctly
model.load_state_dict(learn.model.state_dict(), strict=False)

# Save CLEAN weights
torch.save(model.state_dict(), "resnet50_weights_clean.pth")

print("Export complete!")