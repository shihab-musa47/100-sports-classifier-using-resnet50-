from fastai.vision.all import *

# Setup data
data_path = Path('data')
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms()
)
dls = dblock.dataloaders(data_path, bs=8, num_workers=0)

# Train model
learn = vision_learner(dls, resnet50, metrics=error_rate)
learn.fine_tune(5)

# Save weights (safe for Hugging Face)
learn.save("resnet50_weights")   # creates resnet50_weights.pth
