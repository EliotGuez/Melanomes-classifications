#%% IMPORT 
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt 
from model import UNET

import numpy as np
from PIL import Image

from utils import get_loaders


#%% WORKING DIRECTORY 
Working_directory = "C:\\Users\\eliot\\OneDrive\\Documents\\AI_code\\IMA\\challenge_kaggle"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

train_dir = os.path.join(Working_directory, "Train", "train")
test_dir = os.path.join(Working_directory, "Test", "test")

#%% Hyperparameters

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 1
IMAGE_HEIGHT = 80 
IMAGE_WIDTH = 110

def save_masks(model, dataloader, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        for idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            preds = model(data)
            preds = (preds > 0.5).float()
            for i, pred in enumerate(preds):
                pred_img = pred.squeeze().cpu().numpy()
                pred_img = (pred_img * 255).astype(np.uint8)
                Image.fromarray(pred_img).save(os.path.join(save_dir, f"mask_{idx*BATCH_SIZE + i}.png"))


#%%
def main():

    train_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Rotate(limit = 35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
            ),
            ToTensorV2(),
        ],
    )

    test_transform = A.Compose(
        [
            A.Resize(height= IMAGE_HEIGHT, width  = IMAGE_WIDTH),
            A.Normalize(
                mean = [.0, .0, .0],
                std = [1.0, 1.0, 1.0],
            ),
            ToTensorV2(),
        ]
    )

    model = UNET(in_channels = 3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, test_loader = get_loaders(
        train_dir,
        test_dir,
        train_transform,
        test_transform,
        BATCH_SIZE,
    )
    # some plots 

    dataiter = iter(train_loader)
    images, masks = next(dataiter)
    print(f"Shape of the loader: {images.shape}, {masks.shape}")
    fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        ax[0, i].imshow(images[i].permute(1, 2, 0))
        ax[1, i].imshow(masks[i].squeeze(), cmap="gray")
    plt.show()

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader)
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.float().unsqueeze(1).to(device)
            # forward
            preds = model(data)
            loss = loss_fn(preds, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
    
    torch.save(model.state_dict(), "model.pth", _use_new_zipfile_serialization=True)

    # check accuracy
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            preds = model(data)
            preds = (preds > 0.5).float()
            accuracy = (preds == target).float().mean()
            print(f"Accuracy: {accuracy}")

    # Predict and save masks
    save_masks(model, test_loader, "saved_masks")

if __name__ == "__main__":
    main()

# %%
