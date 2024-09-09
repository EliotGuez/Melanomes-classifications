import torch 
import torchvision
from dataset import SegmentationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#for the loss we want the dice to be the closest possible to 0 
def dice_coeff(inputs, target): 
    inputs = inputs > 0
    target = target > 0

    intersection = inputs * target
    dice = (2. * intersection.sum()) / (inputs.sum() + target.sum())
    
    # On s'assure que le dice soit entre 0 et 1
    dice = torch.clamp(dice, 0, 1)
    
    return 1 - dice



def get_loaders(train_dir,test_dir,train_transform,test_transform,batch_size ):
    train_ds = SegmentationDataset(image_dir= train_dir,transform = train_transform)
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    
    test_ds = SegmentationDataset(image_dir = test_dir,transform = test_transform)
    test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader

