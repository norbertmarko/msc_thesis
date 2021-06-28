import os
import sys
import glob

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

from training.data_pipelines.bdd100k.utils import ConvertToIntLabels_dd

# Transformations and Augmentations

# mean and std values
mean_R, mean_G, mean_B = 0.27387958514512184, 0.2854202855131391, 0.28135438858273215
std_R, std_G, std_B = 0.18293644075057092, 0.1825258390237436, 0.18159887469669117

# Path to images
image_dir = "/home/rtx/datasets/bdd100k/images/100k/train" 
# Path to labels
label_dir = "/home/rtx/datasets/bdd100k/drivable_maps/color_labels/train"

# Path to images (val)
image_dir_val = "/home/rtx/datasets/bdd100k/images/100k/val" 
# Path to labels (val)
label_dir_val = "/home/rtx/datasets/bdd100k/drivable_maps/color_labels/val"

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 512)),
    # augmentations here (same transforms in label)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) # Note: This wouldn't do anything.
]) 

label_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 512)),
    # augmentations here
    ConvertToIntLabels_dd(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 512)),
    transforms.ToTensor()
])

validation_split = 0.1
split_idx = int(len(os.listdir(image_dir_val)) * (1 - validation_split))


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, label_dir, image_transform=None, label_transform=None):
        super(TrainDataset, self).__init__()
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))#[:split_idx]
        self.label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))#[:split_idx]
        self.image_transform = image_transform
        self.label_transform = label_transform

    def combined_transform(self, image, label):

        # Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(256, 512)
        )
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Random Color Jitter
        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1)
        color_jitter_transform = transforms.ColorJitter.get_params(
            color_jitter.brightness, color_jitter.contrast, color_jitter.saturation, color_jitter.hue
        )
        image = color_jitter_transform(image)

        return (image, label)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)

        if self.image_transform is not None:
            image = self.image_transform(image)
            
        if self.label_transform is not None:
            label = self.label_transform(label)

        (image, label) = self.combined_transform(image, label)

        return (image, label)

    def __len__(self):
        return len(self.image_files)


class ValDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, label_dir, image_transform=None, label_transform=None):
        super(ValDataset, self).__init__()
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))#[split_idx:]
        self.label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))#[split_idx:]
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)

        if self.image_transform is not None:
            image = self.image_transform(image)
            
        if self.label_transform is not None:
            label = self.label_transform(label)
        return (image, label)

    def __len__(self):
        return len(self.image_files)


train_set = TrainDataset(
    image_dir=image_dir,
    label_dir=label_dir, 
    image_transform=image_transform,
    label_transform=label_transform
)

val_set = ValDataset(
    image_dir=image_dir_val,
    label_dir=label_dir_val,
    image_transform=image_transform,
    label_transform=label_transform
)

class DatasetTools:
    def __init__(self, dataset, batch_size=8):
        self._dataset = dataset

        self._batch_size = batch_size

        self._loader = torch.utils.data.DataLoader(
            self._dataset, batch_size=self._batch_size,
            shuffle=False, num_workers=4
        )

    def calculate_batch_metrics(self):
        """
        Calculates Mean and Standard Deviation
        across one mini-batch (good for batch normalization).
        """
        pass

    def calculate_dataset_metrics(self):
        """
        Calculates Mean and Standard Deviation
        across the entire dataset.
        """
        pass

    def show_data(self, samples=5):
        """
        Helper function to check the training data visually.
        """
        self._samples = samples
            
        idx = 0
        batch = next(iter(self._loader))
        while idx <= self._samples:

            (img_batch, label_batch) = batch

            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            ax1.imshow(np.transpose(img_batch[idx], (1, 2, 0)))
            ax2.imshow(np.transpose(label_batch[idx], (1, 2, 0)).squeeze(dim=2))

            plt.show()
            idx += 1
                

if __name__ == '__main__':
    tools = DatasetTools(train_set)
    tools.show_data()
