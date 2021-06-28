import os
from pathlib import Path
import random

import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np

from training.data_pipelines.a2d2.helpers import ConvertToIntLabels, audi_cmap


image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]
    )
]) 

label_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 512), interpolation=0),
    ConvertToIntLabels(),
    transforms.ToTensor()
])

# exmaple path
#root_path = Path(r"./demo/camera_lidar_semantic/")
# RTX dataset path
root_path = Path(r"/home/rtx/datasets/a2d2/camera_lidar_semantic/")

# Pattern to images
image_path = r"**/camera/cam_front_center/*.png"
# Pattern to labels
label_path = r"**/label/cam_front_center/*.png"

image_path_list = list(sorted(root_path.glob(image_path), 
        key= lambda path: int(path.stem.rsplit("_", 1)[1])))

label_path_list = list(sorted(root_path.glob(label_path),
        key= lambda path: int(path.stem.rsplit("_", 1)[1])))

# length of dataset calculation
dataset_length = len(list(root_path.glob(image_path)))

# debug print
print(dataset_length)

validation_split = 0.1
split_idx = int(dataset_length * (1 - validation_split))


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, image_dir, label_dir, image_transform=None, label_transform=None):
        super(TrainDataset, self).__init__()
        self.image_files = sorted(root_path.glob(image_path))[:split_idx]
                                    #key= lambda path: int(path.stem.rsplit("_", 1)[1]))
        self.label_files = sorted(root_path.glob(label_path))[:split_idx]
                                    #key= lambda path: int(path.stem.rsplit("_", 1)[1]))
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
        image_path = str(self.image_files[index].resolve())
        label_path = str(self.label_files[index].resolve())

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.label_transform is not None:
            label = self.label_transform(label) 

        (image, label) = self.combined_transform(image, label)

        # image = image.permute(1, 2, 0)
        # label = label.permute(1, 2, 0)

        return (image, label)

    def __len__(self):
        return len(self.image_files)


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, image_dir, label_dir, image_transform=None, label_transform=None):
        super(ValDataset, self).__init__()
        self.image_files = sorted(root_path.glob(image_path))[split_idx:]
                                    #key= lambda path: int(path.stem.rsplit("_", 1)[1]))
        self.label_files = sorted(root_path.glob(label_path))[split_idx:]
                                    #key= lambda path: int(path.stem.rsplit("_", 1)[1]))
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        image_path = str(self.image_files[index].resolve())
        label_path = str(self.label_files[index].resolve())

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)

        if self.image_transform is not None:
            image = self.image_transform(image)
            #image = image.permute(1, 2, 0)

        if self.label_transform is not None:
            label = self.label_transform(label)
            #label = label.permute(1, 2, 0)

        return (image, label)

    def __len__(self):
        return len(self.image_files)


train_set = TrainDataset(
    root_path, image_path, label_path, image_transform, label_transform
)

val_set = ValDataset(
    root_path, image_path, label_path, image_transform, label_transform
)


if __name__ == '__main__':
    dataset = TrainDataset(root_path, image_path, label_path, image_transform, label_transform)

    # print(len(train_set))
    # print(len(val_set))

    # for (image, label) in train_set:

    #     plt.figure(figsize=(15, 15))

    #     plt.subplot(1, 2, 1)

    #     plt.axis('off')
    #     plt.title('Camera')
    #     plt.imshow(image)

    #     plt.subplot(1, 2, 2)

    #     plt.axis('off')
    #     plt.title('Label')
    #     plt.imshow(label)

    #     plt.show()


