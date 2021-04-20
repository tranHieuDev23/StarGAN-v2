import os
from random import sample
from solvers.rgb_to_lab import lab_normalize, rgb_to_lab
import torch
import torch.utils.data as data
import torch.functional as F
from torchvision import transforms, datasets
from PIL import Image
import numpy as np


class DefaultDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.samples = [os.path.join(root_dir, item)
                        for item in os.listdir(root_dir)]
        self.samples.sort()
        self.transform = transform

    def __getitem__(self, index):
        filename = self.samples[index]
        img = Image.open(filename).convert("RGB")
        if (self.transform is not None):
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root_dir, transform):
        super().__init__()
        self._make_dataset(root_dir)
        self.transform = transform

    def _make_dataset(self, root_dir):
        self.classes = os.listdir(root_dir)
        self.classes.sort()
        filenames1, filenames2, labels = [], [], []
        for idx, domain in enumerate(self.classes):
            class_dir = os.path.join(root_dir, domain)
            class_filenames = self.samples = [os.path.join(
                class_dir, item) for item in os.listdir(class_dir)]
            filenames1.extend(class_filenames)
            filenames2.extend(sample(class_filenames, len(class_filenames)))
            labels.extend([idx] * len(class_filenames))
        self.samples = list(zip(filenames1, filenames2))
        self.labels = labels

    def __getitem__(self, index):
        filename, filename2 = self.samples[index]
        label = self.labels[index]
        img = Image.open(filename).convert('RGB')
        img2 = Image.open(filename2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.samples)


class ToTensor(object):
    def __call__(self, image):
        target = torch.ByteTensor(
            torch.ByteStorage.from_buffer(image.tobytes()))
        target = target.view(image.size[1], image.size[0], 3)
        # put it from HWC to CHW format
        target = target.to(torch.float).permute((2, 0, 1)).contiguous()
        return target


def get_source_loader(root_dir, img_size=256, batch_size=8, num_workers=4):
    print("Preparing DataLoader to fetch source images...")
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(rgb_to_lab),
        ToTensor(),
        lab_normalize()
    ])
    dataset = datasets.ImageFolder(root_dir, transform=data_transform)
    dataset_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    return dataset_loader, dataset.classes


def get_reference_loader(root_dir, img_size=256, batch_size=8, num_workers=4):
    print("Preparing DataLoader to fetch referral images...")
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(rgb_to_lab),
        ToTensor(),
        lab_normalize()
    ])
    dataset = ReferenceDataset(root_dir, transform=data_transform)
    dataset_loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    return dataset_loader, dataset.classes


class InputFetcher:
    def __init__(self, dataloader: data.DataLoader):
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)

    def __next__(self):
        try:
            value = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            value = next(self.iter)
        return value
