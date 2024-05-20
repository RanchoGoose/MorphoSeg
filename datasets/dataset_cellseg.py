import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
# from scipy.ndimage.interpolation import zoom
from scipy.ndimage import zoom

from torch.utils.data import Dataset
from einops import repeat
from icecream import ic

# remove this after debugging
import argparse
from torchvision import transforms
import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from icecream import ic
from PIL import Image

#########

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        
    def check_image(self, image):
        # Check if the image is a numpy array
        if not isinstance(image, np.ndarray):
            raise TypeError("Image is not a numpy array.")

        # Check the image pixel range
        if image.max() > 1:
            if image.max() <= 255:
                # Assuming the image is in the range [0, 255], normalize it to [0, 1]
                # print("Normalizing image from [0, 255] to [0, 1].")
                return image / 255.0
            else:
                raise ValueError("Image pixel values are outside the expected range.")
        return image

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
        # Check and normalize image
        image = self.check_image(image)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
    
class CellSeg_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, test_split_ratio=0.2):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        self.test_split_ratio = test_split_ratio
        self.sample_list = open(os.path.join(list_dir, f'{self.split}.txt')).readlines()
        # Filter the list to exclude samples with zero masks
        if self.split in ["train", "test"]:
            self.filter_samples_with_nonzero_masks()
        
    def __len__(self):
        return len(self.sample_list)
    
    # def __getitem__(self, idx):        
    #     if self.split == "eval":
    #         file_stem = self.sample_list[idx].strip().rstrip('.tif')
    #         image_path = os.path.join(self.data_dir, file_stem + '.tif')
    #         image = Image.open(image_path).convert("L")
            
    #         # Initialize a default mask in case it's needed
    #         default_mask = np.zeros_like(np.array(image), dtype=np.float32)
        
    #         mask_path = os.path.join(self.data_dir, file_stem + '_mask.tif')
    #         # Check if the mask file exists
    #         if os.path.exists(mask_path):
    #             mask = Image.open(mask_path).convert("L")
    #         else:
    #             # If no corresponding mask file is found, use the default mask
    #             mask = Image.fromarray(default_mask)
    #     else:
    #         file_stem = self.sample_list[idx].strip().rstrip('.png')
    #         image_path = os.path.join(self.data_dir, file_stem + '.png')
    #         image = Image.open(image_path).convert("L")
            
    #         mask_path = os.path.join(self.data_dir, file_stem + '_mask.png')
    #         mask = Image.open(mask_path).convert("L")
        
    #     # Convert the mask to a numpy array
    #     mask_array = np.array(mask)

    #     # Normalize the mask to have values of 0 and 1 for training and validation splits
    #     if self.split in ["train", "test"]:
    #         normalized_mask = (mask_array > 0).astype(np.uint8)
    #     else:  # For eval, the mask might be the default one, so it's already normalized
    #         normalized_mask = mask_array
        
    #     sample = {'image': np.array(image), 'label': normalized_mask}
    #     if self.transform:
    #         sample = self.transform(sample)
            
    #     sample['case_name'] = file_stem
    #     return sample
    
    def __getitem__(self, idx):
        file_stem = self.sample_list[idx].strip().rstrip('.tif').rstrip('.png')
        file_extension = '.tif' if self.split == 'eval' else '.png'
        image_path = os.path.join(self.data_dir, file_stem + file_extension)
        image = Image.open(image_path).convert("L")
        
        mask_path = os.path.join(self.data_dir, file_stem + '_mask' + file_extension)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            default_mask = np.zeros_like(np.array(image), dtype=np.float32)
            mask = Image.fromarray(default_mask)

        mask_array = np.array(mask)
        normalized_mask = (mask_array > 0).astype(np.uint8) if self.split in ["train", "test"] else mask_array
        
        sample = {'image': np.array(image), 'label': normalized_mask}
        if self.transform:
            sample = self.transform(sample)
        
        sample['case_name'] = file_stem
        return sample
    
    def filter_samples_with_nonzero_masks(self):
        filtered_list = []
        for file_name in self.sample_list:
            file_stem = file_name.strip().rstrip('.tif').rstrip('.png')
            mask_path = os.path.join(self.data_dir, f'{file_stem}_mask' + ('.tif' if self.split == 'eval' else '.png'))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                if np.any(np.array(mask)):
                    filtered_list.append(file_name)
        self.sample_list = filtered_list
        
    def generate_or_load_splits(self, list_dir):
        train_list_path = os.path.join(list_dir, 'train.txt')
        test_list_path = os.path.join(list_dir, 'test.txt')

        if os.path.exists(train_list_path) and os.path.exists(test_list_path):
            return
        
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.png') and not f.endswith('_mask.png')]
        np.random.shuffle(all_files)
        split_idx = int(len(all_files) * (1 - self.test_split_ratio))
        train_files, test_files = all_files[:split_idx], all_files[split_idx:]

        with open(train_list_path, 'w') as f:
            for item in train_files:
                f.write("%s\n" % item)
                
        with open(test_list_path, 'w') as f:
            for item in test_files:
                f.write("%s\n" % item)


