# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 15:46:42 2023

@author: sourg
"""

from torch.utils.data import Dataset
import skimage.io as io
import torch.nn as nn
import torch
import glob

class CamusEDImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None,test=False):
        self.transform = transform
        self.target_transform = target_transform
        if not test:
            self.img_path = glob.glob("./data/training/**/*_ED.mhd")
            self.gt_path = glob.glob("./data/training/**/*_ED_gt.mhd")
        else:
            self.img_path = glob.glob("./data/testing/**/*_ED.mhd")
            self.gt_path = glob.glob("./data/testing/**/*_ED_gt.mhd")
            
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        image = io.imread(img_path,plugin='simpleitk')
        image = image[0]

        gt_path = self.gt_path[idx]
        gt_image = io.imread(gt_path,plugin='simpleitk')
        gt_image = gt_image[0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            gt_image = self.target_transform(gt_image)
        return image,gt_image