import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as nnfunc
import torch.optim as optim
from PIL import Image

'''
Dataset of images and the number of ellipses/other polygons.
'''
class EllipsesDataset(Dataset):
    # dataset of images to sort
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file: Path to csv file of annotations
            root_dir: Directory of images
            transform: Optional transformation
        """
        # creates a data frame from the contents of the csv
        self.num_of_elements_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    # returns the number of elements in a dataset as the length of frame
    def __len__(self):
        return len(self.num_of_elements_frame)

    # returns an object definition of a sample given a tensor
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # finds the image by searching for the index in the frame
        img_name = os.path.join(self.root_dir,
        str(self.num_of_elements_frame.iloc[idx, 2]),
        str(self.num_of_elements_frame.iloc[idx, 0]))
        # loads the image as a tensor
        image = Image.open(img_name)
        img_tensor = self.transform(image)

        polygons = self.num_of_elements_frame.iloc[idx, 1]
        ellipses = self.num_of_elements_frame.iloc[idx, 2]
        sample = {'img_tensor' : img_tensor, 'img_name': img_name,
        'polygons': polygons, 'ellipses': ellipses}

        return sample
