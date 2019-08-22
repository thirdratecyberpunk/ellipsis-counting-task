# classifying a set of images according to the number of ellipses
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

    # gets a specified item given a tensor
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # finds the image by searching for the given tensor in the frame
        img_name = os.path.join(self.root_dir,
        str(self.num_of_elements_frame.iloc[idx, 2]),
        str(self.num_of_elements_frame.iloc[idx, 0]))
        print(img_name)
        polygons = self.num_of_elements_frame.iloc[idx, 1]
        ellipses = self.num_of_elements_frame.iloc[idx, 2]
        sample = {'image': img_name, 'polygons': polygons, 'ellipses': ellipses}

        if self.transform:
            sample = self.transform(sample)

        return sample

parser= argparse.ArgumentParser(description='Classify an image according to the number of ellipses it contains.')
parser.add_argument('--csv_file', default='data/shapes_dataset_MR/labels.csv')
parser.add_argument('--root_dir', default='data/shapes_dataset_MR/')

args = parser.parse_args()

# transforms act as image transformations
# compose chains the transitions together
# this turns an image into a tensor, then normalizes it
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = EllipsesDataset(csv_file=args.csv_file, root_dir=args.root_dir)

# showing images
fig = plt.figure()

for i in range(len(trainset)):
    sample = trainset[i]
    print (i, sample['image'], sample['polygons'].shape, sample['polygons'].shape)
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    img = io.imread(sample['image'])
    plt.imshow(img)
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')

    if i == 3:
        plt.show()
        break

# loading and displaying the images
