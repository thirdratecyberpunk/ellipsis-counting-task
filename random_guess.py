'''
Randomly assigns a class label to the dataset to obtain a benchmark score.
'''
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
from tabulate import tabulate
from EllipsesDataset import EllipsesDataset
from sklearn.model_selection import train_test_split
import random

parser= argparse.ArgumentParser(description=
'Randomly guess a class label.')
parser.add_argument('--csv_file', default='data/shapes_dataset_MR/labels.csv')
parser.add_argument('--root_dir', default='data/shapes_dataset_MR/')
parser.add_argument('--display', action='store_true')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

# Random seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

dataset = EllipsesDataset(csv_file=args.csv_file, root_dir=args.root_dir)

train_data, test_data = train_test_split(dataset, test_size=0.1)

train_loader = DataLoader(train_data, batch_size = 5, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size = 5, shuffle=True, num_workers=2)

classes = (0,1,2,3,4,5)

print("Finished training")

correct = 0
total = 0

# for every item in the testing dataset, randomly pick a class value
with torch.no_grad():
    for data in test_loader:
        labels = data.get('ellipses')
        predicted = torch.from_numpy(np.array([random.choice(classes) for i in range(5)]))
        total += labels.size(0)
        print(predicted)
        print(labels)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))


