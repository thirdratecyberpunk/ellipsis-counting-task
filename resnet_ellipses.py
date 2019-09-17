'''
Using the ResNet architecture to classify images according to the number of ellipses.
'''

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import ResNet, BasicBlock
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
from CsvNameGen import generate_csv_name
import time
import csv
from CsvNameGen import generate_csv_name

'''
Implementation of ResNet class.
'''
class EllipsesResNet(ResNet):
    def __init__(self):
        super(EllipsesResNet, self).__init__(BasicBlock, [2,2,2,2], num_classes=6)
        self.conv1 = torch.nn.Conv2d(3,64, kernel_size=2, stride=1, padding=0, bias=False)

    def forward(self,x):
        return torch.softmax(super(EllipsesResNet, self).forward(x), dim=-1)

parser= argparse.ArgumentParser(description='Predict the number of ellipses contained in an image using Resnet architecture.')
parser.add_argument('--csv_file', default='data/shapes_dataset_MR/labels.csv', help="Directory of .csv file containing class labels.")
parser.add_argument('--root_dir', default='data/shapes_dataset_MR/', help="Root directory of class images.")
parser.add_argument('--seed', type=int, default=0, help="Value used as the seed for random values.")
parser.add_argument('--display', action='store_true', help="Boolean for displaying a sample of images.")
parser.add_argument('--num_test_samples', type=int, default=5, help="Number of images in a batch.")
parser.add_argument('--epochs', type=int, default=5, help="Amount of generations to train the model for.")
parser.add_argument('--output_csv', default = 'output/resnet/', help="Directory to save experiment results to.")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

# Random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# dataset
dataset = EllipsesDataset(csv_file=args.csv_file, root_dir=args.root_dir, transform = transforms.Compose(
        [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229,0.224,0.225))
        ]))

train_data, test_data = train_test_split(dataset, test_size=0.1)

train_loader = DataLoader(train_data, batch_size = 5, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size = 5, shuffle=True, num_workers=2)

classes = ('0','1','2','3','4','5')

# model
net = EllipsesResNet().to(device)

loss_function = nn.CrossEntropyLoss()
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Training...")

with open(generate_csv_name(args.output_csv, "resnet", args.epochs, args.seed), 'w') as csvfile:
    fieldnames = ['epoch','seed', 'batch_size','time', 'running_loss', 'accuracy']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    startts = time.time()
    for epoch in range(args.epochs):
        print("Starting epoch " + str(epoch))
        running_loss = 0.0
        net.train()
        # training the network
        for i, data in enumerate(train_loader, 0):
            # gets inputs
            image = data.get('img_tensor').to(device)
            label = data.get('ellipses').to(device)
            # zero parameter gradients
            optimiser.zero_grad()
            # forward, back and optimise
            outputs = net(image)
            loss = loss_function(outputs, label)
            loss.backward()
            optimiser.step()
            # print statistical information
            running_loss += loss.item()

        net.eval()
        # evaluating performance at this epoch
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images = data.get('img_tensor').to(device)
                labels = data.get('ellipses').to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100.00 * correct / total
        writer.writerow({'epoch': epoch, 'seed': args.seed, 'batch_size': args.num_test_samples, 'time': time.time() - startts, 'running_loss': running_loss, 'accuracy' : accuracy})
                
print("Finished training")
