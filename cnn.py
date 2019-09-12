'''
classifying a set of images according to the number of ellipses
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

class Net(nn.Module):
        # constructing the neural network's structure
        def __init__(self):
            '''
            super(Net, self).__init__()
            # 3 input channels
            # 6 output channels
            # filters are 3 * 5 * 5
            self.conv1 = nn.Conv2d(3,6,5)
            # 2 by 2 kernel with a stride of 2
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6,16,5)
            # TODO: change image size definition in nn
            # rather than just resize it
            # 16 input channels
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84,10)
            '''
            super(Net, self).__init__()
            # 3 input channels
            # 6 output channels
            # filters are 3 * 5 * 5
            self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
            # 2 by 2 kernel with a stride of 2
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6,16,5)
            # TODO: change image size definition in nn
            # rather than just resize it
            # 16 input channels
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84,10)

        def forward(self, x):
            x = self.pool(nnfunc.relu(self.conv1(x)))
            x = self.pool(nnfunc.relu(self.conv2(x)))
            # reshapes the vector to anticipated size
            x = x.view(-1, 16 * 5 * 5)
            x = nnfunc.relu(self.fc1(x))
            x = nnfunc.relu(self.fc2(x))
            x = self.fc3(x)
            return x

'''
Function that shows an image.
'''
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

parser= argparse.ArgumentParser(description='Predict the number of ellipses contained in an image.')
parser.add_argument('--csv_file', default='data/shapes_dataset_MR/labels.csv', help="Directory of .csv file containing class labels.")
parser.add_argument('--root_dir', default='data/shapes_dataset_MR/', help="Root directory of class images.")
parser.add_argument('--seed', type=int, default=0, help="Value used as the seed for random values.")
parser.add_argument('--display', action='store_true', help="Boolean for displaying a sample of images.")
parser.add_argument('--num_test_samples', type=int, default=5, help="Number of images in a batch.")
parser.add_argument('--epochs', type=int, default=5, help="Amount of generations to train the model for.")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

# Random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

dataset = EllipsesDataset(csv_file=args.csv_file, root_dir=args.root_dir, transform = transforms.Compose(
        [
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]))

train_data, test_data = train_test_split(dataset, test_size=0.1)

train_loader = DataLoader(train_data, batch_size = 5, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size = 5, shuffle=True, num_workers=2)

classes = ('0','1','2','3','4','5')

net = Net()
# cross entrophy loss: using the distribution of classes in the dataset to
# reduce prediction errors
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Training...")
# training the network
for epoch in range(args.epochs):
    print("Starting epoch " + str(epoch))
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # gets inputs
        # input, labels = data
        input = data.get('img_tensor')
        labels = { data.get('polygons'), data.get('ellipses')}
        # zero parameter gradients
        optimiser.zero_grad()
        # forward, back and optimise
        outputs = net(input)
        loss = criterion(outputs, data.get('ellipses'))
        loss.backward()
        optimiser.step()
        # print statistical information
        running_loss += loss.item()

print("Finished training")

# testing performance
data_iter = iter(test_loader)
next = data_iter.next()
images = next.get('img_tensor')
labels = next.get('ellipses')
polygon_labels = next.get('polygons')

if (args.display):
	imshow(torchvision.utils.make_grid(images))

outputs = net(images)

# obtaining the index of the highest energy
_, predicted = torch.max(outputs, 1)

true_val = (classes[labels[j]] for j in range(5))
pred_val = (classes[predicted[j]] for j in range(5))
polygon_val = (classes[polygon_labels[j]] for j in range(5))

print(tabulate({"True value": true_val,"Predicted value": pred_val,"Polygons": polygon_val}, headers="keys", showindex="always"))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images = data.get('img_tensor')
        labels = data.get('ellipses')
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
