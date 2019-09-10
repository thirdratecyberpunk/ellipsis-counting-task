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
from torch.utils.data.sampler import SubsetRandomSampler
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
        # transforms act as image transformations
        # compose chains the transitions together
        # this turns an image into a tensor, then normalizes it
        # TODO: change image size definition in nn rather than just resize it
        #self.transform = transforms.Compose(
        #[transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        self.transform = transforms.Compose(
        [transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

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

class Net(nn.Module):
        # constructing the neural network's structure
        def __init__(self):
            super(Net, self).__init__()
            # 3 input channels
            # 6 output channels
            # 3 x 3 convolution
            self.conv1 = nn.Conv2d(3,6,5)
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6,16,5)
            # TODO: change image size definition in nn
            # rather than just resize it
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84,10)

        def forward(self, x):
            x = self.pool(nnfunc.relu(self.conv1(x)))
            x = self.pool(nnfunc.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = nnfunc.relu(self.fc1(x))
            x = nnfunc.relu(self.fc2(x))
            x = self.fc3(x)
            return x

'''
Function that returns the training and testing samples
'''
def load_split_train_test(valid_size = 0.2):
    # splits dataset into training and testing at a random point
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.num_test_samples,
    shuffle=False, num_workers=2, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.num_test_samples,
    shuffle=False, num_workers=2, sampler=test_sampler)
    return train_loader, test_loader

'''
Function that shows an image.
'''
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

parser= argparse.ArgumentParser(description=
'Classify an image according to the number of ellipses it contains.')
parser.add_argument('--csv_file', default='data/shapes_dataset_MR/labels.csv')
parser.add_argument('--root_dir', default='data/shapes_dataset_MR/')
parser.add_argument('--display', action='store_true')
parser.add_argument('--num_test_samples', type=int, default=5)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

train_data = EllipsesDataset(csv_file=args.csv_file, root_dir=args.root_dir)
test_data = EllipsesDataset(csv_file=args.csv_file, root_dir=args.root_dir)

train_loader, test_loader = load_split_train_test(0.2)

#train_loader = torch.utils.data.DataLoader(train_data, batch_size=4,
#shuffle=True, num_workers=2)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=4,
#shuffle=True, num_workers=2)

classes = ('0','1','2','3','4','5')

net = Net()
# cross entrophy loss: using the distribution of classes in the dataset to
# reduce prediction errors
criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training the network
for epoch in range(2):
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
# images, labels = data_iter.next()
next = data_iter.next()
images = next.get('img_tensor')
labels = next.get('ellipses')

if (args.display):
	imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(args.num_test_samples)))

outputs = net(images)

# obtaining the index of the highest energy
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(args.num_test_samples)))
