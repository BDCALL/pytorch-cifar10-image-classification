import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.functional as F

# Categories

class_map = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# Hyperparameters

batch_size = 64
input_size = 1024   # 32*32
hidden_size = 128
num_classes = 10

# Data Loading

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],std  = [0.2023, 0.1994, 0.2010])
])

train_dataset = torchvision.datasets.CIFAR10(root='./data',train=True, transform=transform,download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data',train=False, transform=transform, download = True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Test of image loading

examples = iter(train_loader)
samples, labels = next(examples)

for i in range (6):
    plt.subplot(2,3, i+1)
    plt.imshow(samples[i][0], cmap="gray")
    plt.show()

# CNN

class NeuralNet(nn.Module):
    def __init(self):
        super(self,NeuralNet).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*8*8, 128)
        self.fc2 = nn.Linear(128,10)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = F.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NeuralNet()

# Training loop