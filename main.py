import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
num_epochs = 30

# Data Loading

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

# for i in range (6):
#     plt.subplot(2,3, i+1)
#     plt.imshow(samples[i].permute(1,2,0))
#     plt.show()

# CNN

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256,10)
        self.dropout = nn.Dropout(p = 0.25)
    
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x,2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = NeuralNet()

# Loss and optimiser

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=15, gamma=0.5)

# Training loop

for epoch in range (num_epochs):
    model.train()
    for i, (images, labels) in enumerate (train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i+1) % 100 == 0:
            print(f'epoch : {epoch+1} / {num_epochs},step : {i+1} / {len(train_loader)}, loss : {loss.item():.4f}')

    scheduler.step()

# Testing 

# Make sure model is in eval mode
model.eval()
with torch.no_grad():
    n_correct = 0
    n_sample = 0

    # For per-class accuracy
    num_classes = 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for images, labels in test_loader:
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        n_samples_batch = labels.size(0)
        n_sample += n_samples_batch
        n_correct += (predictions == labels).sum().item()

        # Per-class tracking
        for i in range(n_samples_batch):
            label = labels[i].item()
            pred = predictions[i].item()
            if pred == label:
                class_correct[label] += 1
            class_total[label] += 1

    # Overall accuracy
    print(f'Test Accuracy: {100.0 * n_correct / n_sample:.2f}%')

    # Accuracy per class
    print("Per-class accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f'Class {i} ({class_map[i]}): {acc:.2f}%')
        else:
            print(f'Class {i} ({class_map[i]}): No samples')

# Saving the model
torch.save(model.state_dict(),"cifar10.pth")

# Load the model 
model = NeuralNet()
model.load_state_dict(torch.load("cifar10.pth"))
model.eval()