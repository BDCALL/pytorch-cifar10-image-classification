# PyTorch CIFAR-10 CNN Image Classifier

This project implements a simple **Convolutional Neural Network (CNN)** using PyTorch to classify images from the **CIFAR-10 dataset**.

The goal of this project is to practice the core PyTorch workflow while working with a real image dataset.

---

## Project Objective

The objective of this project is to learn how to:

- Work with image datasets using `torchvision`
- Apply image transforms for preprocessing
- Build CNN architectures with `nn.Module`
- Implement training and evaluation loops
- Save and load trained models

---

## Dataset

We use the **CIFAR-10 dataset**, which contains:

- 60,000 32×32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images, 10,000 test images

The dataset is automatically downloaded using `torchvision`.

---

## Project Workflow

1. **Install dependencies**  
   ```bash
   pip install torch torchvision matplotlib

2. **Prepare the Dataset**
Apply the following transforms to the dataset for preprocessing and augmentation:

- `RandomHorizontalFlip()` – for data augmentation  
- `RandomCrop()` – to vary image regions  
- `ColorJitter()` – to adjust brightness, contrast, saturation, and hue  
- `ToTensor()` – to convert images to PyTorch tensors  
- `Normalize()` – using CIFAR-10 mean `[0.4914, 0.4822, 0.4465]` and standard deviation `[0.2023, 0.1994, 0.2010]`

3. **Create data loaders**

- Use torch.utils.data.DataLoader
- Set batch size (default: 64) and shuffle the training data