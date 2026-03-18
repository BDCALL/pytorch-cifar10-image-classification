# PyTorch CIFAR-10 CNN Image Classifier

This project implements a simple **Convolutional Neural Network (CNN)** using PyTorch to classify images from the **CIFAR-10 dataset**.

The goal of this project is to practice the core PyTorch workflow while working with a real image dataset.

---

## Project Objective

This project demonstrates the development of an end-to-end image classification system using PyTorch. It covers:

- Loading and processing image datasets using `torchvision`
- Applying data augmentation and preprocessing techniques
- Designing and implementing a CNN architecture with `nn.Module`
- Building training and evaluation pipelines
- Saving and loading trained models for reuse

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
   ```

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

4. **Define the CNN Model**

- 3 convolutional layers with batch normalisation and ReLU
- Max Pooling after each convolution
- Fully connected layer with dropout for regularization

5. **Define loss and optimiser**

- Loss function: nn.CrossEntropyLoss()
- Optimiser: torch.optim.Adam()
- Learning Rate Scheduler: torch.optim.lr_scheduler.StepLR() 

6. **Training Loop**

- Iterate over epochs and batches
- Forward pass -> compute loss -> backward pass -> update weights
- Step the scheduler at the end of each epoch
- Print training loss periodically

7. **Evaluation**

- Set model.eval() and wrap in torch.no_grad()
- Compute overall accuracy on test set
- Compute per-class accuracy

8. **Save the model**

## Results
The model was trained over epochs using Adam optimiser and data augmentation

- Test Accuracy: ~78% - 82%
- Best performance: Achived using data augmentation and learning rate scheduling

Per Class Accuracy on Run1
- Class 0 (airplane): 83.10%
- Class 1 (automobile): 92.70%
- Class 2 (bird): 70.20%
- Class 3 (cat): 68.90%
- Class 4 (deer): 79.30%
- Class 5 (dog): 69.50%
- Class 6 (frog): 83.20%
- Class 7 (horse): 81.30%
- Class 8 (ship): 87.40%
- Class 9 (truck): 87.00%

## Reflection

Through this project, I gained practical experience in building and training convolutional neural networks using PyTorch. I developed a deeper understanding of how data preprocessing and augmentation techniques, such as random cropping and flipping, can significantly improve model generalisation.

One key challenge was improving model performance beyond baseline accuracy. This was addressed by introducing batch normalisation, dropout, and a learning rate scheduler, which helped stabilise training and reduce overfitting. I also learned the importance of monitoring both overall and per-class accuracy to better understand model weaknesses.

Additionally, this project strengthened my understanding of structuring clean and maintainable code, particularly when implementing training and evaluation pipelines. It also highlighted the importance of experimentation and iterative improvement in machine learning workflows.

Overall, this project provided a strong foundation in developing production-ready deep learning systems and reinforced best practices in software engineering and model optimisation.