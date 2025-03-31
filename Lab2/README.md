# Lab Report: Computer Vision with PyTorch

## 📌 Overview
This lab explores different deep learning architectures for image classification using the MNIST dataset. We implemented and compared CNN, Faster R-CNN, VGG16, AlexNet, and Vision Transformer (ViT) models in PyTorch.

---

## 🔹 **Part 1: CNN & Faster R-CNN**

### ✅ **Step 1: Data Loading (MNIST)**
- Used `torchvision.datasets` to load the MNIST dataset.
- Applied transformations (normalization, tensor conversion).
- Created DataLoaders for efficient batch training.

### ✅ **Step 2: Implementing a CNN**
- Defined a CNN model using `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, and `nn.Linear` layers.
- Configured hyperparameters (kernel size, padding, stride, optimizer, regularization).
- Trained the model on GPU using `torch.cuda`.

### ✅ **Step 3: Implementing Faster R-CNN**
- Used `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
- Adapted the model for MNIST (adjusting the backbone and input layers).
- Trained and evaluated the model.

### ✅ **Step 4: Model Comparison**
- Measured **Accuracy**, **F1-score**, **Loss**, and **Training Time** for CNN and Faster R-CNN.

### ✅ **Step 5: Fine-Tuning VGG16 & AlexNet**
- Loaded pre-trained `torchvision.models.vgg16` and `torchvision.models.alexnet` models.
- Adapted the final layers for MNIST classification.
- Compared results with CNN and Faster R-CNN.

---

## 🔹 **Part 2: Vision Transformer (ViT)**

### ✅ **Step 1: Implementing ViT**
- Followed a tutorial to build a ViT model from scratch using PyTorch.
- Adjusted the model to classify MNIST images.

### ✅ **Step 2: Training & Evaluation**
- Compared ViT’s performance against CNN, Faster R-CNN, VGG16, and AlexNet.
- Discussed the strengths and weaknesses of ViT in comparison to traditional architectures.

---

## 🎯 **Key Takeaways**
- **CNN** performs well for image classification but struggles with object detection.
- **Faster R-CNN** is better for object detection but is computationally expensive.
- **VGG16 & AlexNet** benefit from transfer learning, improving performance with pre-trained features.
- **ViT** shows promising results in image classification but requires larger datasets for optimal performance.

This lab provided hands-on experience in implementing and evaluating different deep learning models for computer vision tasks.

