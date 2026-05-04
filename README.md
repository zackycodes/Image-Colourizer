# 🎨 Neural Colorizer: U-Net Image Restoration
A minimalist PyTorch implementation of a deep learning model designed to restore color to grayscale images. By utilizing a U-Net style encoder-decoder architecture, this project learns to map single-channel intensity values back to the full RGB spectrum using the CIFAR-10 dataset.  

## 🔬 Architecture & Logic
The model leverages a deep convolutional structure to understand both low-level textures and high-level semantics:  

* The Encoder: A series of Conv2d layers with BatchNorm2d and ReLU activations that compress the input, doubling feature depth at each stage (64 → 128 → 256 → 512).  

* The Decoder: Uses Upsample (bilinear mode) and convolutional layers to reconstruct the spatial resolution while reducing feature depth back to the 3-channel RGB output.  

* Final Layer: Employs a Sigmoid activation function to ensure all output pixel values are normalized between 0 and 1.  

## 🛠️ Technical Specifications
Framework: PyTorch  

* Dataset: CIFAR-10 (sampled to 10,000 images for efficient training)  

* Preprocessing: Images are resized to 32x32 and converted into grayscale/color pairs.  

* Loss Function: L1Loss (Mean Absolute Error) to promote sharper reconstructions.  

* Optimizer: Adam with a learning rate of 0.0005.  

* Hardware: Optimized for CUDA execution.  

## 🚀 Usage
1. Training
The training loop runs for 100 epochs, optimizing the model to minimize the difference between the predicted colorization and the ground truth.  
The model is automatically saved after training
torch.save(model.state_dict(), "unet_colorization.pth")

3. Inference & Visualization
The script includes a testing suite that utilizes matplotlib to display a side-by-side comparison of:  

* Grayscale Input: The single-channel source image.  

* Predicted Colorization: The model's interpretation of the scene.  

* Ground Truth: The original color image for accuracy reference.  

## 📊 Performance Visuals
The model outputs a grid of 8 sample images after training to demonstrate its ability to generalize colors across various CIFAR-10 categories like automobiles, animals, and aircraft.
