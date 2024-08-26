# Slide, Project, Learn, Repeat (ProjConv2D)

**Status**: 🚧 Draft - Work in Progress 🚧

## Overview

This repository contains the implementation of a novel neural network layer, **ProjConv2D**, designed as an alternative or complementary module to the standard convolutional layer in Convolutional Neural Networks (CNNs). The ProjConv2D layer leverages a learnable projection matrix to transform sliding patches of the input image, offering a unique mechanism for feature extraction. This approach is particularly effective when used as the first layer in a CNN, followed by traditional convolutional layers.

## Abstract

ProjConv2D introduces a novel method for feature extraction by combining the sliding operation of CNNs with a learnable projection step. This hybrid approach bridges the gap between localized feature extraction and the flexibility offered by learned projections, providing a new avenue for enhancing neural network architectures. The layer is particularly beneficial in tasks requiring efficient and powerful feature extraction, as demonstrated in experiments on the FashionMNIST and CIFAR-10 datasets.

## Key Features

- **Learnable Projection Matrix**: Each sliding patch of the input is projected into a new feature space using a learnable matrix.
- **Localized Feature Extraction**: Retains spatial locality by operating in a manner similar to traditional convolutions, making it suitable for tasks where spatial structure is important.
- **Comparative Efficiency**: Offers a middle ground between the computational efficiency of CNNs and the global context modeling of Vision Transformers (ViTs).

## Mathematical Formulation

The ProjConv2D layer operates by first unfolding the input tensor into patches of size \( C_{\text{in}} \times K \times K \). Each patch is then projected into a new feature space using a learnable matrix:

\( y_{\text{projected}} = W \cdot x_{\text{unfolded}} \)

This operation results in a tensor of shape \( (B, C_{\text{out}}, P) \), which is then reshaped to \( (B, C_{\text{out}}, H', W') \) to match the spatial dimensions of the output.

## Comparison with Vision Transformers (ViTs)

- **ProjConv2D**: Focuses on localized feature extraction with a learnable projection, retaining spatial structure.
- **ViTs**: Utilizes self-attention mechanisms for capturing long-range dependencies, often requiring more computational resources due to the complexity of the attention mechanism.

## Experimental Setup

The ProjConv2D layer was evaluated on FashionMNIST and CIFAR-10 datasets. The experimental setups included:
1. **ProjConv2D as the first layer followed by traditional convolutional layers.**
2. **All layers as ProjConv2D.**
3. **All layers as standard Conv2D (baseline).**

### Example Network Configuration

The network used for testing included ProjConv2D as the first layer followed by several traditional convolutional layers, batch normalization, and activation functions such as LeakyReLU.

## Results

The table below summarizes the results of the experiments on the FashionMNIST dataset:

| Model Configuration           | Validation Accuracy | F1-Score | Training Time (seconds) |
|-------------------------------|---------------------|----------|-------------------------|
| ProjConv2D (1st Layer) + Conv Layers  | 90.97%              | 0.9100   | 19.00                    |
| All ProjConv2D Layers                 | 88.92%              | 0.8896   | 25.14                    |
| All Conv Layers (Baseline)     | 90.66%              | 0.9065   | 19.37                    |

## Feature Maps Comparison

Here is a comparison of feature maps produced by the ProjConv2D layer versus standard Conv2D:

- **ProjConv2D Feature Maps**:
  ![ProjConv2D Feature Maps](readme/projconv2d_fm_1.png)

- **Conv2D Feature Maps**:
  ![Conv2D Feature Maps](readme/conv2d_fm_1.png)

These visualizations highlight the differences in how each layer captures features at different levels of the network.

## How to Use

1. **Clone the repository**:
git clone https://github.com/jirkapivrnec/ProjConv2d.git

2. **Install dependencies**:
pip install -r requirements.txt 
TODO add requirements.txt

3. **Run the experiments**:
python train.py
TODO add arguments and clean up the file


## Contributing

This project is in the draft stage and is a work in progress. Contributions are welcome, particularly in the areas of refining the implementation, testing on additional datasets, and extending the comparative analysis with other architectures.
