# CIFAR-10 Classification with LeNet

A PyTorch implementation of a LeNet-style convolutional neural network for classifying images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

## Overview

CIFAR-10 contains 60,000 color images across 10 classes. This project trains a convolutional neural network to recognize them using PyTorch. The 50,000 training images are split into training and validation sets; the official 10,000-image test set is reserved for the final evaluation.

Classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

## Model architecture

The model is adapted from LeNet for 32 × 32 RGB images.

| Stage | Layer / output |
| --- | --- |
| Input | 3 × 32 × 32 RGB image |
| Feature extractor | Conv2d (3 → 6, 5 × 5, padding 2) → ReLU → AvgPool2d (2 × 2) |
| Feature extractor | Conv2d (6 → 16, 5 × 5) → ReLU → AvgPool2d (2 × 2) |
| Classifier | Flatten (16 × 6 × 6) → Linear 120 → ReLU → Linear 84 → ReLU → Linear 10 |
| Output | Logits for the 10 CIFAR-10 classes |

## Data processing

Training images use data augmentation:

- Random 32 × 32 crop with four pixels of padding
- Random horizontal flip
- Tensor conversion and channel-wise normalization

Validation and test images should use only tensor conversion and normalization—no random augmentation. This keeps evaluation consistent and reproducible.

Normalization statistics:

```python
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]
```

## Requirements

- Python 3.9 or later
- PyTorch
- Torchvision

Install dependencies:

```bash
pip install torch torchvision
```

## Run

Save the training script as `CIFAR-10.py`, then run:

```bash
python CIFAR-10.py
```

The CIFAR-10 data is downloaded automatically to `./data` the first time the script runs. CUDA is used automatically when a compatible GPU is available; otherwise, training runs on CPU.

The best validation-loss checkpoint is saved to:

```text
model/lenet_model.pt
```

## Suggested project structure

```text
cifar10-lenet/
├── CIFAR-10.py          # Data loading, model, training, and evaluation
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── data/                # Downloaded dataset (ignored by Git)
└── model/               # Saved checkpoints (ignored by Git)
```

## Notes and possible improvements

- Use a fixed random seed when making the train/validation split to reproduce results.
- Evaluate the saved best checkpoint on the test set, rather than only the final-epoch model.
- A LeNet-style model is a useful baseline; larger CIFAR-10 architectures such as ResNet generally achieve better accuracy.

## License

This project is intended for educational use. Add a license file before publishing if you want to define reuse terms.
