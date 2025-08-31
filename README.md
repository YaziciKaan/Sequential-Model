# Sequential Model - MNIST CNN Classifier

A simple Convolutional Neural Network (CNN) implementation using PyTorch for classifying handwritten digits from the MNIST dataset. This project demonstrates a straightforward approach to image classification using sequential layers.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [File Structure](#file-structure)
- [Expected Performance](#expected-performance)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a CNN classifier for the MNIST dataset using PyTorch's Sequential API. The model achieves good accuracy on digit classification through a combination of convolutional and fully connected layers with appropriate regularization techniques.

**Key Highlights:**
- Simple and clean PyTorch implementation
- Automatic GPU/CPU device detection
- Progress tracking during training
- Model checkpointing
- Comprehensive testing with accuracy metrics

## ✨ Features

- **Convolutional Neural Network**: Two-layer CNN with ReLU activation and max pooling
- **Dropout Regularization**: Prevents overfitting with 50% dropout
- **Automatic Device Detection**: Seamlessly works on both CPU and GPU
- **Progress Monitoring**: Real-time training progress with tqdm
- **Model Persistence**: Saves trained model weights for later use
- **Comprehensive Testing**: Evaluates model performance on test dataset

## 🔧 Prerequisites

- Python 3.7 or higher
- pip package manager

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YaziciKaan/Sequential-Model.git
   cd Sequential-Model
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   The project requires:
   - `torch>=2.0.0` - PyTorch deep learning framework
   - `torchvision>=0.15.0` - Computer vision datasets and transforms
   - `tqdm>=4.65.0` - Progress bar for training loops

## 🚀 Usage

### Training and Testing

Run the complete training and testing pipeline:

```bash
python train.py
```

This will:
1. Download the MNIST dataset (if not already present)
2. Train the model for 10 epochs (default)
3. Save the trained model as `model.pth`
4. Evaluate the model on the test dataset
5. Display the final test accuracy

### Training Output Example

```
Training Model...
Epoch 1/10: 100%|██████████| 1875/1875 [00:45<00:00, 41.67it/s, Loss: 0.1234]
Epoch 2/10: 100%|██████████| 1875/1875 [00:43<00:00, 43.21it/s, Loss: 0.0987]
...
Testing Model...
Test Loss: 0.0543
Test Accuracy: 0.9876
```

### Custom Training

You can modify the training parameters by editing the `train()` function call in `train.py`:

```python
if __name__ == "__main__":
    print("Training Model...")
    train(epoch=20)  # Train for 20 epochs instead of 10
    print("Testing Model...")
    test()
```

## 🏗️ Model Architecture

The `SimpleNN` class implements a CNN with the following architecture:

```
Input (28x28x1) MNIST Image
    ↓
Conv2D (1→32 channels, 3x3 kernel, padding=1)
    ↓
ReLU Activation
    ↓
MaxPool2D (2x2, stride=2) → (14x14x32)
    ↓
Conv2D (32→64 channels, 3x3 kernel, padding=1)
    ↓
ReLU Activation
    ↓
MaxPool2D (2x2, stride=2) → (7x7x64)
    ↓
Flatten → (3136,)
    ↓
Linear (3136 → 128)
    ↓
ReLU Activation
    ↓
Dropout (p=0.5)
    ↓
Linear (128 → 10) [Output layer for 10 digit classes]
```

**Total Parameters:** ~421,642

**Key Design Decisions:**
- **Two Convolutional Layers**: Extract hierarchical features from images
- **Max Pooling**: Reduce spatial dimensions and computational load
- **Dropout**: Prevent overfitting during training
- **ReLU Activation**: Non-linear activation for better learning capacity

## 📁 File Structure

```
Sequential-Model/
├── model.py           # Neural network architecture definition
├── train.py          # Training and testing logic
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

### File Descriptions

- **`model.py`**: Contains the `SimpleNN` class defining the CNN architecture
- **`train.py`**: Main script with functions for:
  - Data loading (`get_data_loaders()`)
  - Model initialization (`get_model()`)
  - Device selection (`get_device()`)
  - Training loop (`train()`)
  - Testing and evaluation (`test()`)
- **`requirements.txt`**: Lists all required Python packages
- **`README.md`**: This documentation file

## 📊 Expected Performance

With the default configuration, you can expect:

- **Training Time**: ~5-10 minutes on CPU (faster on GPU)
- **Test Accuracy**: ~98-99% on MNIST test set
- **Model Size**: ~1.6 MB (saved weights)

**Performance Notes:**
- The model typically converges within 10 epochs
- GPU training significantly reduces training time
- Higher accuracy can be achieved with data augmentation and hyperparameter tuning

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

1. **Bug Fixes**: Report and fix any issues you encounter
2. **Features**: Add new functionality like:
   - Data augmentation
   - Learning rate scheduling
   - Model architecture improvements
   - Visualization tools
3. **Documentation**: Improve or expand the documentation
4. **Testing**: Add unit tests for better code reliability

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test them
4. Commit your changes: `git commit -am 'Add some feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

For questions or suggestions, please open an issue on GitHub or contact the repository owner.

---

**Happy Learning! 🚀**