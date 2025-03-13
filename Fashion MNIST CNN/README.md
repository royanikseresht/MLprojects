# Fashion-MNIST CNN Classifier

This project implements a Convolutional Neural Network (CNN) model to classify images from the Fashion-MNIST dataset. The model is trained using TensorFlow/Keras and achieves a high accuracy on the test set.

## Overview

Fashion-MNIST is a dataset consisting of 60,000 28x28 grayscale images of 10 different fashion categories, with 10,000 test images. This project uses a CNN model to classify these images and evaluate the model's performance.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

You can install the required libraries with:

```bash
pip install tensorflow numpy matplotlib
```

## Dataset

The dataset is part of TensorFlow and is automatically loaded using `tensorflow.keras.datasets.fashion_mnist`. It includes the following classes:

- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

## Model Architecture

The model architecture consists of:
1. **Conv2D** layers for feature extraction.
2. **MaxPooling2D** layers to reduce dimensionality.
3. **Flatten** layer to convert the 2D matrix into a 1D vector.
4. **Dense** layers for classification.
5. **Softmax** activation to output class probabilities.

## Training

The model is trained using the following parameters:
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Metrics: Accuracy

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## Evaluation

After training, the model's performance is evaluated on the test set, yielding an accuracy of approximately 90.21% and a loss of 0.3906.

## How to Use

### Training the model:

To train the model, run the following script:

```bash
python train_model.py
```

This will load the Fashion-MNIST dataset, preprocess the data, and train the CNN model.

### Saving the model:

The trained model will be saved in the `fashion_mnist_cnn_model.keras` file format.

```python
model.save('fashion_mnist_cnn_model.keras')
```

### Loading the model:

You can load the model with the following code:

```python
from tensorflow.keras.models import load_model

model = load_model('fashion_mnist_cnn_model.keras')
```

### Predicting with the model:

To predict an image from the test dataset:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load a sample image from the test set
image = test_images[0]
label = test_labels[0]

# Reshape the image to fit the model
image = np.expand_dims(image, axis=0)

# Predict the class
predicted_label = model.predict(image)
print(f"Predicted label: {predicted_label}")
```
