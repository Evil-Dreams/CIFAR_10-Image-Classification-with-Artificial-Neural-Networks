# CIFAR-10 Image Classification with Artificial Neural Networks

## Project Description
This project aims to classify images from the CIFAR-10 dataset using an Artificial Neural Network (ANN). The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to build and train an ANN model that can accurately identify the class of a given image.

## Dataset
The CIFAR-10 dataset is a widely used benchmark for image classification. It contains:
- **Training images:** 50,000 images, each 32x32 pixels with 3 color channels (RGB).
- **Testing images:** 10,000 images, each 32x32 pixels with 3 color channels (RGB).
- **Number of classes:** 10 (e.g., airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

### Preprocessing Steps
Before feeding the images into the neural network, the following preprocessing steps were applied:
1.  **Normalization:** Image pixel values (ranging from 0-255) were scaled down to a range of 0-1 by dividing by 255.0. This helps in faster convergence during training.
    - `x_train` normalized min value: 0.0
    - `x_train` normalized max value: 1.0
    - `x_test` normalized min value: 0.0
    - `x_test` normalized max value: 1.0
2.  **Flattening:** The 3D image data (32x32x3) was reshaped into a 1D vector (3072 features) for input into the dense layers of the ANN.
    - `x_train` shape after flattening: (50000, 3072)
    - `x_test` shape after flattening: (10000, 3072)

## Model Architecture
The Artificial Neural Network (ANN) model used for classification is a Sequential Keras model with the following layers:
-   **Input Layer:** Implicitly defined by the first Dense layer with `input_shape=(3072,)`.
-   **Hidden Layer 1:** A `Dense` layer with 128 neurons and `ReLU` activation function.
-   **Hidden Layer 2:** A `Dense` layer with 64 neurons and `ReLU` activation function.
-   **Output Layer:** A `Dense` layer with 10 neurons (corresponding to the 10 classes) and `Softmax` activation function to output class probabilities.

### Model Summary
Model: "sequential_1"
Total params: 402250 (1.54 MB) Trainable params: 402250 (1.54 MB) Non-trainable params: 0 (0.00 Byte)


## Training and Evaluation Results
The model was compiled using the `adam` optimizer and `sparse_categorical_crossentropy` loss function, with `accuracy` as the evaluation metric. It was trained for 10 epochs.

### Training History (Last Epoch)
-   **Training Accuracy:** 0.4638
-   **Training Loss:** 1.4911
-   **Validation Accuracy:** 0.4526
-   **Validation Loss:** 1.5363

### Final Test Evaluation
After training, the model's performance on the unseen test dataset (used as validation set during training) yielded:
-   **Test Accuracy:** 0.4526
-   **Test Loss:** 1.5363

## Setup and Installation
To set up and run this project, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment. Install the required packages using pip:
    ```bash
    pip install tensorflow keras scikit-learn pandas numpy matplotlib
    ```

3.  **Run the script:**
    Execute the main Python script containing the model definition, training, and evaluation logic:
    ```bash
    python your_script_name.py
    ```

## Dependencies
-   `tensorflow` (>=2.x)
-   `keras` (usually bundled with TensorFlow)
-   `scikit-learn`
-   `pandas`
-   `numpy`
-   `matplotlib`
