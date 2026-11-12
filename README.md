# Convolutional Neural Network (CNN) Built From Scratch in NumPy

## Project Overview
This project implements a complete deep learning framework and Convolutional Neural Network (`MNIST_CNN`) entirely from scratch using only **NumPy**. Built without high-level deep learning libraries like PyTorch or TensorFlow, the notebook covers forward propagation, manual backpropagation using the spatial chain rule, and Stochastic Gradient Descent (SGD) optimization to classify handwritten digits from the **MNIST** dataset.

---

## Custom Framework Architecture
The codebase is structured into modular, object-oriented components mirroring modern deep learning APIs:

### 1. Core Mathematics & Base Module
* **`NuralMaths`:** Handles scaled random weight/bias initialization and 3D spatial dot-product operations (`CNeuron`).
* **`Module`:** Base class inherited by all layers, managing `.train()` and `.eval()` execution states along with callable forward routing.

### 2. Neural Network Layers
* **`Conv2d`:** 2D convolutional layer supporting configurable input/output channels, kernel size, stride, and zero-padding. Implements manual spatial backpropagation for weight gradients, bias gradients, and padded input error routing.
* **`MaxPool2d`:** Spatial downsampling layer that extracts maximum window values during the forward pass and routes gradients strictly to active peak coordinates via boolean masking during backpropagation.
* **`Linear`:** Fully connected dense layer performing matrix multiplication ($W\mathbf{x} + \mathbf{b}$) and computing parameter and input gradients.
* **`Flatten`:** Reshapes multidimensional spatial feature maps into 1D vectors while preserving original tensor dimensions for backward gradient reconstruction.
* **`Dropout`:** Regularization layer that randomly zeroes neurons during training using inverted scaling ($\frac{1}{1-p}$) and passes inputs unchanged during evaluation.

### 3. Activations, Loss & Optimizer
* **Activation Functions (`F` & Layer Wrappers):** Implemented `ReLU`, `Sigmoid`, `Tanh`, and numerically stable `Softmax` along with their analytical derivatives.
* **`CrossEntropyLoss`:** Computes categorical cross-entropy penalty with $\epsilon$-stability and combined Softmax-CrossEntropy backward gradients.
* **`SGD`:** Stochastic Gradient Descent optimizer managing parameter updates and gradient clearing (`zero_grad`).

---

## Model Architecture (`MNIST_CNN`)
The network processes $28 \times 28$ grayscale MNIST digit images through the following pipeline:
1. **`Conv2d`:** 1 input channel $\rightarrow$ 8 output filters ($3 \times 3$ kernel, `padding=1`, `stride=1`)
2. **`ReLU`:** Element-wise non-linear activation
3. **`MaxPool2d`:** $2 \times 2$ pooling window with `stride=2` (reduces spatial dimensions to $8 \times 14 \times 14$)
4. **`Flatten`:** Reshapes feature maps into a 1,568-element vector
5. **`Linear`:** Maps 1,568 features to 10 digit class logits
6. **`Softmax`:** Outputs class probability distributions

---

## Training & Evaluation Results
* **Dataset:** MNIST handwritten digits fetched via `scikit-learn` (`fetch_openml`), normalized to $[0, 1]$ and one-hot encoded (2,500 training images, 500 test images).
* **Hyperparameters:** 5 epochs, SGD optimizer with learning rate `0.01`.
* **Training Progression:**
  * **Epoch 1:** Loss `0.8506` | Accuracy `72.68%`
  * **Epoch 2:** Loss `0.4219` | Accuracy `87.28%`
  * **Epoch 3:** Loss `0.3597` | Accuracy `89.32%`
  * **Epoch 4:** Loss `0.3137` | Accuracy `90.44%`
  * **Epoch 5:** Loss `0.2672` | Accuracy `92.32%`
* **Final Out-of-Sample Test Accuracy:** **`89.00%`**

---

## Tech Stack & Dependencies
* **Python 3.x**
* `numpy` (Core tensor operations and manual backpropagation)
* `scikit-learn` (MNIST dataset retrieval via `fetch_openml`)
* `matplotlib` (Digit visualization and error analysis plots)

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/TavishMassy/CODEC_HandwrittenDigitRecognizer
   cd CODEC_HandwrittenDigitRecognizer
   ```
2. Install dependencies:
   ```bash
   pip install numpy scikit-learn matplotlib
   ```
3. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
