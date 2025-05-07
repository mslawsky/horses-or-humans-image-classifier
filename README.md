# Horses vs Humans Image Classifier 🐴👨

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3+-yellow.svg)](https://matplotlib.org/)

This repository contains a convolutional neural network (CNN) implementation for classifying images of horses and humans using TensorFlow and Keras. The project demonstrates the effect of image size on model training and performance.

![Horse vs Human Examples](https://github.com/tensorflow/datasets/raw/master/tensorflow_datasets/image_classification/horses_or_humans/figure/horses_or_humans.png)

---

## Table of Contents 📋
- [Project Overview](#project-overview-)
- [Dataset Details](#dataset-details-)
- [Model Architecture](#model-architecture-)
- [Training Process](#training-process-)
- [Results](#results-)
- [Visualizations](#visualizations-)
- [Getting Started](#getting-started-)
- [Key Observations](#key-observations-)
- [Future Improvements](#future-improvements-)
- [Contact](#contact-)
- [License](#license-)
- [Acknowledgments](#acknowledgments-)

---

## Project Overview 🔎

The main goal of this project is to build and train a CNN that can accurately classify whether an image contains a horse or a human. Unlike simpler classification tasks, this project demonstrates how to work with real-world images using neural networks.

This project explores:
- Using compacted images (150x150) to improve training speed
- How changing image size affects the model architecture
- The trade-off between model size, training time, and accuracy
- Techniques for data preprocessing and visualization

---

## Dataset Details 📊

The project uses two datasets:
- `horse-or-human`: Training set with 500 horse images and 527 human images
- `validation-horse-or-human`: Validation set with 128 horse images and 128 human images

These datasets consist of color images (RGB) that will be resized to 150x150 pixels.

**Data Preprocessing:**
- Images are normalized from 0-255 pixel values to 0-1 range
- Data is organized in the following directory structure:
  ```
  horse-or-human/
  ├── horses/
  │   ├── horse01.png
  │   ├── horse02.png
  │   └── ...
  └── humans/
      ├── human01.png
      ├── human02.png
      └── ...
  ```
- Dataset is loaded using TensorFlow's image_dataset_from_directory API

## Model Architecture 🧠

The implemented CNN has the following structure:

```python
model = tf.keras.models.Sequential([
    # Input layer accepting 150x150 RGB images
    tf.keras.Input(shape=(150, 150, 3)),
    # First convolutional layer
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Second convolutional layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Third convolutional layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results
    tf.keras.layers.Flatten(),
    # Dense hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Output layer (0 for horses, 1 for humans)
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

**Architecture Breakdown:**
- **Input Layer**: Accepts 150x150 RGB images (3 channels)
- **Convolutional Layers**: Three layers of increasing complexity (16→32→64 filters)
- **Pooling Layers**: Max pooling with 2x2 windows to reduce spatial dimensions
- **Flatten Layer**: Converts 2D feature maps to 1D vector for dense layers
- **Dense Hidden Layer**: 512 neurons with ReLU activation for non-linearity
- **Output Layer**: Single neuron with sigmoid activation (0 for horse, 1 for human)

The model uses approximately 9.5 million trainable parameters, with most concentrated in the dense layer after flattening.

## Training Process 🔄

The model is trained using:
- RMSprop optimizer with learning rate of 0.001
- Binary cross-entropy loss function
- Normalized image data (scaled to 0-1 range)
- 15 epochs of training

```python
# Compile model with appropriate loss function and optimizer
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset_final,
    epochs=15,
    validation_data=validation_dataset_final,
    verbose=2)
```

The training process includes:
- Data normalization through a rescaling layer (1/255)
- Dataset shuffling with buffer size of 1000
- TensorFlow's data prefetching to optimize training speed
- Verbose logging to track accuracy and loss metrics

## Results 📈

The model achieves approximately 85-90% validation accuracy within 15 epochs. The training demonstrates:
- Fast convergence (high accuracy within few epochs)
- Signs of overfitting (training accuracy reaches 100% while validation stays lower)
- Good generalization to unseen images

**Training Metrics:**

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|-------------------|---------------------|---------------|-----------------|
| 1     | 63.8%             | 85.9%               | 0.758         | 0.379           |
| 5     | 98.1%             | 79.3%               | 0.054         | 1.864           |
| 10    | 100.0%            | 85.9%               | 0.001         | 1.817           |
| 15    | 100.0%            | 85.6%               | 0.000         | 2.672           |

The model starts to show signs of overfitting around epoch 5, where training accuracy continues to improve while validation accuracy plateaus. This suggests that a shorter training process with early stopping might be beneficial.

## Visualizations 📊

The repository includes code for various visualizations to help understand model performance and behavior:

### Training Metrics
![Training Accuracy Curve](https://raw.githubusercontent.com/yourusername/horses-vs-humans-classifier/main/images/accuracy_curve.png)
*Example visualization showing training and validation accuracy across epochs*

### Intermediate Layer Activations
The notebook includes code to visualize what different convolutional layers are "seeing" in the images:

```python
# Display the representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)
    
    # Tile the images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x
      
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
```

### Interactive Predictions
The notebook contains an interactive widget for uploading and classifying new images:

![Prediction Widget](https://raw.githubusercontent.com/yourusername/horses-vs-humans-classifier/main/images/prediction_widget.png)
*Example of the prediction widget interface*

## Getting Started 🚀

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

### Installation
```bash
# Clone this repository
git clone https://github.com/yourusername/horses-vs-humans-classifier.git

# Navigate to project directory
cd horses-vs-humans-classifier

# Install dependencies
pip install tensorflow numpy matplotlib
```

### Running the Code
```bash
# Run the notebook
jupyter notebook C1_W4_Lab_3_compacted_images.ipynb
```

---

## Key Observations 🔍

1. Using compacted 150x150 images (instead of 300x300) significantly reduced model size and training time
2. The model still achieved high accuracy despite the reduced image size
3. The CNN architecture with three convolutional layers showed strong performance
4. The model showed signs of overfitting after ~7 epochs (training accuracy at 100% while validation oscillated around 85-87%)
5. Training with smaller images reduces the parameter count in the model while maintaining good classification performance
6. Early layers in the CNN learn basic features (edges, textures) while deeper layers capture more complex patterns

**Architecture Comparison:**

| Image Size | Parameters | Training Time | Peak Validation Accuracy |
|------------|------------|---------------|--------------------------|
| 150×150    | 9.5M       | ~7s/epoch     | ~87%                     |
| 300×300    | 38M        | ~30s/epoch    | ~88%                     |

Using smaller images (150×150) provides nearly the same accuracy with 75% fewer parameters and significantly faster training times compared to larger (300×300) images.

## Future Improvements 🚀

- **Early Stopping**: Implement callbacks to halt training when validation accuracy plateaus
  ```python
  class AccuracyThresholdCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
          if logs.get('accuracy') >= 0.95:
              print("\nReached 95% accuracy - stopping training!")
              self.model.stop_training = True
  ```

- **Regularization Techniques**:
  - Add dropout layers between dense layers to reduce overfitting
  - Implement L2 regularization on convolutional layers
  - Use batch normalization to stabilize training

- **Architecture Experiments**:
  - Test different numbers of convolutional layers
  - Experiment with various filter counts (16, 32, 64, 128)
  - Try different kernel sizes for convolutions (3×3, 5×5)

- **Data Augmentation**: Implement image augmentation to improve generalization
  ```python
  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
  ])
  ```

- **Transfer Learning**: Leverage pre-trained models like VGG16, ResNet, or MobileNet as feature extractors
  ```python
  base_model = tf.keras.applications.MobileNetV2(
      input_shape=(150, 150, 3),
      include_top=False,
      weights='imagenet'
  )
  base_model.trainable = False
  ```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- The dataset is provided as part of the ["TensorFlow in Practice" specialization on Coursera](https://www.coursera.org/specializations/tensorflow-in-practice)
- Inspired by the work of the TensorFlow team and Laurence Moroney
- Special thanks to the deep learning community for their valuable resources and tutorials
- Image examples courtesy of the [TensorFlow Datasets repository](https://github.com/tensorflow/datasets)

