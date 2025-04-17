# Remote Sensing Image Classification with Data Augmentation and Transfer Learning

This project focuses on classifying remote sensing images by leveraging data augmentation techniques and transfer learning. The approach involves enhancing the training dataset through various augmentation methods, training a Convolutional Neural Network (CNN) on this augmented data, and applying transfer learning using the Inception V3 model from PyTorch's torchvision model hub.

## 🛰️ Objective

- **Data Augmentation**: Enhance the diversity and size of the training dataset using techniques such as random cropping, flipping, rotation, and color jittering.
- **CNN Training**: Train a custom CNN model on the augmented dataset to establish a performance baseline.
- **Transfer Learning**: Utilize the pre-trained Inception V3 model to improve classification accuracy on the augmented dataset.

## 📁 Dataset

The project uses the **UCMerced Land Use Dataset**, which contains 2,100 aerial images (256x256 pixels) from 21 land use categories.

### 🔽 Download Instructions (Google Colab)

Use the following commands in a Colab notebook to download and extract the dataset:

```python
!wget http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
!unzip UCMerced_LandUse.zip -d ./UCMerced_LandUse

## 🧪 Methodology

### 1. Data Augmentation

Applied the following augmentation techniques to the training dataset:

- **Random Resized Crop**: Randomly crop and resize images to a specified size.
- **Random Horizontal Flip**: Flip images horizontally with a certain probability.
- **Random Rotation**: Rotate images within a specified degree range.
- **Color Jitter**: Randomly change the brightness, contrast, saturation, and hue.

These augmentations were implemented using `torchvision.transforms` to increase the robustness of the model against variations in the data.

### 2. CNN Training

Developed a custom CNN architecture consisting of:

- Convolutional layers with ReLU activation
- MaxPooling layers for downsampling
- Fully connected layers leading to the output

Trained this model on the augmented dataset to serve as a baseline for comparison with the transfer learning approach.

### 3. Transfer Learning with Inception V3

Utilized the pre-trained Inception V3 model from PyTorch's torchvision model hub:

- **Model Loading**:
```python
from torchvision import models
inception = models.inception_v3(pretrained=True)
