# Remote Sensing Image Classification with Data Augmentation and Transfer Learning

This project performs remote sensing image classification by leveraging **data augmentation** and **transfer learning**. The workflow involves augmenting the training data, training a custom CNN model, and applying **Inception V3** (pre-trained on ImageNet) to enhance classification accuracy.

---

## 🛰️ Objective

- Augment the original training dataset using multiple techniques to improve model generalization.
- Train a custom CNN model using the augmented dataset.
- Apply **Inception V3** for transfer learning to further improve performance on remote sensing image classification tasks.

---

## 📁 Data

The dataset used is the **UCMerced Land Use Dataset**, containing 2,100 aerial images (256×256 pixels) evenly distributed across 21 land use classes.

### 📥 Download Instructions (for Google Colab)

Use the following commands in Google Colab to download and unzip the dataset:

```python
!wget http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip
!unzip UCMerced_LandUse.zip -d ./UCMerced_LandUse
