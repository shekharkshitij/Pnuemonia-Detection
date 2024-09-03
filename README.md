# Chest X-Ray Pneumonia Detection

This project involves loading, preprocessing, and analyzing chest X-ray images to detect pneumonia using machine learning models. The data is preprocessed to prepare it for model training, testing, and validation.

## Project Overview

The goal of this project is to classify chest X-ray images as either **PNEUMONIA** or **NORMAL** using deep learning techniques. The images are loaded and preprocessed to a consistent format before being fed into a machine learning model for training.

## Dataset

The dataset contains chest X-ray images stored in three directories:

- `train`: Training images
- `test`: Testing images
- `val`: Validation images

Each directory contains two subdirectories:
- `PNEUMONIA`: X-ray images diagnosed with pneumonia.
- `NORMAL`: X-ray images of healthy patients.

## Prerequisites

Make sure you have the following libraries installed:

- Python 3.x
- OpenCV
- NumPy

You can install these libraries using pip:

```sh
pip install opencv-python-headless numpy
```

## Data Preprocessing

The data preprocessing involves loading images, converting them to grayscale, resizing them to a fixed size, and pairing them with their corresponding labels. Non-image files, such as `.DS_Store` (created by macOS), are ignored during the loading process.

### Preprocessing Function

The `get_training_data` function loads and preprocesses the data:

```python
import os
import cv2
import numpy as np

labels = ['PNEUMONIA', 'NORMAL']
img_size = 200

def get_training_data(data_dir):
    data = []  # List to store both images and labels together

    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                # Check if the file is an image
                if img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'):
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))
                    data.append([resized_arr, class_num])  # Append both image and label together
            except Exception as e:
                print(f"Error loading image {img}: {e}")

    # Convert to numpy array and return
    return np.array(data, dtype=object)
```

### Usage

To use the function, call it with the path to the training, testing, or validation directories:

```python
train_data = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/train')
test_data = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
val_data = get_training_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')

# Access images and labels separately
train_images = np.array([item[0] for item in train_data])
train_labels = np.array([item[1] for item in train_data])
```

## Notes

- Ensure the dataset is organized correctly with subdirectories for each class under `train`, `test`, and `val` directories.
- The `get_training_data` function handles errors gracefully and skips non-image files.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgements

- The dataset used in this project is part of the "Chest X-Ray Images (Pneumonia)" dataset available on Kaggle.

---
