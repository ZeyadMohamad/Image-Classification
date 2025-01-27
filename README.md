# **Intel Image Classification**

## **Overview**
This project aims to classify images into various categories based on the **Intel Image Classification dataset**. The dataset contains images representing six different classes of natural scenes:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The project leverages deep learning techniques to train a model capable of achieving high accuracy in image classification tasks. It is designed for research, experimentation, and practical application in computer vision tasks.

---

## **Dataset**
The dataset used in this project is sourced from Kaggle:

[Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

### **Dataset Details:**
- **Training images**: Used to train the model.
- **Validation images**: Used for hyperparameter tuning and early stopping.
- **Test images**: Used to evaluate the model's final performance.
- **Image resolution**: The images are provided in 150x150 resolution.

---

## **Technologies Used**
- **Programming Language**: Python
- **Frameworks and Libraries**:
  - TensorFlow/Keras (Deep Learning Framework)
  - NumPy (Numerical computations)
  - Matplotlib (Data visualization)
  - Scikit-learn (Evaluation metrics and utilities)

---

## **Implementation Steps**
1. **Data Preprocessing**:
   - Loaded the dataset and split it into training, validation, and test sets.
   - Applied image augmentation techniques (e.g., rotation, flipping, scaling) to improve model generalization.

2. **Model Development**:
   - Built a Convolutional Neural Network (CNN) using Keras.
   - Designed a model architecture consisting of convolutional layers, pooling layers, and dense layers for classification.

3. **Model Training**:
   - Used training data to train the CNN model.
   - Implemented callbacks such as early stopping and learning rate reduction.

4. **Evaluation**:
   - Assessed model performance using metrics such as accuracy, precision, recall, and F1-score.
   - Visualized training and validation losses over epochs.

5. **Prediction and Visualization**:
   - Made predictions on test images.
   - Visualized the correctly and incorrectly classified images.

