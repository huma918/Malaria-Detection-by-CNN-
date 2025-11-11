Malaria Cell Classification using Deep Learning

This project uses a Convolutional Neural Network (CNN) to classify red blood cell images as either Parasitized (infected with malaria) or Uninfected. The model is built using TensorFlow and Keras and includes a full pipeline from data exploration and augmentation to training and evaluation.

1. Project Overview

Malaria remains a significant global health issue, and accurate, timely diagnosis is critical for treatment. The standard method, manual inspection of blood smears by a microscopist, is time-consuming, subjective, and requires specialized training.

This project implements a deep learning model to automate this process, providing a fast, objective, and scalable solution for malaria detection.

2. The Dataset

This project uses the "Malaria Cell Images Dataset" from the National Institutes of Health (NIH).

Source: Kaggle: Malaria Cell Images Dataset

Contents: 27,558 cell images (in .png format).

Classes: Two balanced classes:

Parasitized: 13,779 images

Uninfected: 13,779 images

The dataset is organized into cell_images/cell_images/ with Parasitized and Uninfected subfolders, which is compatible with Keras's flow_from_directory method.

3. Methodology

The project follows a standard data science pipeline:

3.1. Exploratory Data Analysis (EDA)

Before training, the script performs an EDA to understand the dataset:

Class Distribution: Verifies the 50/50 split between 'Parasitized' and 'Uninfected' classes.

Image Dimensions: Analyzes the varying heights and widths of the images to determine a standard input size for the model.

Data Visualization: Displays sample images from each class to get a qualitative feel for the data.

3.2. Data Pre-processing and Augmentation

Data pre-processing is handled using tensorflow.keras.preprocessing.image.ImageDataGenerator.

Rescaling: All pixel values are normalized from [0, 255] to [0, 1] by setting rescale=1./255.

Train/Validation Split: 20% of the data is held back for validation (validation_split=0.2).

Data Augmentation: To prevent overfitting and make the model more robust, the following augmentations are applied to the training data only:

shear_range=0.2

zoom_range=0.2

horizontal_flip=True

Batching: Data is fed to the model in batches of 32.


3.4. Training & Evaluation

Compiler: The model is compiled with the adam optimizer and binary_crossentropy loss function.

Class Weights: To handle the slight imbalance in the training split (even though the full dataset is balanced), class_weights are calculated and applied during training.

Callbacks: ReduceLROnPlateau is used to decrease the learning rate if the validation loss stops improving.

Training: The model is trained for 30 epochs.

Evaluation: Model performance is evaluated using:

Accuracy & Loss Curves: Plotted for both training and validation sets.

Confusion Matrix: To visualize True Positives, True Negatives, False Positives, and False Negatives.

Classification Report: Provides precision, recall, and F1-score for each class.

4. Results

The model consistently achieves high accuracy (typically ~95%) on the validation set. The training and validation curves show that the model learns quickly and that the use of data augmentation and dropout successfully prevents major overfitting.

The final classification report and confusion matrix confirm that the model is highly effective at distinguishing between parasitized and uninfected cells, with well-balanced precision and recall.

5. How to Run This Project

Clone the Repository:

git clone https://github.com/huma918/Malaria-Detection-CNN.git
cd Malaria-Detection-CNN


Install Dependencies:
Make sure you have Python 3.8+ and install the required libraries:

pip install -r requirements.txt


Get the Data:

Download the dataset from Kaggle.

Unzip the file cell_images.zip.

Ensure the directory structure matches what the script expects:

your-repo-directory/
├── cell_images/
│   ├── cell_images/
│   │   ├── Parasitized/
│   │   │   ├── ... (images)
│   │   ├── Uninfected/
│   │   │   ├── ... (images)
├── final_project.py
├── final_eda.py
├── README.md
└── requirements.txt


(Note: The script expects the data to be in cell_images/cell_images/)

Run the Script:
This will run the full EDA, model training, and evaluation. All plots and the final model (.h5 file) will be saved to your directory.

python final_project.py


Files in This Repository

final_project.py (or final_eda.py): The main Python script that contains all code for the project.

requirements.txt: A list of all Python libraries needed to run the project.

README.md: This file.

final_eda.py: Complete Eda of the project
