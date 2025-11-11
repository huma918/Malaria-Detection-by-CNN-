import os
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# Load Data
data_dir = 'cell_images/cell_images/'

# Check if the dataset folder exists, if not, unzip the dataset
if not os.path.exists(data_dir):
    zip_file_path = 'cell_images.zip'  # Path to your zip file
    extract_dir = 'cell_images'  # Folder where you want to extract

    if not os.path.exists(extract_dir):  # If folder doesn't exist, unzip the data
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted {zip_file_path} to {extract_dir}")
    else:
        print(f"Directory {extract_dir} already exists, skipping extraction.")

# Data Setup
image_shape = (150, 150, 3)
batch_size = 32

# Data augmentation and preprocessing for EDA
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Distribution of Classes
parasitized_count = len(os.listdir(os.path.join(data_dir, 'Parasitized')))
uninfected_count = len(os.listdir(os.path.join(data_dir, 'Uninfected')))

plt.figure(figsize=(8, 6))
plt.bar(['Parasitized', 'Uninfected'], [parasitized_count, uninfected_count], color=['red', 'green'])
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Distribution of Parasitized and Uninfected Images')
plt.savefig('class_distribution.png')
plt.show()

# Check the size of the images and the number of images
image_sizes = []
for category in ['Parasitized', 'Uninfected']:
    image_folder = os.path.join(data_dir, category)
    for img_name in os.listdir(image_folder):
        try:
            # Add error handling for corrupted images
            image_sizes.append(plt.imread(os.path.join(image_folder, img_name)).shape)
        except Exception as e:
            print(f"Error reading {img_name}: {e}")

image_sizes = np.array([size for size in image_sizes if len(size) == 3]) # Ensure all are 3-channel images
height, width = image_sizes[:, 0], image_sizes[:, 1]

# Histogram of image heights and widths
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(height, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Image Heights')
plt.xlabel('Height')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(width, bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Image Widths')
plt.xlabel('Width')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('image_size_distribution.png')
plt.show()

# Correlation plot (for illustrative purposes)
image_data = {
    'mean_height': height,
    'mean_width': width,
    'pixel_intensity': np.random.rand(len(height))  # Replace with actual pixel intensity data if available
}

df = pd.DataFrame(image_data)

# Plotting correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# Boxplot to check for any outliers in the data distribution
plt.figure(figsize=(8, 6))
sns.boxplot(data=[height, width], orient='h', palette='Set2')
plt.title('Boxplot of Image Heights and Widths')
plt.xlabel('Pixel Value')
plt.yticks([0, 1], ['Height', 'Width'])
plt.savefig('boxplot_image_dimensions.png')
plt.show()

# Data Augmentation (Visualizing Augmented Images)
# Re-define train_datagen for visualization with a small batch size
viz_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

viz_generator = viz_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    subset='training'  # Note: This will take from the training set
)

# Visualizing augmented images
x_batch, y_batch = next(viz_generator)
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_batch[i])
    plt.title('Parasitized' if y_batch[i] == 1 else 'Uninfected')
    plt.axis('off')
plt.suptitle('Augmented Images')
plt.savefig('augmented_images.png')
plt.show()

# Display Sample Images from Dataset
def display_samples(generator):
    # Use the visualization generator
    x, y = next(generator)
    batch_size_viz = x.shape[0]  # Get the actual number of images in the batch (it should be 5 here)
    plt.figure(figsize=(10, 10))
    
    for i in range(batch_size_viz):
        plt.subplot(2, 3, i+1)
        plt.imshow(x[i])
        plt.title('Parasitized' if y[i] == 1 else 'Uninfected')
        plt.axis('off')
    
    plt.savefig('sample_images.png')
    plt.show()

display_samples(viz_generator) # Pass the correct generator

# CNN Model Creation
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Calculate class weights to address class imbalance
# Use the counts from the initial EDA
class_weights = {0: (parasitized_count + uninfected_count) / (2 * uninfected_count), # Weight for class 0 (Uninfected)
                 1: (parasitized_count + uninfected_count) / (2 * parasitized_count)} # Weight for class 1 (Parasitized)

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Model Training
model = create_model()
model.summary()

history = model.fit(
    train_generator,  # Use the main generator with batch_size 32
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=30,
    validation_data=validation_generator, # Use the main validation generator
    validation_steps=validation_generator.samples // batch_size,
    class_weight=class_weights,
    callbacks=[lr_scheduler]
)

# Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()

# Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss.png')
plt.show()

# Model Saving for User App
model.save('malaria_detection_model_keras_v1.h5')

# Loading of Model
loaded_model = tf.keras.models.load_model('malaria_detection_model_keras_v1.h5')

# Prediction
model.evaluate(validation_generator)

pred_probabilities = model.predict(validation_generator).flatten()
predictions = (pred_probabilities > 0.5).astype(np.int32)
true_labels = validation_generator.classes

# Confusion matrix
conf_mat = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Uninfected', 'Parasitized'], yticklabels=['Uninfected', 'Parasitized'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Classification report
print('Classification Report:')
print(classification_report(true_labels, predictions, target_names=['Uninfected', 'Parasitized']))