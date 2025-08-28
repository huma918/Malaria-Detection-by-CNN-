import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CNN Model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

## Load Data
data_dir = 'cell_images/cell_images/'

## Data Setup
image_shape = (150,150,3)
batch_size = 32

# Data augmentation and preprocessing
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

## Distribution of Classes
parasitized_count = len(os.listdir(os.path.join(data_dir, 'Parasitized')))
uninfected_count = len(os.listdir(os.path.join(data_dir, 'Uninfected')))

plt.figure(figsize=(8, 6))
plt.bar(['Parasitized', 'Uninfected'], [parasitized_count, uninfected_count], color=['red', 'green'])
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Distribution of Parasitized and Uninfected Images')
plt.savefig('class_distribution.png')
plt.show()

## Display Data
def display_samples(generator):
    x, y = next(generator)
    plt.figure(figsize=(10, 10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(x[i])
        plt.title('Parasitized' if y[i] == 1 else 'Uninfected')
        plt.axis('off')
    plt.savefig('sample_images.png')
    plt.show()

display_samples(train_generator)

## Pixel value Distribution
def plot_pixel_histogram(image_array, title, filename):
    image = image_array.flatten()
    plt.hist(image, bins=50, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.show()

def visualize_pixel_values(generator):
    x, y = next(generator)
    parasitized_img = x[y == 1][0]
    uninfected_img = x[y == 0][0]

    plot_pixel_histogram(parasitized_img, 'Pixel Value Distribution for Parasitized Image', 'parasitized_pixel_values.png')
    plot_pixel_histogram(uninfected_img, 'Pixel Value Distribution for Uninfected Image', 'uninfected_pixel_values.png')

visualize_pixel_values(train_generator)

## Creating Model
model = create_model()
model.summary()

# Model Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
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