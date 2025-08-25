import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import requests
import os

# =================== Helper Functions =====================

def download_file(url, filename):
    """Download a file from a URL and save it locally."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded: {filename}")
    else:
        raise Exception(f"Failed to download {filename}. Status code: {response.status_code}")

# ================= Project and Dataset =====================
project = "histology"  # Options: "beans", "malaria", "histology", "bees"

dataset_url_prefix_dict = {
    "histology": "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Towards%20Precision%20Medicine/",
    "bees": "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Safeguarding%20Bee%20Health/"
}

if project == "beans":
    data, info = tfds.load('beans', split='train[:1024]', as_supervised=True, with_info=True)
    feature_dict = info.features['label'].names
    images = np.array([tf.image.resize_with_pad(image, 128, 128).numpy() for image, _ in data])
    labels = [feature_dict[int(label)] for _, label in data]

elif project == "malaria":
    data, info = tfds.load('malaria', split='train[:1024]', as_supervised=True, with_info=True)
    images = np.array([tf.image.resize_with_pad(image, 256, 256).numpy() for image, _ in data])
    labels = ['malaria' if label == 1 else 'healthy' for _, label in data]

else:  # For histology and bees datasets
    # Define file URLs
    image_url = f"{dataset_url_prefix_dict[project]}images.npy"
    labels_url = f"{dataset_url_prefix_dict[project]}labels.npy"

    # Download files
    download_file(image_url, "images.npy")
    download_file(labels_url, "labels.npy")

    # Load the downloaded data
    images = np.load("images.npy")
    labels = np.load("labels.npy")

    # Clean up: remove files after loading
    os.remove("images.npy")
    os.remove("labels.npy")

# ==================== Data Preprocessing ====================

# Encode labels to integers
class_names = sorted(list(set(labels)))
class_to_index = {label: i for i, label in enumerate(class_names)}
y = tf.keras.utils.to_categorical([class_to_index[label] for label in labels], num_classes=len(class_names))

X = images.astype("float32") / 255.0  # normalize

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==================== CNN Model =====================
cnn_model = tf.keras.Sequential([
    tf.keras.Input(shape=X_train.shape[1:]),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

cnn_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Data Augmentation (flip images vertically)
X_train_augment = np.array([tf.image.flip_up_down(img).numpy() for img in X_train])
y_train_augment = y_train

X_train_final = np.concatenate((X_train, X_train_augment), axis=0)
y_train_final = np.concatenate((y_train, y_train_augment), axis=0)

# Train the model
cnn_model.fit(X_train_final, y_train_final, epochs=35, validation_data=(X_test, y_test))

# ==================== Image Prediction =====================
image_path = "your_image.jpg"  # Replace with your image file path

# Load and preprocess the image with Keras utils
input_image = tf.keras.utils.load_img(image_path, target_size=X_train.shape[1:3])
input_image = tf.keras.utils.img_to_array(input_image) / 255.0
input_image = np.expand_dims(input_image, axis=0)

# Make prediction
predictions = cnn_model.predict(input_image)
class_index = np.argmax(predictions[0])
confidence = predictions[0][class_index]
class_label = class_names[class_index]

# Display prediction
print(f"Predicted Class: {class_label}")
print(f"Confidence: {confidence:.2f}")
