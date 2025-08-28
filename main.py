import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
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
project = "histology"  

dataset_url_prefix_dict = {
    "histology": "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Towards%20Precision%20Medicine/",
}

# For histology datasets
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

# Train-test split of the dataset
def tf_train_test_split(X, y, test_size=0.2, random_state=42):
    """TensorFlow implementation of train-test split"""
    tf.random.set_seed(random_state)
    n_samples = X.shape[0]
    indices = tf.range(n_samples)
    shuffled_indices = tf.random.shuffle(indices)
    
    test_size = int(n_samples * test_size)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    
    X_train = tf.gather(X, train_indices).numpy()
    X_test = tf.gather(X, test_indices).numpy()
    y_train = tf.gather(y, train_indices).numpy()
    y_test = tf.gather(y, test_indices).numpy()
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = tf_train_test_split(X, y, test_size=0.2, random_state=42)

# ==================== CNN Model =====================
cnn_model = tf.keras.Sequential([
    tf.keras.Input(shape=X_train.shape[1:]),
    
# Model Layerts

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

# Data Augmentation

# Convert to TensorFlow tensors for efficient augmentation
X_train_tf = tf.convert_to_tensor(X_train)

# 1. Vertical flips
X_train_vert = tf.image.flip_up_down(X_train_tf).numpy()
y_train_vert = y_train

# 2. Horizontal flips
X_train_horiz = tf.image.flip_left_right(X_train_tf).numpy()
y_train_horiz = y_train

# 3. Random rotations (90°, 180°, 270°)
def apply_random_rotations(images):
    rotated_images = []
    for img in images:
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        rotated_images.append(tf.image.rot90(img, k=k).numpy())
    return np.array(rotated_images)

X_train_rotate = apply_random_rotations(X_train_tf)
y_train_rotate = y_train

# 4. Random brightness adjustments
X_train_bright = tf.image.random_brightness(X_train_tf, max_delta=0.2).numpy()
y_train_bright = y_train

# 5. Random contrast adjustments
X_train_contrast = tf.image.random_contrast(X_train_tf, lower=0.8, upper=1.2).numpy()
y_train_contrast = y_train

# Combine all augmented datasets
X_train_final = np.concatenate([
    X_train,           # Original
    X_train_vert,      # Vertical flips
    X_train_horiz,     # Horizontal flips  
    X_train_rotate,    # Rotations
    X_train_bright,    # Brightness
    X_train_contrast   # Contrast
], axis=0)

y_train_final = np.concatenate([
    y_train,
    y_train_vert,
    y_train_horiz,
    y_train_rotate,
    y_train_bright,
    y_train_contrast
], axis=0)

# Augmenting makes the dataset 6.0x larger

print(f"Original training samples: {len(X_train)}")
print(f"Augmented training samples: {len(X_train_final)}")
print(f"Augmentation factor: {len(X_train_final)/len(X_train):.1f}x")

# Train 
cnn_model.fit(X_train_final, y_train_final, epochs=35, validation_data=(X_test, y_test))

# ==================== Image Prediction =====================
image_path = r"C:\Users\mithr_z9a5h10\COMPUTER SCIENCE\MFI Lab\histologyCNN\slide4.jpg"  

# Load and preprocess 
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
