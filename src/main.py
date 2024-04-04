import glob
import json
import os

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the pre-trained MobileNetV2 model without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Freeze the base model
base_model.trainable = False

# Add your own layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Dense layer
predictions = Dense(1)(x)  # Single output neuron without activation function for regression

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with a regression loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])  # Resize images if needed
    image = image / 255  # Normalize pixel values to [0, 1]
    return image


def get_image_paths() -> list[str]:
    dir_path = 'test_data'
    # Use glob to find all files ending in .json in the directory
    image_paths_jpeg = glob.glob(os.path.join(dir_path, '*.jpeg'))
    image_paths_jpg = glob.glob(os.path.join(dir_path, '*.jpg'))

    return image_paths_jpg + image_paths_jpeg


image_paths = get_image_paths()


def get_image_points(image_paths) -> list[float]:
    points = []
    for path in image_paths:
        stem = path.replace('.jpg', '')
        stem = stem.replace('.jpeg', '')
        json_path = stem + '.jsondumped.json'
        with open(json_path, 'r') as f:
            data = json.load(f)
            x_coordinate = data['shapes'][2]['points'][0][0]
            points.append(x_coordinate)
    return points


points = get_image_points(image_paths)

# Load all images into a list of tensors
images = [load_and_preprocess_image(path) for path in image_paths]
images = tf.convert_to_tensor(images)
points = tf.convert_to_tensor(points)


print(images)
print(points)

# Create a TensorFlow dataset from tensors
dataset = tf.data.Dataset.from_tensor_slices((images, points))

# Calculate the number of samples
num_samples = images.shape[0]

# Determine train and validation sizes
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=num_samples)

# Split the dataset into training and validation
train_dataset = dataset.take(train_size)
print(train_dataset)
val_dataset = dataset.skip(train_size)
print(val_dataset)

batch_size = 32
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# Example usage - remember to prepare your data accordingly
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
