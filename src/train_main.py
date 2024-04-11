import glob
import json
import os

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from process_test_data import DIR_PATH


def create_model():
    # Load the pre-trained MobileNetV2 model without the top layer
    base_model = MobileNetV2(weights='imagenet', include_top=False)

    # Freeze the base model
    base_model.trainable = False

    # Adding custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Global average pooling layer at the end has 1280
    x = Dense(1280, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(1)(x)  # Single output neuron without activation function for regression

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model with a regression loss function
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def load_and_preprocess_image(path):
    image = load_image(path)
    image = preprocess_image(image)
    return image


def preprocess_image(image):
    image = tf.image.resize(image, [224, 224])  # Resize images if needed
    image = image / 255  # Normalize pixel values to [0, 1]
    return image


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def get_image_paths() -> list[str]:
    # Used glob to find all files ending in .json in the directory
    image_paths_jpeg = glob.glob(os.path.join(DIR_PATH, '*.jpeg'))
    image_paths_jpg = glob.glob(os.path.join(DIR_PATH, '*.jpg'))

    return image_paths_jpg + image_paths_jpeg


def get_image_points(image_paths) -> list[float]:
    points = []
    for path in image_paths:
        json_path = image_path_to_json_path(path)
        with open(json_path, 'r') as f:
            data = json.load(f)
            x_coordinate = data['shapes'][2]['points'][0][0]
            points.append(x_coordinate)
    return points


def image_path_to_json_path(path):
    stem = path.replace('.jpg', '')
    stem = stem.replace('.jpeg', '')
    json_path = stem + '.jsondumped.json'
    return json_path


def run_training():
    model = create_model()
    dataset = load_dataset()

    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)

    dataset = dataset.shuffle(buffer_size=num_samples)

    # Split the dataset into training and validation
    train_dataset = dataset.take(train_size)
    print(train_dataset)
    val_dataset = dataset.skip(train_size)
    print(val_dataset)

    batch_size = 32
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    # Epoch size lower than 40 was observed to be too small
    model.fit(train_dataset, validation_data=val_dataset, epochs=40)

    val_loss = model.evaluate(val_dataset)
    print(f'Validation loss: {val_loss}')

    model.save('trained_model.keras')


def load_dataset():
    image_paths = get_image_paths()
    points = get_image_points(image_paths)
    # Load all images into a list of tensors
    images = [load_and_preprocess_image(path) for path in image_paths]
    images = tf.convert_to_tensor(images)
    points = tf.convert_to_tensor(points)
    print(images)
    print(points)
    # Create a TensorFlow dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((images, points))
    return dataset


if __name__ == '__main__':
    run_training()
