import sys

import tensorflow as tf
import numpy as np

from train_main import load_image, preprocess_image


def denormalize(prediction: float, original_image, preprocessed_image):
    print(original_image.shape)
    print(preprocessed_image.shape)
    original_width = original_image.shape[1]
    preprocessed_width = preprocessed_image.shape[1]
    return prediction * (original_width / preprocessed_width)


class CouplingsApp:
    def __init__(self):
        self.model = tf.keras.models.load_model('src/trained_model.keras')

    def run(self, image_paths):
        if image_paths is None or image_paths == []:
            print("No image paths provided")

        results = []
        for image_path in image_paths:
            print("Gonna load", image_path)
            image = load_image(image_path)
            preprocessed_image = preprocess_image(image)
            input_data_batched = np.expand_dims(preprocessed_image, axis=0)
            predictions = self.model.predict(input_data_batched)
            prediction = predictions[0][0]
            print(f'Prediction: {prediction/224}% from left?')
            prediction_denormalized = denormalize(prediction, original_image=image,
                                                  preprocessed_image=preprocessed_image)
            print(prediction_denormalized)
            results.append(prediction_denormalized)
        return results


if __name__ == '__main__':
    app = CouplingsApp()
    image_paths = sys.argv[1:]
    app.run(image_paths)
