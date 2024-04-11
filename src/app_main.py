import sys

import tensorflow as tf
import numpy as np

from train_main import CouplingsAppTrainer


class CouplingsApp:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.model = tf.keras.models.load_model('src/trained_model.keras')
        self.trainer = CouplingsAppTrainer()

    def denormalize(self, prediction: float, original_image, preprocessed_image):
        if self.verbose:
            print(original_image.shape)
            print(preprocessed_image.shape)
        original_width = original_image.shape[1]
        preprocessed_width = preprocessed_image.shape[1]
        return prediction * (original_width / preprocessed_width)

    def run(self, image_paths):
        if image_paths is None or image_paths == []:
            print("No image paths provided")

        results = []
        for image_path in image_paths:
            if self.verbose:
                print("Gonna load", image_path)
            image = self.trainer.load_image(image_path)
            preprocessed_image = self.trainer.preprocess_image(image)
            input_data_batched = np.expand_dims(preprocessed_image, axis=0)
            predictions = self.model.predict(input_data_batched, verbose=0)
            prediction = predictions[0][0]
            if self.verbose:
                print(f'Prediction: {prediction / 224}% from left?')
            prediction_denormalized = self.denormalize(prediction, original_image=image,
                                                       preprocessed_image=preprocessed_image)
            print(round(prediction_denormalized))
            results.append(prediction_denormalized)
        return results

    def print_model_summary(self):
        self.model.summary()


if __name__ == '__main__':
    app = CouplingsApp()
    image_paths = sys.argv[1:]
    app.run(image_paths)
    # app.print_model_summary()
