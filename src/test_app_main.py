import json

from app_main import CouplingsApp
from train_main import CouplingsAppTrainer


def test_run():
    image_paths = ['src/test_data/1586019697.598904_image_96493.jpg']
    app = CouplingsApp()
    predictions = app.run(image_paths)
    print(predictions)
    assert len(predictions) == 1
    assert predictions[0] > 1500
    assert predictions[0] < 2000


def _is_valid_prediction(image_path, prediction: float) -> bool:
    json_path = CouplingsAppTrainer.image_path_to_json_path(image_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
        left_bound = data['shapes'][3]['points'][0][0]
        right_bound = data['shapes'][4]['points'][0][0]
    return left_bound < prediction < right_bound


def validate_run():
    trainer = CouplingsAppTrainer()
    image_paths = trainer.get_image_paths()
    print(image_paths)
    assert len(image_paths) > 0
    correct = 0
    app = CouplingsApp()
    predictions = app.run(image_paths)

    paths_and_predictions = list(zip(image_paths, predictions))
    for image_path, prediction in paths_and_predictions:
        if _is_valid_prediction(image_path, prediction):
            correct += 1

    print("There was a total of", correct, "correct predictions out of", len(image_paths), "images")


if __name__ == '__main__':
    validate_run()
