import glob
import json
import os


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def normalized(self, image_width, image_height):
        width_ratio = image_width / 224
        height_ratio = image_height / 224
        return Point(self.x / width_ratio, self.y / height_ratio)


def compute_point(points: list[list[float]]) -> Point:
    x_sum = 0
    y_sum = 0
    for point in points:
        x_sum += point[0]
        y_sum += point[1]
    x = x_sum / len(points)
    y = y_sum / len(points)
    return Point(x, y)


DIR_PATH = 'src/test_data'


def _get_original_json_paths():
    all_jsons = glob.glob(os.path.join(DIR_PATH, '*.json'))
    for json in all_jsons:
        if 'dumped' in json:
            all_jsons.remove(json)
    return all_jsons


def process_test_data():
    # Use glob to find all files ending in .json in the directory
    json_files = _get_original_json_paths()

    for file_path in json_files:
        process_file(file_path)


def process_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(data)
        point = compute_point(data['shapes'][0]['points'])
        left_upper_point = data['shapes'][0]['points'][0]
        left_upper_point = Point(left_upper_point[0], left_upper_point[1])
        right_lower_point = data['shapes'][0]['points'][2]
        right_lower_point = Point(right_lower_point[0], right_lower_point[1])
        add_point_shape(data, point, label='point')

        normalized_point = point.normalized(image_width=data['imageWidth'], image_height=data['imageHeight'])
        add_point_shape(data, normalized_point, label='normalized_point')

        add_point_shape(data, left_upper_point, label='left_upper_point')
        add_point_shape(data, right_lower_point, label='right_lower_point')

        filename = file_path + 'dumped.json'
        with open(filename, 'w') as dump_f:
            json.dump(data, dump_f, indent=4)
    print(data)


def add_point_shape(data, point, label):
    point_shape = {
        "label": label,
        "points": [
            [
                point.x,
                point.y
            ]
        ],
        "group_id": None,
        "shape_type": "point",
        "flags": {}
    }
    data['shapes'].append(point_shape)


if __name__ == '__main__':
    process_test_data()
