import glob
import json
import os


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def compute_point(points: list[list[float]]) -> Point:
    x_sum = 0
    y_sum = 0
    for point in points:
        x_sum += point[0]
        y_sum += point[1]
    x = x_sum / len(points)
    y = y_sum / len(points)
    return Point(x, y)


def process_test_data():
    dir_path = 'test_data'
    # Use glob to find all files ending in .json in the directory
    json_files = glob.glob(os.path.join(dir_path, '*.json'))

    for file_path in json_files:
        process_file(file_path)


def process_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(data)
        point = compute_point(data['shapes'][0]['points'])
        point_shape = {
            "label": "point",
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

        filename = file_path + 'dumped.json'
        with open(filename, 'w') as dump_f:
            json.dump(data, dump_f, indent=4)
    print(data)


if __name__ == '__main__':
    process_test_data()
