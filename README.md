# This is an example interview project

## Installation

1. Clone the repo & move to it
### Using pip & venv
2. (Optional) Activate venv

```console
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies

```console
pip install --no-input -r requirements.txt
``` 
4. Run it

```console
python src/app_main.py "src/test_data/1586019697.598904_image_96493.jpg"
```
### Using Docker
2. Build & run container 

One liner:
```console
docker run --rm $(docker build -q .) 2>/dev/null
```
In separate steps:
```console
docker build .
docker run --rm your_image_hash 2>/dev/null
```

## The task
The aim of this task is to locate car couplings, that is the place where one train car attaches to the other. More
specifically, for each image, we want to output the x coordinate (horizontal axis) where the coupling is located.

While the dataset contains a bounding box for each coupling, your final output should only output the x coordinates.
Your solution should take a list of images as input and output the x coordinates, one per line:

```console
./find_couplings image1.jpg image2.jpg
435
456
```

How and with which tools you approach the problem is up to you. We only ask that the submitted solution contains all the
code including training (if there is any), that it can be run on a linux/ubuntu computer easily: either using pip
dependencies or a docker image.

If training is involved, it should not take more than an hour on a GTX 1070 comparable GPU (or have a lower accuracy
version that does not)

We are interested in a working solution, but do not expect you to spend large amounts of time on it. A few percent of
accuracy is not crucial as long as the basic algorithm works and you describe how you would go about pushing it further.

## Assumptions

- I assume that we can return some error message when no argument is provided
- I assume we need to return some value for every image