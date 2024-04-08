FROM python:3.10 as base
LABEL authors="Jaroslav Otradovec"
# Base Stage
RUN python3 --version
RUN python --version
COPY ./requirements.txt ./
RUN pip install --disable-pip-version-check --no-input -r requirements.txt
COPY src/ ./src/

# Change the image path below to your needs
ENTRYPOINT ["python","src/app_main.py", "src/test_data/1586019697.598904_image_96493.jpg"]

