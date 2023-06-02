# object_detection

## Introduction
The Object Detection App is a powerful application that leverages the detr-resnet-50 model and Streamlit framework to perform object detection tasks. With this app, users can easily upload images and detect objects present in them, providing valuable insights for various use cases, including computer vision research, content analysis, and more.

This documentation serves as a comprehensive guide for installing, configuring, and using the Object Detection App effectively. It provides step-by-step instructions, tips, and explanations to ensure a smooth experience for both developers and end-users.

## Installation
Create a virtual envrionment and activate it using the following code block:
Below commands will be a bit different for windows (https://medium.com/co-learning-lounge/create-virtual-environment-python-windows-2021-d947c3a3ca78)

```
python -m venv venv
source venv/bin/activate
```

Install the requirements using the following commands:

```
pip install -r requirements.txt
```

## Usage
Run the following command to run the app:
```
streamlit run app.py
```

## Upload Image
Click on the "Browse Files" button in the app interface and select an image from your local machine. Wait for the image to be processed.

## View Object Detection Results
Once the image is processed, the app will display the uploaded image with bounding boxes around the detected objects. The labels for each object will be displayed alongside the bounding boxes.

## TODO:
Fine-tune and train it specific to wildlife species