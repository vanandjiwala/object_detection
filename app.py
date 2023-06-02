import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrImageProcessor, DetrForObjectDetection

def detect_image(image):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(box)
        x, y, width, height = box

        # Create a rectangle patch
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rect)

        label_text = model.config.id2label[label.item()]
        confidence = round(score.item(), 3)
        ax.text(x, y, f"{label_text}: {confidence}", color='r')

        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    return plt

def main():
    st.title('Object Detection App')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    side_bar =  st.sidebar
    uploaded_file = side_bar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image using Pillow
        image = Image.open(uploaded_file)

        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        plt = detect_image(image)
        st.pyplot()

if __name__ == "__main__":
    main()

