import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np
import random

# Load digits dataset from sklearn
digits = load_digits()
images = digits.images
labels = digits.target

st.title("Handwritten Digit Viewer (0–9)")
digit = st.selectbox("Select a digit", list(range(10)))

# Filter images for the selected digit
digit_images = images[labels == digit]

if st.button("Generate Images"):
    selected_images = random.sample(list(digit_images), 5)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for ax, img in zip(axes, selected_images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    st.pyplot(fig)

