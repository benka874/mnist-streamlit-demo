import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.datasets import fetch_openml

st.title("High-Resolution Handwritten Digit Viewer (MNIST 28x28)")

@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    images = mnist['data'].reshape(-1, 28, 28)
    labels = mnist['target'].astype(int)
    return images, labels

images, labels = load_mnist()

digit = st.selectbox("Select a digit", list(range(10)))
digit_images = images[labels == digit]

if st.button("Generate Images"):
    selected_images = random.sample(list(digit_images), 5)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for ax, img in zip(axes, selected_images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    st.pyplot(fig)


