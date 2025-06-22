import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import random

(x_train, y_train), (_, _) = mnist.load_data()

st.title("MNIST Handwritten Digit Generator")
st.write("Select a digit (0–9) and generate 5 MNIST-style handwritten images.")

digit = st.selectbox("Choose a digit", list(range(10)))
digit_images = x_train[y_train == digit]

if st.button("Generate Images"):
    selected_images = random.sample(list(digit_images), 5)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for ax, img in zip(axes, selected_images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
