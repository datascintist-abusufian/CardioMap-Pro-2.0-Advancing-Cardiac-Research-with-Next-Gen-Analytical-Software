import streamlit as st
import os
import scipy.io
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt

@st.cache
def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_data = np.array(img)
    return img, img_data

def main():
    st.title("Image Viewer and Data Analysis")

    # Provide the GitHub API URL for your repository
    repo_url = 'https://api.github.com/repos/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/contents/mat_api'

    response = requests.get(repo_url)
    files = response.json()

    mat_files = [file for file in files if file['name'].endswith('.mat')]

    file_index = st.sidebar.slider("Select an image", 0, len(mat_files) - 1)

    img, img_data = load_image(mat_files[file_index]['download_url'])

    st.image(img, use_column_width=True)

    if st.sidebar.checkbox("Show histogram"):
        fig, ax = plt.subplots()
        ax.hist(img_data.ravel(), bins=256, color='gray')
        ax.set_title("Histogram of pixel values")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
