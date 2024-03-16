import streamlit as st
import scipy.io
from PIL import Image
import numpy as np
import requests
import matplotlib.pyplot as plt
import tempfile
import os

@st.cache
def load_image(url, headers):
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {url}")
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as fp:
        fp.write(response.content)
        fp.flush()
        os.fsync(fp.fileno())
    try:
        mat_content = scipy.io.loadmat(fp.name)
        if 'images' not in mat_content:
            raise KeyError("'images' key not found in the .mat file.")
        img_data = mat_content['images']
        img = Image.fromarray(np.uint8(img_data.squeeze()))
    finally:
        os.remove(fp.name)
    return img, img_data

def main():
    st.title("Image Viewer and Data Analysis")
    
    repo_url = 'https://api.github.com/repos/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/contents/mat_api'
    
    # Use your GitHub personal access token
    headers = {'Authorization': 'token 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'}
    
    response = requests.get(repo_url, headers=headers)
    if response.status_code != 200:
        st.error(f"Failed to fetch data from the GitHub API. Status code: {response.status_code}, Response: {response.text}")
        st.stop()

    try:
        files = response.json()
    except ValueError:
        st.error("Invalid response received from the GitHub API.")
        st.stop()

    if not isinstance(files, list):
        st.error("Unexpected format received from the GitHub API. Expected a list of files.")
        st.stop()

    mat_files = [file for file in files if file['name'].endswith('.mat')]
    if not mat_files:
        st.error("No .mat files found.")
        st.stop()

    file_index = st.sidebar.slider("Select an image", 0, len(mat_files) - 1)
    
    raw_url = mat_files[file_index]['download_url'].replace("https://github.com", "https://raw.githubusercontent.com").replace("/blob", "")
    
    img, img_data = load_image(raw_url, headers)
    
    st.image(img, use_column_width=True)
    
    if st.sidebar.checkbox("Show histogram"):
        fig, ax = plt.subplots()
        ax.hist(img_data.ravel(), bins=256, color='gray')
        ax.set_title("Histogram of pixel values")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
