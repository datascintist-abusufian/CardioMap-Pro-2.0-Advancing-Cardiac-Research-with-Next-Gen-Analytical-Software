import streamlit as st
import scipy.io
from PIL import Image
import numpy as np
import requests
import matplotlib.pyplot as plt
import tempfile
import os

# Streamlit cache decorator to ensure the function is only rerun when the inputs change.
@st.cache
def load_image(url):
    # Get the response from the URL
    response = requests.get(url)
    if response.status_code != 200:
        # If the response is not successful, raise an exception
        raise Exception(f"Failed to download image from {url}")
    
    # Create a temporary file to store the downloaded .mat file
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as fp:
        fp.write(response.content)
        # Ensure the file is flushed and written to disk
        fp.flush()
        os.fsync(fp.fileno())
        
    try:
        # Load the .mat file content
        mat_content = scipy.io.loadmat(fp.name)
        # Check if 'images' key exists in the .mat file
        if 'images' not in mat_content:
            raise KeyError("'images' key not found in the .mat file.")
        img_data = mat_content['images']
        # Convert the numpy array to an image
        img = Image.fromarray(np.uint8(img_data.squeeze()))
    finally:
        # Remove the temporary file after loading its content
        os.remove(fp.name)
    
    return img, img_data

def main():
    st.title("Image Viewer and Data Analysis")
    
    # GitHub API URL for the repository containing .mat files
    repo_url = 'https://api.github.com/repos/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/contents/mat_api'
    
    response = requests.get(repo_url)
    files = response.json()
    
    # Filter for .mat files
    mat_files = [file for file in files if file['name'].endswith('.mat')]
    
    if not mat_files:
        st.write("No .mat files found in the repository.")
        return
    
    # Create a slider in the sidebar to select a file
    file_index = st.sidebar.slider("Select an image", 0, len(mat_files) - 1)
    
    # Construct the raw GitHub URL for the selected file
    raw_url = mat_files[file_index]['download_url'].replace("https://github.com", "https://raw.githubusercontent.com").replace("/blob", "")
    
    # Load the image from the URL
    img, img_data = load_image(raw_url)
    
    # Display the image
    st.image(img, use_column_width=True)
    
    # Option to show histogram of the image data
    if st.sidebar.checkbox("Show histogram"):
        fig, ax = plt.subplots()
        ax.hist(img_data.ravel(), bins=256, color='gray')
        ax.set_title("Histogram of pixel values")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
