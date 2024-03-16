import os
import streamlit as st
import requests
from io import BytesIO
import scipy.io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# GitHub repository and directory details
GITHUB_REPO = "datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software"
MAT_API_DIR = "mat_api"

@st.cache
def get_mat_files_from_github(repo_path, folder_path):
    """Fetch list of .mat files from a specified GitHub directory."""
    api_url = f"https://api.github.com/repos/{repo_path}/contents/{folder_path}"
    response = requests.get(api_url)
    if response.status_code == 200:
        files = response.json()
        return [file for file in files if file['name'].endswith('.mat')]
    else:
        st.error("Failed to fetch files from GitHub.")
        return []

@st.cache(allow_output_mutation=True)
def load_mat_data_from_url(url):
    """Load .mat data from a given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        content = BytesIO(response.content)
        mat_data = scipy.io.loadmat(content)
        return mat_data
    else:
        st.error("Failed to load data from URL.")
        return None

def display_mat_data(mat_data, key='image_data'):
    """Display data from a .mat file."""
    if key in mat_data:
        img_data = mat_data[key]
        if img_data.ndim == 3:
            # Convert to grayscale if necessary
            img_data = np.mean(img_data, axis=2)
        img = Image.fromarray(np.uint8(img_data))
        st.image(img, use_column_width=True)
    else:
        st.write(f"No data found under key '{key}' in the .mat file.")

def activation_map(data):
    # Define a threshold for activation
    threshold = 0.01 * np.max(data)

    # Calculate the activation map
    activation_map = np.argmax(data > threshold, axis=2)

    st.write(f"Activation Map: {activation_map}")

def main():
    st.title("Streamlit MAT File Analyzer")

    mat_files = get_mat_files_from_github(GITHUB_REPO, MAT_API_DIR)
    if not mat_files:
        st.write("No .mat files found in the specified directory.")
        return

    file_names = [file['name'] for file in mat_files]
    selected_file_name = st.selectbox("Select a .mat file:", file_names)
    selected_file = next((file for file in mat_files if file['name'] == selected_file_name), None)

    if selected_file:
        mat_data = load_mat_data_from_url(selected_file['download_url'])
        if mat_data:
            # Display the .mat file data
            display_mat_data(mat_data)
            # Display the activation map
            activation_map(mat_data)

if __name__ == "__main__":
    main()
