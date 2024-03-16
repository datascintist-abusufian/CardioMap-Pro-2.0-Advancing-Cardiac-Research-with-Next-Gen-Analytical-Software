import streamlit as st
import os
import scipy.io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import requests
from io import BytesIO

# GitHub repository details
GITHUB_REPO = "datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software"
MAT_API_DIR = "mat_api"

@st.cache
def get_mat_files_from_github(repo_path, folder_path):
    """Fetch list of .mat files from GitHub directory."""
    api_url = f"https://api.github.com/repos/{repo_path}/contents/{folder_path}"
    response = requests.get(api_url)
    if response.status_code == 200:
        files = response.json()
        return [file for file in files if file['name'].endswith('.mat')]
    else:
        st.error("Failed to fetch files from GitHub")
        return []

@st.cache
def load_image_from_url(url):
    """Load image data from a .mat file URL."""
    response = requests.get(url)
    if response.status_code == 200:
        content = BytesIO(response.content)
        mat_data = scipy.io.loadmat(content)
        img_data = mat_data.get('image_data', None)
        if img_data is not None:
            img = Image.fromarray(np.uint8(img_data.squeeze()))
            return img, img_data
    st.error("Failed to load image from URL")
    return None, None

def main():
    st.title("MAT File Image Analyzer")
    
    mat_files = get_mat_files_from_github(GITHUB_REPO, MAT_API_DIR)
    if not mat_files:
        st.write("No .mat files found.")
        return
    
    file_names = [file['name'] for file in mat_files]
    selected_file_name = st.selectbox("Select a .mat file:", file_names)
    selected_file = next((file for file in mat_files if file['name'] == selected_file_name), None)

    if selected_file:
        img, img_data = load_image_from_url(selected_file['download_url'])
        if img is not None:
            st.image(img, use_column_width=True)
          def velocity_analysis(data):
    # Calculate the mean and standard deviation of the pixel values
    mean = np.mean(data)
    std = np.std(data)
    st.write(f"Mean pixel value: {mean}")
    st.write(f"Standard deviation of pixel values: {std}")

def histogram_analysis(data):
    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Perform your plotting actions
    ax.hist(data.ravel(), bins=256, color='orange')
    ax.hist(data.ravel(), bins=256, color='black')

    # Pass the figure to st.pyplot()
    st.pyplot(fig)
    
def accuracy_display(data):
    # Calculate the proportion of pixels that are above a certain threshold
    threshold = 128  # replace with your actual threshold
    accuracy = np.mean(data > threshold)
    st.write(f"Proportion of pixels above threshold: {accuracy}")
    
def electromapping(data):
    # Apply a Fourier transform to the data
    electromap = np.fft.fft2(data)
    # Take the logarithm of the absolute value of the Fourier transform
    log_electromap = np.log1p(np.abs(electromap))
    # Normalize the data to the range [0.0, 1.0]
    normalized_electromap = (log_electromap - np.min(log_electromap)) / (np.max(log_electromap) - np.min(log_electromap))
    # Apply a colormap to the data
    colored_electromap = cm.hot(normalized_electromap)
    st.image(colored_electromap, use_column_width=True)

def signal_processing(data):
    # Apply an inverse Fourier transform to the data
    processed_signal = np.fft.ifft2(data)
    # Take the absolute value of the processed signal
    abs_processed_signal = np.abs(processed_signal)
    # Apply a logarithmic function to the data
    log_processed_signal = np.log1p(abs_processed_signal)
    # Normalize the data to the range [0.0, 1.0]
    normalized_signal = (log_processed_signal - np.min(log_processed_signal)) / (np.max(log_processed_signal) - np.min(log_processed_signal))
    st.image(normalized_signal, use_column_width=True)

def region_selection(data):
    # Dynamic adjustment based on data dimensions
    max_row, max_col = data.shape[0], data.shape[1]
    
    # Streamlit sliders for dynamic region selection
    start_row = st.sidebar.number_input('Start Row', min_value=0, max_value=max_row-1, value=0)
    end_row = st.sidebar.number_input('End Row', min_value=0, max_value=max_row, value=max_row)
    start_col = st.sidebar.number_input('Start Column', min_value=0, max_value=max_col-1, value=0)
    end_col = st.sidebar.number_input('End Column', min_value=0, max_value=max_col, value=max_col)
    
    # Select and display the region
    region = data[start_row:end_row, start_col:end_col]
    if np.ptp(region) == 0:  # Checking if the selected region is uniform
        st.write("The selected region is uniform or empty.")
    else:
        normalized_region = (region - np.min(region)) / (np.max(region) - np.min(region))
        st.image(normalized_region, caption="Selected Region", use_column_width=True)

def automatically_segmented_signal(data):
    # Define a threshold for segmentation
    threshold = np.mean(data)  # replace with your actual threshold

    # Apply the threshold to the data
    segmented_signal = np.where(data > threshold, 1, 0)

    # Scale the segmented signal to the range [0, 255]
    segmented_signal = segmented_signal * 255

    # Display the segmented signal
    st.image(segmented_signal.astype(np.uint8), use_column_width=True)

def activation_map(data):
    # Define a threshold for activation
    threshold = 0.01 * np.max(data)

    # Calculate the activation map
    activation_map = np.argmax(data > threshold, axis=2)

    st.write(f"Activation Map: {activation_map}")

def display_analysis_option(data):
    # Placeholder for a more complex analysis, e.g., finding areas with specific properties
    threshold = np.mean(data) + np.std(data)  # Example threshold
    areas_above_threshold = np.sum(data > threshold)
    st.write(f"Areas above threshold (mean + std): {areas_above_threshold}")

def diastolic_interval(data):
    # Example calculation: Variability of the signal (assuming variability might relate to diastolic intervals)
    variability = np.std(data)
    st.write(f"Signal variability (potential proxy for diastolic interval variability): {variability}")

def repolarisation(data):
    # Placeholder: Assuming higher values might indicate repolarization regions
    high_value_threshold = np.percentile(data, 90)  # 90th percentile as a high-value threshold
    high_value_areas = np.sum(data > high_value_threshold)
    st.write(f"Areas potentially representing repolarisation (above 90th percentile): {high_value_areas}")

def APD(data):
    # Example assuming APD relates to the duration of certain signal levels
    # This is highly simplified and not directly applicable without knowing data structure
    duration_threshold = np.mean(data)  # Simplified threshold
    potential_apd_areas = np.sum(data > duration_threshold)
    st.write(f"Areas with potential APD (above mean value): {potential_apd_areas}")
def main():
    st.title("Image Viewer and Data Analysis")

    local_dir = '/Users/mdabusufian/Downloads/balil/mat_api'
    mat_files = [os.path.join(local_dir, file) for file in os.listdir(local_dir) if file.endswith('.mat')]
    file_index = st.sidebar.slider("Select an image", 0, len(mat_files) - 1)
    img, img_data = load_image(mat_files[file_index])
    st.image(img, use_column_width=True)

    if st.sidebar.checkbox("Velocity Analysis"):
        velocity_analysis(img_data)

    if st.sidebar.checkbox("Histogram Analysis"):
        histogram_analysis(img_data)

    if st.sidebar.checkbox("Accuracy Display"):
        accuracy_display(img_data)

    if st.sidebar.checkbox("Electromapping"):
        electromapping(img_data)

    if st.sidebar.checkbox("Signal Processing"):
        signal_processing(img_data)

    if st.sidebar.checkbox("Region Selection"):
        region_selection(img_data)

    if st.sidebar.checkbox("Automatically Segmented Signal"):
        automatically_segmented_signal(img_data)

    if st.sidebar.checkbox("Activation Map"):
        activation_map(img_data)

    if st.sidebar.checkbox("Display Analysis Option"):
        display_analysis_option(img_data)

    if st.sidebar.checkbox("Diastolic Interval"):
        diastolic_interval(img_data)

    if st.sidebar.checkbox("Repolarisation"):
        repolarisation(img_data)

    if st.sidebar.checkbox("APD"):
        APD(img_data)

if __name__ == "__main__":
    main()
