import streamlit as st
import requests
from io import BytesIO
import scipy.io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

@st.cache(allow_output_mutation=True)
def load_image(url):
    """Load .mat data from a given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        content = BytesIO(response.content)
        mat_data = scipy.io.loadmat(content)
        img_data = mat_data['image_data']
        img = Image.fromarray(np.uint8(img_data.squeeze()))
        return img, img_data
    else:
        st.error("Failed to load data from URL.")
        return None, None

# The rest of your functions go here...
import streamlit as st
import requests
from io import BytesIO
import scipy.io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

@st.cache(allow_output_mutation=True)
def load_image(url):
    """Load .mat data from a given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        content = BytesIO(response.content)
        mat_data = scipy.io.loadmat(content)
        img_data = mat_data['image_data']
        img = Image.fromarray(np.uint8(img_data.squeeze()))
        return img, img_data
    else:
        st.error("Failed to load data from URL.")
        return None, None

# The rest of your functions go here...

def main():
    st.title("Image Viewer and Data Analysis")

    # List of URLs to the .mat files in the GitHub repository
    mat_files = [
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/Figure_1.mat",
        # Add more URLs as needed
    ]

    file_index = st.sidebar.slider("Select an image", 0, len(mat_files) - 1)
    img, img_data = load_image(mat_files[file_index])
    if img is not None and img_data is not None:
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
def main():
    st.title("Image Viewer and Data Analysis")

    # List of URLs to the .mat files in the GitHub repository
    mat_files = [
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/Figure_1.mat",
        # Add more URLs as needed
    ]

    file_index = st.sidebar.slider("Select an image", 0, len(mat_files) - 1)
    img, img_data = load_image(mat_files[file_index])
    if img is not None and img_data is not None:
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
