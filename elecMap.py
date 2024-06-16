import streamlit as st
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import classification_report
from torchvision import transforms
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool3d(2)
        self.upconv4 = self.upconv(512, 256)
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(128, 64)
        self.conv_last = nn.Conv3d(64, 1, 1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        dec4 = self.upconv4(enc4)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)
        return self.conv_last(dec2)

# Load pre-trained model
model = UNet3D()
model.load_state_dict(torch.load('unet3d.pth'))
model.eval()

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_image_3d(url):
    response = requests.get(url)
    if response.status_code == 200:
        img = np.load(BytesIO(response.content))  # Assuming the 3D image is stored in .npy format
        return img
    else:
        return None

def preprocess_3d(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize the image
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add channel and batch dimensions
    return img

def load_ground_truth(url):
    response = requests.get(url)
    if response.status_code == 200:
        ground_truth = np.load(BytesIO(response.content))
        return ground_truth
    else:
        st.warning("Failed to load ground truth data.")
        return None

def display_classification_report(pred, ground_truth):
    pred = pred.flatten()
    ground_truth = ground_truth.flatten()
    report = classification_report(ground_truth, pred, output_dict=True)
    st.write(report)

def velocity_analysis(data):
    mean = np.mean(data)
    std = np.std(data)
    st.write(f"Mean pixel value: {mean}")
    st.write(f"Standard deviation of pixel values: {std}")

def histogram_analysis(data):
    fig, ax = plt.subplots(figsize=(5, 5))  # Set figure size to 5x5 inches
    ax.hist(data.ravel(), bins=256, color='orange', alpha=0.5)
    st.pyplot(fig)

def accuracy_display(data, ground_truth):
    threshold = 0.5
    prediction = data > threshold
    accuracy = np.mean(prediction == ground_truth)
    st.write(f"Segmentation Accuracy: {accuracy}")

    # Highlight the diseased areas on the original image
    highlighted_image = np.copy(data)
    highlighted_image[prediction == 1] = 1  # Mark the segmented areas

    # Create an RGB image for visualization
    rgb_image = np.stack([highlighted_image]*3, axis=-1)
    st.image(rgb_image, caption="Disease Area Highlighted", use_column_width=False, width=300)

def electromapping(data):
    electromap = np.fft.fft2(data)
    log_electromap = np.log1p(np.abs(electromap))
    normalized_electromap = (log_electromap - np.min(log_electromap)) / (np.max(log_electromap) - np.min(log_electromap))
    colored_electromap = cm.hot(normalized_electromap)
    fig, ax = plt.subplots(figsize=(5, 5))  # Set figure size to 5x5 inches
    ax.imshow(colored_electromap)
    ax.set_title('Electromapping')
    st.pyplot(fig)

def signal_processing(data):
    processed_signal = np.fft.ifft2(data)
    abs_processed_signal = np.abs(processed_signal)
    log_processed_signal = np.log1p(abs_processed_signal)
    normalized_signal = (log_processed_signal - np.min(log_processed_signal)) / (np.max(log_processed_signal) - np.min(log_processed_signal))
    fig, ax = plt.subplots(figsize=(5, 5))  # Set figure size to 5x5 inches
    ax.imshow(normalized_signal, cmap='gray')
    ax.set_title('Signal Processing')
    st.pyplot(fig)

def region_selection(data):
    max_row, max_col = data.shape[0], data.shape[1]

    if max_row <= 1 or max_col <= 1:
        st.write("The data dimensions are too small for region selection.")
        return

    start_row = st.sidebar.number_input('Start Row', min_value=0, max_value=max_row-2, value=0)
    end_row = st.sidebar.number_input('End Row', min_value=start_row+1, max_value=max_row, value=max_row)
    start_col = st.sidebar.number_input('Start Column', min_value=0, max_value=max_col-2, value=0)
    end_col = st.sidebar.number_input('End Column', min_value=start_col+1, max_value=max_col, value=max_col)

    region = data[start_row:end_row, start_col:end_col]
    if np.ptp(region) == 0:
        st.write("The selected region is uniform or empty.")
    else:
        normalized_region = (region - np.min(region)) / (np.max(region) - np.min(region))
        st.image(normalized_region, caption="Selected Region", use_column_width=True)

def automatically_segmented_signal(data):
    threshold = np.mean(data)
    segmented_signal = np.where(data > threshold, 1, 0)
    segmented_signal = segmented_signal * 255
    st.image(segmented_signal.astype(np.uint8), use_column_width=True)

def activation_map(data):
    st.write(f"Data Min: {np.min(data)}, Data Max: {np.max(data)}, Data Mean: {np.mean(data)}, Data Std: {np.std(data)}")

    threshold = np.mean(data) + np.std(data)
    st.write(f"Using threshold: {threshold}")

    if data.ndim == 3:
        activation_map = np.max(data, axis=2) > threshold
    elif data.ndim == 2:
        activation_map = data > threshold
    else:
        st.error("Data should be a 2D or 3D array")
        return

    unique_values = np.unique(activation_map)
    st.write(f"Unique values in the activation map: {unique_values}")

    if len(unique_values) == 1 and unique_values[0] == False:
        st.error("Activation map contains only zero values. Adjusting the threshold might help.")
        return

    activation_map = activation_map.astype(np.float32)
    fig, ax = plt.subplots(figsize=(5, 5))  # Set figure size to 5x5 inches
    cax = ax.imshow(activation_map, cmap='hot', interpolation='nearest')
    ax.set_title('Activation Map')
    fig.colorbar(cax, ax=ax)
    st.pyplot(fig)

def display_analysis_option(data):
    threshold = np.mean(data) + np.std(data)
    areas_above_threshold = np.sum(data > threshold)
    st.write(f"Areas above threshold (mean + std): {areas_above_threshold}")

def diastolic_interval(data):
    variability = np.std(data)
    st.write(f"Signal variability (potential proxy for diastolic interval variability): {variability}")

def repolarisation(data):
    high_value_threshold = np.percentile(data, 90)
    high_value_areas = np.sum(data > high_value_threshold)
    st.write(f"Areas potentially representing repolarisation (above 90th percentile): {high_value_areas}")

def APD(data):
    duration_threshold = np.mean(data)
    potential_apd_areas = np.sum(data > duration_threshold)
    st.write(f"Areas with potential APD (above mean value): {potential_apd_areas}")

def ground_truth_display(data):
    st.image(data, caption="Ground Truth", use_column_width=False, width=300)  # Adjust width to 300 pixels

def log_likelihood_density(data):
    density, bins, _ = plt.hist(data.ravel(), bins=256, density=True)
    log_likelihood = np.log(density + 1e-9)  # Adding a small constant to avoid log(0)
    fig, ax = plt.subplots(figsize=(5, 5))  # Set figure size to 5x5 inches
    ax.plot(bins[:-1], log_likelihood, label='Log Likelihood')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Log Likelihood')
    ax.set_title('Log Likelihood vs Density')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("3D Heart Image Analysis")

    image_url = st.text_input("Enter the 3D image URL:", "")
    ground_truth_url = st.text_input("Enter the Ground Truth URL:", "")

    if image_url and ground_truth_url:
        img = load_image_3d(image_url)
        ground_truth = load_ground_truth(ground_truth_url)

        if img is not None and ground_truth is not None:
            st.write("Original 3D Image")
            st.image(np.max(img, axis=0), caption="Original 3D Image", use_column_width=False, width=300)

            img_tensor = preprocess_3d(img)

            with torch.no_grad():
                pred = model(img_tensor)
                pred = torch.sigmoid(pred).squeeze().numpy()

            st.sidebar.write("Analysis Options")
            if st.sidebar.checkbox("Velocity Analysis"):
                velocity_analysis(img)

            if st.sidebar.checkbox("Histogram Analysis"):
                histogram_analysis(img)

            if st.sidebar.checkbox("Accuracy Display"):
                accuracy_display(pred, ground_truth)

            if st.sidebar.checkbox("Electromapping"):
                electromapping(img)

            if st.sidebar.checkbox("Signal Processing"):
                signal_processing(img)

            if st.sidebar.checkbox("Region Selection"):
                region_selection(img)

            if st.sidebar.checkbox("Automatically Segmented Signal"):
                automatically_segmented_signal(img)

            if st.sidebar.checkbox("Activation Map"):
                activation_map(img)

            if st.sidebar.checkbox("Display Analysis Option"):
                display_analysis_option(img)

            if st.sidebar.checkbox("Diastolic Interval"):
                diastolic_interval(img)

            if st.sidebar.checkbox("Repolarisation"):
                repolarisation(img)

            if st.sidebar.checkbox("APD"):
                APD(img)

            if st.sidebar.checkbox("Ground Truth"):
                ground_truth_display(ground_truth)

            if st.sidebar.checkbox("Log Likelihood vs Density"):
                log_likelihood_density(img)

if __name__ == "__main__":
    main()
