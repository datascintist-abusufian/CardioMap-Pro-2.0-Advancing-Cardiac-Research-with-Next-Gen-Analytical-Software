import streamlit as st
import requests
from io import BytesIO
import scipy.io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from torch.autograd import Variable
from torchvision import models, transforms

@st.cache(allow_output_mutation=True)
def load_image(url):
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

@st.cache(allow_output_mutation=True)
def download_weights():
    url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    response = requests.get(url)
    if response.status_code == 200:
        with open("resnet18-f37072fd.pth", "wb") as f:
            f.write(response.content)
    else:
        st.error("Failed to download weights.")

def velocity_analysis(data):
    mean = np.mean(data)
    std = np.std(data)
    st.write(f"Mean pixel value: {mean}")
    st.write(f"Standard deviation of pixel values: {std}")

def histogram_analysis(data):
    fig, ax = plt.subplots()
    ax.hist(data.ravel(), bins=256, color='orange', alpha=0.5)
    ax.hist(data.ravel(), bins=256, color='black', alpha=0.5)
    st.pyplot(fig)

def accuracy_display(data):
    threshold = 128
    accuracy = np.mean(data > threshold)
    st.write(f"Proportion of pixels above threshold: {accuracy}")

def electromapping(data):
    electromap = np.fft.fft2(data)
    log_electromap = np.log1p(np.abs(electromap))
    normalized_electromap = (log_electromap - np.min(log_electromap)) / (np.max(log_electromap) - np.min(log_electromap))
    colored_electromap = cm.hot(normalized_electromap)
    st.image(colored_electromap, use_column_width=True)

def signal_processing(data):
    processed_signal = np.fft.ifft2(data)
    abs_processed_signal = np.abs(processed_signal)
    log_processed_signal = np.log1p(abs_processed_signal)
    normalized_signal = (log_processed_signal - np.min(log_processed_signal)) / (np.max(log_processed_signal) - np.min(log_processed_signal))
    st.image(normalized_signal, use_column_width=True)

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
    threshold = 0.01 * np.max(data)
    activation_map = np.argmax(data > threshold, axis=2)
    st.write(f"Activation Map: {activation_map}")

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

def grad_cam(data):
    download_weights()
    model = models.resnet18()
    model.load_state_dict(torch.load('resnet18-f37072fd.pth'))
    model.eval()

    # Ensure data is a 2D numpy array and convert to a 3-channel (RGB) image
    if len(data.shape) == 2:
        data = np.stack([data]*3, axis=-1)
    elif data.shape[2] == 1:
        data = np.concatenate([data]*3, axis=-1)
    elif data.shape[2] == 4:  # In case there's an alpha channel
        data = data[:, :, :3]

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(data).unsqueeze(0)
    input_tensor = Variable(input_tensor, requires_grad=True)

    def extract_layer(layer, input, output):
        return output
    
    handle = model.layer4[1].register_forward_hook(extract_layer)
    output = model(input_tensor)
    handle.remove()

    output_idx = output.argmax().item()
    output_max = output[0, output_idx]
    model.zero_grad()
    output_max.backward()

    gradients = input_tensor.grad[0]
    pooled_gradients = torch.mean(gradients, dim=[1, 2])

    activations = model.layer4[1].output[0]
    for i in range(512):
        activations[i, ...] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize((data.shape[1], data.shape[0]))
    heatmap = np.array(heatmap)

    heatmap = cm.jet(heatmap)[:, :, :3]
    superimposed_img = heatmap * 0.4 + data
    st.image(superimposed_img.astype(np.uint8), caption="Grad-CAM Output", use_column_width=True)

def ground_truth(data):
    st.image(data, caption="Ground Truth", use_column_width=True)

def main():
    st.title("Image Viewer and Data Analysis")

    mat_files = [
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/Figure_1.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/Figure_11.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/Figure_2.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/Figure_22.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/Figure_3.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/Figure_33.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/Figure_4.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Activation_2-wk_04-450_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Activation_2-wk_05-400_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Activation_2-wk_05-400_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Activation_2-wk_06-350_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Activation_2-wk_06-350_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Activation_6-wk_01-350_Ca_transient.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Activation_6-wk_02-400_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Activation_6-wk_02-400_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Activation_6-wk_03-350_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Coupling_2-wk old_05-400.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Duration_2-wk_04-450_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Duration_2-wk_04-450_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Duration_2-wk_05-400_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Duration_2-wk_05-400_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Duration_2-wk_06-350_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Duration_2-wk_06-350_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Duration_6-week old_02-400_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Duration_6-wk_01-350_Ca_transient.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_Duration_6-wk_03-350_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_SNR_02-400_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_SNR_2-wk_04-450_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_SNR_2-wk_05-400_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_SNR_2-wk_05-400_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_SNR_2-wk_06-350_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_SNR_2-wk_06-350_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_SNR_6-wk_01-350_Ca_transient.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapPig_SNR_6-wk_03-350_Ca.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapRat_Activation_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapRat_Duration_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/MapRat_SNR_Vm.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/anys_activation.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/anys_duration80.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/anys_duration90.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/integration_MapActivation.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/integration_MapActivation_New.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/integration_MapDuration.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/integration_MapSNR_NEW.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/integration_MapSNR_PigSpot.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/integration_MapSNR_PigWhole.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/integration_MapSNR_Rat.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/prep_mask.mat",
        "https://raw.githubusercontent.com/datascintist-abusufian/CardioMap-Pro-2.0-Advancing-Cardiac-Research-with-Next-Gen-Analytical-Software/main/mat_api/proc_snr.mat"
    ]

    file_index = st.sidebar.selectbox("Select an image", range(len(mat_files)), format_func=lambda x: mat_files[x].split('/')[-1])
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
            
        if st.sidebar.checkbox("Grad-CAM"):
            grad_cam(img_data)
            
        if st.sidebar.checkbox("Ground Truth"):
            ground_truth(img_data)

if __name__ == "__main__":
    main()
