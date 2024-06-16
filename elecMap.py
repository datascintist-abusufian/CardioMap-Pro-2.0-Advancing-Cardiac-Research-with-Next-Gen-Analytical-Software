import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from torch.autograd import Variable
from torchvision import models, transforms
import os
import sys

# Handle potential recursion limits for large images
sys.setrecursionlimit(10000)

# List of image paths provided
image_files = [
    "original_2D_01_2023_jpg.rf.08fb9bb2f436753819831dd5e4b8e0c2.jpg",
    "original_2D_01_2024_jpg.rf.dc5b4863b6e5730b5c59b1d25b0b9611.jpg",
    "original_2D_01_2025_jpg.rf.975b93d5d4d68bb0306fde3ce8e37e86.jpg",
    "original_2D_01_2026_jpg.rf.b727703b43368f2018773abbd3dff73a.jpg",
    "original_2D_01_2028_jpg.rf.64b2b9caea1ace3c99df576f0264c189.jpg",
    "original_2D_01_2031_jpg.rf.14b8264df08246d0e12f212fb7d80e2a.jpg",
    "original_2D_01_2032_jpg.rf.30160c9fecffab87f8d2f1f359ace04c.jpg",
    "original_2D_01_2033_jpg.rf.41f6b753f6ceaeabaffd88ef3dd4942c.jpg",
    "original_2D_01_2034_jpg.rf.912a83e4e856f1832188be74aada0cc1.jpg",
    "original_2D_01_2038_jpg.rf.8413b9278a67b87056db0f71307bacf9.jpg",
    "original_2D_01_2039_jpg.rf.6b93fc11288a7adcd648b22eec2ce6e8.jpg",
    "original_2D_01_2045_jpg.rf.a9cf3117df073692b09abc0898154154.jpg",
    "original_2D_01_2047_jpg.rf.d62d36328c70a05f53b2d124a86253e6.jpg",
    "original_2D_01_2048_jpg.rf.1ab0f86aa5e0893ea5423a4e013322fa.jpg",
    "original_2D_01_2049_jpg.rf.607732dc9489ca7ae3b160bce0e4fe9f.jpg",
    "original_2D_01_2050_jpg.rf.6a2f6c6382fc81957cd627324308191a.jpg",
    "original_2D_01_2052_jpg.rf.7a0cbe73bace6303a05ca7982d7bddc8.jpg",
    "original_2D_01_2110_jpg.rf.a1bda674c1f1ad6d5fd6c22b92a232df.jpg",
    "original_2D_01_2111_jpg.rf.0ed3959c7f54bf9634be31826c50aa45.jpg",
    "original_2D_01_2113_jpg.rf.9a8d97afef2eb832cbd49fc914c022d2.jpg",
    "original_2D_01_2114_jpg.rf.d4e9ba5c9b240bf480bea86d32711f6d.jpg",
    "original_2D_01_2116_jpg.rf.2126370329cb6a15a99f10137630e9be.jpg",
    "original_2D_01_2117_jpg.rf.faae1020fa66d4e93d0364cd695730be.jpg",
    "original_2D_01_2118_jpg.rf.fc02b008e46dfa389c8ea5a42db3218c.jpg",
    "original_2D_01_2119_jpg.rf.f20d895c47556dd46d243ce5181cf054.jpg",
    "original_2D_01_2120_jpg.rf.f27fa6cd777ebd88ad6a887aeeebadb8.jpg",
    "original_2D_01_2124_jpg.rf.67420c3762e88aba197c5e974bb6a910.jpg",
    "original_2D_01_2125_jpg.rf.53eccae3bbb821fd561c4d0710865014.jpg",
    "original_2D_01_2126_jpg.rf.3fb47cae54f4e590a5204bc0391733d8.jpg",
    "original_2D_01_2127_jpg.rf.f2735201c938e4c28537323423175388.jpg",
    "original_2D_01_2128_jpg.rf.80c5af4211487b22a3337abc2357de1b.jpg",
    "original_2D_01_2129_jpg.rf.826e462b55aa541f5531162fd842f18f.jpg",
    "original_2D_01_2130_jpg.rf.919b10f44c5353cb1b7b9167c7d053c1.jpg",
    "original_2D_01_2131_jpg.rf.db0a0fca7d7517257fff4c519be4ac0c.jpg",
    "original_2D_01_2133_jpg.rf.54a628f5c0af050d361ff54c14813f13.jpg",
    "original_2D_01_2134_jpg.rf.e681a4814cde2142144dc9229901771b.jpg",
    "original_2D_01_2135_jpg.rf.0c996ea722d2f1d7e4040bc3144323a4.jpg",
    "original_2D_01_2136_jpg.rf.04548d1acef1e178e9d78c3b7b00b0e2.jpg",
    "original_2D_01_2137_jpg.rf.9b3fa4a0b2c6ded7fc375839a92d7a59.jpg",
    "original_2D_01_2138_jpg.rf.62d17e6007cb54302e0669f29dbd226a.jpg",
    "original_2D_01_2139_jpg.rf.dffe67ae6437f81856c41f0bbbfca21a.jpg",
    "original_2D_01_2140_jpg.rf.a7132f1f56d7aca29717b29649b6cb76.jpg",
    "original_2D_01_2142_jpg.rf.48191c1c4b3af9af3ce50024def4fca5.jpg",
    "original_2D_01_2143_jpg.rf.fffafb3bc737af426ecd7bcf9342d2fa.jpg",
    "original_2D_01_2144_jpg.rf.b968ec80a7cab1b5ae2ee2d9a143f8ce.jpg",
    "original_2D_01_2149_jpg.rf.dc65cb7dcfcd2f7eb7a9df16585970c9.jpg",
    "original_2D_01_2150_jpg.rf.bf1f0fe1deedfc082f3e5ac136a1b0a8.jpg",
    "original_2D_01_2151_jpg.rf.56a2607239b7c962b50da12fce82d663.jpg",
    "original_2D_01_2212_jpg.rf.ab88b8a9ccb4566c825ab1691f64eece.jpg",
    "original_2D_01_2213_jpg.rf.767c796fae2831f840417bc0360e6d77.jpg",
    "original_2D_01_2214_jpg.rf.4ca991645aedef70daca14c383da2fb6.jpg",
    "original_2D_01_2216_jpg.rf.3174e07474ffd55f46905e0f88a8fd83.jpg",
    "original_2D_01_2218_jpg.rf.a6f4f68c816c93955c469a7298d7c151.jpg",
    "original_2D_01_2219_jpg.rf.bc2fca7067e8bd6d56d3828e0ec35a8d.jpg",
    "original_2D_01_2220_jpg.rf.afdc9b6484723344cb9af289a137e828.jpg",
    "original_2D_01_2221_jpg.rf.9ff7eff504f6dfb77ab5070a33bf717e.jpg",
    "original_2D_01_2222_jpg.rf.34fe0e4652ea94a14bac007ec7c48531.jpg",
    "original_2D_01_2223_jpg.rf.f76913a1d6debc34ab09c1bd4303ccf5.jpg",
    "original_2D_01_2224_jpg.rf.ea075971be38aaf8749cda5fe77102dd.jpg",
    "original_2D_01_2225_jpg.rf.f3ce8ab5e8d1485e9babd9dc21661774.jpg",
    "original_2D_01_2226_jpg.rf.efc235a3511f842c329852a8c1fd6236.jpg",
    "original_2D_01_2227_jpg.rf.d9cff590a4deed8096754780d8cedb29.jpg",
    "original_2D_01_2229_jpg.rf.6ed4fb56e4e9440f91cab68c7820e543.jpg",
    "original_2D_01_2230_jpg.rf.1c75746f02a4c0e9cd5bd81194b7ceae.jpg",
    "original_2D_01_2231_jpg.rf.0a290eb89ff1218d9dca980c063943c2.jpg",
    "original_2D_01_2232_jpg.rf.7567efac5e81aa0272f5a98646c184c1.jpg",
    "original_2D_01_2233_jpg.rf.78659d10a3ef81dbaf547c7498d6e952.jpg",
    "original_2D_01_2234_jpg.rf.428f672b653c00204f2cd57518ff21db.jpg",
    "original_2D_01_2235_jpg.rf.27758d1cadb9ee5eb0abb3744fbac3a9.jpg",
    "original_2D_01_2236_jpg.rf.031e9e2ceb2cb9613368a7c1f4423c0c.jpg",
    "original_2D_01_2237_jpg.rf.334c240093cd3fb645a4acfb35702ac7.jpg",
    "original_2D_01_2238_jpg.rf.fa3738f07c7b26fe973cf30fbfe8a590.jpg",
    "original_2D_01_2239_jpg.rf.05364083b7acd80c575e3acc015a4e46.jpg",
    "original_2D_01_2241_jpg.rf.7205683b425d724d64d174a097c40d14.jpg",
    "original_2D_01_2242_jpg.rf.8a16c7af7b500c8e648733d8b68db78a.jpg",
    "original_2D_01_2243_jpg.rf.5508721f27710beffd4aefdb9296dd49.jpg",
    "original_2D_01_2244_jpg.rf.3fa80a1fbc2fe300f328a37d27b3a68b.jpg",
    "original_2D_01_2245_jpg.rf.e53af9e2e20249e7a1166b9d76e30079.jpg",
    "original_2D_01_2247_jpg.rf.6373f64d0a8275e1bcfc90adc19f5b25.jpg",
    "original_2D_01_2249_jpg.rf.80cccee6719fc7cee2bfc8426316e36e.jpg",
    "original_2D_01_2250_jpg.rf.41d2b8a6d16305a452a863ebb89a9ac8.jpg",
    "original_2D_01_2310_jpg.rf.3776153faed0eb40ac85500a0e395780.jpg",
    "original_2D_01_2311_jpg.rf.09dfddf11f04c1d1f86dd4c9a3a3c291.jpg",
    "original_2D_01_2312_jpg.rf.e47c0e9f0184d8dd12d16a56a997c661.jpg",
    "original_2D_01_2313_jpg.rf.2b9a43c5e80cd169cbc2c099c34750be.jpg",
    "original_2D_01_2315_jpg.rf.7990a82da9c40969223cd81f4e778ee4.jpg",
    "original_2D_01_2316_jpg.rf.d68ea884aa381883b1f35e779bdee556.jpg",
    "original_2D_01_2317_jpg.rf.ae9870bb583fdd93ef1e337eac26353c.jpg",
    "original_2D_01_2319_jpg.rf.e4f7068a937427c5eea991126c98a003.jpg",
    "original_2D_01_2322_jpg.rf.f59933f021c03295c0d80da76cfa845f.jpg"
]

@st.cache(allow_output_mutation=True)
def load_image(file_name):
    file_path = f'path_to_images/{file_name}'
    if os.path.exists(file_path):
        img = Image.open(file_path).convert('L')
        img_data = np.array(img)
        return img, img_data
    else:
        st.error(f"Failed to load image: {file_name}")
        return None, None

@st.cache(allow_output_mutation=True)
def load_ground_truth_data(image_files):
    ground_truth_data = {}
    for image_file in image_files:
        image_name = image_file.split('.')[0]  # Assuming ground truth files have a similar name
        ground_truth_path = f'path_to_ground_truth/{image_name}_ground_truth.npy'
        if os.path.exists(ground_truth_path):
            ground_truth_data[image_name] = np.load(ground_truth_path)
        else:
            st.warning(f"Ground truth not found for {image_name}")
    return ground_truth_data

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
    st.pyplot(fig)

def accuracy_display(data, ground_truth):
    threshold = 128
    prediction = data > threshold
    accuracy = np.mean(prediction == ground_truth)
    st.write(f"Segmentation Accuracy: {accuracy}")

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
    fig, ax = plt.subplots(figsize=(10, 6))
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

def grad_cam(data):
    download_weights()
    model = models.resnet18()
    model.load_state_dict(torch.load('resnet18-f37072fd.pth'))
    model.eval()

    if len(data.shape) == 2:
        data = np.stack([data]*3, axis=-1)
    elif data.shape[2] == 1:
        data = np.concatenate([data]*3, axis=-1)
    elif data.shape[2] == 4:
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

def ground_truth_display(data):
    st.image(data, caption="Ground Truth", use_column_width=True)

def log_likelihood_density(data):
    density, bins, _ = plt.hist(data.ravel(), bins=256, density=True)
    log_likelihood = np.log(density + 1e-9)  # Adding a small constant to avoid log(0)
    fig, ax = plt.subplots()
    ax.plot(bins[:-1], log_likelihood, label='Log Likelihood')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Log Likelihood')
    ax.set_title('Log Likelihood vs Density')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Image Viewer and Data Analysis")

    # Load ground truth data
    ground_truth_data = load_ground_truth_data(image_files)

    file_index = st.sidebar.selectbox("Select an image", range(len(image_files)), format_func=lambda x: image_files[x].split('/')[-1])
    img, img_data = load_image(image_files[file_index])
    image_name = image_files[file_index].split('.')[0]
    ground_truth = ground_truth_data.get(image_name)

    if img is not None and img_data is not None:
        st.image(img, use_column_width=True)

        if st.sidebar.checkbox("Velocity Analysis"):
            velocity_analysis(img_data)

        if st.sidebar.checkbox("Histogram Analysis"):
            histogram_analysis(img_data)

        if ground_truth is not None and st.sidebar.checkbox("Accuracy Display"):
            accuracy_display(img_data, ground_truth)

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
            
        if ground_truth is not None and st.sidebar.checkbox("Ground Truth"):
            ground_truth_display(ground_truth)
            
        if st.sidebar.checkbox("Log Likelihood vs Density"):
            log_likelihood_density(img_data)

if __name__ == "__main__":
    main()
