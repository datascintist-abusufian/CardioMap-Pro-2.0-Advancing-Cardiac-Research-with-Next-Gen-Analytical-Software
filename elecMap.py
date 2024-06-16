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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128, 128)),  # Resize to the model input size
    ])
    return transform(img)

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

def main():
    st.title("3D Image Segmentation and Classification Report")

    image_url = st.text_input("Enter the 3D image URL:", "")
    ground_truth_url = st.text_input("Enter the Ground Truth URL:", "")

    if image_url and ground_truth_url:
        img = load_image_3d(image_url)
        ground_truth = load_ground_truth(ground_truth_url)

        if img is not None and ground_truth is not None:
            st.write("Original 3D Image")
            st.image(img, use_column_width=True)  # Visualize the 3D image

            img_tensor = preprocess_3d(img).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                pred = model(img_tensor)
                pred = torch.sigmoid(pred).numpy()

            display_classification_report(pred, ground_truth)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(np.max(img, axis=0), cmap='gray')
            ax[0].set_title("Original Image")
            ax[1].imshow(np.max(pred, axis=0), cmap='jet')
            ax[1].set_title("Segmented Image")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
