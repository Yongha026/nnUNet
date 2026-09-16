import torch
import torch.nn.functional as F

import os
import cv2
import sys
import numpy as np
import glob
import argparse

import torchvision

import PIL.Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

# model_path_adgbc = os.path.join(plugins_dir, "model_ckpts","adgbc_nn_best.pth")
# num_balls = 16
model_path_adgbc = "/mnt/hdd1/nnunetv2_openEDS/nnUNet_results/Dataset250_OpenEDS2019/nnUNetTrainerGBC_S_16__nnUNetPlans__2d/fold_0/checkpoint_best.pth"


# adgbc_encoder.py 만들어서 인코더까지만 로드

from enc_GBC import GBC_S_EncDec
try:
    model = GBC_S_EncDec(num_classes=4, input_channels=1, deep_supervision=False, gbc_num_balls=16).to(device)
    if os.path.exists(model_path_adgbc):
        checkpoint = torch.load(model_path_adgbc, map_location=device, weights_only=False)
        state_dict = checkpoint["network_weights"] if (
                    isinstance(checkpoint, dict) and "network_weights" in checkpoint) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print(f"ADGBC ckpt file not found at {model_path_adgbc}")
except Exception as e:
    print(f"Error loading adgbc: {e}")
    raise e
model.eval()

# Get images
transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ])
clahe = cv2.createCLAHE(
    clipLimit=1.5, tileGridSize=(8, 8)
)
def get_img(img_path: str) -> torch.Tensor:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (192,192),cv2.INTER_AREA)
    table = float(255) * (np.linspace(0, 1, 256) ** 0.8)
    img_gamma = cv2.LUT(img_resized.astype(np.uint8), table.astype(np.uint8))
    img_clahe = clahe.apply(img_gamma)
    pil_img = PIL.Image.fromarray(img_clahe)
    return transform(pil_img).unsqueeze(0).to(device)

openeds_1 = get_img("./dataset_images/openeds_1.png")
# openeds_2 = get_img("./dataset_images/openeds_2.png")
# pupil = get_img("./dataset_images/pupil.png")
# swirski = get_img("./dataset_images/swirski.png")

# sigma => torch.Size([1, 1, 16, 64]) : B, N, num_balls, features space dim
# ball centers => torch.Size([16, 64]): num_balls, coordinates

# 필요한 정보: 1. Encoder features(GBC이전)=t3 2. GB 중심 3. GB radii
# (out, t3, t3_gbc, att_t3, dec_feature,sigma,enc_balls_centers)
_,t3_eds1,_,_,_,sigma_eds1,enc_centers_eds1 = model(openeds_1, return_details=True)
t3_eds1_arr = t3_eds1.squeeze().cpu().detach().numpy()
sigma_eds1_arr = sigma_eds1.squeeze().cpu().detach().numpy()
enc_centers_eds1_arr = enc_centers_eds1.squeeze().cpu().detach().numpy()

print("t3_eds1_arr: ", t3_eds1_arr.shape)
print("sigma_eds1_arr: ", sigma_eds1_arr.shape)
print("enc_centers_eds1_arr: ", enc_centers_eds1_arr.shape)

np.save("./toy_tensors/features.npy", t3_eds1_arr)
np.save("./toy_tensors/sigma.npy", sigma_eds1_arr)
np.save("./toy_tensors/centers.npy", enc_centers_eds1_arr)



