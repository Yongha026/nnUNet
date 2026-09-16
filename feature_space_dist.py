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
openeds_2 = get_img("./dataset_images/openeds_2.png")
pupil = get_img("./dataset_images/pupil.png")
swirski = get_img("./dataset_images/swirski.png")

# (out, t3, t3_gbc, att_t3, dec_feature, sigma, enc_ball_centers) sigma: radii
# sigma => torch.Size([1, 1, 16, 64]) : B, N, num_balls, features space dim
sigma_eds_1 = model(openeds_1)[-1].squeeze()
sigma_eds_2 = model(openeds_2)[-1].squeeze()
sigma_pupil = model(pupil)[-1].squeeze()
sigma_swir = model(swirski)[-1].squeeze()

### GB는 훈련된 파라미터(broadcast에만 사용하니까) => 정해져있겠지 멍청아
# 그러면 아예 인코더 나온 feature 들을 16개로 군집화시키면?
# 문제: 얘는 Anisotropic, K-means는 Isotropic. 초기화 문제도 있음.
print(sigma_eds_1[0])
print(sigma_swir[0])

# 알고싶은거: Deep feature의 군집 정보가 유의미하게 Segmentation metric을 올려주는가?

# 계획1: GB16개로 EDS, Swir 50epo씩만 훈련 -  Pupil & 그외만.
# EDS, Aniso
# EDS, Iso
# Swir, Aniso
# Swir, Iso

# 계획2. GB16, Iso로 AD-GBC재훈련 - GB 중심 그냥 줘버리고 Finetuning만 할까
# vs 해당 중심으로 K-means 돌려서 결과 비슷한지.
# 1. AD-GBC_S 동일 인코더, 디코더. GB 파라미터만 Isotropic으로 바꾸고 파인튜닝.
# torch kmeans로 GBC 빼고 중심점만 주고 broadcast해서 정보 더해서 결과내기.


# +@. Domain gap있는 데이터 줄 때 feature만 주고 새로 군집화시켜. 그게 finetuning과 뭐가 다르지
# TTA가 뭔지 읽어봐라...