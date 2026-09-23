"""
extract_sam_prototypes.py
=========================
Offline utility to extract GBC cluster centers (4, 64) and empirical dispersion radii (4, 64)
from pre-trained SAM image encoder features on the OpenEDS2019 dataset.

Workflow:
1. Loads an accumulated batch of OpenEDS images and corresponding 4-class ground-truth masks.
2. Passes images through pretrained SAM ViT encoder (samvit_base_patch16.sa1b) -> (B, 256, 64, 64).
3. Spatially resamples features from (64, 64) -> (48, 48) and projects channels 256 -> 64 via PCA.
4. Performs masked average pooling per class to compute class prototype centers (4, 64).
5. Computes empirical class standard deviations to initialize log_sigma (4, 64).
6. Saves `centers.npy` and `sigma.npy` ready for GBC parameter loading or clustering analysis.

Usage on remote server:
    python feature_cluster_exp/extract_sam_prototypes.py \
        --images_dir /path/to/OpenEDS2019/images \
        --labels_dir /path/to/OpenEDS2019/labels \
        --num_samples 32 \
        --save_dir ./toy_tensors \
        --device cuda
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

try:
    from nnunetv2.training.nnUNetTrainer.GBC_utils import (
        shrink_sam_to_gbc_space,
        compute_class_prototypes_and_radii,
    )
except ImportError:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(repo_root)
    from nnunetv2.training.nnUNetTrainer.GBC_utils import (
        shrink_sam_to_gbc_space,
        compute_class_prototypes_and_radii,
    )

import timm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract GBC prototype centers and radii from SAM features.")
    parser.add_argument("--images_dir", type=str, default="./dataset_images", help="Path to OpenEDS images directory")
    parser.add_argument("--labels_dir", type=str, default=None, help="Path to OpenEDS labels directory (default: inferred from images)")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of accumulated images for prototype extraction")
    parser.add_argument("--save_dir", type=str, default="./sam_centers", help="Directory to save openeds_centers_{K}.npy and openeds_sigma_{K}.npy")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def load_sample_batch(images_dir, labels_dir=None, num_samples=32):
    """
    Loads and preprocesses an accumulated batch of images (1024x1024 for SAM)
    and corresponding 4-class masks (0=bg, 1=pupil, 2=iris, 3=sclera).
    """
    img_files = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))
    if not img_files:
        raise FileNotFoundError(f"No image files found in {images_dir}")

    img_files = img_files[:num_samples]
    print(f"Loading {len(img_files)} accumulated images for prototype extraction...")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    table = float(255) * (np.linspace(0, 1, 256) ** 0.8)

    images_list = []
    masks_list = []

    for img_path in img_files:
        # 1. Preprocess image for SAM (1024x1024, 3ch RGB)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_gamma = cv2.LUT(img.astype(np.uint8), table.astype(np.uint8))
        img_clahe = clahe.apply(img_gamma)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        pil_img = PIL.Image.fromarray(img_resized)
        images_list.append(transform(pil_img))

        # 2. Load corresponding mask
        if labels_dir is not None:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_candidates = [
                os.path.join(labels_dir, f"{base_name}.npy"),
                os.path.join(labels_dir, f"{base_name}.png"),
            ]
        else:
            mask_candidates = [
                img_path.replace("images", "labels").replace(".png", ".npy"),
                img_path.replace("images", "labels").replace(".jpg", ".npy"),
                img_path.replace("images", "labels"),
            ]

        mask_loaded = False
        for m_path in mask_candidates:
            if os.path.exists(m_path):
                if m_path.endswith(".npy"):
                    msk = np.load(m_path).astype(np.uint8)
                else:
                    msk = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                masks_list.append(torch.from_numpy(msk).long())
                mask_loaded = True
                break

        if not mask_loaded:
            raise FileNotFoundError(f"Could not find matching label mask for image: {img_path}")

    images_tensor = torch.stack(images_list, dim=0)  # [B, 3, 1024, 1024]
    masks_tensor = torch.stack(masks_list, dim=0)    # [B, H, W]
    return images_tensor, masks_tensor


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Executing SAM Prototype Extraction on device: {device}")

    # 1. Load data batch
    images, masks = load_sample_batch(args.images_dir, args.labels_dir, args.num_samples)
    images = images.to(device)
    masks = masks.to(device)

    # 2. Instantiate pre-trained SAM image encoder
    print("Loading pretrained SAM ViT encoder (samvit_base_patch16.sa1b)...")
    sam_model = timm.create_model("samvit_base_patch16.sa1b", pretrained=True, num_classes=4)
    sam_model = sam_model.eval().to(device)

    # 3. Extract SAM features
    print("Forwarding accumulated batch through SAM encoder...")
    with torch.no_grad():
        sam_features = sam_model.forward_features(images)  # [B, 256, 64, 64]
        print(f"SAM output features shape: {tuple(sam_features.shape)}")

        # 4. Shrink space: [B, 256, 64, 64] -> [B, 64, 48, 48]
        print("Shrinking SAM feature space: (256, 64, 64) -> (64, 48, 48) via PCA...")
        gbc_features, pca = shrink_sam_to_gbc_space(sam_features, target_hw=(48, 48), target_dim=64)
        print(f"GBC space features shape: {tuple(gbc_features.shape)}")

        # 5. Extract 4-class prototypes and empirical radii
        print("Computing 4-class prototypes and empirical dispersion radii...")
        centers, log_sigma = compute_class_prototypes_and_radii(gbc_features, masks, num_classes=4)

    # Calculate actual positive sigma: softplus(log_sigma) + 1e-6
    sigma = F.softplus(log_sigma) + 1e-6

    centers_np = centers.cpu().numpy()
    sigma_np = sigma.cpu().numpy()

    # 6. Save results
    os.makedirs(args.save_dir, exist_ok=True)
    centers_path = os.path.join(args.save_dir, "openeds_centers_4.npy")
    sigma_path = os.path.join(args.save_dir, "openeds_sigma_4.npy")

    np.save(centers_path, centers_np)
    np.save(sigma_path, sigma_np)
    # Also save generic centers.npy / sigma.npy for backward compatibility
    np.save(os.path.join(args.save_dir, "centers.npy"), centers_np)
    np.save(os.path.join(args.save_dir, "sigma.npy"), sigma_np)

    print("\n" + "=" * 60)
    print("         SAM PROTOTYPE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Centers shape: {centers_np.shape} -> Saved to: {centers_path}")
    print(f"Sigma shape:   {sigma_np.shape} -> Saved to: {sigma_path}")
    print("\nClass Statistics:")
    class_names = ["Background", "Pupil", "Iris", "Sclera"]
    for k in range(4):
        c_norm = np.linalg.norm(centers_np[k])
        s_mean = sigma_np[k].mean()
        print(f"  Class {k} ({class_names[k]:<10}): Center Norm = {c_norm:.3f}, Mean Radius (sigma) = {s_mean:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
