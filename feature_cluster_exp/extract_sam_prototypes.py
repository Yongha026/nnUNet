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

# Robust multi-path resolution for GBC_utils
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
trainer_dir = os.path.join(repo_root, "nnunetv2", "training", "nnUNetTrainer")
for p in [repo_root, trainer_dir, script_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from nnunetv2.training.nnUNetTrainer.GBC_utils import (
        shrink_sam_to_gbc_space,
        compute_class_prototypes_and_radii,
        extract_sam_kmeans_clusters,
    )
except ImportError:
    try:
        from GBC_utils import (
            shrink_sam_to_gbc_space,
            compute_class_prototypes_and_radii,
            extract_sam_kmeans_clusters,
        )
    except ImportError:
        import importlib.util
        gbc_utils_path = os.path.join(trainer_dir, "GBC_utils.py")
        if not os.path.isfile(gbc_utils_path):
            gbc_utils_path = os.path.join(script_dir, "GBC_utils.py")
        if os.path.isfile(gbc_utils_path):
            spec = importlib.util.spec_from_file_location("GBC_utils", gbc_utils_path)
            gbc_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gbc_utils)
            shrink_sam_to_gbc_space = gbc_utils.shrink_sam_to_gbc_space
            compute_class_prototypes_and_radii = gbc_utils.compute_class_prototypes_and_radii
            extract_sam_kmeans_clusters = getattr(gbc_utils, "extract_sam_kmeans_clusters", None)
        else:
            raise ModuleNotFoundError(
                f"Could not find 'GBC_utils.py' in {trainer_dir} or {script_dir}. "
                f"Please ensure 'git pull' has been run to sync latest files."
            )

import timm



def parse_args():
    parser = argparse.ArgumentParser(description="Extract GBC prototype centers, radii, and precompute SAM features.")
    parser.add_argument("--images_dir", type=str, default="./dataset_images", help="Path to OpenEDS images directory")
    parser.add_argument("--labels_dir", type=str, default=None, help="Path to OpenEDS labels directory (default: inferred from images)")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of accumulated images for prototype extraction")
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size for SAM forward pass to prevent CUDA OOM (default: 4)")
    parser.add_argument("--save_dir", type=str, default="./sam_centers", help="Directory to save features.npy, openeds_centers_{K}.npy, and openeds_sigma_{K}.npy")
    parser.add_argument("--cluster_k", type=str, default="2,4,8,16,32,64", help="Comma-separated cluster counts K to extract (default: 2,4,8,16,32,64)")
    parser.add_argument("--save_features", action="store_true", default=True, help="Save precomputed SAM features.npy and masks.npy (default: True)")
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

    # 1. Load data batch (kept on CPU until fed in mini-batches)
    images, masks = load_sample_batch(args.images_dir, args.labels_dir, args.num_samples)

    # 2. Instantiate pre-trained SAM image encoder
    print("Loading pretrained SAM ViT encoder (samvit_base_patch16.sa1b)...")
    sam_model = timm.create_model("samvit_base_patch16.sa1b", pretrained=True, num_classes=4)
    sam_model = sam_model.eval().to(device)

    # 3. Extract SAM features in mini-batches to prevent CUDA OOM
    print(f"Forwarding {len(images)} images through SAM in mini-batches (batch_size={args.batch_size})...")
    sam_features_list = []
    with torch.no_grad():
        for i in range(0, len(images), args.batch_size):
            batch_imgs = images[i:i + args.batch_size].to(device)
            feats = sam_model.forward_features(batch_imgs)  # [b, 256, 64, 64]
            sam_features_list.append(feats.cpu())
            del batch_imgs, feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sam_features = torch.cat(sam_features_list, dim=0).to(device)
        masks = masks.to(device)
        print(f"Aggregated SAM output features shape: {tuple(sam_features.shape)}")

        # Clean up SAM model from GPU memory before PCA / clustering
        del sam_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Shrink space: [B, 256, 64, 64] -> [B, 64, 48, 48]
        print("Shrinking SAM feature space: (256, 64, 64) -> (64, 48, 48) via PCA...")
        gbc_features, pca = shrink_sam_to_gbc_space(sam_features, target_hw=(48, 48), target_dim=64)
        print(f"GBC space features shape: {tuple(gbc_features.shape)}")

        # 5. Extract 4-class prototypes and empirical radii (if masks provided)
        if masks is not None:
            print("Computing 4-class prototypes and empirical dispersion radii from masks...")
            centers_4, log_sigma_4 = compute_class_prototypes_and_radii(gbc_features, masks, num_classes=4)
        else:
            centers_4, log_sigma_4 = None, None

    os.makedirs(args.save_dir, exist_ok=True)

    # 6. Save precomputed SAM features
    if args.save_features:
        features_np = gbc_features.cpu().numpy()
        features_path = os.path.join(args.save_dir, "features.npy")
        np.save(features_path, features_np)
        print(f"[*] Precomputed SAM Features shape: {features_np.shape} -> Saved to: {features_path}")

        if masks is not None:
            if masks.shape[-2:] != (48, 48):
                masks_resampled = F.interpolate(masks.unsqueeze(1).float(), size=(48, 48), mode="nearest").squeeze(1).long()
            else:
                masks_resampled = masks.long()
            masks_np = masks_resampled.cpu().numpy()
            masks_path = os.path.join(args.save_dir, "masks.npy")
            np.save(masks_path, masks_np)
            print(f"[*] Aligned Masks shape:            {masks_np.shape} -> Saved to: {masks_path}")

    # 7. Generate and save cluster prototypes for all requested K values
    cluster_k_list = [int(k.strip()) for k in args.cluster_k.split(",") if k.strip()] if args.cluster_k else [4]
    print(f"\n[*] Generating prototype clusters for K in {cluster_k_list}...")

    for k in cluster_k_list:
        if k == 4 and centers_4 is not None:
            k_centers = centers_4
            k_sigma = F.softplus(log_sigma_4) + 1e-6
            print(f"  - K=4: Computed via ground-truth masked pooling (4 classes)")
        elif extract_sam_kmeans_clusters is not None:
            k_centers, k_log_sigma = extract_sam_kmeans_clusters(gbc_features, num_clusters=k)
            k_sigma = F.softplus(k_log_sigma) + 1e-6
            print(f"  - K={k}: Computed via feature token K-Means clustering")
        else:
            continue

        k_centers_np = k_centers.cpu().numpy()
        k_sigma_np = k_sigma.cpu().numpy()

        k_centers_path = os.path.join(args.save_dir, f"openeds_centers_{k}.npy")
        k_sigma_path = os.path.join(args.save_dir, f"openeds_sigma_{k}.npy")

        np.save(k_centers_path, k_centers_np)
        np.save(k_sigma_path, k_sigma_np)

        if k == 4:
            # Backward-compatible generic aliases
            np.save(os.path.join(args.save_dir, "centers.npy"), k_centers_np)
            np.save(os.path.join(args.save_dir, "sigma.npy"), k_sigma_np)

        print(f"    Saved: {k_centers_path} {k_centers_np.shape} | {k_sigma_path} {k_sigma_np.shape}")

    print("\n" + "=" * 60)
    print("         SAM FEATURE & PROTOTYPE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Save Directory: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
