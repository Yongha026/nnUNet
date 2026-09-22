"""
tsne_vit.py — Visualise ViT encoder patch features via t-SNE / PCA.

Plots patch-wise feature clustering extracted from Vision Transformer (ViT-L/16).

Usage
-----
python tsne_vit.py /path/to/images [options]

Options
-------
--datas             Number of images to sample (default: 1024)
--samples_per_class Max points per class for t-SNE (default: 2000)
--pupil_only        Binary: Pupil vs Else (default)
--all_classes       Full 4-class: Background / Sclera / Iris / Pupil
--output            Output path (default: auto-generated in tsne_results/)
--perplexity        t-SNE perplexity (default: 30)
--seed              Random seed (default: 42)
--pca               Use PCA instead of t-SNE
"""

import os
import sys
import glob
import argparse
import random
import numpy as np
import cv2
import PIL.Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# ─── Dataset ─────────────────────────────────────────────────────────────────
class ViTImageDataset(Dataset):
    def __init__(self, image_paths, grid_size=14):
        self.image_paths = image_paths
        self.grid_size = grid_size
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # ImageNet standard normalization for pretrained ViT
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        self.table = float(255) * (np.linspace(0, 1, 256) ** 0.8)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Image Preprocessing (192 -> 224x224, 1ch -> 3ch)
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img_gamma = cv2.LUT(img_resized.astype(np.uint8), self.table.astype(np.uint8))
        img_clahe = self.clahe.apply(img_gamma)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        pil_img = PIL.Image.fromarray(img_rgb)

        # 2. Mask Preprocessing (Aligned to ViT patch grid 14x14)
        msk_path = self.image_paths[idx].replace("images", "labels").replace("png", "npy")
        msk = np.load(msk_path).astype(np.uint8)
        msk_resized = cv2.resize(
            msk, (self.grid_size, self.grid_size), interpolation=cv2.INTER_NEAREST
        )

        return self.transform(pil_img), msk_resized


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise ViT encoder patch features using t-SNE / PCA"
    )
    parser.add_argument("IMG_PATH", type=str, help="Path to image folder")
    parser.add_argument(
        "--datas", default=1024, type=int, help="Number of images to sample"
    )
    parser.add_argument(
        "--samples_per_class",
        default=2000,
        type=int,
        help="Max points per class for t-SNE",
    )

    class_group = parser.add_mutually_exclusive_group()
    class_group.add_argument(
        "--pupil_only",
        action="store_true",
        default=True,
        help="Binary: Pupil vs Else (default)",
    )
    class_group.add_argument(
        "--all_classes",
        action="store_true",
        default=False,
        help="Full 4-class: Background / Sclera / Iris / Pupil",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Output path (default: auto-generated in tsne_results/)",
    )
    parser.add_argument(
        "--perplexity", default=30, type=int, help="t-SNE perplexity"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--pca",
        action="store_true",
        default=False,
        help="Use PCA dimensionality reduction instead of t-SNE",
    )
    parser.add_argument("--untrained", "-u", default=False, action="store_true", help="Load untrained(random init) vit model")
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    trained_prefix = "UNTRAINED_" if args.untrained else ""
    model_prefix = trained_prefix + "vit_large_patch16_224"

    # Dataset identification
    EDS_match = "Openedsdata2019" in args.IMG_PATH
    Pupil_match = "jw_" in args.IMG_PATH
    Swir_match = "nnunetv2_swir" in args.IMG_PATH

    if EDS_match:
        dataset_prefix = "OpenEDS2019"
    elif Pupil_match:
        dataset_prefix = "PupilLabs"
    elif Swir_match:
        dataset_prefix = "Swirski"
    else:
        dataset_prefix = "CustomData"

    if args.all_classes:
        args.pupil_only = False

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Model loading ────────────────────────────────────────────────────────
    if args.untrained:
        print(f"[INFO] Loading random initialized {model_prefix} from timm ...")
    else:
        print(f"[INFO] Loading pretrained {model_prefix} from timm ...")
    model = timm.create_model("vit_large_patch16_224", pretrained=args.untrained)
    model.eval().to(device)

    # ── Image sampling ───────────────────────────────────────────────────────
    image_path = os.path.join(args.IMG_PATH, "*.png")
    full_images = glob.glob(image_path)
    try:
        rand_images = random.sample(full_images, args.datas)
    except ValueError:
        rand_images = full_images
    print(f"[INFO] Dataset: {dataset_prefix} — Using {len(rand_images)} images")

    dataset = ViTImageDataset(rand_images, grid_size=14)
    dataloader = DataLoader(
        dataset, batch_size=16, num_workers=4, pin_memory=True
    )

    NUM_CLASSES = 4
    SAMPLES_PER_CLASS = args.samples_per_class
    feats_by_class = {c: [] for c in range(NUM_CLASSES)}

    # ── Feature extraction ───────────────────────────────────────────────────
    print("[INFO] Extracting ViT patch tokens ...")
    with torch.no_grad():
        for batch_imgs, batch_masks in tqdm(dataloader, desc="Extracting"):
            batch_imgs = batch_imgs.to(device)

            # timm ViT forward_features 출력: [B, num_tokens, embed_dim] (num_tokens = 1 + 196)
            tokens = model.forward_features(batch_imgs)

            # [CLS] 토큰 제거 후 196개 패치 토큰 슬라이싱: [B, 196, 1024]
            patch_tokens = tokens[:, 1:, :]
            B, num_patches, embed_dim = patch_tokens.shape

            # 1D 토큰을 [B * 14 * 14, embed_dim] 2D 행렬 형태로 Flatten
            patch_pixels = patch_tokens.reshape(-1, embed_dim).cpu().numpy()
            labels_pixels = batch_masks.reshape(-1).numpy()

            # 클래스별 특징 수집
            for c in range(NUM_CLASSES):
                current_len = sum(len(x) for x in feats_by_class[c])
                if current_len < SAMPLES_PER_CLASS:
                    c_mask = labels_pixels == c
                    c_feats = patch_pixels[c_mask]
                    if len(c_feats) > 0:
                        feats_by_class[c].append(c_feats)

    # ── Balance & Merge ──────────────────────────────────────────────────────
    selected_feats, selected_labels = [], []
    for c in range(NUM_CLASSES):
        if len(feats_by_class[c]) == 0:
            continue
        c_all = np.vstack(feats_by_class[c])

        if len(c_all) > SAMPLES_PER_CLASS:
            idx = np.random.choice(len(c_all), SAMPLES_PER_CLASS, replace=False)
            c_all = c_all[idx]

        selected_feats.append(c_all)
        selected_labels.append(np.full(len(c_all), c))

    X = np.vstack(selected_feats)    # [N_total, 1024]
    y = np.concatenate(selected_labels)

    # ── Dimensionality Reduction ─────────────────────────────────────────────
    if args.pca:
        print(f"[INFO] Running PCA on {X.shape[0]} patch vectors (Dim: {X.shape[1]}) …")
        pca = PCA(n_components=2, random_state=args.seed)
        X_embedded = pca.fit_transform(X)
        var_pct = pca.explained_variance_ratio_.sum() * 100
        method_detail = f"PCA (Explained Var: {var_pct:.1f}%)"
        reduction_tag = "pca"
    else:
        # ViT-Large의 1024차원을 t-SNE로 직접 축소하기 전 PCA 64차원으로 선행 축소하여 가속
        print(f"[INFO] Pre-reducing dim (1024 -> 64) with PCA before t-SNE …")
        X_pca = PCA(n_components=64, random_state=args.seed).fit_transform(X)

        print(f"[INFO] Running t-SNE on {X.shape[0]} patch vectors (perplexity={args.perplexity}) …")
        tsne = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            n_jobs=-1,
            random_state=args.seed,
        )
        X_embedded = tsne.fit_transform(X_pca)
        method_detail = f"t-SNE (perplexity={args.perplexity})"
        reduction_tag = "tsne"

    # ── Label Remapping & Colormap ───────────────────────────────────────────
    tab10_cmap = plt.cm.get_cmap("tab10")
    if args.pupil_only:
        y_plot = np.where(y == 3, 1, 0)
        plot_cmap = ListedColormap(tab10_cmap.colors[:2])
        ticks = [0, 1]
        tick_labels = ["Else", "Pupil"]
        suffix = "pupil_only"
    else:
        y_plot = y
        plot_cmap = ListedColormap(tab10_cmap.colors[:4])
        ticks = [0, 1, 2, 3]
        tick_labels = ["Background", "Sclera", "Iris", "Pupil"]
        suffix = "all_classes"

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y_plot,
        cmap=plot_cmap,
        alpha=0.4,
        s=6,
    )

    cbar = plt.colorbar(scatter, ticks=ticks)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("Class ID")

    plt.title(
        f"[{dataset_prefix}] ViT Patch Feature Clustering\n"
        f"{model_prefix} • {X.shape[0]} points • {method_detail}",
        fontsize=11,
    )
    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(exp_dir, "tsne_results")
    os.makedirs(results_dir, exist_ok=True)

    if args.output:
        save_path = args.output
    else:
        save_path = os.path.join(
            results_dir,
            f"{model_prefix}_{dataset_prefix}_{reduction_tag}_{suffix}.png",
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {save_path}")