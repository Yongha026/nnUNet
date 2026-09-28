"""
tsne_sam.py — Visualise Segment Anything Model (SAM) image encoder features via t-SNE / PCA.

Plots patch-wise feature clustering extracted from SAM ViT image encoder (samvit_base_patch16.sa1b).
Supports native SAM representation space (256-d, 64x64 grid) and GBC-projected space (64-d, 48x48 grid),
with optional overlay of precomputed cluster centers.

Usage
-----
python feature_cluster_exp/tsne_sam.py /path/to/images [options]

Options
-------
--datas             Number of images to sample (default: 1024)
--samples_per_class Max points per class for t-SNE (default: 2000)
--batch_size        Mini-batch size for SAM forward pass (default: 4 to prevent CUDA OOM)
--pupil_only        Binary: Pupil vs Else (default)
--all_classes       Full 4-class: Background / Sclera / Iris / Pupil
--output            Output path (default: auto-generated in tsne_results/)
--perplexity        t-SNE perplexity (default: 30)
--seed              Random seed (default: 42)
--pca               Use PCA dimensionality reduction instead of t-SNE
--untrained         Load untrained (random init) SAM model
--model_name        SAM model architecture from timm (default: samvit_base_patch16.sa1b)
--shrink_gbc        Project SAM features to GBC space (48x48 spatial, 64 PCA channels)
--centers_path      Path to precomputed centers .npy (e.g. sam_centers/openeds_centers_4.npy)
--labels_dir        Explicit directory for segmentation label masks (optional)
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
import torch.nn.functional as F

import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ─── Path setup & GBC utils import ───────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
trainer_dir = os.path.join(repo_root, "nnunetv2", "training", "nnUNetTrainer")
for p in [repo_root, trainer_dir, script_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from nnunetv2.training.nnUNetTrainer.GBC_utils import shrink_sam_to_gbc_space
except ImportError:
    try:
        from GBC_utils import shrink_sam_to_gbc_space
    except ImportError:
        shrink_sam_to_gbc_space = None


# ─── Dataset ─────────────────────────────────────────────────────────────────
class SAMImageDataset(Dataset):
    def __init__(self, image_paths, labels_dir=None, grid_size=64):
        self.image_paths = image_paths
        self.labels_dir = labels_dir
        self.grid_size = grid_size
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # ImageNet standard normalization expected by SAM encoder in timm
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        self.table = float(255) * (np.linspace(0, 1, 256) ** 0.8)

    def __len__(self):
        return len(self.image_paths)

    def _load_mask(self, img_path):
        candidates = []
        if self.labels_dir is not None:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            candidates.extend([
                os.path.join(self.labels_dir, f"{base_name}.npy"),
                os.path.join(self.labels_dir, f"{base_name}.png"),
            ])
        else:
            candidates.extend([
                img_path.replace("images", "labels").replace(".png", ".npy"),
                img_path.replace("images", "labels").replace(".jpg", ".npy"),
                img_path.replace("images", "labels"),
            ])

        for c_path in candidates:
            if os.path.isfile(c_path):
                if c_path.endswith(".npy"):
                    return np.load(c_path).astype(np.uint8)
                else:
                    return cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)

        raise FileNotFoundError(f"Mask not found for image: {img_path}")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # 1. Image Preprocessing (grayscale -> CLAHE gamma -> 3ch RGB -> 1024x1024 for SAM)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load image from {img_path}")
        img_gamma = cv2.LUT(img.astype(np.uint8), self.table.astype(np.uint8))
        img_clahe = self.clahe.apply(img_gamma)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        pil_img = PIL.Image.fromarray(img_resized)

        # 2. Mask Preprocessing (Aligned to SAM feature grid, e.g. 64x64 or 48x48)
        msk = self._load_mask(img_path)
        msk_resized = cv2.resize(
            msk, (self.grid_size, self.grid_size), interpolation=cv2.INTER_NEAREST
        )

        return self.transform(pil_img), msk_resized


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise SAM image encoder patch features using t-SNE / PCA"
    )
    parser.add_argument("IMG_PATH", type=str, help="Path to image folder")
    parser.add_argument(
        "--labels_dir", default=None, type=str, help="Path to label folder (optional)"
    )
    parser.add_argument(
        "--datas", default=1024, type=int, help="Number of images to sample"
    )
    parser.add_argument(
        "--samples_per_class",
        default=2000,
        type=int,
        help="Max points per class for t-SNE",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Mini-batch size for SAM forward pass (default: 4)",
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
    parser.add_argument(
        "--untrained", "-u", default=False, action="store_true", help="Load untrained (random init) SAM model"
    )
    parser.add_argument(
        "--model_name",
        default="samvit_base_patch16.sa1b",
        type=str,
        help="Timm SAM model name (default: samvit_base_patch16.sa1b)",
    )
    parser.add_argument(
        "--shrink_gbc",
        action="store_true",
        default=False,
        help="Project SAM features to GBC space (resample 48x48, PCA 64 channels)",
    )
    parser.add_argument(
        "--centers_path",
        default=None,
        type=str,
        help="Optional path to precomputed cluster centers .npy to overlay on the plot",
    )

    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    trained_prefix = "UNTRAINED_" if args.untrained else ""
    model_prefix = trained_prefix + args.model_name
    if args.shrink_gbc:
        model_prefix += "_GBC64"

    # Dataset identification
    img_path_str = args.IMG_PATH.lower()
    if "openeds" in img_path_str or "250" in img_path_str:
        dataset_prefix = "OpenEDS2019"
    elif "jw_" in img_path_str or "pupil" in img_path_str:
        dataset_prefix = "PupilLabs"
    elif "swir" in img_path_str:
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
        print(f"[INFO] Loading randomly initialized {args.model_name} from timm ...")
    else:
        print(f"[INFO] Loading pretrained {args.model_name} from timm ...")

    model = timm.create_model(args.model_name, pretrained=not args.untrained)
    model.eval().to(device)

    # ── Grid resolution ──────────────────────────────────────────────────────
    target_grid = 48 if args.shrink_gbc else 64

    # ── Image sampling ───────────────────────────────────────────────────────
    image_patterns = [
        os.path.join(args.IMG_PATH, "*.png"),
        os.path.join(args.IMG_PATH, "*.jpg"),
    ]
    full_images = []
    for pat in image_patterns:
        full_images.extend(glob.glob(pat))
    full_images = sorted(full_images)

    if not full_images:
        raise FileNotFoundError(f"No image files found in {args.IMG_PATH}")

    try:
        rand_images = random.sample(full_images, args.datas)
    except ValueError:
        rand_images = full_images
    print(f"[INFO] Dataset: {dataset_prefix} — Using {len(rand_images)} images (Grid: {target_grid}x{target_grid})")

    dataset = SAMImageDataset(
        rand_images, labels_dir=args.labels_dir, grid_size=target_grid
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=2, pin_memory=(device_str == "cuda")
    )

    NUM_CLASSES = 4
    SAMPLES_PER_CLASS = args.samples_per_class
    feats_by_class = {c: [] for c in range(NUM_CLASSES)}

    # ── Feature extraction ───────────────────────────────────────────────────
    print(f"[INFO] Extracting SAM features with {args.model_name} ...")
    pca_gbc = None

    with torch.no_grad():
        for batch_imgs, batch_masks in tqdm(dataloader, desc="Extracting SAM"):
            batch_imgs = batch_imgs.to(device)

            # timm SAM forward_features output: [B, 256, 64, 64]
            sam_feats = model.forward_features(batch_imgs)

            if args.shrink_gbc:
                if shrink_sam_to_gbc_space is not None:
                    feats, pca_gbc = shrink_sam_to_gbc_space(
                        sam_feats, target_hw=(48, 48), target_dim=64, pca=pca_gbc
                    )
                else:
                    # Inline fallback if shrink_sam_to_gbc_space is unavailable
                    if sam_feats.shape[-2:] != (48, 48):
                        sam_feats = F.interpolate(sam_feats, size=(48, 48), mode="bilinear", align_corners=False)
                    B_s, C_s, H_s, W_s = sam_feats.shape
                    flat_s = sam_feats.permute(0, 2, 3, 1).reshape(-1, C_s).cpu().numpy()
                    if pca_gbc is None:
                        pca_gbc = PCA(n_components=64, random_state=args.seed)
                        flat_pca = pca_gbc.fit_transform(flat_s)
                    else:
                        flat_pca = pca_gbc.transform(flat_s)
                    feats = torch.from_numpy(flat_pca).float().view(B_s, H_s, W_s, 64).permute(0, 3, 1, 2).to(device)
            else:
                feats = sam_feats

            B, C, H, W = feats.shape

            # Flatten spatial tokens: [B * H * W, C]
            patch_pixels = feats.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            labels_pixels = batch_masks.reshape(-1).numpy()

            # Class-conditional feature accumulation
            for c in range(NUM_CLASSES):
                current_len = sum(len(x) for x in feats_by_class[c])
                if current_len < SAMPLES_PER_CLASS:
                    c_mask = labels_pixels == c
                    c_feats = patch_pixels[c_mask]
                    if len(c_feats) > 0:
                        feats_by_class[c].append(c_feats)

    # Free SAM model from GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    if not selected_feats:
        raise RuntimeError("No features extracted. Please check image paths and label mask alignments.")

    X = np.vstack(selected_feats)    # [N_total, D] (D=256 or D=64)
    y = np.concatenate(selected_labels)
    print(f"[INFO] Collected feature matrix X: {X.shape} across {len(np.unique(y))} classes")

    # ── Optional Center Overlay ──────────────────────────────────────────────
    centers = None
    if args.centers_path and os.path.isfile(args.centers_path):
        centers = np.load(args.centers_path)
        print(f"[INFO] Loaded cluster centers from {args.centers_path}: shape {centers.shape}")
        if centers.shape[1] != X.shape[1]:
            print(f"[WARN] Center dimension ({centers.shape[1]}) does not match feature dimension ({X.shape[1]}). Skipping center overlay.")
            centers = None

    show_centers = centers is not None
    if show_centers:
        all_points = np.vstack([X, centers])
    else:
        all_points = X

    # ── Dimensionality Reduction ─────────────────────────────────────────────
    if args.pca:
        print(f"[INFO] Running 2D PCA on {all_points.shape[0]} points (Dim: {all_points.shape[1]}) …")
        pca_2d = PCA(n_components=2, random_state=args.seed)
        all_embedded = pca_2d.fit_transform(all_points)
        var_pct = pca_2d.explained_variance_ratio_.sum() * 100
        method_detail = f"PCA (Explained Var: {var_pct:.1f}%)"
        reduction_tag = "pca"
    else:
        # Pre-reduce to 64 components via PCA if input dimension > 64, then run t-SNE
        if all_points.shape[1] > 64:
            print(f"[INFO] Pre-reducing dim ({all_points.shape[1]} -> 64) with PCA before t-SNE …")
            all_pre = PCA(n_components=64, random_state=args.seed).fit_transform(all_points)
        else:
            all_pre = all_points

        print(f"[INFO] Running t-SNE on {all_points.shape[0]} points (perplexity={args.perplexity}) …")
        tsne = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            n_jobs=-1,
            random_state=args.seed,
        )
        all_embedded = tsne.fit_transform(all_pre)
        method_detail = f"t-SNE (perplexity={args.perplexity})"
        reduction_tag = "tsne"

    if show_centers:
        X_embedded = all_embedded[:len(X)]
        centers_embedded = all_embedded[len(X):]
    else:
        X_embedded = all_embedded
        centers_embedded = None

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
        zorder=1,
    )

    if show_centers and centers_embedded is not None:
        ball_cmap = plt.cm.get_cmap("tab20")
        num_centers = centers_embedded.shape[0]
        for k in range(num_centers):
            color_k = ball_cmap(k % 20)
            ck = centers_embedded[k]
            plt.scatter(
                ck[0],
                ck[1],
                marker="*",
                s=140,
                c=[color_k],
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            plt.annotate(
                str(k),
                xy=(ck[0], ck[1]),
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="black",
                zorder=6,
            )

    cbar = plt.colorbar(scatter, ticks=ticks)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("Class ID")

    center_suffix = f" • {centers_embedded.shape[0]} Centers (*)" if show_centers else ""
    plt.title(
        f"[{dataset_prefix}] SAM Patch Feature Clustering\n"
        f"{model_prefix} • {X.shape[0]} points • {method_detail}{center_suffix}",
        fontsize=11,
    )
    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    results_dir = os.path.join(script_dir, "tsne_results")
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
