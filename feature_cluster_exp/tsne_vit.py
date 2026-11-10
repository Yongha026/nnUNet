"""
tsne_vit.py — Visualise ViT encoder patch features via t-SNE / PCA.

Plots patch-wise feature clustering extracted from Vision Transformer (e.g., ViT-L/16, DINOv3 ViT-B/16).
Supports native ViT patch representation space (e.g., 1024-d 14x14 or 768-d 16x16) and
GBC-projected space (64-d, 48x48 grid via --shrink_gbc), with optional cluster center overlay.

Usage
-----
python feature_cluster_exp/tsne_vit.py /path/to/images [options]

Options
-------
--datas             Number of images to sample (default: 1024)
--samples_per_class Max points per class for t-SNE (default: 2000)
--batch_size        Mini-batch size for ViT forward pass (default: 16)
--pupil_only        Binary: Pupil vs Else (default)
--all_classes       Full 4-class: Background / Sclera / Iris / Pupil
--output            Output path (default: auto-generated in tsne_results/)
--perplexity        t-SNE perplexity (default: 30)
--seed              Random seed (default: 42)
--pca               Use PCA instead of t-SNE
--untrained         Load untrained (random init) ViT model
--model_name        ViT model architecture from timm (default: vit_large_patch16_224, e.g. vit_base_patch16_dinov3.lvd1689m)
--img_size          Optional input image resolution override (default: resolved from model)
--shrink_gbc        Project ViT features to GBC space (48x48 spatial, 64 PCA channels)
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
class ViTImageDataset(Dataset):
    def __init__(self, image_paths, img_size=(224, 224), grid_size=(14, 14), labels_dir=None):
        self.image_paths = image_paths
        self.labels_dir = labels_dir
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.grid_size = (grid_size, grid_size) if isinstance(grid_size, int) else tuple(grid_size)
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # ImageNet standard normalization for pretrained ViT / DINOv3
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
                img_path.replace("images", "labels").replace("png", "npy"),
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
        # 1. Image Preprocessing (192 -> H_img x W_img, 1ch -> 3ch)
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load image from {img_path}")
        img_resized = cv2.resize(
            img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_AREA
        )
        img_gamma = cv2.LUT(img_resized.astype(np.uint8), self.table.astype(np.uint8))
        img_clahe = self.clahe.apply(img_gamma)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        pil_img = PIL.Image.fromarray(img_rgb)

        # 2. Mask Preprocessing (Aligned to ViT patch grid or 48x48 GBC grid)
        msk = self._load_mask(img_path)
        msk_resized = cv2.resize(
            msk, (self.grid_size[1], self.grid_size[0]), interpolation=cv2.INTER_NEAREST
        )

        return self.transform(pil_img), msk_resized


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise ViT encoder patch features using t-SNE / PCA"
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
        default=16,
        type=int,
        help="Mini-batch size for ViT forward pass (default: 16)",
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
        "--untrained", "-u", default=False, action="store_true", help="Load untrained(random init) vit model"
    )
    parser.add_argument(
        "--model_name",
        default="vit_large_patch16_224",
        type=str,
        help="Timm ViT model name (default: vit_large_patch16_224: vit, e.g. vit_base_patch16_dinov3.lvd1689m: dino)",
    )
    parser.add_argument(
        "--img_size",
        default=None,
        type=int,
        help="Optional override for input image resolution (default: uses model's native resolution)",
    )
    parser.add_argument(
        "--shrink_gbc",
        action="store_true",
        default=False,
        help="Project ViT features to GBC space (resample 48x48, PCA 64 channels)",
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
    EDS_match = "openeds" in img_path_str or "250" in img_path_str
    Pupil_match = "jw_" in img_path_str or "pupil" in img_path_str
    Swir_match = "swir" in img_path_str

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
        print(f"[INFO] Loading random initialized {args.model_name} from timm ...")
    else:
        print(f"[INFO] Loading pretrained {args.model_name} from timm ...")

    create_kwargs = {"pretrained": not args.untrained}
    if args.img_size is not None:
        create_kwargs["img_size"] = args.img_size

    if args.model_name == "vit": model_name = "vit_large_patch16_224"
    elif args.model_name == "dino": model_name = "vit_base_patch16_dinov3.lvd1689m"
    else: model_name = "vit_large_patch16_224"

    model = timm.create_model(model_name, **create_kwargs)
    model.eval().to(device)

    # ── Resolve input image size, patch grid size & prefix token count ───────
    if hasattr(model, "patch_embed"):
        native_img_size = getattr(model.patch_embed, "img_size", (224, 224))
        if isinstance(native_img_size, int):
            native_img_size = (native_img_size, native_img_size)
        native_grid_size = getattr(model.patch_embed, "grid_size", None)
        if native_grid_size is None:
            patch_size = getattr(model.patch_embed, "patch_size", (16, 16))
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size)
            native_grid_size = (
                native_img_size[0] // patch_size[0],
                native_img_size[1] // patch_size[1],
            )
        elif isinstance(native_grid_size, int):
            native_grid_size = (native_grid_size, native_grid_size)
    else:
        input_size = model.default_cfg.get("input_size", (3, 224, 224))
        native_img_size = (input_size[1], input_size[2])
        native_grid_size = (native_img_size[0] // 16, native_img_size[1] // 16)

    num_prefix_tokens = getattr(model, "num_prefix_tokens", 1)
    target_mask_grid = (48, 48) if args.shrink_gbc else native_grid_size

    print(
        f"[INFO] Model Config: img_size={native_img_size}, patch_grid={native_grid_size}, "
        f"num_prefix_tokens={num_prefix_tokens}, mask_grid={target_mask_grid}"
    )

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
    print(f"[INFO] Dataset: {dataset_prefix} — Using {len(rand_images)} images")

    dataset = ViTImageDataset(
        rand_images,
        img_size=native_img_size,
        grid_size=target_mask_grid,
        labels_dir=args.labels_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=(device_str == "cuda"),
    )

    NUM_CLASSES = 4
    SAMPLES_PER_CLASS = args.samples_per_class
    feats_by_class = {c: [] for c in range(NUM_CLASSES)}

    # ── Feature extraction ───────────────────────────────────────────────────
    print(f"[INFO] Extracting ViT patch tokens with {args.model_name} ...")
    pca_gbc = None
    num_patches_expected = native_grid_size[0] * native_grid_size[1]

    with torch.no_grad():
        for batch_imgs, batch_masks in tqdm(dataloader, desc="Extracting"):
            batch_imgs = batch_imgs.to(device)

            # timm ViT forward_features 출력: [B, num_prefix_tokens + num_patches, embed_dim]
            tokens = model.forward_features(batch_imgs)

            if tokens.ndim == 4:
                # 4D spatial feature map: [B, C, H, W]
                vit_feats_4d = tokens
                B, embed_dim, H_f, W_f = vit_feats_4d.shape
                patch_tokens = vit_feats_4d.flatten(2).transpose(1, 2)
            else:
                # [CLS] 및 Register 토큰(num_prefix_tokens) 제거 후 공간 패치 토큰만 슬라이싱
                if tokens.shape[1] == num_prefix_tokens + num_patches_expected:
                    patch_tokens = tokens[:, num_prefix_tokens:, :]
                elif tokens.shape[1] > num_patches_expected:
                    patch_tokens = tokens[:, -num_patches_expected:, :]
                else:
                    patch_tokens = tokens
                B, num_patches, embed_dim = patch_tokens.shape

            if args.shrink_gbc:
                if tokens.ndim != 4:
                    vit_feats_4d = (
                        patch_tokens.transpose(1, 2)
                        .reshape(B, embed_dim, native_grid_size[0], native_grid_size[1])
                        .contiguous()
                    )
                if shrink_sam_to_gbc_space is not None:
                    feats_gbc, pca_gbc = shrink_sam_to_gbc_space(
                        vit_feats_4d, target_hw=(48, 48), target_dim=64, pca=pca_gbc
                    )
                else:
                    # Inline fallback if shrink_sam_to_gbc_space is unavailable
                    if vit_feats_4d.shape[-2:] != (48, 48):
                        vit_feats_4d = F.interpolate(
                            vit_feats_4d, size=(48, 48), mode="bilinear", align_corners=False
                        )
                    B_s, C_s, H_s, W_s = vit_feats_4d.shape
                    flat_s = vit_feats_4d.permute(0, 2, 3, 1).reshape(-1, C_s).cpu().numpy()
                    if pca_gbc is None:
                        pca_gbc = PCA(n_components=64, random_state=args.seed)
                        flat_pca = pca_gbc.fit_transform(flat_s)
                    else:
                        flat_pca = pca_gbc.transform(flat_s)
                    feats_gbc = (
                        torch.from_numpy(flat_pca)
                        .float()
                        .view(B_s, H_s, W_s, 64)
                        .permute(0, 3, 1, 2)
                        .to(device)
                    )
                B_g, C_g, H_g, W_g = feats_gbc.shape
                patch_pixels = feats_gbc.permute(0, 2, 3, 1).reshape(-1, C_g).cpu().numpy()
            else:
                # 1D 토큰을 [B * H_grid * W_grid, embed_dim] 2D 행렬 형태로 Flatten
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

    # Free ViT model from GPU memory
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

    X = np.vstack(selected_feats)    # [N_total, D] (D=1024, 768, or 64)
    y = np.concatenate(selected_labels)
    print(f"[INFO] Collected feature matrix X: {X.shape} across {len(np.unique(y))} classes")

    # ── Optional Center Overlay ──────────────────────────────────────────────
    centers = None
    if args.centers_path and os.path.isfile(args.centers_path):
        centers = np.load(args.centers_path)
        print(f"[INFO] Loaded cluster centers from {args.centers_path}: shape {centers.shape}")
        if centers.shape[1] != X.shape[1]:
            print(
                f"[WARN] Center dimension ({centers.shape[1]}) does not match feature dimension ({X.shape[1]}). "
                f"Skipping center overlay."
            )
            centers = None

    show_centers = centers is not None
    if show_centers:
        all_points = np.vstack([X, centers])
    else:
        all_points = X

    # ── Dimensionality Reduction ─────────────────────────────────────────────
    if args.pca:
        print(f"[INFO] Running PCA on {all_points.shape[0]} patch vectors (Dim: {all_points.shape[1]}) …")
        pca = PCA(n_components=2, random_state=args.seed)
        all_embedded = pca.fit_transform(all_points)
        var_pct = pca.explained_variance_ratio_.sum() * 100
        method_detail = f"PCA (Explained Var: {var_pct:.1f}%)"
        reduction_tag = "pca"
    else:
        # 고차원(1024/768차원)을 t-SNE로 직접 축소하기 전 PCA 64차원으로 선행 축소하여 가속
        if all_points.shape[1] > 64:
            print(f"[INFO] Pre-reducing dim ({all_points.shape[1]} -> 64) with PCA before t-SNE …")
            all_pre = PCA(n_components=64, random_state=args.seed).fit_transform(all_points)
        else:
            all_pre = all_points

        print(f"[INFO] Running t-SNE on {all_points.shape[0]} patch vectors (perplexity={args.perplexity}) …")
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
        f"[{dataset_prefix}] ViT Patch Feature Clustering\n"
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
