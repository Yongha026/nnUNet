"""
hdbscan_cluster.py — Cluster UNet deep features via HDBSCAN and visualise via 2D projection.

Performs density-based clustering (HDBSCAN) directly in 64D feature space,
evaluates the natural number of clusters, and visualizes the results side-by-side
(Ground Truth vs HDBSCAN Clusters) using t-SNE or PCA 2D embedding.

Usage
-----
python feature_cluster_exp/hdbscan_cluster.py /path/to/images /path/to/checkpoint.pth [options]

Options
-------
--datas             Number of images to sample (default: 1024)
--samples_per_class Max points per class for sampling (default: 2000)
--min_cluster_size  Minimum size of clusters for HDBSCAN (default: 50)
--min_samples       HDBSCAN min_samples for conservative core points (default: None)
--projection        2D projection method for plotting: 'tsne' or 'pca' (default: 'tsne')
--perplexity        t-SNE perplexity if projection is tsne (default: 30)
--pupil_only        Binary: Pupil vs Else (default)
--all_classes       Full 4-class: Background / Sclera / Iris / Pupil
--output            Output path (default: auto-generated in hdbscan_results/)
--seed              Random seed (default: 42)
"""

import os
import sys
import glob
import argparse
import random
import re
import numpy as np
import cv2
import PIL.Image
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

try:
    import hdbscan
except ImportError:
    from sklearn.cluster import HDBSCAN as hdbscan_fallback
    hdbscan = None

# ─── path setup ──────────────────────────────────────────────────────────────
exp_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(exp_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


# ─── dataset ─────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.table = float(255) * (np.linspace(0, 1, 256) ** 0.8)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)
        img_gamma = cv2.LUT(
            img_resized.astype(np.uint8), self.table.astype(np.uint8)
        )
        img_clahe = self.clahe.apply(img_gamma)
        pil_img = PIL.Image.fromarray(img_clahe)

        msk_path = (
            self.image_paths[idx].replace("images", "labels").replace("png", "npy")
        )
        msk = np.load(msk_path).astype(np.uint8)
        msk_resized = cv2.resize(msk, (48, 48), interpolation=cv2.INTER_NEAREST)
        return self.transform(pil_img), msk_resized


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HDBSCAN clustering and visualization on UNet deep features"
    )
    parser.add_argument("IMG_PATH", type=str, help="Path to image folder")
    parser.add_argument("MODEL_PATH", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--datas", default=1024, type=int, help="Number of images to sample"
    )
    parser.add_argument(
        "--samples_per_class",
        default=2000,
        type=int,
        help="Max points per class for sampling",
    )
    # HDBSCAN parameters
    parser.add_argument(
        "--min_cluster_size",
        default=50,
        type=int,
        help="HDBSCAN minimum cluster size (default: 50)",
    )
    parser.add_argument(
        "--min_samples",
        default=None,
        type=int,
        help="HDBSCAN min_samples for core distance (default: None)",
    )
    # Projection for 2D visualization
    parser.add_argument(
        "--projection",
        choices=["tsne", "pca"],
        default="tsne",
        help="2D reduction method to visualize HDBSCAN clusters (default: tsne)",
    )
    parser.add_argument(
        "--perplexity", default=30, type=int, help="t-SNE perplexity if used"
    )
    # Class toggle
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
        help="Output path",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--untrained", "-u",
        action="store_true",
        default=False,
        help="Inference with Random initialized model"
    )
    parser.add_argument(
        "--dec",
        action="store_true",
        default=False,
        help="Extract decoder features instead of encoder"
    )
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model_path_adgbc = args.MODEL_PATH
    ADGBC_match = "nnUNetTrainerGBC_" in args.MODEL_PATH
    DGBC_match = "nnUNetTrainerDGBC_" in args.MODEL_PATH
    KMeans_match = "nnUNetTrainerKMeans_" in args.MODEL_PATH
    Next_match = any(x in args.MODEL_PATH for x in ["nnUNetTrainer_Next_", "Next", "next", "UNext", "unext"])

    use_diag_cov = False
    if Next_match:
        model_prefix = "UNeXt"
    elif ADGBC_match:
        model_prefix = "ADGBC"
        use_diag_cov = True
    elif DGBC_match:
        model_prefix = "DGBC"
        use_diag_cov = False
    elif KMeans_match:
        model_prefix = "KMeans"
    else:
        model_prefix = "Model"

    untrained_prefix = "UNTRAINED_" if args.untrained else ""
    model_prefix = untrained_prefix + model_prefix

    EDS_match = "Openedsdata2019" in args.IMG_PATH
    Pupil_match = "jw_" in args.IMG_PATH
    Swir_match = "nnunetv2_swir" in args.IMG_PATH

    if not EDS_match and not Pupil_match and not Swir_match:
        parser.error("IMG_PATH must contain either 'OpenEDS2019', 'jw_' or 'nnunetv2_swir'.")

    if EDS_match: dataset_prefix = "OpenEDS2019"
    elif Pupil_match: dataset_prefix = "PupilLabs"
    else: dataset_prefix = "Swirski"

    if args.all_classes:
        args.pupil_only = False

    if not Next_match:
        match = re.search(r"_S_(2|4|8|16|32|64)(?=__|$)", str(model_path_adgbc))
        gbc_num_balls = int(match.group(1)) if match else 32
    else:
        gbc_num_balls = None

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ─── model loading ───────────────────────────────────────────────────────────
    def initialize_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
            if getattr(m, 'weight', None) is not None:
                torch.nn.init.constant_(m.weight, 1.0)
            if getattr(m, 'bias', None) is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    from nnunetv2.training.nnUNetTrainer.ADGBC_encoder import GBC_S_EncDec
    from nnunetv2.training.nnUNetTrainer.archs_K_means_UNet import Kmeans_encoder
    from nnunetv2.training.nnUNetTrainer.archs_unext import UNext

    if Next_match:
        model = UNext(num_classes=4, input_channels=1, deep_supervision=False, enc_dec=True).to(device)
    elif ADGBC_match or DGBC_match:
        model = GBC_S_EncDec(
            num_classes=4, input_channels=1, deep_supervision=False, gbc_num_balls=gbc_num_balls, use_diag_cov=use_diag_cov
        ).to(device)
    elif KMeans_match:
        model = Kmeans_encoder(
            num_classes=4, input_channels=1, deep_supervision=False, gbc_num_balls=gbc_num_balls
        ).to(device)
    else:
        raise Exception(f"Unknown model architecture from: {args.MODEL_PATH}")

    checkpoint = torch.load(model_path_adgbc, map_location=device, weights_only=False)
    state_dict = checkpoint["network_weights"] if isinstance(checkpoint, dict) and "network_weights" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    if args.untrained:
        print("[INFO] Initializing model with random weights...")
        model.apply(initialize_weights)

    model.eval().to(device)

    # ── image sampling ────────────────────────────────────────────────────
    image_path = os.path.join(args.IMG_PATH, "*.png")
    full_images = glob.glob(image_path)
    try:
        rand_images = random.sample(full_images, args.datas)
    except ValueError:
        rand_images = full_images
    print(f"[INFO] Dataset: {dataset_prefix} — Using {len(rand_images)} images from {args.IMG_PATH}")

    dataset = ImageDataset(rand_images)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)

    NUM_CLASSES = 4
    SAMPLES_PER_CLASS = args.samples_per_class
    feats_by_class = {c: [] for c in range(NUM_CLASSES)}

    # ── feature extraction ────────────────────────────────────────────────
    target_layer = "Decoder" if args.dec else "Encoder"
    print(f"[INFO] Extracting {target_layer} processed features …")

    with torch.no_grad():
        for batch_imgs, batch_masks in tqdm(dataloader, desc="Extracting"):
            batch_imgs = batch_imgs.to(device)
            if args.dec:
                _, _, enc = model(batch_imgs)
            else:
                _, enc, _ = model(batch_imgs)
            B, C, H, W = enc.shape

            enc_pixels = enc.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()

            if batch_masks.shape[1] != H or batch_masks.shape[2] != W:
                masks_t = batch_masks.unsqueeze(1).float()
                masks_resized = (
                    torch.nn.functional.interpolate(masks_t, size=(H, W), mode="nearest")
                    .squeeze(1)
                    .byte()
                    .numpy()
                )
            else:
                masks_resized = batch_masks.numpy()
            labels_pixels = masks_resized.reshape(-1)

            for c in range(NUM_CLASSES):
                current_len = sum(len(x) for x in feats_by_class[c])
                if current_len < SAMPLES_PER_CLASS:
                    c_mask = labels_pixels == c
                    c_feats = enc_pixels[c_mask]
                    if len(c_feats) > 0:
                        feats_by_class[c].append(c_feats)

    # ── balance & merge ───────────────────────────────────────────────────
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

    X = np.vstack(selected_feats)    # [N, 64]
    y = np.concatenate(selected_labels)

    # ── L2 Normalization & HDBSCAN Clustering ────────────────────────────
    # High-dimensional feature cosine space mapping via L2 norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    print(f"\n[INFO] Running HDBSCAN directly on {X_norm.shape[0]} points in 64D space ...")
    print(f"       min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}")

    if hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric="euclidean",  # L2 정규화 상태에서는 Euclidean 거리가 Cosine 거리와 단조 비례함
            core_dist_n_jobs=-1,
        )
    else:
        clusterer = hdbscan_fallback(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            metric="euclidean",
            n_jobs=-1,
        )

    cluster_labels = clusterer.fit_predict(X_norm)

    # ── Quantitative Analysis ─────────────────────────────────────────────
    unique_clusters = sorted(list(set(cluster_labels) - {-1}))
    num_natural_clusters = len(unique_clusters)
    noise_count = int(np.sum(cluster_labels == -1))
    noise_ratio = noise_count / len(cluster_labels)

    print("\n" + "=" * 60)
    print(f"  [RESULT] 자연 군집 개수 (Natural Clusters): {num_natural_clusters}")
    print(f"  [RESULT] 노이즈 포인트: {noise_count}/{len(cluster_labels)} ({noise_ratio:.2%})")
    print("=" * 60)

    # Cluster vs Ground Truth Class Confusion Matrix
    class_names = ["Background", "Sclera", "Iris", "Pupil"]
    print("\n[INFO] 군집별 Ground Truth 클래스 분포:")
    header = f"{'Cluster ID':<12} | " + " | ".join([f"{name:<10}" for name in class_names]) + " | Total"
    print(header)
    print("-" * len(header))

    for k in unique_clusters + ([-1] if noise_count > 0 else []):
        mask_k = cluster_labels == k
        counts = [int(np.sum((y == c) & mask_k)) for c in range(NUM_CLASSES)]
        total_k = sum(counts)
        k_str = f"Cluster {k}" if k != -1 else "Noise (-1)"
        counts_str = " | ".join([f"{cnt:<10}" for cnt in counts])
        print(f"{k_str:<12} | {counts_str} | {total_k}")

    # Cluster agreement with GT (excluding noise for ARI/NMI calculation)
    non_noise = cluster_labels != -1
    if np.sum(non_noise) > 0 and num_natural_clusters > 1:
        ari = adjusted_rand_score(y[non_noise], cluster_labels[non_noise])
        nmi = normalized_mutual_info_score(y[non_noise], cluster_labels[non_noise])
        print(f"\n[METRICS (non-noise)] Adjusted Rand Index (ARI): {ari:.4f} | NMI: {nmi:.4f}")

    # ── 2D Projection for Visualization ──────────────────────────────────
    if args.projection == "pca":
        print(f"\n[INFO] Running PCA for 2D visualization ...")
        reducer = PCA(n_components=2, random_state=args.seed)
        X_2d = reducer.fit_transform(X_norm)
        proj_label = f"PCA (Var: {reducer.explained_variance_ratio_.sum()*100:.1f}%)"
    else:
        print(f"\n[INFO] Running t-SNE for 2D visualization (perplexity={args.perplexity}) ...")
        reducer = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            n_jobs=-1,
            random_state=args.seed,
        )
        X_2d = reducer.fit_transform(X_norm)
        proj_label = f"t-SNE (Perplexity: {args.perplexity})"

    # ── Plotting: Side-by-Side (Ground Truth vs HDBSCAN) ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Subplot 1: Ground Truth
    tab10 = plt.cm.get_cmap("tab10")
    if args.pupil_only:
        y_gt = np.where(y == 3, 1, 0)
        gt_cmap = ListedColormap(tab10.colors[:2])
        gt_ticks = [0, 1]
        gt_labels = ["Else", "Pupil"]
    else:
        y_gt = y
        gt_cmap = ListedColormap(tab10.colors[:4])
        gt_ticks = [0, 1, 2, 3]
        gt_labels = class_names

    sc1 = axes[0].scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=y_gt, cmap=gt_cmap,
        alpha=0.4, s=6, zorder=1
    )
    axes[0].set_title(f"Ground Truth Semantic Classes ({proj_label})", fontsize=12)
    cb1 = fig.colorbar(sc1, ax=axes[0], ticks=gt_ticks)
    cb1.ax.set_yticklabels(gt_labels)

    # Subplot 2: HDBSCAN Natural Clusters
    # 군집 라벨 컬러 매핑: 노이즈(-1)는 회색(#bbbbbb), 유효 군집은 고유 색상 부여
    color_palette = plt.cm.get_cmap("Spectral", max(num_natural_clusters, 1))
    point_colors = np.zeros((len(cluster_labels), 4))

    for idx, c_id in enumerate(cluster_labels):
        if c_id == -1:
            point_colors[idx] = [0.75, 0.75, 0.75, 0.25]  # 회색 투명 노이즈
        else:
            color = color_palette(c_id / max(num_natural_clusters - 1, 1))
            point_colors[idx] = [color[0], color[1], color[2], 0.7]

    # 노이즈를 먼저 그리고, 유효 군집을 위에 표시
    noise_mask = cluster_labels == -1
    axes[1].scatter(
        X_2d[noise_mask, 0], X_2d[noise_mask, 1],
        c=point_colors[noise_mask], s=4, label="Noise (-1)", zorder=1
    )
    for c_id in unique_clusters:
        c_mask = cluster_labels == c_id
        axes[1].scatter(
            X_2d[c_mask, 0], X_2d[c_mask, 1],
            c=point_colors[c_mask], s=7, label=f"Cluster {c_id} (n={np.sum(c_mask)})", zorder=2
        )

    axes[1].set_title(
        f"HDBSCAN Predicted Clusters (k={num_natural_clusters}, Noise: {noise_ratio:.1%})",
        fontsize=12,
    )
    if num_natural_clusters <= 12:
        axes[1].legend(markerscale=3, loc="upper right", fontsize=8)

    fig.suptitle(
        f"[{dataset_prefix}] {model_prefix} {target_layer} Features (N={X.shape[0]}, Dim=64)\n"
        f"HDBSCAN Natural Cluster Extraction vs Ground Truth",
        fontsize=14,
    )
    plt.tight_layout()

    # ── Save Results ──────────────────────────────────────────────────────
    results_dir = os.path.join(exp_dir, "hdbscan_results")
    os.makedirs(results_dir, exist_ok=True)

    if args.output:
        save_path = args.output
    else:
        save_path = os.path.join(
            results_dir,
            f"{model_prefix}_{dataset_prefix}_{target_layer}_HDBSCAN_min{args.min_cluster_size}_{args.projection}.png",
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[INFO] Figure saved successfully → {save_path}")