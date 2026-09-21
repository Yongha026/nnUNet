"""
tsne_GBC.py — Visualise GBC anisotropic balls with UNet deep encoder features
via t-SNE.

Produces a side-by-side plot:
  Left  : Pre-GBC features (t3)  + Granular Ball ellipses / centers
  Right : Post-GBC features (t3_gbc) + Granular Ball ellipses / centers

Ball centers for Pre-GBC come directly from model.gbc.centers.
Ball centers for Post-GBC are membership-weighted centroids of the
post-GBC pixel features.

Usage
-----
python pupil_src/tests/tsne_GBC.py  /path/to/images  [options]

Options
-------
--datas             Number of images to sample (default: 1024)
--samples_per_class Max points per class for t-SNE (default: 2000)
--percentile        Percentile boundary for ball ellipses (default: 80.0)
--pupil_only        Binary: Pupil vs Else (default)
--all_classes       Full 4-class: Background / Sclera / Iris / Pupil
--output            Output path (default: auto-generated in tsne_results/)
--perplexity        t-SNE perplexity (default: 30)
--seed              Random seed (default: 42)
"""

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
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader
import random

from sklearn.manifold import TSNE

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
        # Image
        img_path = self.image_paths[idx]
        # if img_path.endswith("_0000.png"):
        #     img_path = img_path.replace("_0000", "")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)
        img_gamma = cv2.LUT(
            img_resized.astype(np.uint8), self.table.astype(np.uint8)
        )
        img_clahe = self.clahe.apply(img_gamma)
        pil_img = PIL.Image.fromarray(img_clahe)

        # Mask
        msk_path = (
            self.image_paths[idx].replace("images", "labels").replace("png", "npy")
        )
        msk = np.load(msk_path).astype(np.uint8)
        msk_resized = cv2.resize(msk, (48, 48), interpolation=cv2.INTER_NEAREST)
        return self.transform(pil_img), msk_resized


# ─── ellipse drawing helpers ─────────────────────────────────────────────────
def _percentile_ellipse(
    points_2d: np.ndarray,
    center_2d: np.ndarray,
    percentile: float,
    color,
    alpha: float = 0.15,
    linewidth: float = 1.5,
):
    """
    Return a matplotlib Ellipse patch fitted to *points_2d* around
    *center_2d*, scaled so that *percentile* % of points fall inside.

    Returns None if there are < 5 points (degenerate covariance).
    """
    if len(points_2d) < 5:
        return None

    # 2D sample covariance of assigned points
    diff = points_2d - center_2d
    cov = np.cov(diff, rowvar=False)  # (2, 2)

    # eigen-decomposition → orientation
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Mahalanobis distance of each point
    cov_inv = np.linalg.inv(cov + np.eye(2) * 1e-8)
    mahal = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
    radius = np.percentile(mahal, percentile)

    # Ellipse axes = radius * sqrt(eigenvalue) in each direction
    width = 2 * radius * np.sqrt(max(eigvals[0], 1e-8))
    height = 2 * radius * np.sqrt(max(eigvals[1], 1e-8))

    # Rotation angle (degrees)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    return Ellipse(
        xy=center_2d,
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        linewidth=linewidth,
        linestyle="--",
    )


def _plot_panel(
    ax,
    X_embedded,       # (N, 2)  — pixel embeddings
    centers_embedded,  # (K, 2)  — ball center embeddings
    labels,            # (N,)    — class labels for each pixel
    assignments,       # (N,)    — hard ball assignment per pixel
    num_balls,
    percentile,
    cmap,
    ticks,
    tick_labels,
    title,
):
    """Draw one panel (pre- or post-GBC) of the side-by-side figure."""
    # Scatter pixel features coloured by class
    scatter = ax.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=labels,
        cmap=cmap,
        alpha=0.35,
        s=3,
        zorder=1,
    )

    # Distinct colours for the 32 ball ellipses / markers
    ball_cmap = plt.cm.get_cmap("tab20", num_balls)

    for k in range(num_balls):
        mask_k = assignments == k
        color_k = ball_cmap(k)
        center_k = centers_embedded[k]

        if mask_k.sum() >= 5:
            pts_k = X_embedded[mask_k]
            ell = _percentile_ellipse(
                pts_k, center_k, percentile, color=color_k, alpha=0.15
            )
            if ell is not None:
                ax.add_patch(ell)

        # Ball center marker
        ax.scatter(
            center_k[0],
            center_k[1],
            marker="^",
            s=60,
            c=[color_k],
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        ax.annotate(
            str(k),
            xy=(center_k[0], center_k[1]),
            fontsize=5,
            fontweight="bold",
            ha="center",
            va="bottom",
            color="black",
            zorder=6,
        )

    cbar = plt.colorbar(scatter, ax=ax, ticks=ticks, shrink=0.8)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("Class ID")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise GBC anisotropic balls with UNet deep encoder "
        "features using t-SNE (side-by-side Pre-GBC vs Post-GBC)"
    )
    parser.add_argument("IMG_PATH", type=str, help="Path to image folder")
    parser.add_argument(
        "--datas", default=1024, type=int, help="Number of images to sample"
    )
    parser.add_argument("MODEL_PATH", type=str, help="Path to adgbc checkpoint")
    parser.add_argument(
        "--samples_per_class",
        default=2000,
        type=int,
        help="Max points per class for t-SNE",
    )
    parser.add_argument(
        "--percentile",
        default=80.0,
        type=float,
        help="Percentile boundary for ball ellipses",
    )
    # Class toggle — default is --pupil_only
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
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model_path_adgbc = args.MODEL_PATH

    # ── dataset validation ────────────────────────────────────────────────
    # IMG_PATH must contain exactly one of "OpenEDS2019" or "jw_"
    EDS_match = "OpenEDS2019" in args.IMG_PATH
    Pupil_match = "jw_" in args.IMG_PATH

    if EDS_match and Pupil_match:
        parser.error(
            "IMG_PATH contains both 'OpenEDS2019' and 'jw_'. "
            "Please specify a path belonging to only one dataset."
        )
    if not EDS_match and not Pupil_match:
        parser.error(
            "IMG_PATH must contain either 'OpenEDS2019' or 'jw_' "
            "to identify the dataset."
        )

    if not os.path.exists(model_path_adgbc):
        parser.error(f"ADGBC checkpoint file not found at: {model_path_adgbc}")

    dataset_prefix = "OpenEDS2019" if EDS_match else "PupilLabs"

    # If --all_classes is set, override pupil_only
    if args.all_classes:
        args.pupil_only = False

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ─── model loading ───────────────────────────────────────────────────────────
    from nnunetv2.training.nnUNetTrainer.ADGBC_encoder import GBC_S_EncDec

    try:
        model = GBC_S_EncDec(
            num_classes=4, input_channels=1, deep_supervision=False
        ).to(device)
        checkpoint = torch.load(
            model_path_adgbc, map_location=device, weights_only=False
        )
        state_dict = (
            checkpoint["network_weights"]
            if (isinstance(checkpoint, dict) and "network_weights" in checkpoint)
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"Error loading adgbc: {e}")
        raise e

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
    dataloader = DataLoader(
        dataset, batch_size=16, num_workers=4, pin_memory=True
    )

    NUM_CLASSES = 4
    SAMPLES_PER_CLASS = args.samples_per_class
    K = model.gbc.num_balls  # 32

    # Accumulators — one per class, for pre- and post-GBC features
    pre_feats_by_class = {c: [] for c in range(NUM_CLASSES)}
    post_feats_by_class = {c: [] for c in range(NUM_CLASSES)}

    # For post-GBC weighted centroids
    post_weighted_sum = torch.zeros(K, model.gbc.proj_dim, device="cpu")
    att_sum = torch.zeros(K, device="cpu")

    # ── feature extraction ────────────────────────────────────────────────
    print("[INFO] Extracting features …")
    with torch.no_grad():
        for batch_imgs, batch_masks in tqdm(dataloader, desc="Extracting"):
            batch_imgs = batch_imgs.to(device)

            _, t3, t3_gbc, att_t3, _ = model(batch_imgs, return_details=True)
            # t3, t3_gbc: [B, C, H, W]   att_t3: [B, H*W, K]

            B, C, H, W = t3.shape

            # Flatten to [B*H*W, C]
            pre_pixels = t3.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            post_pixels = t3_gbc.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            labels_pixels = batch_masks.reshape(-1).numpy()
            att_flat = att_t3.reshape(-1, K).cpu()  # [B*H*W, K]

            # Accumulate for post-GBC weighted centroid
            post_flat_t = t3_gbc.permute(0, 2, 3, 1).reshape(-1, C).cpu()  # [N, C]
            att_sum += att_flat.sum(dim=0)  # [K]
            post_weighted_sum += torch.einsum(
                "nk,nd->kd", att_flat, post_flat_t
            )  # [K, C]

            # Per-class accumulation with cap
            for c in range(NUM_CLASSES):
                current_len_pre = sum(len(x) for x in pre_feats_by_class[c])
                if current_len_pre < SAMPLES_PER_CLASS:
                    c_mask = labels_pixels == c
                    c_pre = pre_pixels[c_mask]
                    c_post = post_pixels[c_mask]
                    if len(c_pre) > 0:
                        pre_feats_by_class[c].append(c_pre)
                        post_feats_by_class[c].append(c_post)

    # ── balance & merge ───────────────────────────────────────────────────
    selected_pre, selected_post, selected_labels = [], [], []

    for c in range(NUM_CLASSES):
        c_pre_all = np.vstack(pre_feats_by_class[c])
        c_post_all = np.vstack(post_feats_by_class[c])

        if len(c_pre_all) > SAMPLES_PER_CLASS:
            idx = np.random.choice(len(c_pre_all), SAMPLES_PER_CLASS, replace=False)
            c_pre_all = c_pre_all[idx]
            c_post_all = c_post_all[idx]

        selected_pre.append(c_pre_all)
        selected_post.append(c_post_all)
        selected_labels.append(np.full(len(c_pre_all), c))

    X_pre = np.vstack(selected_pre)    # [N_total, C]
    X_post = np.vstack(selected_post)  # [N_total, C]
    y = np.concatenate(selected_labels)

    # ── ball centers ──────────────────────────────────────────────────────
    centers_pre = model.gbc.centers.detach().cpu().numpy()  # [K, C]

    # Post-GBC: membership-weighted centroids
    att_sum_safe = att_sum.clamp(min=1e-6)
    centers_post = (post_weighted_sum / att_sum_safe.unsqueeze(1)).numpy()  # [K, C]

    # ── t-SNE fitting (independent for pre and post) ──────────────────────
    print(f"[INFO] Running t-SNE (perplexity={args.perplexity}) …")
    N = X_pre.shape[0]

    # Pre-GBC: append centers, fit, split
    X_pre_with_centers = np.vstack([X_pre, centers_pre])  # [N+K, C]
    tsne_pre = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        n_jobs=-1,
        random_state=args.seed,
    )
    emb_pre = tsne_pre.fit_transform(X_pre_with_centers)
    X_pre_2d = emb_pre[:N]
    centers_pre_2d = emb_pre[N:]

    # Post-GBC: append centers, fit, split
    X_post_with_centers = np.vstack([X_post, centers_post])
    tsne_post = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        n_jobs=-1,
        random_state=args.seed,
    )
    emb_post = tsne_post.fit_transform(X_post_with_centers)
    X_post_2d = emb_post[:N]
    centers_post_2d = emb_post[N:]

    # ── hard ball assignment for ellipses ─────────────────────────────────
    # Re-compute assignments for balanced/selected pixels using pre-GBC
    # distances (same logic as GranularBall.forward)
    sigma_np = (
        F.softplus(model.gbc.log_sigma).detach().cpu().numpy() + 1e-6
    )  # [K, C]
    dif = X_pre[:, None, :] - centers_pre[None, :, :]  # [N, K, C]
    dif_scaled = dif / sigma_np[None, :, :]
    dist2 = (dif_scaled**2).sum(axis=-1)  # [N, K]
    assignments_pre = dist2.argmin(axis=1)  # [N]

    # Post-GBC assignments via same distance in post space
    dif_post = X_post[:, None, :] - centers_post[None, :, :]
    dif_post_scaled = dif_post / sigma_np[None, :, :]
    dist2_post = (dif_post_scaled**2).sum(axis=-1)
    assignments_post = dist2_post.argmin(axis=1)

    # ── class label remapping ─────────────────────────────────────────────
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

    # ── plot ──────────────────────────────────────────────────────────────
    fig, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(22, 9))

    _plot_panel(
        ax=ax_pre,
        X_embedded=X_pre_2d,
        centers_embedded=centers_pre_2d,
        labels=y_plot,
        assignments=assignments_pre,
        num_balls=K,
        percentile=args.percentile,
        cmap=plot_cmap,
        ticks=ticks,
        tick_labels=tick_labels,
        title=f"Pre-GBC (t3) — {args.percentile:.0f}th-pctl Ellipses",
    )

    _plot_panel(
        ax=ax_post,
        X_embedded=X_post_2d,
        centers_embedded=centers_post_2d,
        labels=y_plot,
        assignments=assignments_post,
        num_balls=K,
        percentile=args.percentile,
        cmap=plot_cmap,
        ticks=ticks,
        tick_labels=tick_labels,
        title=f"Post-GBC (t3_gbc) — {args.percentile:.0f}th-pctl Ellipses",
    )

    fig.suptitle(
        f"[{dataset_prefix}] GBC Anisotropic Ball Visualisation  •  K={K} balls  •  "
        f"perplexity={args.perplexity}  •  {len(rand_images)} images",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ── save ──────────────────────────────────────────────────────────────
    results_dir = os.path.join(exp_dir, "tsne_results")
    os.makedirs(results_dir, exist_ok=True)

    if args.output:
        save_path = args.output
    else:
        save_path = os.path.join(
            results_dir,
            f"{dataset_prefix}_GBC_anisotropic_balls_tsne_{suffix}.png",
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {save_path}")
