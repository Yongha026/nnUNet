"""
check_clustering_gbc_kmeans.py
==============================
Standalone comparison tool between:
1. Trained Granular Ball Clustering (GBC) [archs_GBC.py]
2. 0-iteration K-Means (nearest centroid with fixed initial centers; isolates anisotropic sigma effect)
3. Converged K-Means (iteratively fitted centroids via torch_kmeans)

This script is located in `./feature_cluster_exp/` and is designed to run on the remote server
with either GPU or CPU.

Usage:
    # From workspace root:
    python feature_cluster_exp/check_clustering_gbc_kmeans.py \
        --features ./toy_tensors/features.npy \
        --centers ./toy_tensors/centers.npy \
        --sigma ./toy_tensors/sigma.npy \
        --output_dir ./feature_cluster_exp/clustering_results \
        --tau 1.0 \
        --device auto

    # Or from within feature_cluster_exp/:
    cd feature_cluster_exp
    python check_clustering_gbc_kmeans.py \
        --features ../toy_tensors/features.npy \
        --centers ../toy_tensors/centers.npy \
        --sigma ../toy_tensors/sigma.npy \
        --output_dir ./clustering_results \
        --tau 1.0 \
        --device auto
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
# plt.rcParams['text.usetex'] = True

# Scikit-learn & Scipy imports
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment

# Robust multi-path import for torch_kmeans
try:
    from torch_kmeans import KMeans
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    found = False
    for base in [script_dir, parent_dir]:
        cand = os.path.join(base, "torch_kmeans", "src")
        if os.path.exists(cand):
            sys.path.append(cand)
            found = True
            break
    try:
        from torch_kmeans import KMeans
    except ImportError:
        raise ImportError(
            "Could not import 'torch_kmeans'. Please install it via 'pip install torch-kmeans' "
            "or ensure the 'torch_kmeans' directory is accessible."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check and compare clustering of deep features with GBC and torch_kmeans."
    )
    parser.add_argument(
        "--features",
        type=str,
        default="./toy_tensors/features.npy",
        help="Path to features npy file (shape: (64, 48, 48) or (N, D)).",
    )
    parser.add_argument(
        "--centers",
        type=str,
        default="./toy_tensors/centers.npy",
        help="Path to initial centers npy file (shape: (16, 64)).",
    )
    parser.add_argument(
        "--sigma",
        type=str,
        default="./toy_tensors/sigma.npy",
        help="Path to sigma/radii npy file (shape: (16, 64) or (1, 1, 16, 64)).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./feature_cluster_exp/clustering_results",
        help="Directory to save comparison figures and metrics.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Temperature parameter for GBC softmax membership (default: 1.0).",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum iterations for K-Means fitting (default: 100).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance for K-Means (default: 1e-4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run computations on (default: auto).",
    )
    return parser.parse_args()


def resolve_file_path(path_str):
    """
    Attempts to resolve path relative to current working directory,
    and falls back to resolving relative to parent directory if running from subfolder.
    """
    if os.path.exists(path_str):
        return path_str
    # Fallback checking parent dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    cand1 = os.path.join(script_dir, path_str)
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(parent_dir, path_str)
    if os.path.exists(cand2):
        return cand2
    return path_str


def load_and_preprocess_inputs(features_path, centers_path, sigma_path, device):
    """
    Loads npy files and formats shapes:
      - features: (N, D) where N = H * W, D = 64
      - centers: (K, D) where K = 16, D = 64
      - sigma:   (K, D) where K = 16, D = 64
    """
    features_resolved = resolve_file_path(features_path)
    centers_resolved = resolve_file_path(centers_path)
    sigma_resolved = resolve_file_path(sigma_path)

    if not os.path.exists(features_resolved):
        raise FileNotFoundError(f"Features file not found at: {features_path} (checked: {features_resolved})")
    if not os.path.exists(centers_resolved):
        raise FileNotFoundError(f"Centers file not found at: {centers_path} (checked: {centers_resolved})")
    if not os.path.exists(sigma_resolved):
        raise FileNotFoundError(f"Sigma file not found at: {sigma_path} (checked: {sigma_resolved})")

    feat_np = np.load(features_resolved)
    centers_np = np.load(centers_resolved)
    sigma_np = np.load(sigma_resolved)

    # Clean shapes
    centers_np = np.squeeze(centers_np)
    sigma_np = np.squeeze(sigma_np)

    # Check features shape
    H, W = None, None
    if feat_np.ndim == 4:  # (1, C, H, W)
        feat_np = feat_np.squeeze(0)
    if feat_np.ndim == 3:
        if feat_np.shape[0] == centers_np.shape[1]:  # (C, H, W) e.g., (64, 48, 48)
            C, H, W = feat_np.shape
            feat_flat = np.transpose(feat_np, (1, 2, 0)).reshape(-1, C)
        elif feat_np.shape[2] == centers_np.shape[1]:  # (H, W, C)
            H, W, C = feat_np.shape
            feat_flat = feat_np.reshape(-1, C)
        else:
            raise ValueError(f"Unable to infer feature dimensions from shape {feat_np.shape}")
    elif feat_np.ndim == 2:  # (N, D)
        feat_flat = feat_np
        side = int(np.sqrt(feat_flat.shape[0]))
        if side * side == feat_flat.shape[0]:
            H, W = side, side
        else:
            H, W = feat_flat.shape[0], 1
    else:
        raise ValueError(f"Unsupported features shape: {feat_np.shape}")

    # Convert to PyTorch tensors
    feat_t = torch.from_numpy(feat_flat).float().to(device)
    centers_t = torch.from_numpy(centers_np).float().to(device)
    sigma_t = torch.from_numpy(sigma_np).float().to(device)

    K, D = centers_t.shape
    assert sigma_t.shape == (K, D), f"Sigma shape {sigma_t.shape} does not match centers shape {(K, D)}"
    assert feat_t.shape[1] == D, f"Feature dimension {feat_t.shape[1]} does not match center dimension {D}"

    print(f"Loaded inputs successfully:")
    print(f"  - Features: {feat_flat.shape[0]} points of dim {D} (Spatial grid: {H}x{W})")
    print(f"  - Centers:  {K} initial clusters of dim {D}")
    print(f"  - Sigma:    {K} anisotropic radii sets of dim {D}")
    print(f"  - Device:   {device}")

    return feat_t, centers_t, sigma_t, H, W, K, D


def compute_gbc_clustering(z, centers, sigma, tau=1.0):
    """
    Computes GBC normalized Mahalanobis-like distances and fuzzy assignments:
        dif_scaled = (z - c) / sigma
        dist2 = sum(dif_scaled ** 2, dim=-1)
        att = softmax(-dist2 / tau)
        label = argmin(dist2)
    """
    N, D = z.shape
    K = centers.shape[0]

    dif = z.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, D)
    dif_scaled = dif / (sigma.unsqueeze(0) + 1e-6)  # (N, K, D)
    dist2 = (dif_scaled ** 2).sum(dim=-1)  # (N, K)

    # Soft membership
    att = F.softmax(-dist2 / max(1e-6, tau), dim=-1)  # (N, K)

    # Hard cluster assignment
    labels = torch.argmin(dist2, dim=-1)  # (N,)

    # Metrics
    min_dist2, _ = torch.min(dist2, dim=-1)
    inside_mask = min_dist2 <= 1.0  # Point lies inside the Granular Ball boundary
    confidence, _ = torch.max(att, dim=-1)
    entropy = -torch.sum(att * torch.log(att + 1e-12), dim=-1)

    return {
        "labels": labels.cpu().numpy(),
        "att": att.cpu().numpy(),
        "dist2": dist2.cpu().numpy(),
        "min_dist2": min_dist2.cpu().numpy(),
        "inside_mask": inside_mask.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
        "entropy": entropy.cpu().numpy(),
    }


def compute_kmeans_step0(z, centers):
    """
    0-iteration K-Means: computes nearest center under isotropic Euclidean distance
    without updating centers. Isolates the anisotropic sigma effect.
    """
    dif = z.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, D)
    dist2_euc = (dif ** 2).sum(dim=-1)  # (N, K)
    labels = torch.argmin(dist2_euc, dim=-1)  # (N,)
    min_dist2_euc, _ = torch.min(dist2_euc, dim=-1)

    return {
        "labels": labels.cpu().numpy(),
        "dist2_euc": dist2_euc.cpu().numpy(),
        "inertia": min_dist2_euc.sum().item(),
    }


def compute_kmeans_converged(z, centers, max_iter=100, tol=1e-4, seed=42):
    """
    Converged K-Means via torch_kmeans: iteratively updates centroids
    starting from initial centers.
    """
    K, D = centers.shape
    x_batched = z.unsqueeze(0)  # (1, N, D)
    centers_batched = centers.unsqueeze(0)  # (1, K, D)

    km = KMeans(
        n_clusters=K,
        num_init=1,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
        seed=seed,
    )
    result = km(x_batched, centers=centers_batched)

    conv_labels = result.labels.squeeze(0).cpu().numpy()
    conv_centers = result.centers.squeeze(0).cpu().numpy()
    inertia = result.inertia.item() if hasattr(result, "inertia") else None

    if inertia is None:
        dif = z.unsqueeze(1) - torch.from_numpy(conv_centers).to(z.device).unsqueeze(0)
        dist2 = (dif ** 2).sum(dim=-1)
        min_dist2, _ = torch.min(dist2, dim=-1)
        inertia = min_dist2.sum().item()

    return {
        "labels": conv_labels,
        "centers": conv_centers,
        "inertia": inertia,
    }


def align_labels_hungarian(ref_labels, target_labels, K):
    """
    Solves linear sum assignment (Hungarian matching) to align target_labels
    to ref_labels based on maximum overlap in contingency matrix.
    """
    cm = confusion_matrix(ref_labels, target_labels, labels=np.arange(K))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned_labels = np.array([mapping.get(lbl, lbl) for lbl in target_labels])
    return aligned_labels, mapping, cm


def generate_color_palette(K=16):
    """
    Generates a consistent, distinct colormap for K clusters.
    """
    base_cmap = plt.get_cmap("tab20" if K <= 20 else "gist_ncar")
    colors = [base_cmap(i / (K - 1) if K > 1 else 0) for i in range(K)]
    return ListedColormap(colors)


def plot_and_save_all(
    feat_np,
    H,
    W,
    K,
    gbc_res,
    km_init_res,
    km_conv_res,
    km_init_aligned,
    km_conv_aligned,
    centers_init,
    centers_conv,
    metrics,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cmap = generate_color_palette(K)

    # 1. Master Multi-panel Figure
    fig = plt.figure(figsize=(24, 16), dpi=150)
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.3, hspace=0.35)

    # (0, 0) GBC Hard Partition
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(gbc_res["labels"].reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax0.set_title(f"1. GBC Hard Clustering\n(Anisotropic $\sigma$, Tau={metrics['tau']})", fontsize=11, fontweight="bold")
    ax0.axis("off")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # (0, 1) 0-iter K-Means (Aligned)
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(km_init_aligned.reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax1.set_title("2. 0-iter K-Means (Aligned)\n(Isotropic Euclidean, Fixed Centers)", fontsize=11, fontweight="bold")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # (0, 2) Converged K-Means (Aligned)
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(km_conv_aligned.reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax2.set_title("3. Converged K-Means (Aligned)\n(Fitted Centroids via torch_kmeans)", fontsize=11, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # (0, 3) Disagreement Map (GBC vs Converged KM)
    disagreement = (gbc_res["labels"] != km_conv_aligned).astype(np.float32).reshape(H, W)
    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(disagreement, cmap="coolwarm", vmin=0, vmax=1)
    ax3.set_title(f"4. Disagreement Map\n(GBC vs Conv KM: {metrics['disagreement_pct']:.1f}% mismatch)", fontsize=11, fontweight="bold")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # (1, 0) Granular Ball Coverage Map (Inside vs Outside)
    coverage = gbc_res["inside_mask"].astype(np.float32).reshape(H, W)
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(coverage, cmap="Greens", vmin=0, vmax=1)
    ax4.set_title(f"5. GBC Ball Coverage ($d^2 \\leq 1.0$)\n({metrics['ball_coverage_pct']:.1f}% pixels inside)",
                  fontsize=11, fontweight="bold")
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # (1, 1) GBC Membership Confidence Map
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(gbc_res["confidence"].reshape(H, W), cmap="viridis", vmin=0, vmax=1)
    ax5.set_title("6. GBC Membership Confidence\n($\\max_k \\alpha_{i,k}$)", fontsize=11, fontweight="bold")
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # (1, 2) GBC Membership Entropy Map
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(gbc_res["entropy"].reshape(H, W), cmap="magma")
    ax6.set_title("7. GBC Fuzzy Assignment Entropy\n(High = High Boundary Uncertainty)", fontsize=11, fontweight="bold")
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # (1, 3) Centroid Drift Bar Chart
    ax7 = fig.add_subplot(gs[1, 3])
    shifts = metrics["centroid_shifts_per_cluster"]
    cluster_idx = np.arange(K)
    bars = ax7.bar(cluster_idx, shifts, color="royalblue", edgecolor="black", alpha=0.8)
    ax7.axhline(np.mean(shifts), color="red", linestyle="--", label=f"Mean: {np.mean(shifts):.3f}")
    ax7.set_title("8. Centroid Drift per Cluster\n($\\|\\mathbf{c}_k^* - \\mathbf{c}_k\\|_2$)", fontsize=11, fontweight="bold")
    ax7.set_xlabel("Cluster ID", fontsize=10)
    ax7.set_ylabel("L2 Shift", fontsize=10)
    ax7.set_xticks(cluster_idx)
    ax7.legend(loc="upper right", fontsize=9)
    ax7.grid(axis="y", linestyle=":", alpha=0.6)

    # PCA 2D Projection
    pca = PCA(n_components=2, random_state=42)
    feat_pca = pca.fit_transform(feat_np)
    centers_init_pca = pca.transform(centers_init)
    centers_conv_pca = pca.transform(centers_conv)

    # (2, 0) PCA Feature Space Scatter
    ax8 = fig.add_subplot(gs[2, 0])
    scatter8 = ax8.scatter(
        feat_pca[:, 0], feat_pca[:, 1], c=gbc_res["labels"], cmap=cmap, s=8, alpha=0.5, vmin=0, vmax=K - 1
    )
    ax8.scatter(
        centers_init_pca[:, 0], centers_init_pca[:, 1], c="black", marker="x", s=60, linewidths=2, label="Init GBC Centers"
    )
    ax8.scatter(
        centers_conv_pca[:, 0], centers_conv_pca[:, 1], c="red", marker="+", s=70, linewidths=2, label="Conv KM Centers"
    )
    for k in range(K):
        ax8.annotate(
            "",
            xy=(centers_conv_pca[k, 0], centers_conv_pca[k, 1]),
            xytext=(centers_init_pca[k, 0], centers_init_pca[k, 1]),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2, alpha=0.8),
        )
    ax8.set_title(f"9. PCA Projection (Var Expl: {pca.explained_variance_ratio_.sum()*100:.1f}%)\nPoints Colored by GBC", fontsize=11, fontweight="bold")
    ax8.legend(loc="upper right", fontsize=8)
    ax8.grid(True, linestyle=":", alpha=0.5)

    # t-SNE 2D Projection
    all_points = np.vstack([feat_np, centers_init, centers_conv])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_proj = tsne.fit_transform(all_points)
    feat_tsne = tsne_proj[: len(feat_np)]
    c_init_tsne = tsne_proj[len(feat_np) : len(feat_np) + K]
    c_conv_tsne = tsne_proj[len(feat_np) + K :]

    # (2, 1) t-SNE Scatter
    ax9 = fig.add_subplot(gs[2, 1])
    ax9.scatter(
        feat_tsne[:, 0], feat_tsne[:, 1], c=gbc_res["labels"], cmap=cmap, s=8, alpha=0.5, vmin=0, vmax=K - 1
    )
    ax9.scatter(c_init_tsne[:, 0], c_init_tsne[:, 1], c="black", marker="x", s=60, linewidths=2, label="Init Centers")
    ax9.scatter(c_conv_tsne[:, 0], c_conv_tsne[:, 1], c="red", marker="+", s=70, linewidths=2, label="Conv Centers")
    ax9.set_title("10. t-SNE Manifold Projection\n(Points Colored by GBC)", fontsize=11, fontweight="bold")
    ax9.legend(loc="upper right", fontsize=8)
    ax9.grid(True, linestyle=":", alpha=0.5)

    # (2, 2) Cluster Size Distribution Comparison
    ax10 = fig.add_subplot(gs[2, 2])
    counts_gbc = np.bincount(gbc_res["labels"], minlength=K)
    counts_km_init = np.bincount(km_init_aligned, minlength=K)
    counts_km_conv = np.bincount(km_conv_aligned, minlength=K)

    width = 0.26
    ax10.bar(cluster_idx - width, counts_gbc, width=width, label="GBC", color="teal", alpha=0.8)
    ax10.bar(cluster_idx, counts_km_init, width=width, label="0-iter KM", color="orange", alpha=0.8)
    ax10.bar(cluster_idx + width, counts_km_conv, width=width, label="Conv KM", color="purple", alpha=0.8)
    ax10.set_title("11. Cluster Size Distribution\n(Pixel Counts per Cluster)", fontsize=11, fontweight="bold")
    ax10.set_xlabel("Cluster ID (Aligned)", fontsize=10)
    ax10.set_ylabel("Number of Pixels", fontsize=10)
    ax10.set_xticks(cluster_idx)
    ax10.legend(loc="upper right", fontsize=8)
    ax10.grid(axis="y", linestyle=":", alpha=0.6)

    # (2, 3) Metrics Summary Table Card
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.axis("off")
    table_data = [
        ["Metric", "Value"],
        ["GBC vs 0-iter KM ARI", f"{metrics['ari_gbc_vs_km_init']:.4f}"],
        ["GBC vs 0-iter KM NMI", f"{metrics['nmi_gbc_vs_km_init']:.4f}"],
        ["0-iter vs Conv KM ARI", f"{metrics['ari_km_init_vs_km_conv']:.4f}"],
        ["0-iter vs Conv KM NMI", f"{metrics['nmi_km_init_vs_km_conv']:.4f}"],
        ["GBC vs Conv KM ARI", f"{metrics['ari_gbc_vs_km_conv']:.4f}"],
        ["GBC vs Conv KM NMI", f"{metrics['nmi_gbc_vs_km_conv']:.4f}"],
        ["Pixel Agreement (GBC/Conv)", f"{metrics['agreement_pct_gbc_vs_km_conv']:.1f}%"],
        ["Pixels inside GB (d² ≤ 1)", f"{metrics['ball_coverage_pct']:.1f}%"],
        ["Mean Centroid Drift", f"{metrics['mean_centroid_shift']:.4f}"],
        ["Max Centroid Drift", f"{metrics['max_centroid_shift']:.4f}"],
        ["GBC Inertia (WCSS)", f"{metrics['inertia_gbc']:.2f}"],
        ["Conv KM Inertia (WCSS)", f"{metrics['inertia_km_conv']:.2f}"],
    ]
    t = ax11.table(cellText=table_data, loc="center", cellLoc="left", colWidths=[0.65, 0.35])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1.1, 1.4)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f8f9fa")
    ax11.set_title("Quantitative Summary", fontsize=11, fontweight="bold")

    # Save Master Overview
    overview_path = os.path.join(output_dir, "clustering_overview.png")
    fig.savefig(overview_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved master overview plot: {overview_path}")

    # 2. Save Individual High-Res Figures
    # 01 Spatial GBC
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(gbc_res["labels"].reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax.set_title("GBC Hard Cluster Partition")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "01_spatial_gbc.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 02 Spatial KM Init
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(km_init_aligned.reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax.set_title("0-iteration K-Means Partition (Aligned)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "02_spatial_kmeans_init.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 03 Spatial KM Conv
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(km_conv_aligned.reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax.set_title("Converged K-Means Partition (Aligned)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "03_spatial_kmeans_converged.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 04 Disagreement
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(disagreement, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_title("Disagreement Map (GBC vs Converged KM)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "04_spatial_disagreement.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 05 Coverage
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(coverage, cmap="Greens", vmin=0, vmax=1)
    ax.set_title("GBC Granular Ball Coverage ($d^2 \\leq 1.0$)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "05_gbc_ball_coverage.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 06 Confidence
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(gbc_res["confidence"].reshape(H, W), cmap="viridis", vmin=0, vmax=1)
    ax.set_title("GBC Membership Confidence ($\\max_k \\alpha_k$)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "06_gbc_confidence.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 07 Entropy
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(gbc_res["entropy"].reshape(H, W), cmap="magma")
    ax.set_title("GBC Membership Entropy")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "07_gbc_entropy.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 08 Centroid Drift
    fig_ind, ax = plt.subplots(figsize=(7, 4), dpi=200)
    ax.bar(cluster_idx, shifts, color="royalblue", edgecolor="black", alpha=0.8)
    ax.axhline(np.mean(shifts), color="red", linestyle="--", label=f"Mean: {np.mean(shifts):.3f}")
    ax.set_title("Centroid Drift per Cluster ($\\|\\mathbf{c}_k^* - \\mathbf{c}_k\\|_2$)")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("L2 Shift")
    ax.set_xticks(cluster_idx)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig_ind.savefig(os.path.join(fig_dir, "08_centroid_drift.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 09 PCA
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.scatter(feat_pca[:, 0], feat_pca[:, 1], c=gbc_res["labels"], cmap=cmap, s=8, alpha=0.5, vmin=0, vmax=K - 1)
    ax.scatter(centers_init_pca[:, 0], centers_init_pca[:, 1], c="black", marker="x", s=60, linewidths=2, label="Init Centers")
    ax.scatter(centers_conv_pca[:, 0], centers_conv_pca[:, 1], c="red", marker="+", s=70, linewidths=2, label="Conv Centers")
    for k in range(K):
        ax.annotate(
            "",
            xy=(centers_conv_pca[k, 0], centers_conv_pca[k, 1]),
            xytext=(centers_init_pca[k, 0], centers_init_pca[k, 1]),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2, alpha=0.8),
        )
    ax.set_title("PCA Feature Space Projection")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig_ind.savefig(os.path.join(fig_dir, "09_pca_projection.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 10 t-SNE
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.scatter(feat_tsne[:, 0], feat_tsne[:, 1], c=gbc_res["labels"], cmap=cmap, s=8, alpha=0.5, vmin=0, vmax=K - 1)
    ax.scatter(c_init_tsne[:, 0], c_init_tsne[:, 1], c="black", marker="x", s=60, linewidths=2, label="Init Centers")
    ax.scatter(c_conv_tsne[:, 0], c_conv_tsne[:, 1], c="red", marker="+", s=70, linewidths=2, label="Conv Centers")
    ax.set_title("t-SNE Manifold Projection")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig_ind.savefig(os.path.join(fig_dir, "10_tsne_projection.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 11 Cluster Sizes
    fig_ind, ax = plt.subplots(figsize=(8, 4), dpi=200)
    ax.bar(cluster_idx - width, counts_gbc, width=width, label="GBC", color="teal", alpha=0.8)
    ax.bar(cluster_idx, counts_km_init, width=width, label="0-iter KM", color="orange", alpha=0.8)
    ax.bar(cluster_idx + width, counts_km_conv, width=width, label="Conv KM", color="purple", alpha=0.8)
    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Cluster ID (Aligned)")
    ax.set_ylabel("Number of Pixels")
    ax.set_xticks(cluster_idx)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig_ind.savefig(os.path.join(fig_dir, "11_cluster_sizes.png"), bbox_inches="tight")
    plt.close(fig_ind)

    print(f"Saved 11 individual figures in: {fig_dir}")


def print_rich_terminal_table(metrics):
    border = "=" * 65
    subborder = "-" * 65
    print(border)
    print("           GBC vs. K-MEANS CLUSTERING COMPARISON REPORT")
    print(border)
    print(f"  Configuration:")
    print(f"    - Temperature (Tau):              {metrics['tau']}")
    print(f"    - Number of Clusters (K):         {metrics['num_clusters']}")
    print(f"    - Feature Dimensions (N, D):      ({metrics['num_samples']}, {metrics['feature_dim']})")
    print(f"    - Spatial Grid Size:              {metrics['spatial_shape']}")
    print(subborder)
    print(f"  Pairwise Agreement Metrics:")
    print(f"    - GBC vs. 0-iter K-Means (Anisotropic effect):")
    print(f"        * Adjusted Rand Index (ARI):  {metrics['ari_gbc_vs_km_init']:.4f}")
    print(f"        * Normalized Mutual Info (NMI):{metrics['nmi_gbc_vs_km_init']:.4f}")
    print(f"        * Pixel Agreement:            {metrics['agreement_pct_gbc_vs_km_init']:.2f}%")
    print(f"    - 0-iter KM vs. Converged KM (Centroid shift effect):")
    print(f"        * Adjusted Rand Index (ARI):  {metrics['ari_km_init_vs_km_conv']:.4f}")
    print(f"        * Normalized Mutual Info (NMI):{metrics['nmi_km_init_vs_km_conv']:.4f}")
    print(f"        * Pixel Agreement:            {metrics['agreement_pct_km_init_vs_km_conv']:.2f}%")
    print(f"    - GBC vs. Converged KM (Overall comparison):")
    print(f"        * Adjusted Rand Index (ARI):  {metrics['ari_gbc_vs_km_conv']:.4f}")
    print(f"        * Normalized Mutual Info (NMI):{metrics['nmi_gbc_vs_km_conv']:.4f}")
    print(f"        * Pixel Agreement:            {metrics['agreement_pct_gbc_vs_km_conv']:.2f}%")
    print(f"        * Pixel Disagreement:         {metrics['disagreement_pct']:.2f}%")
    print(subborder)
    print(f"  Granular Ball Geometry & Coverage:")
    print(f"    - Pixels inside Granular Balls (d² <= 1.0): {metrics['ball_coverage_pct']:.2f}%")
    print(f"    - Pixels outside all balls:                 {100.0 - metrics['ball_coverage_pct']:.2f}%")
    print(f"    - Mean GBC Membership Confidence:          {metrics['mean_confidence']:.4f}")
    print(f"    - Mean GBC Assignment Entropy:             {metrics['mean_entropy']:.4f}")
    print(subborder)
    print(f"  Centroid Optimization & Drift:")
    print(f"    - Mean Centroid Shift (L2):       {metrics['mean_centroid_shift']:.4f}")
    print(f"    - Max Centroid Shift (L2):        {metrics['max_centroid_shift']:.4f} (Cluster {metrics['max_shift_cluster']})")
    print(f"    - GBC Euclidean Inertia (WCSS):   {metrics['inertia_gbc']:.2f}")
    print(f"    - 0-iter KM Inertia (WCSS):       {metrics['inertia_km_init']:.2f}")
    print(f"    - Converged KM Inertia (WCSS):    {metrics['inertia_km_conv']:.2f}")
    print(border)


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 1. Load inputs
    feat_t, centers_t, sigma_t, H, W, K, D = load_and_preprocess_inputs(
        args.features, args.centers, args.sigma, device
    )

    feat_np = feat_t.cpu().numpy()
    centers_init_np = centers_t.cpu().numpy()
    sigma_np = sigma_t.cpu().numpy()

    # 2. GBC Clustering
    print("\n[1/3] Computing GBC clustering and soft membership...")
    gbc_res = compute_gbc_clustering(feat_t, centers_t, sigma_t, tau=args.tau)

    # 3. 0-iteration K-Means
    print("[2/3] Computing 0-iteration K-Means (fixed initial centers)...")
    km_init_res = compute_kmeans_step0(feat_t, centers_t)

    # 4. Converged K-Means via torch_kmeans
    print("[3/3] Fitting Converged K-Means via torch_kmeans...")
    km_conv_res = compute_kmeans_converged(
        feat_t, centers_t, max_iter=args.max_iter, tol=args.tol, seed=args.seed
    )

    centers_conv_np = km_conv_res["centers"]

    # 5. Hungarian matching
    km_init_aligned, mapping_init, cm_init = align_labels_hungarian(
        gbc_res["labels"], km_init_res["labels"], K
    )
    km_conv_aligned, mapping_conv, cm_conv = align_labels_hungarian(
        gbc_res["labels"], km_conv_res["labels"], K
    )

    # 6. Metrics Calculation
    ari_gbc_init = float(adjusted_rand_score(gbc_res["labels"], km_init_res["labels"]))
    nmi_gbc_init = float(normalized_mutual_info_score(gbc_res["labels"], km_init_res["labels"]))
    agree_gbc_init = float((gbc_res["labels"] == km_init_aligned).mean() * 100)

    ari_init_conv = float(adjusted_rand_score(km_init_res["labels"], km_conv_res["labels"]))
    nmi_init_conv = float(normalized_mutual_info_score(km_init_res["labels"], km_conv_res["labels"]))
    agree_init_conv = float((km_init_aligned == km_conv_aligned).mean() * 100)

    ari_gbc_conv = float(adjusted_rand_score(gbc_res["labels"], km_conv_res["labels"]))
    nmi_gbc_conv = float(normalized_mutual_info_score(gbc_res["labels"], km_conv_res["labels"]))
    agree_gbc_conv = float((gbc_res["labels"] == km_conv_aligned).mean() * 100)
    disagree_gbc_conv = 100.0 - agree_gbc_conv

    # Centroid shifts
    shifts = np.linalg.norm(centers_conv_np - centers_init_np, axis=-1).tolist()
    mean_shift = float(np.mean(shifts))
    max_shift = float(np.max(shifts))
    max_shift_cluster = int(np.argmax(shifts))

    # GBC inertia (WCSS with respect to initial centers under Euclidean metric)
    dif_gbc = feat_np - centers_init_np[gbc_res["labels"]]
    inertia_gbc = float(np.sum(dif_gbc ** 2))

    ball_coverage_pct = float(gbc_res["inside_mask"].mean() * 100)

    metrics = {
        "tau": args.tau,
        "num_clusters": K,
        "feature_dim": D,
        "num_samples": feat_np.shape[0],
        "spatial_shape": f"{H}x{W}",
        "ari_gbc_vs_km_init": ari_gbc_init,
        "nmi_gbc_vs_km_init": nmi_gbc_init,
        "agreement_pct_gbc_vs_km_init": agree_gbc_init,
        "ari_km_init_vs_km_conv": ari_init_conv,
        "nmi_km_init_vs_km_conv": nmi_init_conv,
        "agreement_pct_km_init_vs_km_conv": agree_init_conv,
        "ari_gbc_vs_km_conv": ari_gbc_conv,
        "nmi_gbc_vs_km_conv": nmi_gbc_conv,
        "agreement_pct_gbc_vs_km_conv": agree_gbc_conv,
        "disagreement_pct": disagree_gbc_conv,
        "ball_coverage_pct": ball_coverage_pct,
        "mean_confidence": float(np.mean(gbc_res["confidence"])),
        "mean_entropy": float(np.mean(gbc_res["entropy"])),
        "centroid_shifts_per_cluster": shifts,
        "mean_centroid_shift": mean_shift,
        "max_centroid_shift": max_shift,
        "max_shift_cluster": max_shift_cluster,
        "inertia_gbc": inertia_gbc,
        "inertia_km_init": float(km_init_res["inertia"]),
        "inertia_km_conv": float(km_conv_res["inertia"]),
        "hungarian_mapping_init_to_gbc": {int(k): int(v) for k, v in mapping_init.items()},
        "hungarian_mapping_conv_to_gbc": {int(k): int(v) for k, v in mapping_conv.items()},
    }

    # 7. Print Terminal Table
    print_rich_terminal_table(metrics)

    # 8. Save Metrics JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics summary JSON: {json_path}")

    # 9. Save Compressed Array Archive
    npz_path = os.path.join(args.output_dir, "assignments.npz")
    np.savez_compressed(
        npz_path,
        features=feat_np,
        gbc_labels=gbc_res["labels"].reshape(H, W),
        gbc_soft_membership=gbc_res["att"],
        gbc_coverage_mask=gbc_res["inside_mask"].reshape(H, W),
        gbc_confidence=gbc_res["confidence"].reshape(H, W),
        gbc_entropy=gbc_res["entropy"].reshape(H, W),
        km_init_labels_raw=km_init_res["labels"].reshape(H, W),
        km_init_labels_aligned=km_init_aligned.reshape(H, W),
        km_conv_labels_raw=km_conv_res["labels"].reshape(H, W),
        km_conv_labels_aligned=km_conv_aligned.reshape(H, W),
        centers_init=centers_init_np,
        centers_conv=centers_conv_np,
        sigma=sigma_np,
    )
    print(f"Saved array archive: {npz_path}")

    # 10. Generate and Save Visual Plots
    print("\nGenerating and saving visualization figures...")
    plot_and_save_all(
        feat_np=feat_np,
        H=H,
        W=W,
        K=K,
        gbc_res=gbc_res,
        km_init_res=km_init_res,
        km_conv_res=km_conv_res,
        km_init_aligned=km_init_aligned,
        km_conv_aligned=km_conv_aligned,
        centers_init=centers_init_np,
        centers_conv=centers_conv_np,
        metrics=metrics,
        output_dir=args.output_dir,
    )
    print("\nClustering comparison complete! All outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
