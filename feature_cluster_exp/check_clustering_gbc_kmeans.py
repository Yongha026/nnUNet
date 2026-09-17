"""
check_clustering_gbc_kmeans.py
==============================
Standalone comparison tool between:
1. Trained Granular Ball Clustering (GBC) [archs_GBC.py]
2. 0-iteration K-Means (nearest centroid with fixed initial centers; isolates anisotropic sigma effect)
3. Converged K-Means (iteratively fitted centroids via torch_kmeans)

Supports multiple normalization modes:
- none:   Raw unnormalized features and centers
- zscore: Channel-wise standardization ((x - mean) / std) aligning positive features with zero-mean centers
- unit:   L2 unit-sphere normalization (directional / cosine clustering, resolving cluster collapse)
- all:    Runs all three modes sequentially and generates side-by-side cross-mode comparisons

Usage:
    # Run all normalization modes and generate comparative synthesis:
    python feature_cluster_exp/check_clustering_gbc_kmeans.py --normalize all

    # Run specific normalization mode:
    python feature_cluster_exp/check_clustering_gbc_kmeans.py --normalize unit
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
        "--normalize",
        type=str,
        default="all",
        choices=["none", "zscore", "unit", "all"],
        help="Feature normalization mode: none, zscore, unit, or all (default: all).",
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
    if os.path.exists(path_str):
        return path_str
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

    centers_np = np.squeeze(centers_np)
    sigma_np = np.squeeze(sigma_np)

    H, W = None, None
    if feat_np.ndim == 4:
        feat_np = feat_np.squeeze(0)
    if feat_np.ndim == 3:
        if feat_np.shape[0] == centers_np.shape[1]:  # (C, H, W)
            C, H, W = feat_np.shape
            feat_flat = np.transpose(feat_np, (1, 2, 0)).reshape(-1, C)
        elif feat_np.shape[2] == centers_np.shape[1]:  # (H, W, C)
            H, W, C = feat_np.shape
            feat_flat = feat_np.reshape(-1, C)
        else:
            raise ValueError(f"Unable to infer feature dimensions from shape {feat_np.shape}")
    elif feat_np.ndim == 2:
        feat_flat = feat_np
        side = int(np.sqrt(feat_flat.shape[0]))
        if side * side == feat_flat.shape[0]:
            H, W = side, side
        else:
            H, W = feat_flat.shape[0], 1
    else:
        raise ValueError(f"Unsupported features shape: {feat_np.shape}")

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


def apply_normalization(feat_t, centers_t, sigma_t, mode="none"):
    """
    Applies the chosen normalization mode across features, centers, and sigma:
    - none:   Returns original raw tensors.
    - zscore: Standardizes features along each channel (mean=0, std=1), aligns centers
              to the same standardized coordinates, and scales sigma accordingly.
    - unit:   L2-normalizes each feature vector and center vector onto the unit hypersphere
              (directional cosine clustering, eliminating magnitude collapse).
    """
    if mode == "none":
        return feat_t, centers_t, sigma_t

    elif mode == "zscore":
        mean = feat_t.mean(dim=0, keepdim=True)
        std = feat_t.std(dim=0, keepdim=True) + 1e-6
        feat_norm = (feat_t - mean) / std
        centers_norm = (centers_t - mean) / std
        sigma_norm = sigma_t / std
        return feat_norm, centers_norm, sigma_norm

    elif mode == "unit":
        feat_norm = F.normalize(feat_t, p=2, dim=-1)
        centers_norm = F.normalize(centers_t, p=2, dim=-1)
        c_norms = torch.norm(centers_t, p=2, dim=-1, keepdim=True) + 1e-6
        sigma_norm = sigma_t / c_norms
        return feat_norm, centers_norm, sigma_norm

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def compute_gbc_clustering(z, centers, sigma, tau=1.0):
    N, D = z.shape
    K = centers.shape[0]

    dif = z.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, D)
    dif_scaled = dif / (sigma.unsqueeze(0) + 1e-6)  # (N, K, D)
    dist2 = (dif_scaled ** 2).sum(dim=-1)  # (N, K)

    att = F.softmax(-dist2 / max(1e-6, tau), dim=-1)  # (N, K)
    labels = torch.argmin(dist2, dim=-1)  # (N,)

    min_dist2, _ = torch.min(dist2, dim=-1)
    confidence, _ = torch.max(att, dim=-1)
    entropy = -torch.sum(att * torch.log(att + 1e-12), dim=-1)

    # Multi-radius coverage masks
    inside_mask_r1 = min_dist2 <= 1.0   # 1-sigma boundary
    inside_mask_r2 = min_dist2 <= 4.0   # 2-sigma boundary
    inside_mask_r3 = min_dist2 <= 9.0   # 3-sigma boundary
    median_d2 = torch.median(min_dist2).item()
    inside_mask_med = min_dist2 <= median_d2

    return {
        "labels": labels.cpu().numpy(),
        "att": att.cpu().numpy(),
        "dist2": dist2.cpu().numpy(),
        "min_dist2": min_dist2.cpu().numpy(),
        "inside_mask": inside_mask_r1.cpu().numpy(),
        "inside_mask_r2": inside_mask_r2.cpu().numpy(),
        "inside_mask_r3": inside_mask_r3.cpu().numpy(),
        "inside_mask_med": inside_mask_med.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
        "entropy": entropy.cpu().numpy(),
        "median_dist2": median_d2,
    }


def compute_kmeans_step0(z, centers):
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
    cm = confusion_matrix(ref_labels, target_labels, labels=np.arange(K))
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned_labels = np.array([mapping.get(lbl, lbl) for lbl in target_labels])
    return aligned_labels, mapping, cm


def generate_color_palette(K=16):
    base_cmap = plt.get_cmap("tab20" if K <= 20 else "gist_ncar")
    colors = [base_cmap(i / (K - 1) if K > 1 else 0) for i in range(K)]
    return ListedColormap(colors)


def compute_joint_pca(feat_np, centers_init, centers_conv):
    """
    Fits PCA on the joint matrix of features, initial centers, and converged centers.
    This ensures that initial centers are not artificially squashed into a clump.
    """
    all_points = np.vstack([feat_np, centers_init, centers_conv])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_points)

    feat_pca = pca.transform(feat_np)
    c_init_pca = pca.transform(centers_init)
    c_conv_pca = pca.transform(centers_conv)
    return feat_pca, c_init_pca, c_conv_pca, pca


def compute_joint_tsne(feat_np, centers_init, centers_conv, seed=42):
    K = len(centers_init)
    all_points = np.vstack([feat_np, centers_init, centers_conv])
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    tsne_proj = tsne.fit_transform(all_points)

    feat_tsne = tsne_proj[: len(feat_np)]
    c_init_tsne = tsne_proj[len(feat_np) : len(feat_np) + K]
    c_conv_tsne = tsne_proj[len(feat_np) + K :]
    return feat_tsne, c_init_tsne, c_conv_tsne


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
    mode_name="none",
):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cmap = generate_color_palette(K)

    # 1. Joint Dimensionality Reduction
    feat_pca, centers_init_pca, centers_conv_pca, pca = compute_joint_pca(
        feat_np, centers_init, centers_conv
    )
    feat_tsne, c_init_tsne, c_conv_tsne = compute_joint_tsne(
        feat_np, centers_init, centers_conv, seed=metrics.get("seed", 42)
    )

    # 2. Master Multi-panel Figure
    fig = plt.figure(figsize=(24, 16), dpi=150)
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.3, hspace=0.35)

    mode_title_suffix = f" [Mode: {mode_name.upper()}]"

    # (0, 0) GBC Hard Partition
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(gbc_res["labels"].reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax0.set_title(f"1. GBC Hard Clustering\n(Tau={metrics['tau']}){mode_title_suffix}", fontsize=11, fontweight="bold")
    ax0.axis("off")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # (0, 1) 0-iter K-Means (Aligned)
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(km_init_aligned.reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax1.set_title(f"2. 0-iter K-Means (Aligned)\n(Fixed Centers){mode_title_suffix}", fontsize=11, fontweight="bold")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # (0, 2) Converged K-Means (Aligned)
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(km_conv_aligned.reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax2.set_title(f"3. Converged K-Means (Aligned)\n(torch_kmeans fit){mode_title_suffix}", fontsize=11, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # (0, 3) Disagreement Map (GBC vs Converged KM)
    disagreement = (gbc_res["labels"] != km_conv_aligned).astype(np.float32).reshape(H, W)
    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(disagreement, cmap="coolwarm", vmin=0, vmax=1)
    ax3.set_title(f"4. Disagreement Map\n(Mismatch: {metrics['disagreement_pct']:.1f}%)", fontsize=11, fontweight="bold")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # (1, 0) Granular Ball Coverage Map
    coverage = gbc_res["inside_mask"].astype(np.float32).reshape(H, W)
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(coverage, cmap="Greens", vmin=0, vmax=1)
    ax4.set_title(f"5. GBC Ball Coverage (d² <= 1.0)\n({metrics['ball_coverage_pct']:.1f}% inside, 2σ: {metrics['coverage_pct_r2']:.1f}%)", fontsize=11, fontweight="bold")
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # (1, 1) GBC Membership Confidence Map
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(gbc_res["confidence"].reshape(H, W), cmap="viridis", vmin=0, vmax=1)
    ax5.set_title(f"6. GBC Membership Confidence\n(Mean: {metrics['mean_confidence']:.3f})", fontsize=11, fontweight="bold")
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # (1, 2) GBC Membership Entropy Map
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(gbc_res["entropy"].reshape(H, W), cmap="magma")
    ax6.set_title(f"7. GBC Fuzzy Assignment Entropy\n(Mean: {metrics['mean_entropy']:.3f})", fontsize=11, fontweight="bold")
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # (1, 3) Centroid Drift Bar Chart
    ax7 = fig.add_subplot(gs[1, 3])
    shifts = metrics["centroid_shifts_per_cluster"]
    cluster_idx = np.arange(K)
    ax7.bar(cluster_idx, shifts, color="royalblue", edgecolor="black", alpha=0.8)
    ax7.axhline(np.mean(shifts), color="red", linestyle="--", label=f"Mean: {np.mean(shifts):.3f}")
    ax7.set_title(f"8. Centroid Drift per Cluster\n(Max: {metrics['max_centroid_shift']:.3f})", fontsize=11, fontweight="bold")
    ax7.set_xlabel("Cluster ID", fontsize=10)
    ax7.set_ylabel("L2 Shift", fontsize=10)
    ax7.set_xticks(cluster_idx)
    ax7.legend(loc="upper right", fontsize=9)
    ax7.grid(axis="y", linestyle=":", alpha=0.6)

    # (2, 0) Joint PCA Feature Space Scatter
    ax8 = fig.add_subplot(gs[2, 0])
    ax8.scatter(
        feat_pca[:, 0], feat_pca[:, 1], c=gbc_res["labels"], cmap=cmap, s=8, alpha=0.5, vmin=0, vmax=K - 1
    )
    ax8.scatter(
        centers_init_pca[:, 0], centers_init_pca[:, 1], c="black", marker="x", s=60, linewidths=2, label="Init Centers"
    )
    ax8.scatter(
        centers_conv_pca[:, 0], centers_conv_pca[:, 1], c="red", marker="+", s=70, linewidths=2, label="Conv Centers"
    )
    for k in range(K):
        ax8.annotate(
            "",
            xy=(centers_conv_pca[k, 0], centers_conv_pca[k, 1]),
            xytext=(centers_init_pca[k, 0], centers_init_pca[k, 1]),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2, alpha=0.8),
        )
    ax8.set_title(f"9. Joint PCA (Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)\n(Jointly Fitted on Features & Centers)", fontsize=11, fontweight="bold")
    ax8.legend(loc="upper right", fontsize=8)
    ax8.grid(True, linestyle=":", alpha=0.5)

    # (2, 1) Joint t-SNE Scatter
    ax9 = fig.add_subplot(gs[2, 1])
    ax9.scatter(
        feat_tsne[:, 0], feat_tsne[:, 1], c=gbc_res["labels"], cmap=cmap, s=8, alpha=0.5, vmin=0, vmax=K - 1
    )
    ax9.scatter(c_init_tsne[:, 0], c_init_tsne[:, 1], c="black", marker="x", s=60, linewidths=2, label="Init Centers")
    ax9.scatter(c_conv_tsne[:, 0], c_conv_tsne[:, 1], c="red", marker="+", s=70, linewidths=2, label="Conv Centers")
    ax9.set_title("10. Joint t-SNE Manifold\n(Points Colored by GBC)", fontsize=11, fontweight="bold")
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
        ["Normalization Mode", mode_name.upper()],
        ["GBC vs 0-iter KM ARI", f"{metrics['ari_gbc_vs_km_init']:.4f}"],
        ["GBC vs 0-iter KM NMI", f"{metrics['nmi_gbc_vs_km_init']:.4f}"],
        ["0-iter vs Conv KM ARI", f"{metrics['ari_km_init_vs_km_conv']:.4f}"],
        ["0-iter vs Conv KM NMI", f"{metrics['nmi_km_init_vs_km_conv']:.4f}"],
        ["GBC vs Conv KM ARI", f"{metrics['ari_gbc_vs_km_conv']:.4f}"],
        ["GBC vs Conv KM NMI", f"{metrics['nmi_gbc_vs_km_conv']:.4f}"],
        ["Pixel Agreement (GBC/Conv)", f"{metrics['agreement_pct_gbc_vs_km_conv']:.1f}%"],
        ["Ball Coverage (d² ≤ 1.0)", f"{metrics['ball_coverage_pct']:.1f}%"],
        ["Ball Coverage (d² ≤ 4.0)", f"{metrics['coverage_pct_r2']:.1f}%"],
        ["Median Mahalanobis d²", f"{metrics['median_dist2']:.2f}"],
        ["Mean Centroid Drift", f"{metrics['mean_centroid_shift']:.4f}"],
        ["Active GBC Clusters (>0 pts)", f"{np.count_nonzero(counts_gbc)} / {K}"],
        ["Active Conv KM Clusters", f"{np.count_nonzero(counts_km_conv)} / {K}"],
    ]
    t = ax11.table(cellText=table_data, loc="center", cellLoc="left", colWidths=[0.65, 0.35])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1.1, 1.35)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f8f9fa")
    ax11.set_title(f"Summary Table ({mode_name.upper()})", fontsize=11, fontweight="bold")

    # Save Master Overview
    overview_path = os.path.join(output_dir, "clustering_overview.png")
    fig.savefig(overview_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved master overview plot: {overview_path}")

    # 3. Save Individual High-Res Figures
    # 01 Spatial GBC
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(gbc_res["labels"].reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax.set_title(f"GBC Hard Cluster Partition ({mode_name.upper()})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "01_spatial_gbc.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 02 Spatial KM Init
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(km_init_aligned.reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax.set_title(f"0-iteration K-Means Partition (Aligned, {mode_name.upper()})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "02_spatial_kmeans_init.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 03 Spatial KM Conv
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(km_conv_aligned.reshape(H, W), cmap=cmap, vmin=0, vmax=K - 1)
    ax.set_title(f"Converged K-Means Partition (Aligned, {mode_name.upper()})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "03_spatial_kmeans_converged.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 04 Disagreement
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(disagreement, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_title(f"Disagreement Map ({mode_name.upper()})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "04_spatial_disagreement.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 05 Coverage
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(coverage, cmap="Greens", vmin=0, vmax=1)
    ax.set_title(f"GBC Granular Ball Coverage (d² <= 1.0, {mode_name.upper()})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "05_gbc_ball_coverage.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 06 Confidence
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(gbc_res["confidence"].reshape(H, W), cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"GBC Membership Confidence ({mode_name.upper()})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "06_gbc_confidence.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 07 Entropy
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(gbc_res["entropy"].reshape(H, W), cmap="magma")
    ax.set_title(f"GBC Membership Entropy ({mode_name.upper()})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_ind.savefig(os.path.join(fig_dir, "07_gbc_entropy.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 08 Centroid Drift
    fig_ind, ax = plt.subplots(figsize=(7, 4), dpi=200)
    ax.bar(cluster_idx, shifts, color="royalblue", edgecolor="black", alpha=0.8)
    ax.axhline(np.mean(shifts), color="red", linestyle="--", label=f"Mean: {np.mean(shifts):.3f}")
    ax.set_title(f"Centroid Drift per Cluster ({mode_name.upper()})")
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
    ax.set_title(f"Joint PCA Projection ({mode_name.upper()})")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig_ind.savefig(os.path.join(fig_dir, "09_pca_projection.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 10 t-SNE
    fig_ind, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.scatter(feat_tsne[:, 0], feat_tsne[:, 1], c=gbc_res["labels"], cmap=cmap, s=8, alpha=0.5, vmin=0, vmax=K - 1)
    ax.scatter(c_init_tsne[:, 0], c_init_tsne[:, 1], c="black", marker="x", s=60, linewidths=2, label="Init Centers")
    ax.scatter(c_conv_tsne[:, 0], c_conv_tsne[:, 1], c="red", marker="+", s=70, linewidths=2, label="Conv Centers")
    ax.set_title(f"Joint t-SNE Projection ({mode_name.upper()})")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig_ind.savefig(os.path.join(fig_dir, "10_tsne_projection.png"), bbox_inches="tight")
    plt.close(fig_ind)

    # 11 Cluster Sizes
    fig_ind, ax = plt.subplots(figsize=(8, 4), dpi=200)
    ax.bar(cluster_idx - width, counts_gbc, width=width, label="GBC", color="teal", alpha=0.8)
    ax.bar(cluster_idx, counts_km_init, width=width, label="0-iter KM", color="orange", alpha=0.8)
    ax.bar(cluster_idx + width, counts_km_conv, width=width, label="Conv KM", color="purple", alpha=0.8)
    ax.set_title(f"Cluster Size Distribution ({mode_name.upper()})")
    ax.set_xlabel("Cluster ID (Aligned)")
    ax.set_ylabel("Number of Pixels")
    ax.set_xticks(cluster_idx)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig_ind.savefig(os.path.join(fig_dir, "11_cluster_sizes.png"), bbox_inches="tight")
    plt.close(fig_ind)

    print(f"Saved 11 individual figures in: {fig_dir}")


def print_rich_terminal_table(metrics, mode_name="none"):
    border = "=" * 68
    subborder = "-" * 68
    print(border)
    print(f"       GBC vs. K-MEANS CLUSTERING REPORT [MODE: {mode_name.upper()}]")
    print(border)
    print(f"  Configuration:")
    print(f"    - Normalization Mode:             {mode_name.upper()}")
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
    print(f"    - Pixels inside 1-sigma ball (d² <= 1.0):   {metrics['ball_coverage_pct']:.2f}%")
    print(f"    - Pixels inside 2-sigma ball (d² <= 4.0):   {metrics['coverage_pct_r2']:.2f}%")
    print(f"    - Pixels inside 3-sigma ball (d² <= 9.0):   {metrics['coverage_pct_r3']:.2f}%")
    print(f"    - Median Mahalanobis d²:                    {metrics['median_dist2']:.3f}")
    print(f"    - Mean GBC Membership Confidence:          {metrics['mean_confidence']:.4f}")
    print(f"    - Mean GBC Assignment Entropy:             {metrics['mean_entropy']:.4f}")
    print(subborder)
    print(f"  Centroid Optimization & Cluster Balance:")
    print(f"    - Mean Centroid Shift (L2):       {metrics['mean_centroid_shift']:.4f}")
    print(f"    - Max Centroid Shift (L2):        {metrics['max_centroid_shift']:.4f} (Cluster {metrics['max_shift_cluster']})")
    print(f"    - Active GBC Clusters (>0 pts):   {metrics['active_clusters_gbc']} / {metrics['num_clusters']}")
    print(f"    - Active Conv KM Clusters:        {metrics['active_clusters_km_conv']} / {metrics['num_clusters']}")
    print(f"    - GBC Euclidean Inertia (WCSS):   {metrics['inertia_gbc']:.2f}")
    print(f"    - Converged KM Inertia (WCSS):    {metrics['inertia_km_conv']:.2f}")
    print(border)


def run_single_pipeline(feat_raw_t, centers_raw_t, sigma_raw_t, H, W, K, D, mode, args, output_dir):
    """
    Executes the full pipeline under a specific normalization mode.
    """
    print(f"\n=======================================================")
    print(f"  RUNNING PIPELINE WITH NORMALIZATION: {mode.upper()}")
    print(f"=======================================================")

    # 1. Apply Normalization
    feat_t, centers_t, sigma_t = apply_normalization(feat_raw_t, centers_raw_t, sigma_raw_t, mode=mode)

    feat_np = feat_t.cpu().numpy()
    centers_init_np = centers_t.cpu().numpy()
    sigma_np = sigma_t.cpu().numpy()

    # 2. GBC Clustering
    print("[1/3] Computing GBC clustering and soft membership...")
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

    # GBC inertia (WCSS with respect to initial centers)
    dif_gbc = feat_np - centers_init_np[gbc_res["labels"]]
    inertia_gbc = float(np.sum(dif_gbc ** 2))

    counts_gbc = np.bincount(gbc_res["labels"], minlength=K)
    counts_km_conv = np.bincount(km_conv_aligned, minlength=K)

    ball_coverage_pct = float(gbc_res["inside_mask"].mean() * 100)
    coverage_pct_r2 = float(gbc_res["inside_mask_r2"].mean() * 100)
    coverage_pct_r3 = float(gbc_res["inside_mask_r3"].mean() * 100)

    metrics = {
        "mode": mode,
        "tau": args.tau,
        "seed": args.seed,
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
        "coverage_pct_r2": coverage_pct_r2,
        "coverage_pct_r3": coverage_pct_r3,
        "median_dist2": float(gbc_res["median_dist2"]),
        "mean_confidence": float(np.mean(gbc_res["confidence"])),
        "mean_entropy": float(np.mean(gbc_res["entropy"])),
        "centroid_shifts_per_cluster": shifts,
        "mean_centroid_shift": mean_shift,
        "max_centroid_shift": max_shift,
        "max_shift_cluster": max_shift_cluster,
        "active_clusters_gbc": int(np.count_nonzero(counts_gbc)),
        "active_clusters_km_conv": int(np.count_nonzero(counts_km_conv)),
        "cluster_counts_gbc": counts_gbc.tolist(),
        "cluster_counts_km_conv": counts_km_conv.tolist(),
        "inertia_gbc": inertia_gbc,
        "inertia_km_init": float(km_init_res["inertia"]),
        "inertia_km_conv": float(km_conv_res["inertia"]),
        "hungarian_mapping_init_to_gbc": {int(k): int(v) for k, v in mapping_init.items()},
        "hungarian_mapping_conv_to_gbc": {int(k): int(v) for k, v in mapping_conv.items()},
    }

    # Print Terminal Table
    print_rich_terminal_table(metrics, mode_name=mode)

    # Save Metrics JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save Compressed Array Archive
    npz_path = os.path.join(output_dir, "assignments.npz")
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

    # Generate and Save Visual Plots
    print("Generating visualization figures...")
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
        output_dir=output_dir,
        mode_name=mode,
    )

    return {
        "metrics": metrics,
        "gbc_labels": gbc_res["labels"].reshape(H, W),
        "km_conv_labels": km_conv_aligned.reshape(H, W),
        "counts_gbc": counts_gbc,
        "counts_km_conv": counts_km_conv,
        "centers_init": centers_init_np,
        "centers_conv": centers_conv_np,
        "feat_np": feat_np,
    }


def generate_cross_mode_comparison(results_by_mode, H, W, K, master_output_dir):
    """
    Generates a high-level comparative plot and metrics table contrasting
    raw (none), zscore, and unit normalization modes.
    """
    modes = [m for m in ["none", "zscore", "unit"] if m in results_by_mode]
    if len(modes) < 2:
        return

    os.makedirs(master_output_dir, exist_ok=True)
    cmap = generate_color_palette(K)

    num_modes = len(modes)
    fig, axes = plt.subplots(4, num_modes, figsize=(6 * num_modes, 18), dpi=150)

    for col, m in enumerate(modes):
        res = results_by_mode[m]
        met = res["metrics"]
        col_title = f"Mode: {m.upper()}"

        # Row 0: GBC Spatial Map
        im0 = axes[0, col].imshow(res["gbc_labels"], cmap=cmap, vmin=0, vmax=K - 1)
        axes[0, col].set_title(f"{col_title}\nGBC Hard Map ({met['active_clusters_gbc']}/{K} active)", fontsize=11, fontweight="bold")
        axes[0, col].axis("off")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        # Row 1: Converged K-Means Map
        im1 = axes[1, col].imshow(res["km_conv_labels"], cmap=cmap, vmin=0, vmax=K - 1)
        axes[1, col].set_title(f"{col_title}\nConv K-Means ({met['active_clusters_km_conv']}/{K} active)", fontsize=11, fontweight="bold")
        axes[1, col].axis("off")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

        # Row 2: Cluster Size Histogram
        width = 0.35
        c_idx = np.arange(K)
        axes[2, col].bar(c_idx - width/2, res["counts_gbc"], width=width, label="GBC", color="teal", alpha=0.8)
        axes[2, col].bar(c_idx + width/2, res["counts_km_conv"], width=width, label="Conv KM", color="purple", alpha=0.8)
        axes[2, col].set_title(f"{col_title}\nCluster Balance (GBC vs Conv KM)", fontsize=11, fontweight="bold")
        axes[2, col].set_xlabel("Cluster ID", fontsize=10)
        axes[2, col].set_ylabel("Pixel Count", fontsize=10)
        axes[2, col].set_xticks(c_idx)
        axes[2, col].legend(fontsize=9)
        axes[2, col].grid(axis="y", linestyle=":", alpha=0.6)

        # Row 3: Joint PCA Projection
        feat_pca, c_init_pca, c_conv_pca, pca = compute_joint_pca(
            res["feat_np"], res["centers_init"], res["centers_conv"]
        )
        axes[3, col].scatter(feat_pca[:, 0], feat_pca[:, 1], c=res["gbc_labels"].reshape(-1), cmap=cmap, s=6, alpha=0.4, vmin=0, vmax=K - 1)
        axes[3, col].scatter(c_init_pca[:, 0], c_init_pca[:, 1], c="black", marker="x", s=50, linewidths=2, label="Init")
        axes[3, col].scatter(c_conv_pca[:, 0], c_conv_pca[:, 1], c="red", marker="+", s=60, linewidths=2, label="Conv")
        for k in range(K):
            axes[3, col].annotate(
                "",
                xy=(c_conv_pca[k, 0], c_conv_pca[k, 1]),
                xytext=(c_init_pca[k, 0], c_init_pca[k, 1]),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0, alpha=0.7),
            )
        axes[3, col].set_title(f"{col_title}\nJoint PCA (Var: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontsize=11, fontweight="bold")
        axes[3, col].legend(fontsize=8)
        axes[3, col].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    comparison_fig_path = os.path.join(master_output_dir, "comparison_across_modes.png")
    fig.savefig(comparison_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n>>> Saved multi-mode comparison figure: {comparison_fig_path}")

    # Cross-Mode Summary JSON
    summary_dict = {}
    for m in modes:
        met = results_by_mode[m]["metrics"]
        summary_dict[m] = {
            "ari_gbc_vs_km_conv": met["ari_gbc_vs_km_conv"],
            "nmi_gbc_vs_km_conv": met["nmi_gbc_vs_km_conv"],
            "agreement_pct_gbc_vs_km_conv": met["agreement_pct_gbc_vs_km_conv"],
            "active_clusters_gbc": met["active_clusters_gbc"],
            "active_clusters_km_conv": met["active_clusters_km_conv"],
            "ball_coverage_pct_r1": met["ball_coverage_pct"],
            "ball_coverage_pct_r2": met["coverage_pct_r2"],
            "median_dist2": met["median_dist2"],
            "mean_centroid_shift": met["mean_centroid_shift"],
            "max_centroid_shift": met["max_centroid_shift"],
        }

    summary_json_path = os.path.join(master_output_dir, "comparison_across_modes.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=4)
    print(f">>> Saved multi-mode summary JSON: {summary_json_path}")

    # Print Comparative Synthesis Table
    print("\n" + "=" * 80)
    print("                 CROSS-MODE COMPARATIVE SYNTHESIS")
    print("=" * 80)
    header = f"{'Metric':<35} | " + " | ".join([f"{m.upper():<12}" for m in modes])
    print(header)
    print("-" * len(header))
    rows = [
        ("Active Clusters GBC (>0 pts)", lambda m: f"{summary_dict[m]['active_clusters_gbc']}/{K}"),
        ("Active Clusters Conv KM", lambda m: f"{summary_dict[m]['active_clusters_km_conv']}/{K}"),
        ("Pixel Agreement GBC/Conv KM", lambda m: f"{summary_dict[m]['agreement_pct_gbc_vs_km_conv']:.1f}%"),
        ("ARI (GBC vs. Conv KM)", lambda m: f"{summary_dict[m]['ari_gbc_vs_km_conv']:.4f}"),
        ("NMI (GBC vs. Conv KM)", lambda m: f"{summary_dict[m]['nmi_gbc_vs_km_conv']:.4f}"),
        ("Ball Coverage (d² <= 1.0)", lambda m: f"{summary_dict[m]['ball_coverage_pct_r1']:.1f}%"),
        ("Ball Coverage (d² <= 4.0)", lambda m: f"{summary_dict[m]['ball_coverage_pct_r2']:.1f}%"),
        ("Median Mahalanobis d²", lambda m: f"{summary_dict[m]['median_dist2']:.2f}"),
        ("Mean Centroid Shift", lambda m: f"{summary_dict[m]['mean_centroid_shift']:.4f}"),
    ]
    for label, fn in rows:
        line = f"{label:<35} | " + " | ".join([f"{fn(m):<12}" for m in modes])
        print(line)
    print("=" * 80 + "\n")


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 1. Load inputs
    feat_raw_t, centers_raw_t, sigma_raw_t, H, W, K, D = load_and_preprocess_inputs(
        args.features, args.centers, args.sigma, device
    )

    # 2. Execution depending on --normalize
    if args.normalize == "all":
        modes_to_run = ["none", "zscore", "unit"]
        results_by_mode = {}
        for mode in modes_to_run:
            sub_output_dir = os.path.join(args.output_dir, f"mode_{mode}")
            res = run_single_pipeline(
                feat_raw_t, centers_raw_t, sigma_raw_t, H, W, K, D, mode, args, sub_output_dir
            )
            results_by_mode[mode] = res

        # Generate Cross-Mode Comparative Analysis
        generate_cross_mode_comparison(results_by_mode, H, W, K, args.output_dir)
        print(f"\nAll modes completed successfully! Check master outputs in: {args.output_dir}")

    else:
        run_single_pipeline(
            feat_raw_t, centers_raw_t, sigma_raw_t, H, W, K, D, args.normalize, args, args.output_dir
        )
        print(f"\nClustering pipeline completed for mode '{args.normalize}'. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
