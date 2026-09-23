import os
import argparse
from typing import Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from scipy.ndimage import distance_transform_edt as distance


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def one_hot2dist(posmask):
    # Input: Mask. Will be converted to Bool.
    # Author: Rakshit Kothari
    assert len(posmask.shape) == 2
    h, w = posmask.shape
    res = np.zeros_like(posmask)
    posmask = posmask.astype(bool)
    mxDist = np.sqrt((h - 1) ** 2 + (w - 1) ** 2)
    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res / mxDist


def shrink_sam_to_gbc_space(
    sam_features: torch.Tensor,
    target_hw: Tuple[int, int] = (48, 48),
    target_dim: int = 64,
    pca: Optional[PCA] = None,
) -> Tuple[torch.Tensor, PCA]:
    """
    Shrinks SAM feature tensor from [B, 256, 64, 64] to GBC feature space [B, 64, 48, 48].
    
    Workflow:
      1. Spatial interpolation: Resamples SAM patches [B, 256, 64, 64] -> [B, 256, 48, 48]
      2. Channel reduction: Projects 256 channels to 64 channels via PCA (>95% variance retained)
      
    Args:
        sam_features: [B, 256, H_sam, W_sam] float tensor from SAM forward_features
        target_hw: Spatial dimensions of GBC stage (default: 48, 48)
        target_dim: Target channel dimension (default: 64)
        pca: Optional pre-fitted sklearn PCA model. If None, fits a new PCA on the batch.
        
    Returns:
        gbc_features: [B, target_dim, target_hw[0], target_hw[1]] tensor on the same device
        pca: The fitted/used PCA model
    """
    B, C, H, W = sam_features.shape
    device = sam_features.device

    # 1. Spatial Resampling: (64, 64) -> (48, 48)
    if (H, W) != target_hw:
        feat_spatial = F.interpolate(
            sam_features, size=target_hw, mode="bilinear", align_corners=False
        )
    else:
        feat_spatial = sam_features

    # 2. Channel Reduction: 256 -> 64 via PCA
    # Flatten spatial tokens across batch: [B * H_tgt * W_tgt, C]
    feat_flat = feat_spatial.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu().numpy()

    if pca is None:
        pca = PCA(n_components=target_dim, random_state=42)
        feat_pca_flat = pca.fit_transform(feat_flat)
    else:
        feat_pca_flat = pca.transform(feat_flat)

    gbc_features = (
        torch.from_numpy(feat_pca_flat)
        .float()
        .view(B, target_hw[0], target_hw[1], target_dim)
        .permute(0, 3, 1, 2)
        .to(device)
    )

    return gbc_features, pca


def compute_class_prototypes_and_radii(
    gbc_features: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes class prototype centers [K, d] and empirical dispersion radii [K, d]
    via masked average and standard-deviation pooling across an accumulated batch.
    
    Args:
        gbc_features: [B, D, H, W] tensor in GBC space (e.g. [B, 64, 48, 48])
        masks: [B, H_m, W_m] or [B, H, W] integer tensor containing class labels {0..num_classes-1}
        num_classes: Number of semantic classes / balls (default: 4 for OpenEDS: 0=bg, 1=pupil, 2=iris, 3=sclera)
        
    Returns:
        centers: [num_classes, D] tensor of class prototype centroids
        log_sigma: [num_classes, D] tensor ready for GranularBall.log_sigma parameter
                   (initialized via inverse softplus: log(exp(std) - 1))
    """
    B, D, H, W = gbc_features.shape
    device = gbc_features.device

    # Ensure mask is aligned to (H, W) of gbc_features
    if masks.shape[-2:] != (H, W):
        masks_aligned = (
            F.interpolate(
                masks.unsqueeze(1).float(),
                size=(H, W),
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )
    else:
        masks_aligned = masks.long()

    # Flatten spatial tokens across all batch images: [N_total, D]
    feats_flat = gbc_features.permute(0, 2, 3, 1).reshape(-1, D)
    masks_flat = masks_aligned.reshape(-1)

    centers = torch.zeros(num_classes, D, device=device)
    log_sigma = torch.zeros(num_classes, D, device=device)

    for k in range(num_classes):
        k_indices = (masks_flat == k).nonzero(as_tuple=True)[0]
        if len(k_indices) > 0:
            k_feats = feats_flat[k_indices]
            c_k = k_feats.mean(dim=0)
            std_k = k_feats.std(dim=0).clamp(min=1e-4)

            centers[k] = c_k
            # Inverse softplus: softplus(x) = log(1 + exp(x)) => x = log(exp(softplus) - 1)
            # For numerical stability when std_k > 20: inv_softplus(x) ≈ x
            log_sigma[k] = torch.where(
                std_k > 20.0,
                std_k,
                torch.log(torch.exp(std_k) - 1.0 + 1e-6),
            )
        else:
            # Fallback if a class is entirely absent in the batch
            centers[k] = torch.randn(D, device=device) * 0.01
            log_sigma[k] = torch.zeros(D, device=device)

    return centers, log_sigma


def sam_init_gbc_from_batch(
    gbc_module: nn.Module,
    sam_model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int = 4,
    target_hw: Tuple[int, int] = (48, 48),
    target_dim: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts SAM features for a batch of images, shrinks them to GBC space (target_dim, target_hw),
    computes class prototype centers and radii using ground-truth masks, and initializes
    the GBC module's parameters (centers and log_sigma / log_radius) in-place.
    
    Args:
        gbc_module: GranularBall module (with .centers and .log_sigma / .log_radius)
        sam_model: Pretrained SAM model (e.g. from timm.create_model('samvit_base_patch16.sa1b'))
        images: [B, 3, 1024, 1024] normalized image batch
        masks: [B, H, W] integer masks with class values in 0..num_classes-1
        num_classes: Number of classes / balls (default: 4)
        target_hw: Spatial grid size of GBC layer (default: 48, 48)
        target_dim: Channel dimension of GBC layer (default: 64)
        
    Returns:
        centers: [num_classes, target_dim]
        log_sigma: [num_classes, target_dim]
    """
    sam_model.eval()
    with torch.no_grad():
        # 1. SAM forward features: [B, 256, 64, 64]
        sam_features = sam_model.forward_features(images)

        # 2. Shrink SAM space: [B, 256, 64, 64] -> [B, 64, 48, 48]
        gbc_features, _ = shrink_sam_to_gbc_space(
            sam_features, target_hw=target_hw, target_dim=target_dim
        )

        # 3. Compute Class Prototypes & Radii across the accumulated batch
        centers, log_sigma = compute_class_prototypes_and_radii(
            gbc_features, masks, num_classes=num_classes
        )

    # 4. In-place parameter initialization of GBC module
    if hasattr(gbc_module, "centers"):
        gbc_module.centers.data.copy_(centers.to(gbc_module.centers.device))

    if hasattr(gbc_module, "use_diag_cov") and gbc_module.use_diag_cov:
        if hasattr(gbc_module, "log_sigma"):
            gbc_module.log_sigma.data.copy_(log_sigma.to(gbc_module.log_sigma.device))
    else:
        if hasattr(gbc_module, "log_radius"):
            # Scalar std across channels: [K, 1]
            scalar_radius = log_sigma.mean(dim=-1, keepdim=True)
            gbc_module.log_radius.data.copy_(scalar_radius.to(gbc_module.log_radius.device))

    return centers, log_sigma


def sam_init(
    m: nn.Module,
    dataloader_or_batch: Optional[Union[torch.utils.data.DataLoader, Tuple[torch.Tensor, torch.Tensor]]] = None,
    num_classes: int = 4,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-level initialization function for GBC centers using a pretrained SAM encoder.
    Can accept an accumulated batch (images, masks) or a DataLoader.
    
    If dataloader_or_batch is provided:
      Extracts SAM prototypes and initializes m.gbc (or m) directly.
    """
    # Identify target GBC layer
    gbc_layer = m.gbc if hasattr(m, "gbc") else m
    target_dim = getattr(gbc_layer, "proj_dim", 64)

    # Instantiate SAM encoder
    sam_model = timm.create_model(
        "samvit_base_patch16.sa1b",
        pretrained=True,
        num_classes=num_classes,
    )
    sam_model = sam_model.eval().to(device)

    if dataloader_or_batch is None:
        print("[sam_init] No batch provided. Instantiated SAM model only.")
        return None, None

    if isinstance(dataloader_or_batch, tuple):
        images, masks = dataloader_or_batch
        images = images.to(device)
        masks = masks.to(device)
    else:
        # Accumulate across the first batch of the dataloader
        batch = next(iter(dataloader_or_batch))
        if isinstance(batch, dict):
            images = batch["data"].to(device)
            masks = batch["target"].to(device)
        else:
            images, masks = batch[0].to(device), batch[1].to(device)

    # Resize images to SAM expected 1024x1024 if needed
    if images.shape[-2:] != (1024, 1024):
        images = F.interpolate(images, size=(1024, 1024), mode="bilinear", align_corners=False)

    # Ensure 3 channels for SAM
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    centers, log_sigma = sam_init_gbc_from_batch(
        gbc_module=gbc_layer,
        sam_model=sam_model,
        images=images,
        masks=masks,
        num_classes=num_classes,
        target_hw=(48, 48),
        target_dim=target_dim,
    )

    print(f"[sam_init] Successfully initialized GBC centers {tuple(centers.shape)} and radii from SAM features.")
    # Clean up SAM model from GPU memory
    del sam_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return centers, log_sigma


def extract_sam_kmeans_clusters(
    gbc_features: torch.Tensor,
    num_clusters: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Clusters shrinked SAM feature tokens into num_clusters centroids and empirical standard deviations.
    
    Args:
        gbc_features: [B, D, H, W] tensor in GBC space (e.g. [B, 64, 48, 48])
        num_clusters: Target number of clusters/balls (K)
        
    Returns:
        centers: [num_clusters, D] tensor of centroids
        log_sigma: [num_clusters, D] tensor of dispersion radii (inverse softplus)
    """
    B, D, H, W = gbc_features.shape
    device = gbc_features.device
    feats_flat = gbc_features.permute(0, 2, 3, 1).reshape(-1, D).detach().cpu().numpy()

    # Subsample if token count is very large to speed up KMeans fitting
    max_samples = 50000
    if len(feats_flat) > max_samples:
        indices = np.random.choice(len(feats_flat), max_samples, replace=False)
        sample_feats = feats_flat[indices]
    else:
        sample_feats = feats_flat

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=2048, random_state=42, n_init=3)
    kmeans.fit(sample_feats)
    cluster_labels = kmeans.predict(feats_flat)

    centers = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
    log_sigma = torch.zeros(num_clusters, D, device=device)

    feats_torch = torch.from_numpy(feats_flat).float().to(device)
    labels_torch = torch.from_numpy(cluster_labels).long().to(device)

    for k in range(num_clusters):
        k_indices = (labels_torch == k).nonzero(as_tuple=True)[0]
        if len(k_indices) > 1:
            std_k = feats_torch[k_indices].std(dim=0).clamp(min=1e-4)
        else:
            std_k = torch.ones(D, device=device) * 0.1

        log_sigma[k] = torch.where(
            std_k > 20.0,
            std_k,
            torch.log(torch.exp(std_k) - 1.0 + 1e-6),
        )

    return centers, log_sigma


def get_or_compute_sam_prototypes(
    num_clusters: int,
    sam_centers_dir: str = "./sam_centers",
    dataloader_or_batch = None,
    device: str = "cuda",
    target_hw: Tuple[int, int] = (48, 48),
    target_dim: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieves SAM prototype centers and dispersion radii for num_clusters (K).
    1. Checks if sam_centers/openeds_centers_{num_clusters}.npy and openeds_sigma_{num_clusters}.npy exist.
    2. If found, loads and returns them directly without invoking SAM or consuming GPU VRAM.
    3. If not found, runs online SAM inference on batch 0, clusters features into num_clusters,
       saves openeds_centers_{num_clusters}.npy and openeds_sigma_{num_clusters}.npy, cleans up VRAM,
       and returns the tensors.
    """
    os.makedirs(sam_centers_dir, exist_ok=True)
    centers_path = os.getenv("SAM_GBC_CENTERS_PATH", os.path.join(sam_centers_dir, f"openeds_centers_{num_clusters}.npy"))
    sigma_path = os.getenv("SAM_GBC_SIGMA_PATH", os.path.join(sam_centers_dir, f"openeds_sigma_{num_clusters}.npy"))

    if os.path.isfile(centers_path) and os.path.isfile(sigma_path):
        print(f"[SAM Prototype] Found precomputed prototypes in {sam_centers_dir} for K={num_clusters}:")
        print(f"  - Centers: {centers_path}")
        print(f"  - Sigma:   {sigma_path}")
        centers_np = np.load(centers_path)
        sigma_np = np.load(sigma_path)
        centers = torch.from_numpy(centers_np).float().to(device)
        log_sigma = torch.from_numpy(sigma_np).float().to(device)
        return centers, log_sigma

    print(f"[SAM Prototype] Precomputed prototypes for K={num_clusters} not found in {sam_centers_dir}.")
    print(f"[SAM Prototype] Running online inference with SAM to extract K={num_clusters} clusters...")

    if dataloader_or_batch is None:
        raise ValueError(
            f"Cannot compute SAM prototypes for K={num_clusters} because neither precomputed files "
            f"nor dataloader/batch were provided."
        )

    # Instantiate SAM encoder
    sam_model = timm.create_model(
        "samvit_base_patch16.sa1b",
        pretrained=True,
        num_classes=num_clusters,
    )
    sam_model = sam_model.eval().to(device)

    # Extract batch
    if isinstance(dataloader_or_batch, tuple):
        images, masks = dataloader_or_batch
        images = images.to(device)
        masks = masks.to(device) if masks is not None else None
    else:
        batch = next(iter(dataloader_or_batch))
        if isinstance(batch, dict):
            images = batch["data"].to(device)
            masks = batch["target"]
            if isinstance(masks, list):
                masks = masks[0]
            masks = masks.to(device) if masks is not None else None
        else:
            images = batch[0].to(device)
            masks = batch[1].to(device) if len(batch) > 1 and batch[1] is not None else None

    # Handle single channel and spatial size
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if images.shape[-2:] != (1024, 1024):
        images = F.interpolate(images, size=(1024, 1024), mode="bilinear", align_corners=False)

    with torch.no_grad():
        sam_features = sam_model.forward_features(images)
        gbc_features, _ = shrink_sam_to_gbc_space(
            sam_features, target_hw=target_hw, target_dim=target_dim
        )

        if num_clusters == 4 and masks is not None:
            # For 4 classes with masks provided, compute class prototypes
            centers, log_sigma = compute_class_prototypes_and_radii(
                gbc_features, masks, num_classes=4
            )
        else:
            # For arbitrary K, cluster features directly via K-Means
            centers, log_sigma = extract_sam_kmeans_clusters(
                gbc_features, num_clusters=num_clusters
            )

    # Save to sam_centers_dir for future runs
    save_centers_path = os.path.join(sam_centers_dir, f"openeds_centers_{num_clusters}.npy")
    save_sigma_path = os.path.join(sam_centers_dir, f"openeds_sigma_{num_clusters}.npy")
    np.save(save_centers_path, centers.detach().cpu().numpy())
    np.save(save_sigma_path, log_sigma.detach().cpu().numpy())
    print(f"[SAM Prototype] Saved extracted prototypes for K={num_clusters} to:")
    print(f"  - {save_centers_path}")
    print(f"  - {save_sigma_path}")

    # Immediately release SAM model and free GPU memory
    del sam_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return centers, log_sigma

