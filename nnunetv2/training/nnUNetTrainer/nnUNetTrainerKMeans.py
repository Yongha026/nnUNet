"""
nnUNetTrainerKMeans.py
======================
Fine-tuning trainer variants for Soft K-Means Rolling-UNet with SAM parameter initialization.
Re-exports the unified implementation from nnUNetTrainerGBC.
"""

try:
    from .nnUNetTrainerGBC import (
        load_pretrained_weights_backbone_only,
        nnUNetTrainerSAM_KMeans,
        nnUNetTrainerSAM_KMeans_S_2,
        nnUNetTrainerSAM_KMeans_S_4,
        nnUNetTrainerSAM_KMeans_S_8,
        nnUNetTrainerSAM_KMeans_S_16,
        nnUNetTrainerSAM_KMeans_S_32,
        nnUNetTrainerSAM_KMeans_S_64,
        nnUNetTrainerKMeans_SAM_S_4,
        nnUNetTrainerKMeans_FT_S_4,
        nnUNetTrainerKMeans_FT_S_16,
        nnUNetTrainerKMeans_FT_S_32,
    )
except (ImportError, ValueError):
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerGBC import (
            load_pretrained_weights_backbone_only,
            nnUNetTrainerSAM_KMeans,
            nnUNetTrainerSAM_KMeans_S_2,
            nnUNetTrainerSAM_KMeans_S_4,
            nnUNetTrainerSAM_KMeans_S_8,
            nnUNetTrainerSAM_KMeans_S_16,
            nnUNetTrainerSAM_KMeans_S_32,
            nnUNetTrainerSAM_KMeans_S_64,
            nnUNetTrainerKMeans_SAM_S_4,
            nnUNetTrainerKMeans_FT_S_4,
            nnUNetTrainerKMeans_FT_S_16,
            nnUNetTrainerKMeans_FT_S_32,
        )
    except (ImportError, ValueError):
        from nnUNetTrainerGBC import (
            load_pretrained_weights_backbone_only,
            nnUNetTrainerSAM_KMeans,
            nnUNetTrainerSAM_KMeans_S_2,
            nnUNetTrainerSAM_KMeans_S_4,
            nnUNetTrainerSAM_KMeans_S_8,
            nnUNetTrainerSAM_KMeans_S_16,
            nnUNetTrainerSAM_KMeans_S_32,
            nnUNetTrainerSAM_KMeans_S_64,
            nnUNetTrainerKMeans_SAM_S_4,
            nnUNetTrainerKMeans_FT_S_4,
            nnUNetTrainerKMeans_FT_S_16,
            nnUNetTrainerKMeans_FT_S_32,
        )

# Aliases
nnUNetTrainerKMeans = nnUNetTrainerSAM_KMeans
nnUNetTrainerKMeans_S_2 = nnUNetTrainerSAM_KMeans_S_2
nnUNetTrainerKMeans_S_4 = nnUNetTrainerSAM_KMeans_S_4
nnUNetTrainerKMeans_S_8 = nnUNetTrainerSAM_KMeans_S_8
nnUNetTrainerKMeans_S_16 = nnUNetTrainerSAM_KMeans_S_16
nnUNetTrainerKMeans_S_32 = nnUNetTrainerSAM_KMeans_S_32
nnUNetTrainerKMeans_S_64 = nnUNetTrainerSAM_KMeans_S_64
