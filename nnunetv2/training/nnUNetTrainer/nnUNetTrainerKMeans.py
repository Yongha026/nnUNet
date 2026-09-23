"""
nnUNetTrainerKMeans.py
======================
Fine-tuning trainer variants for Soft K-Means Rolling-UNet.
Re-exports the unified implementation from nnUNetTrainerGBC with:
- One-time SAM parameter initialization from sam_centers/openeds_centers_{K}.npy (or dynamic SAM inference)
- Frozen backbone with AdamW optimizer on clustering parameters only
- Soft K-Means clustering loss and architecture across cluster counts K
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerGBC import (
    load_pretrained_weights_backbone_only,
    nnUNetTrainerKMeans,
    nnUNetTrainerKMeans_S_2,
    nnUNetTrainerKMeans_S_4,
    nnUNetTrainerKMeans_S_8,
    nnUNetTrainerKMeans_S_16,
    nnUNetTrainerKMeans_S_32,
    nnUNetTrainerKMeans_S_64,
    nnUNetTrainerKMeans_FT_S_4,
    nnUNetTrainerKMeans_FT_S_16,
    nnUNetTrainerKMeans_FT_S_32,
)
