"""
nnUNeTrainerGBC_FT.py
=====================
Fine-tuning trainer variants for GBC and DGBC.
Re-exports the unified implementation from nnUNetTrainerGBC with:
- One-time SAM parameter initialization from sam_centers/openeds_centers_{K}.npy (or dynamic SAM inference)
- Frozen backbone with AdamW optimizer on clustering parameters only
- Hyperspheric (DGBC) and Anisotropic (GBC) variants across cluster counts K
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerGBC import (
    load_pretrained_weights_backbone_only,
    nnUNetTrainerGBC,
    nnUNetTrainerDGBC,
    nnUNetTrainerDGBC_S_2,
    nnUNetTrainerDGBC_S_4,
    nnUNetTrainerDGBC_S_8,
    nnUNetTrainerDGBC_S_16,
    nnUNetTrainerDGBC_S_32,
    nnUNetTrainerDGBC_S_64,
    nnUNetTrainerGBC_S_2,
    nnUNetTrainerGBC_S_4,
    nnUNetTrainerGBC_S_8,
    nnUNetTrainerGBC_S_16,
    nnUNetTrainerGBC_S_32,
    nnUNetTrainerGBC_S_64,
    nnUNetTrainerRGBC_S_4,
    nnUNetTrainerGBC_M,
    nnUNetTrainerGBC_L,
    nnUNetTrainerRoll_L,
    nnUNetTrainer_Next,
)
