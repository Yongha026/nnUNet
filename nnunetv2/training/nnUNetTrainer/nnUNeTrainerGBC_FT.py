"""
nnUNeTrainerGBC_FT.py
=====================
Fine-tuning trainer variants for GBC and DGBC with SAM parameter initialization.
Re-exports the unified implementation from nnUNetTrainerGBC.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerGBC import (
    load_pretrained_weights_backbone_only,
    # Baseline
    nnUNetTrainerGBC,
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
    # Dedicated SAM Trainers
    nnUNetTrainerSAM_GBC,
    nnUNetTrainerSAM_GBC_S_2,
    nnUNetTrainerSAM_GBC_S_4,
    nnUNetTrainerSAM_GBC_S_8,
    nnUNetTrainerSAM_GBC_S_16,
    nnUNetTrainerSAM_GBC_S_32,
    nnUNetTrainerSAM_GBC_S_64,
    nnUNetTrainerSAM_DGBC,
    nnUNetTrainerSAM_DGBC_S_2,
    nnUNetTrainerSAM_DGBC_S_4,
    nnUNetTrainerSAM_DGBC_S_8,
    nnUNetTrainerSAM_DGBC_S_16,
    nnUNetTrainerSAM_DGBC_S_32,
    nnUNetTrainerSAM_DGBC_S_64,
    # Infix aliases
    nnUNetTrainerGBC_SAM_S_4,
    nnUNetTrainerDGBC_SAM_S_4,
    nnUNetTrainerDGBC_SAM_S_16,
    nnUNetTrainerDGBC_SAM_S_32,
)

# Backward-compatible DGBC aliases pointing to SAM-initialized DGBC
nnUNetTrainerDGBC = nnUNetTrainerSAM_DGBC
nnUNetTrainerDGBC_S_2 = nnUNetTrainerSAM_DGBC_S_2
nnUNetTrainerDGBC_S_4 = nnUNetTrainerSAM_DGBC_S_4
nnUNetTrainerDGBC_S_8 = nnUNetTrainerSAM_DGBC_S_8
nnUNetTrainerDGBC_S_16 = nnUNetTrainerSAM_DGBC_S_16
nnUNetTrainerDGBC_S_32 = nnUNetTrainerSAM_DGBC_S_32
nnUNetTrainerDGBC_S_64 = nnUNetTrainerSAM_DGBC_S_64
