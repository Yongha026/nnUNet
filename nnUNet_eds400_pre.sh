#!/bin/bash
# source nnUNet_preliminary.sh to run

# 1. Standard nnU-Net Paths
export nnUNet_raw="/mnt/hdd1/nnunetv2_openEDS_400/nnUNet_raw"
export nnUNet_preprocessed="/mnt/hdd1/nnunetv2_openEDS_400/nnUNet_preprocessed"
export nnUNet_results="/mnt/hdd1/nnunetv2_openEDS_400/nnUNet_results"

# 2. Custom Trainer Code Integration
export nnUNet_extTrainer="/home/iulab9/PycharmProjects/nnUNet/nnunetv2/training/nnUNetTrainer"

# 3. Weights & Biases Logging Settings
export nnUNet_wandb_enabled=1
export nnUNet_wandb_project="MambaLiteUNet"
export WANDB_MODE=online

# 4. Model Training Tweaks
export nnUNet_compile=false # Disable compile to avoid Triton/Dynamo conflicts with custom GBC layers

# 5. Hardware Allocation
export CUDA_VISIBLE_DEVICES=1

echo "========================================================"
echo " nnU-Net Environment Settings Configured Successfully!  "
echo "========================================================"
echo "nnUNet_raw:         $nnUNet_raw"
echo "nnUNet_preprocessed:$nnUNet_preprocessed"
echo "nnUNet_results:     $nnUNet_results"
echo "nnUNet_extTrainer:  $nnUNet_extTrainer"
echo "WandB Project:      $nnUNet_wandb_project (Enabled: $nnUNet_wandb_enabled)"
echo "Target GPU:         Device $CUDA_VISIBLE_DEVICES"
echo "=============================================== commands ================================================"
echo "nohup nnUNetv2_train 250 2d 0 -tr nnUNetTrainerMLU_Run > nnUNet_mlu_v1.log 2>&1 &"
echo "nnUNetv2_predict -i ./nnUNet_test_imgs/ -o ./nnUNet_infer_res/ -d 250 -c 2d -tr nnUNetTrainerMLU_Run -f 0"
echo "nnUNetv2_predict -i /mnt/hdd1/nnunetv2_swir/images/ -o /mnt/hdd1/nnunetv2_swir/labels/ -d 250 -c 2d -tr nnUNetTrainerMLU_Run -f 0"
