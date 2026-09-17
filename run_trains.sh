#!/bin/bash

echo "====No.1: GB 16===="
export nnUNet_wandb_name="nn_D-GBC_S_16_FT"
nohup nnUNetv2_train 250 2d 0 -tr nnUNetTrainerDGBC_S_16 -pretrained_weights /mnt/hdd1/nnunetv2_openEDS/nnUNet_results/Dataset250_OpenEDS2019/nnUNetTrainerGBC_S_16__nnUNetPlans__2d/fold_0/checkpoint_best.pth


echo "====No.2: GB 4(Best so far)"
export nnUNet_wandb_name="nn_D-GBC_S_4_FT"
nohup nnUNetv2_train 250 2d 0 -tr nnUNetTrainerDGBC_S_4 -pretrained_weights /mnt/hdd1/nnunetv2_openEDS/nnUNet_results/Dataset250_OpenEDS2019/nnUNetTrainerGBC_S_4__nnUNetPlans__2d/fold_0/checkpoint_best.pth

echo "====No.3: GB64===="
export nnUNet_wandb_name="nn_D-GBC_S_64_FT"
nohup nnUNetv2_train 250 2d 0 -tr nnUNetTrainerDGBC_S_64 -pretrained_weights /mnt/hdd1/nnunetv2_openEDS/nnUNet_results/Dataset250_OpenEDS2019/nnUNetTrainerGBC_S_64__nnUNetPlans__2d/fold_0/checkpoint_best.pth

echo "====No.4: GB 2===="
export nnUNet_wandb_name="nn_D-GBC_S_2_FT"
nohup nnUNetv2_train 250 2d 0 -tr nnUNetTrainerDGBC_S_2 -pretrained_weights /mnt/hdd1/nnunetv2_openEDS/nnUNet_results/Dataset250_OpenEDS2019/nnUNetTrainerGBC_S_2__nnUNetPlans__2d/fold_0/checkpoint_best.pth
