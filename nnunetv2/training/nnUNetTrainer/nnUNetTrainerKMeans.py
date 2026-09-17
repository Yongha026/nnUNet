import os
import shutil
from typing import Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn
from torch import autocast
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.helpers import dummy_context

try:
    from nnunetv2.training.nnUNetTrainer.archs_K_means_UNet import KMeans_Rolling_Unet_S
except ImportError:
    from archs_K_means_UNet import KMeans_Rolling_Unet_S


class nnUNetTrainerKMeans(nnUNetTrainer):
    """
    nnU-Net Trainer for Soft K-Means Rolling-UNet architecture.
    Features differentiable feature clustering with Cosine Similarity and Soft K-Means EM iterations.
    Optimizes combined loss: L_total = L_seg + 0.1 * L_div + 0.1 * L_inertia.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Deep supervision is disabled because Soft K-Means Rolling-UNet outputs a single resolution segmentation
        self.enable_deep_supervision = False

        # Custom hyperparameters matching GBC / Rolling-UNet
        self.initial_lr = 1e-4
        self.weight_decay = 0.01
        self.num_epochs = 300

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
        return optimizer, lr_scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def _get_actual_network(self) -> nn.Module:
        net = self.network
        if hasattr(net, 'module'):  # DDP wrapping
            net = net.module
        if isinstance(net, OptimizedModule):  # torch.compile wrapping
            net = net._orig_mod
        return net

    def compute_kmeans_losses(self, loss_intermediates: dict, net_module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        kmeans_block = getattr(net_module, 'kmeans_block', getattr(net_module, 'gbc', None))
        centers = kmeans_block.centers  # (K, d)
        K = centers.shape[0]

        # 1. Diversity Loss on initial anchor centers (prevent cluster center collapse)
        if K > 1:
            dist_matrix = torch.cdist(centers, centers, p=2)  # (K, K)
            mask = ~torch.eye(K, dtype=torch.bool, device=centers.device)
            l_div = torch.exp(-dist_matrix)[mask].mean()
        else:
            l_div = torch.tensor(0.0, device=centers.device)

        # 2. Soft K-Means Clustering Inertia / Distortion Loss
        losses_inertia = []
        for suffix in ["_1", "_2"]:
            for key in [f"inertia{suffix}", f"dif{suffix}"]:
                if key in loss_intermediates:
                    losses_inertia.append(loss_intermediates[key])
                    break

        l_inertia = torch.mean(torch.stack(losses_inertia)) if losses_inertia else torch.tensor(0.0, device=centers.device)

        return l_div, l_inertia

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            net_outputs = self.network(data)
            if isinstance(net_outputs, tuple):
                output, loss_intermediates = net_outputs
                l_seg = self.loss(output, target)

                net_module = self._get_actual_network()
                l_div, l_inertia = self.compute_kmeans_losses(loss_intermediates, net_module)

                # Total loss = L_seg + 0.1 * L_div + 0.1 * L_inertia
                l = l_seg + 0.1 * l_div + 0.1 * l_inertia
            else:
                output = net_outputs
                l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return KMeans_Rolling_Unet_S(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size,
            num_clusters=16,
        )


class nnUNetTrainerKMeans_S_2(nnUNetTrainerKMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return KMeans_Rolling_Unet_S(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size,
            num_clusters=2,
        )


class nnUNetTrainerKMeans_S_4(nnUNetTrainerKMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return KMeans_Rolling_Unet_S(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size,
            num_clusters=4,
        )


class nnUNetTrainerKMeans_S_8(nnUNetTrainerKMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return KMeans_Rolling_Unet_S(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size,
            num_clusters=8,
        )


class nnUNetTrainerKMeans_S_16(nnUNetTrainerKMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return KMeans_Rolling_Unet_S(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size,
            num_clusters=16,
        )


class nnUNetTrainerKMeans_S_32(nnUNetTrainerKMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return KMeans_Rolling_Unet_S(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size,
            num_clusters=32,
        )


class nnUNetTrainerKMeans_S_64(nnUNetTrainerKMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return KMeans_Rolling_Unet_S(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size,
            num_clusters=64,
        )


# =========================================================================================
# Fine-Tuning Variants (Frozen Backbone, Trainable KMeans Cluster Parameters Only)
# =========================================================================================

class nnUNetTrainerKMeans_FT(nnUNetTrainerKMeans):
    """
    Fine-tuning variant of nnUNetTrainerKMeans that freezes the backbone and
    only optimizes the Soft K-Means clustering parameters.
    """
    def freeze_backbone(self):
        net = self._get_actual_network()
        for name, module in net.named_children():
            if name not in ("kmeans_block", "gbc"):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            else:
                module.train()
                for param in module.parameters():
                    param.requires_grad = True

    def configure_optimizers(self):
        self.freeze_backbone()
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in self.network.parameters())
        self.print_to_log_file(
            f"[*] KMeans Fine-Tuning Setup: {total_count - trainable_count:,} backbone params frozen, "
            f"{trainable_count:,} KMeans params trainable."
        )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.freeze_backbone()


class nnUNetTrainerKMeans_FT_S_4(nnUNetTrainerKMeans_FT, nnUNetTrainerKMeans_S_4):
    pass

class nnUNetTrainerKMeans_FT_S_16(nnUNetTrainerKMeans_FT, nnUNetTrainerKMeans_S_16):
    pass

class nnUNetTrainerKMeans_FT_S_32(nnUNetTrainerKMeans_FT, nnUNetTrainerKMeans_S_32):
    pass
