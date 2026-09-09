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
    from nnunetv2.training.nnUNetTrainer.archs_GBC import GBC_Rolling_Unet_S, GBC_Rolling_Unet_M, GBC_Rolling_Unet_L, Rolling_Unet_L
except ImportError:
    from archs_GBC import GBC_Rolling_Unet_S, GBC_Rolling_Unet_M, GBC_Rolling_Unet_L, Rolling_Unet_L

try:
    from nnunetv2.training.nnUNetTrainer.archs_unext import UNext
except ImportError:
    from archs_unext import UNext

# try:
#     from nnunetv2.training.nnUNetTrainer.archs_LBUNet import LBUNet
# except ImportError:
#     from archs_unext import LBUNet

class nnUNetTrainerGBC(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Deep supervision is disabled because GBC outputs a single resolution segmentation
        self.enable_deep_supervision = False

        # Custom hyperparameters for hybrid Transformer/MLP/GBC architectures
        self.initial_lr = 1e-4
        self.weight_decay = 0.01
        self.num_epochs = 300  # GBC/Rolling-UNet commonly converges around 300-400 epochs

    def configure_optimizers(self):
        # Use AdamW as recommended for hybrid Transformer/MLP architectures (SGD at 0.01 often diverges)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        # Cosine Annealing scheduler (as used in the Rolling-UNet paper)
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

    def compute_gbc_losses(self, loss_intermediates: dict, net_module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        model_gbc = net_module.gbc
        centers = model_gbc.centers  # (K, d)
        K = centers.shape[0]

        # 1. Wasserstein-based Diversity Loss (prevent center collapse)
        dist_matrix = torch.cdist(centers, centers, p=2)  # (K, K)
        mask = ~torch.eye(K, dtype=torch.bool, device=centers.device)
        l_div = torch.exp(-dist_matrix)[mask].mean()

        # 2. Scale / Radius-Dispersion Consistency Loss
        if hasattr(model_gbc, 'log_sigma') and model_gbc.log_sigma is not None:
            sigma = torch.functional.F.softplus(model_gbc.log_sigma) + 1e-6  # (K, d)
        else:
            sigma = torch.functional.F.softplus(model_gbc.log_radius) + 1e-6  # (K, 1)
        sigma_sq = sigma ** 2

        losses_scale = []
        for suffix in ["_1", "_2"]:
            att_key = f"att{suffix}"
            dif_key = f"dif{suffix}"
            if att_key in loss_intermediates and dif_key in loss_intermediates:
                att = loss_intermediates[att_key]  # (B, N, K)
                dif = loss_intermediates[dif_key]  # (B, N, K, d)

                # Weighted dispersion: sum_i (att_i * (x_i - c)^2) / sum_i att_i
                dif_sq = dif ** 2  # (B, N, K, d)
                num = (att.unsqueeze(-1) * dif_sq).sum(dim=1)  # (B, K, d)
                den = att.sum(dim=1).unsqueeze(-1) + 1e-6  # (B, K, 1)
                weighted_dispersion = num / den  # (B, K, d)

                # Mean squared error between dispersion and scale
                l_s = torch.mean((weighted_dispersion - sigma_sq.unsqueeze(0)) ** 2)
                losses_scale.append(l_s)

        l_scale = torch.mean(torch.stack(losses_scale)) if losses_scale else torch.tensor(0.0, device=centers.device)

        return l_div, l_scale

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
            # Check if the network returns (output, loss_intermediates) or just output (baseline)
            net_outputs = self.network(data)
            if isinstance(net_outputs, tuple):
                output, loss_intermediates = net_outputs

                # 1. Main task segmentation loss
                l_seg = self.loss(output, target)

                # 2. Compute GBC regularizers
                net_module = self._get_actual_network()
                l_div, l_scale = self.compute_gbc_losses(loss_intermediates, net_module)

                # Combine losses
                l = l_seg + 0.1 * l_div + 0.1 * l_scale
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


class nnUNetTrainerGBC_S(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return GBC_Rolling_Unet_S(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size,
            gbc_num_balls=32,
        )
# class GBC_Rolling_Unet_L(nn.Module):
#     def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224,
#                  embed_dims=[64, 128, 256, 512, 1024],
#                  num_heads=[1, 2, 4, 8], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1],
#                  gbc_num_balls=32, gbc_proj_dim=None, use_diag_cov=True, tau=1.0, **kwargs):
#         super().__init__()

# class MambaLiteUNet(nn.Module):
#     def __init__(
#         self,
#         num_classes=1,
#         input_channels=3,
#         c_list=[16, 32, 48, 64, 96, 128]
#     ):

class nnUNetTrainerGBC_M(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return GBC_Rolling_Unet_M(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size
        )


class nnUNetTrainerGBC_L(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return GBC_Rolling_Unet_L(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size,
            gbc_num_balls=16,
        )

class nnUNetTrainerRoll_L(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return Rolling_Unet_L(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size
        )
class nnUNetTrainer_Next(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return UNext(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,
            img_size=img_size
        )
