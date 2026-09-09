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
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss

try:
    from nnunetv2.training.nnUNetTrainer.archs_MambaLiteUNet import MambaLiteUNet
except ImportError:
    from archs_MambaLiteUNet import MambaLiteUNet
try:
    from nnunetv2.training.nnUNetTrainer.archs_ULVMUNet import UltraLight_VM_UNet
except ImportError:
    from archs_ULVMUNet import UltraLight_VM_UNet

class nnUNetTrainerMLU(nnUNetTrainer):
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

    def compute_mlu_losses(self, output, target) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.loss(output, target)


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

            output = self.network(data)

            l = self.compute_mlu_losses(output, target)

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


class nnUNetTrainerMLU_Run(nnUNetTrainerMLU):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return MambaLiteUNet(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            # deep_supervision=False,
            # img_size=img_size
        )
# class MambaLiteUNet(nn.Module):
#     def __init__(
#         self,
#         num_classes=1,
#         input_channels=3,
#         c_list=[16, 32, 48, 64, 96, 128]
#     ):

class _nnUNetULVM(nnUNetTrainer):
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

    def compute_ulvm_losses(self, output, target) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.loss(output, target)

    def _build_loss(self):
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

        # Rolling UNet base loss weight: 0.5 * BCE/CE + 1.0 * Dice
        w_ce = 1.0
        w_dice = 0.0

        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                bce_kwargs={},
                soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                                  'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                weight_ce=w_ce,
                weight_dice=w_dice,
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            loss = DC_and_CE_loss(
                soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                                  'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs={},
                weight_ce=w_ce,
                weight_dice=w_dice,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss
            )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
        return loss

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

            output = self.network(data)

            l = self.compute_ulvm_losses(output, target)

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

class nnUNetTrainerULVM(_nnUNetULVM):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return UltraLight_VM_UNet(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
        )