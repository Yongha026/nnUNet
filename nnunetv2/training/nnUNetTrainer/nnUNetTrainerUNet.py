from typing import Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.helpers import dummy_context

try:
    from nnunetv2.training.nnUNetTrainer.archs_unet import UNet
except ImportError:
    from archs_unet import UNet


class PixelWiseSoftmaxCELoss(nn.Module):
    """
    Pixel-wise Softmax over the final feature map combined with Cross-Entropy loss,
    as formulated in Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
    Image Segmentation" (MICCAI 2015, https://arxiv.org/pdf/1505.04597):
        p_k(x) = exp(a_k(x)) / sum_{k'=1}^K exp(a_{k'}(x))
        E = - sum_{x in Omega} log(p_{l(x)}(x))
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim:
            assert target.shape[1] == 1, (
                f"Expected target channel dimension to be 1, got shape {tuple(target.shape)}"
            )
            target = target[:, 0]
        target = target.long()

        # Pixel-wise log-softmax over the channel dimension (dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # Cross-entropy loss over all spatial pixel positions
        return F.nll_loss(log_probs, target, ignore_index=self.ignore_index, reduction='mean')


class nnUNetTrainerUNet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Original U-Net outputs single-resolution logits without deep supervision
        self.enable_deep_supervision = False

        # Baseline hyperparameters aligned with other comparison trainers (300 epochs)
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

    def _build_loss(self):
        ignore_index = (
            self.label_manager.ignore_label
            if self.label_manager.has_ignore_label
            else -100
        )
        return PixelWiseSoftmaxCELoss(ignore_index=ignore_index)

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
        return UNet(
            n_channels=num_input_channels,
            n_classes=num_output_channels,
            bilinear=False
        )
