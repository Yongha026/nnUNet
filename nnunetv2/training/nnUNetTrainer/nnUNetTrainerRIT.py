import os
import shutil
from typing import Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn
from torch import autocast
from torch._dynamo import OptimizedModule
import torch.nn.functional as F
import cv2
from scipy.ndimage import distance_transform_edt as edt

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.helpers import dummy_context

try:
    from nnunetv2.training.nnUNetTrainer.archs_ritnet import DenseNet2D
except ImportError:
    from archs_ritnet import DenseNet2D


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, weight=None, softmax=True, reduction=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = []
        self.reduction = reduction
        if softmax:
            self.norm = nn.Softmax(dim=1)
        else:
            self.norm = nn.Sigmoid()

    def forward(self, outputs, targets):
        probs = self.norm(outputs)

        if targets.ndim == 3 or (targets.ndim == 4 and targets.shape[1] == 1):
            if targets.ndim == 4:
                targets = targets.squeeze(1)
            targets_one_hot = F.one_hot(targets.long(), num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets.float()

        B, C, H, W = probs.shape
        probs_flat = probs.view(B, C, -1)
        targets_flat = targets_one_hot.view(B, C, -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=2)
        union = torch.sum(probs_flat, dim=2) + torch.sum(targets_flat, dim=2)

        class_weights = 1.0 / (torch.sum(targets_flat, dim=2) ** 2).clamp(min=self.epsilon)

        numerator = torch.sum(class_weights * intersection, dim=1)
        denominator = torch.sum(class_weights * union, dim=1)

        loss = 1.0 - 2.0 * numerator / (denominator + self.epsilon)

        if self.reduction:
            return loss.mean()
        else:
            return loss


class SurfaceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, softmax=True):
        super(SurfaceLoss, self).__init__()
        self.weight_map = []

    def forward(self, x, distmap):
        x = torch.softmax(x, dim=1)
        self.weight_map = distmap
        score = x.flatten(start_dim=2) * distmap.flatten(start_dim=2)
        score = torch.mean(score, dim=2)
        score = torch.mean(score, dim=1)
        return score


class RITLoss(nn.Module):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def forward(self, output, target):
        spatialWeights, maxDist = self.trainer.compute_spatial_weights_and_dist_map(target)

        CE_loss = self.trainer.criterion(output, target.squeeze(1).long() if target.ndim == 4 else target.long())
        loss_ce = CE_loss * (torch.ones_like(spatialWeights) + spatialWeights)
        loss_ce = torch.mean(loss_ce)

        loss_dice = self.trainer.criterion_DICE(output, target)
        loss_sl = torch.mean(self.trainer.criterion_SL(output, maxDist))

        return (1.0 - self.trainer.factor) * loss_sl + self.trainer.factor * loss_dice + loss_ce


# try:
#     from nnunetv2.training.nnUNetTrainer.archs_LBUNet import LBUNet
# except ImportError:
#     from archs_unext import LBUNet

class _nnUNetTrainerRIT(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

        self.initial_lr = 1e-4
        self.weight_decay = 0.01
        self.num_epochs = 300

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

    def _build_loss(self):
        self.criterion = CrossEntropyLoss2d()
        self.criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True)
        self.criterion_SL = SurfaceLoss()
        self.factor = 0.5
        return RITLoss(self)

    def compute_spatial_weights_and_dist_map(self, target):
        target_np = target.cpu().numpy()
        if target_np.ndim == 4:
            target_np = target_np[:, 0]  # shape (B, H, W)
        B, H, W = target_np.shape

        spatialWeights_list = []
        for b in range(B):
            lbl = target_np[b].astype(np.uint8)
            edges = cv2.Canny(lbl, 0, 1) / 255.0
            dilated = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
            spatialWeights_list.append(dilated * 20.0)
        spatialWeights = torch.from_numpy(np.stack(spatialWeights_list)).to(self.device).float()

        maxDist_list = []
        num_classes = self.label_manager.num_segmentation_heads
        for b in range(B):
            lbl = target_np[b]
            seg = (np.arange(num_classes)[:, None, None] == lbl[None, :, :]).astype(np.float32)
            dist = np.zeros_like(seg)
            for k in range(num_classes):
                posmask = seg[k].astype(bool)
                if posmask.any():
                    negmask = ~posmask
                    dist[k] = edt(negmask) * negmask - (edt(posmask) - 1) * posmask
            maxDist_list.append(dist)
        maxDist = torch.from_numpy(np.stack(maxDist_list)).to(self.device).float()

        return spatialWeights, maxDist

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
            output = net_outputs[0] if isinstance(net_outputs, tuple) else net_outputs
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


class nnUNetTrainerRIT(_nnUNetTrainerRIT):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        patch_size = configuration_manager.patch_size
        img_size = patch_size[0]

        return DenseNet2D(
            dropout=True,
            prob=0.2,
            deep_supervision=False,
        )
# model_dict['densenet'] = DenseNet2D(dropout=True,prob=0.2)