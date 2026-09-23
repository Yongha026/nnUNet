import os
import sys
import shutil
from typing import Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn
from torch import autocast
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.helpers import dummy_context
import nnunetv2.run.load_pretrained_weights as lpw

try:
    from .archs_GBC import (
        GBC_Rolling_Unet_S,
        GBC_Rolling_Unet_M,
        GBC_Rolling_Unet_L,
        Rolling_Unet_L,
    )
except (ImportError, ValueError):
    try:
        from nnunetv2.training.nnUNetTrainer.archs_GBC import (
            GBC_Rolling_Unet_S,
            GBC_Rolling_Unet_M,
            GBC_Rolling_Unet_L,
            Rolling_Unet_L,
        )
    except (ImportError, ValueError):
        from archs_GBC import (
            GBC_Rolling_Unet_S,
            GBC_Rolling_Unet_M,
            GBC_Rolling_Unet_L,
            Rolling_Unet_L,
        )

try:
    from .archs_unext import UNext
except (ImportError, ValueError):
    try:
        from nnunetv2.training.nnUNetTrainer.archs_unext import UNext
    except (ImportError, ValueError):
        from archs_unext import UNext

try:
    from .archs_K_means_UNet import KMeans_Rolling_Unet_S
except (ImportError, ValueError):
    try:
        from nnunetv2.training.nnUNetTrainer.archs_K_means_UNet import KMeans_Rolling_Unet_S
    except (ImportError, ValueError):
        try:
            from archs_K_means_UNet import KMeans_Rolling_Unet_S
        except (ImportError, ValueError):
            KMeans_Rolling_Unet_S = None

try:
    from .GBC_utils import get_or_compute_sam_prototypes
except (ImportError, ValueError):
    try:
        from nnunetv2.training.nnUNetTrainer.GBC_utils import get_or_compute_sam_prototypes
    except (ImportError, ValueError):
        from GBC_utils import get_or_compute_sam_prototypes


def load_pretrained_weights_backbone_only(network, fname, verbose=False):
    """
    Transfers backbone weights between matching keys in state_dicts.
    Pretrained clustering parameters ('gbc.', 'kmeans_block.') and segmentation heads ('.seg_layers.')
    are strictly excluded and discarded, avoiding any key or shape mismatch
    and allowing fresh hyperspherical/anisotropic parameter initialization.
    """
    if dist.is_initialized():
        saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()), weights_only=False)
    else:
        saved_model = torch.load(fname, weights_only=False)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        '.seg_layers.',
        'gbc.',
        'kmeans_block.',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be compatible with your network."
            assert model_dict[key].shape == pretrained_dict[key].shape, \
                f"The shape of the parameters of key {key} is not the same. Pretrained model: {pretrained_dict[key].shape}; your network: {model_dict[key].shape}."

    pretrained_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])
    }

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained BACKBONE weights from file ", fname, '###################')
    print(f"[*] Successfully transferred {len(pretrained_dict)} backbone layers. Clustering parameters discarded & initialized fresh.")
    if verbose:
        print("Below is the list of overlapping backbone blocks in pretrained model:")
        for key, value in pretrained_dict.items():
            print(f"  {key}: {tuple(value.shape)}")
    print("################### Done ###################")
    mod.load_state_dict(model_dict)


# Dynamically patch load_pretrained_weights across imported modules
lpw.load_pretrained_weights = load_pretrained_weights_backbone_only
if 'nnunetv2.run.run_training' in sys.modules:
    sys.modules['nnunetv2.run.run_training'].load_pretrained_weights = load_pretrained_weights_backbone_only


# =========================================================================================
# 1. Baseline From-Scratch GBC Trainer (Preserved to avoid overwriting baseline results)
# =========================================================================================

class nnUNetTrainerGBC(nnUNetTrainer):
    """
    Standard from-scratch baseline GBC trainer.
    Trains all parameters end-to-end with AdamW across 300 epochs.
    Results saved under nnUNetTrainerGBC_* folders.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
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
        if hasattr(net, 'module'):
            net = net.module
        if isinstance(net, OptimizedModule):
            net = net._orig_mod
        return net

    def compute_gbc_losses(self, loss_intermediates: dict, net_module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        model_gbc = net_module.gbc
        centers = model_gbc.centers
        K = centers.shape[0]

        if K > 1:
            dist_matrix = torch.cdist(centers, centers, p=2)
            mask = ~torch.eye(K, dtype=torch.bool, device=centers.device)
            l_div = torch.exp(-dist_matrix)[mask].mean()
        else:
            l_div = torch.tensor(0.0, device=centers.device)

        if hasattr(model_gbc, 'log_sigma') and model_gbc.log_sigma is not None:
            sigma = torch.functional.F.softplus(model_gbc.log_sigma) + 1e-6
        else:
            sigma = torch.functional.F.softplus(model_gbc.log_radius) + 1e-6
        sigma_sq = sigma ** 2

        losses_scale = []
        for suffix in ["_1", "_2"]:
            att_key = f"att{suffix}"
            dif_key = f"dif{suffix}"
            if att_key in loss_intermediates and dif_key in loss_intermediates:
                att = loss_intermediates[att_key]
                dif = loss_intermediates[dif_key]

                dif_sq = dif ** 2
                num = (att.unsqueeze(-1) * dif_sq).sum(dim=1)
                den = att.sum(dim=1).unsqueeze(-1) + 1e-6
                weighted_dispersion = num / den

                if sigma_sq.shape[-1] == 1:
                    target_dispersion = weighted_dispersion.mean(dim=-1, keepdim=True)
                else:
                    target_dispersion = weighted_dispersion

                l_s = torch.mean((target_dispersion - sigma_sq.unsqueeze(0)) ** 2)
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
            net_outputs = self.network(data)
            if isinstance(net_outputs, tuple):
                output, loss_intermediates = net_outputs
                l_seg = self.loss(output, target)
                net_module = self._get_actual_network()
                l_div, l_scale = self.compute_gbc_losses(loss_intermediates, net_module)
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


# Baseline GBC Subclasses (From-scratch)
class nnUNetTrainerGBC_S_2(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=2, use_diag_cov=True)

class nnUNetTrainerGBC_S_4(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=4, use_diag_cov=True)

class nnUNetTrainerGBC_S_8(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=8, use_diag_cov=True)

class nnUNetTrainerGBC_S_16(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=16, use_diag_cov=True)

class nnUNetTrainerGBC_S_32(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=32, use_diag_cov=True)

class nnUNetTrainerGBC_S_64(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=64, use_diag_cov=True)

class nnUNetTrainerRGBC_S_4(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=4, use_residual=False, use_diag_cov=True)

class nnUNetTrainerGBC_M(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_M(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0])

class nnUNetTrainerGBC_L(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_L(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0], gbc_num_balls=16)

class nnUNetTrainerRoll_L(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return Rolling_Unet_L(num_classes=num_output_channels, input_channels=num_input_channels,
                              deep_supervision=False, img_size=configuration_manager.patch_size[0])

class nnUNetTrainer_Next(nnUNetTrainerGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return UNext(num_classes=num_output_channels, input_channels=num_input_channels,
                     deep_supervision=False, img_size=configuration_manager.patch_size[0])


# =========================================================================================
# 2. SAM Fine-Tuning Base Trainer (Frozen Backbone + SAM Initialization)
# =========================================================================================

class nnUNetTrainerSAM_GBC(nnUNetTrainerGBC):
    """
    Dedicated Fine-Tuning Trainer with SAM initialization and frozen backbone.
    Saved under a dedicated results folder: nnUNetTrainerSAM_GBC_* (NEVER overwrites baseline!).
    Features:
      - Frozen backbone (encoder/decoder/skip in eval mode, requires_grad=False)
      - Only clustering parameters trainable with AdamW (lr=5e-5, weight_decay=0.01)
      - One-time SAM initialization from sam_centers/openeds_centers_{K}.npy (or dynamic SAM inference)
      - Cosine annealing scheduler over 50 epochs
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 5e-5
        self.weight_decay = 0.01
        self.num_epochs = 50
        self._clustering_sam_initialized = False

        wandb_name = os.getenv("nnUNet_wandb_name")
        if wandb_name is not None:
            try:
                import wandb
                if wandb is not None:
                    if wandb.run is not None:
                        wandb.run.name = wandb_name
                        wandb.run.save()
                        self.print_to_log_file(f"[*] Set existing W&B run name to: {wandb_name}")
                    else:
                        from nnunetv2.training.logging.nnunet_logger import WandbLogger
                        continue_training = plans.get("continue_training", False)
                        wandb_logger = WandbLogger(self.output_folder, continue_training)
                        if wandb.run is not None:
                            wandb.run.name = wandb_name
                            wandb.run.save()
                        self.logger.loggers.append(wandb_logger)
                        self.print_to_log_file(f"[*] Attached WandbLogger with run name: {wandb_name}")
            except Exception as e:
                self.print_to_log_file(f"[!] Warning: Could not setup W&B logger: {e}")

    def _get_clustering_module_and_k(self) -> Tuple[nn.Module, int]:
        net = self._get_actual_network()
        if hasattr(net, "gbc") and net.gbc is not None:
            mod = net.gbc
            k = getattr(mod, "num_balls", getattr(mod, "num_clusters", 4))
            return mod, k
        elif hasattr(net, "kmeans_block") and net.kmeans_block is not None:
            mod = net.kmeans_block
            k = getattr(mod, "num_clusters", 4)
            return mod, k
        raise AttributeError(f"Network {net.__class__.__name__} has neither 'gbc' nor 'kmeans_block'.")

    def freeze_backbone(self):
        """Freezes all non-clustering backbone parameters and sets their BatchNorm layers to eval mode."""
        net = self._get_actual_network()
        for name, module in net.named_children():
            if name not in ("gbc", "kmeans_block"):
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
            f"[*] SAM Fine-Tuning Setup: {total_count - trainable_count:,} backbone params frozen, "
            f"{trainable_count:,} clustering params trainable."
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

    def initialize_clustering_parameters_with_sam(self):
        if self._clustering_sam_initialized or self.current_epoch > 0:
            return

        cluster_mod, k = self._get_clustering_module_and_k()
        self.print_to_log_file(f"[*] Initializing clustering parameters with SAM for K={k}...")

        centers, log_sigma = get_or_compute_sam_prototypes(
            num_clusters=k,
            sam_centers_dir="./sam_centers",
            dataloader_or_batch=self.dataloader_train,
            device=str(self.device),
        )

        if hasattr(cluster_mod, "centers"):
            cluster_mod.centers.data.copy_(centers.to(cluster_mod.centers.device))

        if hasattr(cluster_mod, "use_diag_cov") and cluster_mod.use_diag_cov:
            if hasattr(cluster_mod, "log_sigma"):
                cluster_mod.log_sigma.data.copy_(log_sigma.to(cluster_mod.log_sigma.device))
        elif hasattr(cluster_mod, "log_radius"):
            scalar_radius = log_sigma.mean(dim=-1, keepdim=True)
            cluster_mod.log_radius.data.copy_(scalar_radius.to(cluster_mod.log_radius.device))

        if self.is_ddp:
            dist.broadcast(cluster_mod.centers.data, src=0)
            if hasattr(cluster_mod, "log_sigma") and cluster_mod.log_sigma is not None:
                dist.broadcast(cluster_mod.log_sigma.data, src=0)
            elif hasattr(cluster_mod, "log_radius") and cluster_mod.log_radius is not None:
                dist.broadcast(cluster_mod.log_radius.data, src=0)

        self._clustering_sam_initialized = True
        self.print_to_log_file(f"[*] Successfully initialized clustering parameters with SAM for K={k}.")

    def on_train_start(self):
        super().on_train_start()
        if self.current_epoch == 0 and not self._clustering_sam_initialized:
            self.initialize_clustering_parameters_with_sam()

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
                l_div, l_scale = self.compute_gbc_losses(loss_intermediates, net_module)
                l = l_seg + 0.1 * l_div + 0.1 * l_scale
            else:
                output = net_outputs
                l = self.loss(output, target)

        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}


# =========================================================================================
# 3. Dedicated SAM GBC Subclasses (Anisotropic, use_diag_cov = True)
# Results saved to: nnUNetTrainerSAM_GBC_S_{K}__*
# =========================================================================================

class nnUNetTrainerSAM_GBC_S_2(nnUNetTrainerSAM_GBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=2, use_diag_cov=True)

class nnUNetTrainerSAM_GBC_S_4(nnUNetTrainerSAM_GBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=4, use_diag_cov=True)

class nnUNetTrainerSAM_GBC_S_8(nnUNetTrainerSAM_GBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=8, use_diag_cov=True)

class nnUNetTrainerSAM_GBC_S_16(nnUNetTrainerSAM_GBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=16, use_diag_cov=True)

class nnUNetTrainerSAM_GBC_S_32(nnUNetTrainerSAM_GBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=32, use_diag_cov=True)

class nnUNetTrainerSAM_GBC_S_64(nnUNetTrainerSAM_GBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=64, use_diag_cov=True)


# =========================================================================================
# 4. Dedicated SAM DGBC Subclasses (Hyperspheric Covariance / Scalar Radius, use_diag_cov = False)
# Results saved to: nnUNetTrainerSAM_DGBC_S_{K}__*
# =========================================================================================

class nnUNetTrainerSAM_DGBC(nnUNetTrainerSAM_GBC):
    """Base class for SAM-initialized Hyperspheric Granular Ball Clustering (use_diag_cov = False)."""
    pass

class nnUNetTrainerSAM_DGBC_S_2(nnUNetTrainerSAM_DGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=2, use_diag_cov=False)

class nnUNetTrainerSAM_DGBC_S_4(nnUNetTrainerSAM_DGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=4, use_diag_cov=False)

class nnUNetTrainerSAM_DGBC_S_8(nnUNetTrainerSAM_DGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=8, use_diag_cov=False)

class nnUNetTrainerSAM_DGBC_S_16(nnUNetTrainerSAM_DGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=16, use_diag_cov=False)

class nnUNetTrainerSAM_DGBC_S_32(nnUNetTrainerSAM_DGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=32, use_diag_cov=False)

class nnUNetTrainerSAM_DGBC_S_64(nnUNetTrainerSAM_DGBC):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return GBC_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                  deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                  gbc_num_balls=64, use_diag_cov=False)


# =========================================================================================
# 5. Dedicated SAM Soft K-Means Subclasses (KMeansBlock)
# Results saved to: nnUNetTrainerSAM_KMeans_S_{K}__*
# =========================================================================================

class nnUNetTrainerSAM_KMeans(nnUNetTrainerSAM_GBC):
    """
    Dedicated Fine-Tuning Trainer for Soft K-Means with SAM-initialized anchor centers.
    Results saved under a dedicated folder: nnUNetTrainerSAM_KMeans_*
    """
    def compute_kmeans_losses(self, loss_intermediates: dict, net_module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        kmeans_block = getattr(net_module, 'kmeans_block', getattr(net_module, 'gbc', None))
        centers = kmeans_block.centers
        K = centers.shape[0]

        if K > 1:
            dist_matrix = torch.cdist(centers, centers, p=2)
            mask = ~torch.eye(K, dtype=torch.bool, device=centers.device)
            l_div = torch.exp(-dist_matrix)[mask].mean()
        else:
            l_div = torch.tensor(0.0, device=centers.device)

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
                l = l_seg + 0.1 * l_div + 0.1 * l_inertia
            else:
                output = net_outputs
                l = self.loss(output, target)

        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

class nnUNetTrainerSAM_KMeans_S_2(nnUNetTrainerSAM_KMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return KMeans_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                    deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                    num_clusters=2)

class nnUNetTrainerSAM_KMeans_S_4(nnUNetTrainerSAM_KMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return KMeans_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                    deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                    num_clusters=4)

class nnUNetTrainerSAM_KMeans_S_8(nnUNetTrainerSAM_KMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return KMeans_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                    deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                    num_clusters=8)

class nnUNetTrainerSAM_KMeans_S_16(nnUNetTrainerSAM_KMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return KMeans_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                    deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                    num_clusters=16)

class nnUNetTrainerSAM_KMeans_S_32(nnUNetTrainerSAM_KMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return KMeans_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                    deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                    num_clusters=32)

class nnUNetTrainerSAM_KMeans_S_64(nnUNetTrainerSAM_KMeans):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        return KMeans_Rolling_Unet_S(num_classes=num_output_channels, input_channels=num_input_channels,
                                    deep_supervision=False, img_size=configuration_manager.patch_size[0],
                                    num_clusters=64)


# =========================================================================================
# 6. Backward-Compatible & Convenience Infix Aliases
# =========================================================================================

# Infix aliases (e.g. nnUNetTrainerGBC_SAM_S_4)
nnUNetTrainerGBC_SAM_S_2 = nnUNetTrainerSAM_GBC_S_2
nnUNetTrainerGBC_SAM_S_4 = nnUNetTrainerSAM_GBC_S_4
nnUNetTrainerGBC_SAM_S_8 = nnUNetTrainerSAM_GBC_S_8
nnUNetTrainerGBC_SAM_S_16 = nnUNetTrainerSAM_GBC_S_16
nnUNetTrainerGBC_SAM_S_32 = nnUNetTrainerSAM_GBC_S_32
nnUNetTrainerGBC_SAM_S_64 = nnUNetTrainerSAM_GBC_S_64

nnUNetTrainerDGBC_SAM_S_2 = nnUNetTrainerSAM_DGBC_S_2
nnUNetTrainerDGBC_SAM_S_4 = nnUNetTrainerSAM_DGBC_S_4
nnUNetTrainerDGBC_SAM_S_8 = nnUNetTrainerSAM_DGBC_S_8
nnUNetTrainerDGBC_SAM_S_16 = nnUNetTrainerSAM_DGBC_S_16
nnUNetTrainerDGBC_SAM_S_32 = nnUNetTrainerSAM_DGBC_S_32
nnUNetTrainerDGBC_SAM_S_64 = nnUNetTrainerSAM_DGBC_S_64

nnUNetTrainerKMeans_SAM_S_2 = nnUNetTrainerSAM_KMeans_S_2
nnUNetTrainerKMeans_SAM_S_4 = nnUNetTrainerSAM_KMeans_S_4
nnUNetTrainerKMeans_SAM_S_8 = nnUNetTrainerSAM_KMeans_S_8
nnUNetTrainerKMeans_SAM_S_16 = nnUNetTrainerSAM_KMeans_S_16
nnUNetTrainerKMeans_SAM_S_32 = nnUNetTrainerSAM_KMeans_S_32
nnUNetTrainerKMeans_SAM_S_64 = nnUNetTrainerSAM_KMeans_S_64

# FT aliases
nnUNetTrainerKMeans_FT_S_4 = nnUNetTrainerSAM_KMeans_S_4
nnUNetTrainerKMeans_FT_S_16 = nnUNetTrainerSAM_KMeans_S_16
nnUNetTrainerKMeans_FT_S_32 = nnUNetTrainerSAM_KMeans_S_32
