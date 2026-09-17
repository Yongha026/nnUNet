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

import sys
import nnunetv2.run.load_pretrained_weights as lpw


def load_pretrained_weights_backbone_only(network, fname, verbose=False):
    """
    Transfers backbone weights between matching keys in state_dicts.
    Pretrained clustering parameters ('kmeans_block.', 'gbc.') and segmentation heads ('.seg_layers.')
    are strictly excluded and discarded, avoiding any key or shape mismatch
    and allowing fresh hyperspherical KMeans parameter initialization.
    """
    from torch._dynamo import OptimizedModule
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

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
    # verify that all backbone layers have matching keys and shapes
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


class nnUNetTrainerKMeans(nnUNetTrainer):
    """
    nnU-Net Trainer for Soft K-Means Rolling-UNet fine-tuning.
    Strictly aligned with nnUNeTrainerGBC_FT:
      - Frozen backbone (encoder/decoder), only KMeans parameters trainable
      - initial_lr: 5e-5
      - num_epochs: 50
      - AdamW optimizer with CosineAnnealingLR
      - Combined loss: L_total = L_seg + 0.1 * L_div + 0.1 * L_inertia
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Deep supervision is disabled because Soft K-Means Rolling-UNet outputs a single resolution segmentation
        self.enable_deep_supervision = False

        # Custom hyperparameters strictly aligned to GBC fine-tuning
        self.initial_lr = 5e-5
        self.weight_decay = 0.01
        self.num_epochs = 50  # Fine-tune for 50 epochs matching GBC FT

        # Ensure W&B run name is set and WandbLogger is attached if nnUNet_wandb_name is exported
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

    def _get_actual_network(self) -> nn.Module:
        net = self.network
        if hasattr(net, 'module'):  # DDP wrapping
            net = net.module
        if isinstance(net, OptimizedModule):  # torch.compile wrapping
            net = net._orig_mod
        return net

    def freeze_backbone(self):
        """Freezes all non-KMeans backbone parameters and sets their BatchNorm layers to eval mode."""
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
        # Enforce backbone freezing before registering optimizer parameters
        self.freeze_backbone()
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in self.network.parameters())
        self.print_to_log_file(
            f"[*] KMeans Fine-Tuning Setup: {total_count - trainable_count:,} backbone params frozen, "
            f"{trainable_count:,} KMeans params trainable."
        )

        # AdamW for hybrid architecture with only trainable KMeans parameters
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
        # Re-enforce eval mode on frozen modules to prevent BatchNorm running stats drift
        self.freeze_backbone()

    def set_deep_supervision_enabled(self, enabled: bool):
        pass

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

                # Total loss = L_seg + 0.1 * L_div + 0.1 * L_inertia (matching GBC regularizer weighting)
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


# =========================================================================================
# Subclasses for Different Cluster Counts K (Aligned directly with nnUNetTrainerDGBC_S_*)
# =========================================================================================

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


# Backward-compatible aliases for fine-tuning naming
nnUNetTrainerKMeans_FT_S_4 = nnUNetTrainerKMeans_S_4
nnUNetTrainerKMeans_FT_S_16 = nnUNetTrainerKMeans_S_16
nnUNetTrainerKMeans_FT_S_32 = nnUNetTrainerKMeans_S_32
