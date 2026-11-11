"""
check_flops.py — Calculate Parameters, Model Size (MB), GMACs, and GFLOPs for nnU-Net v2 trainers.

Parses standard nnU-Net v2 CLI arguments (-d, -c, -f, -tr, -p) to resolve the dataset's
plans.json and dataset.json (from nnUNet_results or nnUNet_preprocessed), determines the
appropriate preprocessed image/patch size (configuration_manager.patch_size), builds the
trainer's network architecture, and profiles its computational complexity.

Usage (on remote GPU server):
    # Profile a single trainer using the dataset's preprocessed patch size
    python check_flops.py -d 250 -c 2d -f 0 -tr nnUNetTrainerGBC_S_16

    # Compare multiple trainers side-by-side in a summary table
    python check_flops.py -d 250 -c 2d -f 0 -tr nnUNetTrainer nnUNetTrainerUNet nnUNetTrainerRoll_S nnUNetTrainerGBC_S_16 nnUNetTrainerDGBC_S_16 nnUNetTrainerKMeans_S_16 nnUNetTrainer_Next nnUNetTrainerUKAN nnUNetTrainerRIT

    # Override spatial resolution explicitly (e.g., 192x192 or 448x448)
    python check_flops.py -d 250 -c 2d -f 0 -tr nnUNetTrainerGBC_S_16 --img_size 192 192
"""

import argparse
import copy
import importlib
import inspect
import os
import pkgutil
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from batchgenerators.utilities.file_and_folder_operations import isdir, isfile, join, load_json

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import nnunetv2
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate GFLOPs, GMACs, and parameter counts for nnU-Net v2 models using -d -c -f -tr."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="d",
        type=str,
        required=True,
        help="Dataset ID (e.g., 250) or name (e.g., Dataset250_OpenEDS2019).",
    )
    parser.add_argument(
        "-c",
        "--configuration",
        dest="c",
        type=str,
        default="2d",
        help="nnU-Net configuration name (e.g., 2d, 3d_fullres). Default: 2d",
    )
    parser.add_argument(
        "-f",
        "--fold",
        dest="f",
        nargs="+",
        type=str,
        default=["0"],
        help="Fold index (e.g., 0, 1, ..., or all). Default: 0",
    )
    parser.add_argument(
        "-tr",
        "--trainer",
        dest="tr",
        nargs="+",
        type=str,
        default=["nnUNetTrainer"],
        help="One or more nnU-Net trainer class names (e.g., nnUNetTrainerGBC_S_16). Default: nnUNetTrainer",
    )
    parser.add_argument(
        "-p",
        "--plans",
        dest="p",
        type=str,
        default="nnUNetPlans",
        help="Plans identifier (default: nnUNetPlans) or path to a plans.json file.",
    )
    parser.add_argument(
        "-chk",
        "--checkpoint",
        dest="chk",
        type=str,
        default="checkpoint_final.pth",
        help="Checkpoint filename inside fold_<f> to inspect if present (default: checkpoint_final.pth).",
    )
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        default=None,
        help="Optional spatial size override (e.g., --img_size 192 192). Defaults to configuration patch_size from plans.json.",
    )
    parser.add_argument(
        "-device",
        "--device",
        dest="device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for profiling ('cuda' or 'cpu'). Default: auto-detect.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display detailed per-layer FLOP breakdown from PyTorch FlopCounterMode.",
    )
    return parser.parse_args()


def find_trainer_class_safe(trainer_name: str):
    """
    Locates a trainer class by name in nnunetv2.training.nnUNetTrainer.
    Falls back to fault-tolerant module scanning if an unrelated trainer file raises
    an ImportError (e.g., mamba_ssm missing when not profiling Mamba trainers).
    """
    try:
        return recursive_find_trainer_class_by_name(trainer_name)
    except (ModuleNotFoundError, ImportError, RuntimeError):
        trainer_dir = join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
        if trainer_dir not in sys.path:
            sys.path.insert(0, trainer_dir)

        missing_dep_errors = []
        for _, modname, ispkg in pkgutil.iter_modules([trainer_dir]):
            if ispkg:
                continue
            full_mod = f"nnunetv2.training.nnUNetTrainer.{modname}"
            try:
                module = importlib.import_module(full_mod)
            except (ModuleNotFoundError, ImportError) as e:
                missing_dep_errors.append((modname, str(e)))
                continue
            if hasattr(module, trainer_name):
                return getattr(module, trainer_name)

        err_detail = "; ".join([f"{m}: {err}" for m, err in missing_dep_errors])
        raise RuntimeError(
            f"Could not find trainer class '{trainer_name}' in {trainer_dir}. "
            f"(Skipped modules with missing dependencies: {err_detail})"
        )


def resolve_dataset_name(dataset_name_or_id: str) -> str:
    """Converts dataset ID or name to DatasetXXX_Name using nnU-Net's converter with directory fallback."""
    try:
        return maybe_convert_to_dataset_name(dataset_name_or_id)
    except Exception:
        if dataset_name_or_id.startswith("Dataset"):
            return dataset_name_or_id
        try:
            ds_id = int(dataset_name_or_id)
            prefix = f"Dataset{ds_id:03d}_"
            for base_dir in [nnUNet_results, nnUNet_preprocessed]:
                if base_dir and isdir(base_dir):
                    for entry in sorted(os.listdir(base_dir)):
                        if entry.startswith(prefix) and isdir(join(base_dir, entry)):
                            return entry
        except ValueError:
            pass
        return dataset_name_or_id


def resolve_plans_and_dataset_json(
    dataset_name_or_id: str,
    trainer_name: str,
    plans_identifier: str,
    configuration: str,
    fold: Union[int, str],
    checkpoint_name: str,
) -> Tuple[dict, dict, str, Optional[str]]:
    """
    Resolves plans.json and dataset.json following the nnU-Net v2 pipeline:
      1. Direct path if -p points to a .json file.
      2. Trained model output folder: $nnUNet_results/<Dataset>/<tr__p__c>/
      3. Preprocessed dataset folder: $nnUNet_preprocessed/<Dataset>/
      4. Any sibling trained model folder under $nnUNet_results/<Dataset>/
      5. Local fallback: ./nnUNetPlans.json in repository root.
    Returns (plans_dict, dataset_json_dict, source_description, checkpoint_path_if_found).
    """
    dataset_name = resolve_dataset_name(dataset_name_or_id)
    chk_found_path: Optional[str] = None

    # Check if trained model folder & checkpoint exist
    try:
        model_folder = get_output_folder(dataset_name_or_id, trainer_name, plans_identifier, configuration)
    except Exception:
        model_folder = (
            join(nnUNet_results, dataset_name, f"{trainer_name}__{plans_identifier}__{configuration}")
            if nnUNet_results
            else None
        )

    if model_folder and isdir(model_folder):
        fold_dir = join(model_folder, f"fold_{fold}")
        for cand_chk in [checkpoint_name, "checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"]:
            cand_path = join(fold_dir, cand_chk)
            if isfile(cand_path):
                chk_found_path = cand_path
                break

        plans_path = join(model_folder, "plans.json")
        dataset_path = join(model_folder, "dataset.json")
        if isfile(plans_path) and isfile(dataset_path):
            return load_json(plans_path), load_json(dataset_path), model_folder, chk_found_path

    # 1. Explicit plans file path
    if isfile(plans_identifier):
        plans = load_json(plans_identifier)
        cand_ds = join(os.path.dirname(os.path.abspath(plans_identifier)), "dataset.json")
        if isfile(cand_ds):
            dataset_json = load_json(cand_ds)
        else:
            dataset_json = {
                "channel_names": {"0": "0"},
                "labels": {"background": 0, "sclera": 1, "iris": 2, "pupil": 3},
            }
        return plans, dataset_json, os.path.abspath(plans_identifier), chk_found_path

    # 2. Preprocessed dataset folder ($nnUNet_preprocessed/<Dataset>/)
    if nnUNet_preprocessed and isdir(join(nnUNet_preprocessed, dataset_name)):
        prep_dir = join(nnUNet_preprocessed, dataset_name)
        plans_path = join(prep_dir, f"{plans_identifier}.json")
        dataset_path = join(prep_dir, "dataset.json")
        if isfile(plans_path) and isfile(dataset_path):
            return load_json(plans_path), load_json(dataset_path), prep_dir, chk_found_path

    # 3. Sibling model folder under $nnUNet_results/<Dataset>/
    if nnUNet_results and isdir(join(nnUNet_results, dataset_name)):
        ds_results_dir = join(nnUNet_results, dataset_name)
        for sub in sorted(os.listdir(ds_results_dir)):
            sub_dir = join(ds_results_dir, sub)
            plans_path = join(sub_dir, "plans.json")
            dataset_path = join(sub_dir, "dataset.json")
            if isdir(sub_dir) and isfile(plans_path) and isfile(dataset_path):
                return load_json(plans_path), load_json(dataset_path), sub_dir, chk_found_path

    # 4. Repository root fallback (./nnUNetPlans.json)
    local_plans = join(repo_root, "nnUNetPlans.json")
    if isfile(local_plans):
        print(
            f"[WARN] Could not find dataset '{dataset_name}' in nnUNet_results or nnUNet_preprocessed. "
            f"Falling back to local {local_plans}."
        )
        plans = load_json(local_plans)
        dataset_json = {
            "channel_names": {"0": "0"},
            "labels": {"background": 0, "sclera": 1, "iris": 2, "pupil": 3},
        }
        return plans, dataset_json, local_plans, chk_found_path

    raise FileNotFoundError(
        f"Could not locate plans.json and dataset.json for dataset '{dataset_name_or_id}' "
        f"(resolved: '{dataset_name}'). Checked nnUNet_results={nnUNet_results} and "
        f"nnUNet_preprocessed={nnUNet_preprocessed}."
    )


def build_model_from_trainer(
    trainer_class,
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    num_input_channels: int,
    num_output_channels: int,
) -> nn.Module:
    """Instantiates the network architecture exactly as nnUNetTrainer.initialize / nnUNetPredictor does."""
    sig = inspect.signature(trainer_class.build_network_architecture)
    if "plans_manager" in sig.parameters:
        network = trainer_class.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=False,
        )
    else:
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision=False,
        )

    if hasattr(network, "decoder") and hasattr(network.decoder, "deep_supervision"):
        network.decoder.deep_supervision = False

    return network


def count_parameters_and_size(network: nn.Module, trainer_name: str) -> Dict[str, Union[int, float]]:
    """Computes total parameters, trainable parameters, clustering head parameters, and memory size in MB."""
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

    # Identify clustering module parameters (GranularBall / KMeansBlock) if present
    clustering_params = 0
    for name, module in network.named_children():
        if name in ("gbc", "kmeans_block") or module.__class__.__name__ in ("GranularBall", "KMeansBlock"):
            clustering_params += sum(p.numel() for p in module.parameters())

    # In SAM_* fine-tuning trainers, backbone is frozen during on_train_start() and only gbc/kmeans_block (+ final) train
    is_frozen_backbone_ft = any(tag in trainer_name for tag in ("SAM_", "_SAM_", "_FT_"))
    if is_frozen_backbone_ft and clustering_params > 0:
        final_params = sum(p.numel() for p in network.final.parameters()) if hasattr(network, "final") else 0
        effective_ft_params = clustering_params + final_params
    else:
        effective_ft_params = trainable_params

    param_bytes = sum(p.numel() * p.element_size() for p in network.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in network.buffers())
    size_fp32_mb = (param_bytes + buffer_bytes) / (1024.0 ** 2)
    size_fp16_mb = size_fp32_mb / 2.0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "effective_ft_params": effective_ft_params,
        "clustering_params": clustering_params,
        "size_fp32_mb": size_fp32_mb,
        "size_fp16_mb": size_fp16_mb,
    }


def compute_flops_and_macs(
    network: nn.Module,
    dummy_input: torch.Tensor,
    verbose: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Computes GFLOPs and GMACs for a single forward pass using:
      1. PyTorch built-in torch.utils.flop_counter.FlopCounterMode (ATen dispatch level: 1 MAC = 2 FLOPs)
         + hook for GranularBall element-wise broadcast distance ops.
      2. Optional thop.profile if 'thop' is installed in the environment.
    """
    from torch.utils.flop_counter import FlopCounterMode

    network.eval()

    # Track element-wise broadcast distance FLOPs inside GranularBall (dif, dif/sigma, dif^2, sum)
    gbc_extra_flops = 0
    hooks = []

    def make_gbc_hook(mod):
        def _hook(m, inputs, outputs):
            nonlocal gbc_extra_flops
            if inputs and isinstance(inputs[0], torch.Tensor):
                x = inputs[0]
                B, C, H, W = x.shape
                N = H * W
                d = getattr(m, "proj_dim", C)
                K = getattr(m, "num_balls", 16)
                # dif (B*N*K*d sub) + scale (B*N*K*d div) + square (B*N*K*d mul) + sum (B*N*K*d add) = 4 * B * N * K * d FLOPs
                gbc_extra_flops += 4 * B * N * K * d
        return _hook

    for module in network.modules():
        if module.__class__.__name__ == "GranularBall":
            hooks.append(module.register_forward_hook(make_gbc_hook(module)))

    with torch.no_grad():
        with FlopCounterMode(network, display=verbose) as flop_counter:
            _ = network(dummy_input)

    for h in hooks:
        h.remove()

    aten_flops = flop_counter.get_total_flops()
    total_flops = aten_flops + gbc_extra_flops
    gflops = total_flops / 1e9
    gmacs = (total_flops / 2.0) / 1e9

    # Optional thop comparison if installed
    thop_gmacs: Optional[float] = None
    try:
        from thop import profile as thop_profile

        net_copy = copy.deepcopy(network).eval()
        with torch.no_grad():
            macs_thop, _ = thop_profile(net_copy, inputs=(dummy_input,), verbose=False)
        thop_gmacs = (macs_thop + (gbc_extra_flops / 2.0)) / 1e9
        del net_copy
    except Exception:
        thop_gmacs = None

    return {
        "gflops": gflops,
        "gmacs": gmacs,
        "gbc_extra_gflops": gbc_extra_flops / 1e9,
        "thop_gmacs": thop_gmacs,
    }


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    fold_str = args.f[0] if isinstance(args.f, list) else str(args.f)
    trainer_list: List[str] = args.tr if isinstance(args.tr, list) else [args.tr]

    summary_rows = []

    for tr_name in trainer_list:
        print("\n" + "=" * 78)
        print(f" Profiling Trainer: {tr_name}")
        print("=" * 78)

        # 1. Resolve plans.json & dataset.json
        plans, dataset_json, source_path, chk_path = resolve_plans_and_dataset_json(
            dataset_name_or_id=args.d,
            trainer_name=tr_name,
            plans_identifier=args.p,
            configuration=args.c,
            fold=fold_str,
            checkpoint_name=args.chk,
        )

        # Create a shallow copy because PlansManager or trainers may pop keys
        plans_copy = copy.deepcopy(plans)
        plans_manager = PlansManager(plans_copy)
        configuration_manager = plans_manager.get_configuration(args.c)
        label_manager = plans_manager.get_label_manager(dataset_json)

        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        num_output_channels = label_manager.num_segmentation_heads

        # 2. Resolve preprocessed image / patch size
        native_patch_size = list(configuration_manager.patch_size)
        median_size = getattr(configuration_manager, "median_image_size_in_voxels", native_patch_size)

        if args.img_size is not None:
            if len(args.img_size) == 1:
                input_spatial_size = [args.img_size[0]] * len(native_patch_size)
            else:
                input_spatial_size = list(args.img_size)
            # Update configuration_manager patch_size so build_network_architecture receives the override
            configuration_manager.configuration["patch_size"] = input_spatial_size
        else:
            input_spatial_size = native_patch_size

        # 3. Locate Trainer Class & Build Network
        trainer_class = find_trainer_class_safe(tr_name)
        network = build_model_from_trainer(
            trainer_class=trainer_class,
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
        ).to(device)
        network.eval()

        arch_class_name = network.__class__.__name__
        input_shape = (1, num_input_channels, *input_spatial_size)
        dummy_input = torch.randn(*input_shape, device=device)

        # 4. Compute Parameter Counts & Model Memory Size
        param_stats = count_parameters_and_size(network, tr_name)

        # 5. Compute GFLOPs & GMACs
        flop_stats = compute_flops_and_macs(network, dummy_input, verbose=args.verbose)

        # 6. Print Detailed Report
        print(f"  Dataset               : {plans_manager.dataset_name} (config: {args.c}, fold: {fold_str})")
        print(f"  Plans Source          : {source_path}")
        if chk_path:
            chk_mb = os.path.getsize(chk_path) / (1024.0 ** 2)
            print(f"  Trained Checkpoint    : {chk_path} ({chk_mb:.2f} MB on disk)")
        else:
            print(f"  Trained Checkpoint    : Not required / not found (profiled from architecture & plans)")
        print(f"  Trainer Class         : {tr_name} -> Architecture: {arch_class_name}")
        print(f"  Preprocessed Patch    : {native_patch_size} (median voxel size: {list(median_size)})")
        print(f"  Profiled Input Shape  : {input_shape} (in_ch={num_input_channels}, classes={num_output_channels})")
        print("-" * 78)
        print(
            f"  Total Parameters      : {param_stats['total_params']:,} "
            f"({param_stats['total_params'] / 1e6:.4f} M | {param_stats['total_params'] / 1e3:.2f} K)"
        )
        if param_stats["clustering_params"] > 0:
            print(
                f"  Clustering Parameters : {param_stats['clustering_params']:,} "
                f"(GBC / KMeans module)"
            )
        if param_stats["effective_ft_params"] != param_stats["total_params"]:
            print(
                f"  Active FT Parameters  : {param_stats['effective_ft_params']:,} "
                f"(when backbone frozen in {tr_name})"
            )
        print(
            f"  Model Weights Size    : {param_stats['size_fp32_mb']:.2f} MB (FP32) | "
            f"{param_stats['size_fp16_mb']:.2f} MB (FP16)"
        )
        print(f"  GFLOPs (PyTorch)      : {flop_stats['gflops']:.4f} GFLOPs")
        print(f"  GMACs  (PyTorch)      : {flop_stats['gmacs']:.4f} GMACs  (1 MAC = 2 FLOPs)")
        if flop_stats["thop_gmacs"] is not None:
            print(f"  GMACs  (thop)         : {flop_stats['thop_gmacs']:.4f} GMACs")
        print("=" * 78)

        summary_rows.append(
            {
                "trainer": tr_name,
                "arch": arch_class_name,
                "input_shape": "x".join(map(str, input_shape)),
                "params_m": param_stats["total_params"] / 1e6,
                "params_raw": param_stats["total_params"],
                "size_mb": param_stats["size_fp32_mb"],
                "gmacs": flop_stats["gmacs"],
                "gflops": flop_stats["gflops"],
            }
        )

        del network, dummy_input
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(summary_rows) > 1:
        print("\n" + "=" * 98)
        print(
            f"{'Trainer':<30} | {'Architecture':<22} | {'Input':<13} | {'Params (M)':>10} | {'GMACs':>9} | {'GFLOPs':>9}"
        )
        print("-" * 98)
        for r in summary_rows:
            print(
                f"{r['trainer']:<30} | {r['arch']:<22} | {r['input_shape']:<13} | "
                f"{r['params_m']:>10.4f} | {r['gmacs']:>9.4f} | {r['gflops']:>9.4f}"
            )
        print("=" * 98)


if __name__ == "__main__":
    main()
