"""
tsne.py — Visualise UNet deep encoder processed features via t-SNE.

Plots pixel-wise t-SNE clustering for encoder processed feature elements only
(no GBC ball centers or ellipses).

Usage
-----
python feature_cluster_exp/tsne.py  /path/to/images  /path/to/checkpoint.pth  [options]

Options
-------
--datas             Number of images to sample (default: 1024)
--samples_per_class Max points per class for t-SNE (default: 2000)
--pupil_only        Binary: Pupil vs Else (default)
--all_classes       Full 4-class: Background / Sclera / Iris / Pupil
--output            Output path (default: auto-generated in tsne_results/)
--perplexity        t-SNE perplexity (default: 30)
--seed              Random seed (default: 42)
"""

import torch
import os
import cv2
import sys
import numpy as np
import glob
import argparse
import random
import re

import torchvision
import PIL.Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ─── path setup ──────────────────────────────────────────────────────────────
exp_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(exp_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


# ─── dataset ─────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.table = float(255) * (np.linspace(0, 1, 256) ** 0.8)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Image
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)
        img_gamma = cv2.LUT(
            img_resized.astype(np.uint8), self.table.astype(np.uint8)
        )
        img_clahe = self.clahe.apply(img_gamma)
        pil_img = PIL.Image.fromarray(img_clahe)

        # Mask
        msk_path = (
            self.image_paths[idx].replace("images", "labels").replace("png", "npy")
        )

        msk = np.load(msk_path).astype(np.uint8)
        msk_resized = cv2.resize(msk, (48, 48), interpolation=cv2.INTER_NEAREST)
        return self.transform(pil_img), msk_resized


# ─── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise UNet deep encoder processed features using t-SNE"
    )
    parser.add_argument("IMG_PATH", type=str, help="Path to image folder")
    parser.add_argument("MODEL_PATH", type=str, help="Path to adgbc checkpoint")
    parser.add_argument(
        "--datas", default=1024, type=int, help="Number of images to sample"
    )
    parser.add_argument(
        "--samples_per_class",
        default=2000,
        type=int,
        help="Max points per class for t-SNE",
    )
    # Class toggle — default is --pupil_only
    class_group = parser.add_mutually_exclusive_group()
    class_group.add_argument(
        "--pupil_only",
        action="store_true",
        default=True,
        help="Binary: Pupil vs Else (default)",
    )
    class_group.add_argument(
        "--all_classes",
        action="store_true",
        default=False,
        help="Full 4-class: Background / Sclera / Iris / Pupil",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Output path (default: auto-generated in tsne_results/)",
    )
    parser.add_argument(
        "--perplexity", default=30, type=int, help="t-SNE perplexity"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--center",
        action="store_true",
        default=False,
        help="Visualize GBC cluster centers on the t-SNE plot",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        default=False,
        help="Use PCA dimensionality reduction and visualize centers as stars (*)",
    )
    parser.add_argument(
        "--untrained", "-u",
        action="store_true",
        default=False,
        help="Inference with Random initialized model(Linear: He, Conv: Xavier, else: normal)"
    )
    parser.add_argument(
        "--dec",
        action="store_true",
        default=False,
        help="Visualize Decoder features; After Cluster refinement addition"
    )
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model_path_adgbc = args.MODEL_PATH
    ADGBC_match = "nnUNetTrainerGBC_" in args.MODEL_PATH
    DGBC_match = "nnUNetTrainerDGBC_" in args.MODEL_PATH
    KMeans_match = "nnUNetTrainerKMeans_" in args.MODEL_PATH

    if ADGBC_match:
        model_prefix = "ADGBC"
        use_diag_cov = True
    elif DGBC_match:
        model_prefix = "DGBC"
        use_diag_cov = False
    else: model_prefix = "KMeans"

    untrained_prefix = "UNTRAINED_" if args.untrained else ""
    model_prefix = untrained_prefix + model_prefix

    # ── dataset validation ────────────────────────────────────────────────
    # IMG_PATH must contain exactly one of "OpenEDS2019" or "jw_"
    EDS_match = "Openedsdata2019" in args.IMG_PATH
    Pupil_match = "jw_" in args.IMG_PATH
    Swir_match = "nnunetv2_swir" in args.IMG_PATH

    if EDS_match and Pupil_match and Swir_match:
        parser.error(
            "IMG_PATH contains all three: 'OpenEDS2019', 'jw_', 'nnunetv2_swir. "
            "Please specify a path belonging to only one dataset."
        )
    if not EDS_match and not Pupil_match and not Swir_match:
        parser.error(
            "IMG_PATH must contain either 'OpenEDS2019', 'jw_' or 'nnunetv2_swir' "
            "to identify the dataset."
        )

    if not os.path.exists(model_path_adgbc):
        parser.error(f"ADGBC checkpoint file not found at: {model_path_adgbc}")

    if EDS_match: dataset_prefix = "OpenEDS2019"
    elif Pupil_match: dataset_prefix = "PupilLabs"
    else: dataset_prefix = "Swirski"

    # If --all_classes is set, override pupil_only
    if args.all_classes:
        args.pupil_only = False

    # Extract gbc_num_balls from model filename if present (e.g. GBC_S_16__)
    match = re.search(r"_S_(2|4|8|16|32|64)(?=__|$)", str(model_path_adgbc))
    gbc_num_balls = int(match.group(1)) if match else 32
    print(f"[INFO] Checkpoint has {gbc_num_balls} clusters")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ─── model loading ───────────────────────────────────────────────────────────
    def initialize_weights(m):
        # 선형(Linear) 계층인 경우
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # He(Kaiming) 초기화
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)  # 편향은 0으로 초기화

        # 합성곱(Conv2d, ConvTranspose2d) 계층인 경우
        elif isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.xavier_normal_(m.weight)  # Xavier 초기화
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

        # 정규화(BatchNorm, LayerNorm, GroupNorm) 계층인 경우
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
            if getattr(m, 'weight', None) is not None:
                torch.nn.init.constant_(m.weight, 1.0)
            if getattr(m, 'bias', None) is not None:
                torch.nn.init.constant_(m.bias, 0.0)

        # GranularBall 계층인 경우
        elif type(m).__name__ == "GranularBall":
            if hasattr(m, 'centers') and m.centers is not None:
                torch.nn.init.normal_(m.centers, 0, 0.01)
            if hasattr(m, 'log_sigma') and m.log_sigma is not None:
                torch.nn.init.zeros_(m.log_sigma)
            if hasattr(m, 'log_radius') and m.log_radius is not None:
                torch.nn.init.zeros_(m.log_radius)

        # 그 외 weight 텐서를 직접 가진 파라미터 계층만 normal dist로 초기화
        elif hasattr(m, 'weight') and isinstance(getattr(m, 'weight', None), torch.Tensor) and m.weight is not None:
            torch.nn.init.normal_(m.weight, 0, 0.02)
            if getattr(m, 'bias', None) is not None and isinstance(m.bias, torch.Tensor):
                torch.nn.init.constant_(m.bias, 0.0)

    from nnunetv2.training.nnUNetTrainer.ADGBC_encoder import GBC_S_EncDec
    from nnunetv2.training.nnUNetTrainer.archs_K_means_UNet import Kmeans_encoder
    try:
        if not KMeans_match:
            model = GBC_S_EncDec(
                num_classes=4, input_channels=1, deep_supervision=False, gbc_num_balls=gbc_num_balls, use_diag_cov=use_diag_cov
            ).to(device)
        else:
            model = Kmeans_encoder(
                num_classes=4, input_channels=1, deep_supervision=False, gbc_num_balls=gbc_num_balls
            )
        checkpoint = torch.load(
            model_path_adgbc, map_location=device, weights_only=False
        )
        state_dict = (
            checkpoint["network_weights"]
            if (isinstance(checkpoint, dict) and "network_weights" in checkpoint)
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"Error loading adgbc: {e}")
        raise e

    if args.untrained:
        print("[INFO] Initializing model ...")
        model.apply(initialize_weights)

    model.eval().to(device)

    # ── image sampling ────────────────────────────────────────────────────
    image_path = os.path.join(args.IMG_PATH, "*.png")
    full_images = glob.glob(image_path)
    try:
        rand_images = random.sample(full_images, args.datas)
    except ValueError:
        rand_images = full_images
    print(f"[INFO] Dataset: {dataset_prefix} — Using {len(rand_images)} images from {args.IMG_PATH}")

    dataset = ImageDataset(rand_images)
    dataloader = DataLoader(
        dataset, batch_size=16, num_workers=4, pin_memory=True
    )

    NUM_CLASSES = 4
    SAMPLES_PER_CLASS = args.samples_per_class

    feats_by_class = {c: [] for c in range(NUM_CLASSES)}

    # ── feature extraction ────────────────────────────────────────────────
    if not args.dec:
        print("[INFO] Extracting encoder processed features …")
    else:print("[INFO] Extracting decoder refined features …")

    with torch.no_grad():
        for batch_imgs, batch_masks in tqdm(dataloader, desc="Extracting"):
            batch_imgs = batch_imgs.to(device)

            # Extract encoder processed feature (enc_feature)
            if args.dec:
                _, _, enc = model(batch_imgs)  # [B, 64, 48, 48]
            else:
                _, enc, _ = model(batch_imgs)  # [B, 64, 48, 48]

            B, C, H, W = enc.shape

            # Flatten to [B*H*W, C]
            enc_pixels = enc.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            labels_pixels = batch_masks.reshape(-1).numpy()

            # Per-class accumulation with cap
            for c in range(NUM_CLASSES):
                current_len = sum(len(x) for x in feats_by_class[c])
                if current_len < SAMPLES_PER_CLASS:
                    c_mask = labels_pixels == c
                    c_feats = enc_pixels[c_mask]
                    if len(c_feats) > 0:
                        feats_by_class[c].append(c_feats)

    # ── balance & merge ───────────────────────────────────────────────────
    selected_feats, selected_labels = [], []

    for c in range(NUM_CLASSES):
        if len(feats_by_class[c]) == 0:
            continue
        c_all = np.vstack(feats_by_class[c])

        if len(c_all) > SAMPLES_PER_CLASS:
            idx = np.random.choice(len(c_all), SAMPLES_PER_CLASS, replace=False)
            c_all = c_all[idx]

        selected_feats.append(c_all)
        selected_labels.append(np.full(len(c_all), c))

    X = np.vstack(selected_feats)    # [N_total, C]
    y = np.concatenate(selected_labels)

    # ── Dimensionality reduction & center projection ─────────────────────────
    show_centers = args.center or args.pca
    centers = model.gbc.centers.detach().cpu().numpy() if show_centers else None

    if args.pca:
        pca = PCA(n_components=2, random_state=args.seed)
        if show_centers:
            all_points = np.vstack([X, centers])
            print(f"[INFO] Running PCA on {X.shape[0]} feature elements + {gbc_num_balls} centers …")
            emb = pca.fit_transform(all_points)
            X_embedded = emb[:len(X)]
            centers_embedded = emb[len(X):]
        else:
            print(f"[INFO] Running PCA on {X.shape[0]} feature elements …")
            X_embedded = pca.fit_transform(X)
            centers_embedded = None
        method_name = "PCA"
        marker_style = "*"
        marker_size = 140
        center_label = "PCA Centers (*)"
        var_pct = pca.explained_variance_ratio_.sum() * 100
        method_detail = f"PCA (Explained Var: {var_pct:.1f}%)"
        reduction_tag = "pca"
    else:
        tsne = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            n_jobs=-1,
            random_state=args.seed,
        )
        if show_centers:
            all_points = np.vstack([X, centers])
            print(f"[INFO] Running t-SNE on {X.shape[0]} feature elements + {gbc_num_balls} GBC centers (perplexity={args.perplexity}) …")
            emb = tsne.fit_transform(all_points)
            X_embedded = emb[:len(X)]
            centers_embedded = emb[len(X):]
        else:
            print(f"[INFO] Running t-SNE on {X.shape[0]} feature elements (perplexity={args.perplexity}) …")
            X_embedded = tsne.fit_transform(X)
            centers_embedded = None
        method_name = "t-SNE"
        marker_style = "^"
        marker_size = 80
        center_label = "GBC Centers (^)"
        method_detail = f"t-SNE (perplexity={args.perplexity})"
        reduction_tag = "tsne"

    # ── class label remapping & colormap ─────────────────────────────────
    tab10_cmap = plt.cm.get_cmap("tab10")
    if args.pupil_only:
        y_plot = np.where(y == 3, 1, 0)
        plot_cmap = ListedColormap(tab10_cmap.colors[:2])
        ticks = [0, 1]
        tick_labels = ["Else", "Pupil"]
        suffix = "pupil_only"
    else:
        y_plot = y
        plot_cmap = ListedColormap(tab10_cmap.colors[:4])
        ticks = [0, 1, 2, 3]
        tick_labels = ["Background", "Sclera", "Iris", "Pupil"]
        suffix = "all_classes"

    # ── plot ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y_plot,
        cmap=plot_cmap,
        alpha=0.4,
        s=5,
        zorder=1,
    )

    if show_centers and centers_embedded is not None:
        ball_cmap = plt.cm.get_cmap("tab20")
        for k in range(gbc_num_balls):
            color_k = ball_cmap(k % 20)
            ck = centers_embedded[k]
            plt.scatter(
                ck[0],
                ck[1],
                marker=marker_style,
                s=marker_size,
                c=[color_k],
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            plt.annotate(
                str(k),
                xy=(ck[0], ck[1]),
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="bottom",
                color="black",
                zorder=6,
            )

    cbar = plt.colorbar(scatter, ticks=ticks)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("Class ID")

    center_title = f" with {center_label}" if show_centers else ""
    Enc_or_Dec = "Decoder" if args.dec else "Encoder"
    plt.title(
        f"[{dataset_prefix}] Pixel-wise {Enc_or_Dec} Feature Clustering{center_title}\n"
        f"Processed Features • {X.shape[0]} points • {method_detail}",
        fontsize=11,
    )
    plt.tight_layout()

    # ── save ──────────────────────────────────────────────────────────────
    results_dir = os.path.join(exp_dir, "tsne_results")
    os.makedirs(results_dir, exist_ok=True)

    center_suffix = "_centers" if show_centers else ""
    if args.output:
        save_path = args.output
    else:
        enc_or_dec = "Dec" if args.dec else "Enc"
        save_path = os.path.join(
            results_dir,
            f"{model_prefix}_{dataset_prefix}_{enc_or_dec}_{gbc_num_balls}_balls_{reduction_tag}_{suffix}{center_suffix}.png",
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved → {save_path}")
