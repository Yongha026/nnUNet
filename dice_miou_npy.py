import numpy as np
import glob
import argparse
import os
from tqdm import tqdm
import cv2
from medpy.metric.binary import hd95

def fit_ellipse_to_mask(binary_mask):
    """
    이진 마스크에서 가장 큰 객체를 찾아 타원 피팅을 수행하고 새로운 마스크를 반환합니다.
    """
    img = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.zeros_like(binary_mask, dtype=bool)

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 5:
        return binary_mask.astype(bool)

    hull = cv2.convexHull(largest_contour)
    ellipse = cv2.fitEllipse(hull)

    fitted_mask = np.zeros_like(img)
    cv2.ellipse(fitted_mask, ellipse, color=255, thickness=-1)

    return (fitted_mask > 0)


def calculate_dice_iou(mask1, mask2):
    """
    이미 이진화(bool)된 두 마스크 간의 Dice 및 IoU를 계산합니다.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum()
    cup = np.logical_or(mask1, mask2).sum()

    epsilon = 1e-6
    dice = (2. * intersection + epsilon) / (union + epsilon)
    iou = (intersection + epsilon) / (cup + epsilon)
    return dice, iou


def compute_metrics(pred, gt):
    """
    이진화된 두 마스크에 대해 Dice, IoU, HD95를 일괄 계산합니다.
    """
    dsc, iou = calculate_dice_iou(pred, gt)
    if pred.sum() > 0 and gt.sum() > 0:
        h_dist = hd95(pred, gt)
    else:
        h_dist = np.nan
    return dsc, iou, h_dist


def print_metric_results(title, dices, ious, hd95s):
    """
    수집된 평가 지표 리스트의 평균을 포맷팅하여 출력합니다.
    """
    print(f"\n==================== {title} ====================")
    print(f"DSC:   {np.nanmean(dices):.4f}")
    print(f"mIOU:  {np.nanmean(ious):.4f}")
    print(f"HD95:  {np.nanmean(hd95s):.4f} (Valid: {np.count_nonzero(~np.isnan(hd95s))}/{len(hd95s)})")


parser = argparse.ArgumentParser()
parser.add_argument("test_dir")
parser.add_argument("GT_dir")
parser.add_argument("--pupil_only", action="store_true", help="Calculate metrics only for pupil class (class ID 3)")
parser.add_argument("--fit_ellipse", action="store_true",
                    help="Apply ellipse fitting when running in single mode")
parser.add_argument("--not_both", action="store_true",
                    help="Disable running both modes; only run single mode specified by --fit_ellipse")
args = parser.parse_args()

test_msks = sorted(glob.glob(os.path.join(args.test_dir, "*.npy")))
gt_msks = sorted(glob.glob(os.path.join(args.GT_dir, "*.npy")))

test_dict = {os.path.basename(p): p for p in test_msks}
gt_dict = {os.path.basename(p): p for p in gt_msks}

common_basenames = set(test_dict.keys()).intersection(set(gt_dict.keys()))

pruned_test_masks = [test_dict[base] for base in sorted(common_basenames)]
pruned_gt_masks = [gt_dict[base] for base in sorted(common_basenames)]

print("##### Before Pruning #####")
print(f"{len(gt_msks)} GT msks")
print(f"{len(test_msks)} test msks")
print("##### After Pruning #####")
print(f"{len(pruned_gt_masks)} GT msks")
print(f"{len(pruned_test_masks)} test msks")

run_both = not args.not_both

if args.pupil_only:
    mode_str = "Both (w/o and w/ Ellipse Fitting)" if run_both else ("w/ Ellipse Fitting" if args.fit_ellipse else "w/o Ellipse Fitting")
    print(f"Evaluating only for Pupil (Class ID: 3) [{mode_str}]...\n")

if run_both:
    raw_dices, raw_ious, raw_hd95s = [], [], []
    fit_dices, fit_ious, fit_hd95s = [], [], []
else:
    dices, ious, hd95s = [], [], []

# for i in tqdm(range(len(pruned_gt_masks))):
for i in range(len(pruned_gt_masks)):
    raw_pred = np.load(pruned_test_masks[i])
    raw_gt = np.load(pruned_gt_masks[i])

    h, w = raw_gt.shape[:2]
    raw_pred = cv2.resize(raw_pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # 1. 대상 클래스 이진화 (bool)
    if args.pupil_only:
        pred_raw = (raw_pred == 3)
        gt_bin = (raw_gt == 3)
    else:
        pred_raw = (raw_pred > 0)
        gt_bin = (raw_gt > 0)

    # 2. 메트릭 연산 분기
    if run_both:
        # (1) 타원 피팅 미적용 메트릭 계산
        d_raw, i_raw, h_raw = compute_metrics(pred_raw, gt_bin)
        raw_dices.append(d_raw)
        raw_ious.append(i_raw)
        raw_hd95s.append(h_raw)

        # (2) 타원 피팅 적용 메트릭 계산 (pupil_only 기준)
        pred_fit = fit_ellipse_to_mask(pred_raw) if args.pupil_only else pred_raw
        d_fit, i_fit, h_fit = compute_metrics(pred_fit, gt_bin)
        fit_dices.append(d_fit)
        fit_ious.append(i_fit)
        fit_hd95s.append(h_fit)
    else:
        # 단일 모드 실행
        pred_eval = fit_ellipse_to_mask(pred_raw) if (args.fit_ellipse and args.pupil_only) else pred_raw
        dsc, iou, hd_val = compute_metrics(pred_eval, gt_bin)
        dices.append(dsc)
        ious.append(iou)
        hd95s.append(hd95_val)

# 3. 최종 결과 출력
if run_both:
    print_metric_results("Without Ellipse Fitting", raw_dices, raw_ious, raw_hd95s)
    print_metric_results("With Ellipse Fitting", fit_dices, fit_ious, fit_hd95s)
else:
    mode_name = "With Ellipse Fitting" if args.fit_ellipse else "Without Ellipse Fitting"
    print_metric_results(mode_name, dices, ious, hd95s)