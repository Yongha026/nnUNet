import os
import shutil
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from PIL import Image
import cv2

if __name__ == '__main__':
    # 1. Update this to point to the semantic segmentation directory on your NAS
    nas_source_dir = "NAS dir here"
    dataset_id = 400
    dataset_name = f"Dataset{dataset_id}_OpenEDS2019_400"

    # Target directories on the local HDD
    output_dir = join(nnUNet_raw, dataset_name)
    imagestr = join(output_dir, 'imagesTr')
    labelstr = join(output_dir, 'labelsTr')

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    # 2. Iterate through splits and copy them to the training set
    # (nnUNet will handle validation via its internal 5-fold cross-validation)
    splits = ['train', 'validation']
    num_cases = 0
    skipped_cases = 0

    # Precompute Gamma LUT and setup CLAHE outside the loops for performance
    invGamma = 1.0 / 0.8
    gamma_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

    for split in splits:
        split_img_dir = join(nas_source_dir, split, 'images')
        split_lbl_dir = join(nas_source_dir, split, 'labels')

        if not os.path.isdir(split_img_dir):
            print(f"Skipping missing directory: {split_img_dir}")
            continue

        img_files = subfiles(split_img_dir, join=False, suffix='.png')

        for f in img_files:
            base_name = f[:-4]
            case_id = f"{split}_{base_name}"

            src_img = join(split_img_dir, f)

            src_lbl_png = join(split_lbl_dir, f)
            src_lbl_npy = join(split_lbl_dir, f"{base_name}.npy")

            # Destination file names: Images require a 4-digit channel identifier suffix (e.g. _0000.png)
            dst_img = join(imagestr, f"{case_id}_0000.png")
            dst_lbl = join(labelstr, f"{case_id}.png")

            img = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (400,400), interpolation=cv2.INTER_AREA)
            img_gamma = cv2.LUT(img_resized, gamma_table)
            img_enhanced = clahe.apply(img_gamma)
            cv2.imwrite(dst_img, img_enhanced)

            if os.path.exists(src_lbl_png):
                lbl = cv2.imread(src_lbl_png, cv2.IMREAD_GRAYSCALE)
            elif os.path.exists(src_lbl_npy):
                lbl = np.load(src_lbl_npy).astype(np.uint8)
            else:
                skipped_cases += 1
                continue

            lbl_resized = cv2.resize(lbl, (400,400), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(dst_lbl, lbl_resized)
            num_cases += 1

    print(f"Successfully copied {num_cases} cases to {output_dir}")
    print(f"Skipped {skipped_cases} images due to missing labels")
    # 3. Generate dataset.json using the helper function
    # OpenEDS2019 standard semantic mapping: 0=background, 1=pupil, 2=iris, 3=sclera
    generate_dataset_json(
        output_folder=output_dir,
        channel_names={0: 'grayscale'},
        labels={
            'background': 0,
            'sclera': 1,
            'iris': 2,
            'pupil': 3
        },
        num_training_cases=num_cases,
        file_ending='.png',
        dataset_name=dataset_name,
        reference='Meta Reality Labs OpenEDS 2019',
        description='Eye semantic segmentation dataset for AD-GBC project',
        converted_by="Yongha Chun"
    )
