import glob
import os
 
PATH = "/mnt/hdd1/adgbc_results/nnUNet/AD-RGBC_S-4_nn/*.npy"

labels = glob.glob(PATH)
for f in labels:
    if not f.endswith("_0000.npy"): os.rename(f, f.replace(".npy","_0000.npy"))

