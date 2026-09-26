import glob
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("PATH", help="Path to files")
args = parser.parse_args()

# PATH = "/mnt/hdd1/adgbc_results/nnUNet/AD-RGBC_S-4_nn/*.npy"
PATH = args.PATH
PATH = os.path.join(PATH,"*.npy")
labels = glob.glob(PATH)
for f in tqdm(labels):
    if not f.endswith("_0000.npy"): os.rename(f, f.replace(".npy","_0000.npy"))

