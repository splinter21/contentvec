import os
import numpy as np
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from npy_append_array import NpyAppendArray
import logging

logger = logging.getLogger(__name__)

def process_npy_files(feat_dir, output_dir, split):
    feat_path = f"{output_dir}/{split}_total.npy"
    leng_path = f"{output_dir}/{split}_total.len"

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    npy_files = glob(os.path.join(feat_dir, f"{split}_*.npy"))
    npy_files_sort = natsorted(npy_files)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for file in tqdm(npy_files_sort):
            file_load = np.load(file)
            feat_f.append(file_load)
            leng_f.write(f"{len(file_load)}\n")
            os.remove(file)  # Delete the .npy file after loading its content
    
    logger.info("Finished generating feature and length files")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_npy_files("data/02_metadata_npy", "data/03_metadata_total", "train")
    process_npy_files("data/02_metadata_npy", "data/03_metadata_total", "valid")