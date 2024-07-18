import logging
import os
import sys
import argparse
import numpy as np

import joblib
import torch
import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")

class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm)
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = ((x ** 2).sum(1, keepdims=True) - 2 * np.matmul(x, self.C_np) + self.Cnorm_np)
            return np.argmin(dist, axis=1)

def get_feat_iterator(feat_dir, split):
    feat_path = f"{feat_dir}/{split}_total.npy"
    leng_path = f"{feat_dir}/{split}_total.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    def iterate():
        feat = np.load(feat_path, mmap_mode="r")
        assert feat.shape[0] == (offsets[-1] + lengs[-1])
        for offset, leng in zip(offsets, lengs):
            yield feat[offset: offset + leng]

    return iterate, len(lengs)

def dump_label(feat_dir, split, km_path, lab_dir):
    apply_kmeans = ApplyKmeans(km_path)
    generator, num = get_feat_iterator(feat_dir, split)
    iterator = generator()

    lab_path = f"{lab_dir}/{split}.km"
    os.makedirs(lab_dir, exist_ok=True)
    with open(lab_path, "w") as f:
        for feat in tqdm.tqdm(iterator, total=num):
            lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, lab)) + "\n")
    logger.info("finished successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", type=str, default="data/03_metadata_total")
    parser.add_argument("--lab_dir",  type=str, default="data/00_filelist")
    args = parser.parse_args()

    dump_label(feat_dir=args.feat_dir, split="train", km_path="data/04_cluster/train_km", lab_dir=args.lab_dir)
    dump_label(feat_dir=args.feat_dir, split="valid", km_path="data/04_cluster/valid_km", lab_dir=args.lab_dir)