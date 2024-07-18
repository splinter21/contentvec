import logging
import os
import sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib
import argparse
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")

def get_km_model(n_clusters, init, max_iter, batch_size, tol, max_no_improvement, n_init, reassignment_ratio):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )

def load_feature(feat_dir, split, percent):
    assert percent <= 1.0
    feat_path = f"{feat_dir}/{split}_total.npy"
    leng_path = f"{feat_dir}/{split}_total.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if percent < 0:
        return np.load(feat_path, mmap_mode="r")
    else:
        nsample = int(np.ceil(len(lengs) * percent))
        indices = np.random.choice(len(lengs), nsample, replace=False)
        feat = np.load(feat_path, mmap_mode="r")
        slices = [feat[offsets[i]: offsets[i] + lengs[i]] for i in tqdm(indices)]
        logger.info("Loading...")
        sampled_feat = np.concatenate(slices, axis=0)
        logger.info((f"sampled {nsample} utterances, {len(sampled_feat)} frames"))
        return sampled_feat

def learn_kmeans(feat_dir, split, km_path, n_clusters, percent, init, max_iter, batch_size, tol, n_init, reassignment_ratio, max_no_improvement,):
    feat = load_feature(feat_dir, split, percent)
    km_model = get_km_model(n_clusters, init, max_iter, batch_size, tol, max_no_improvement, n_init, reassignment_ratio)
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir",           type=str,   default="data/03_metadata_total")
    parser.add_argument("--km_path",            type=str,   default="data/04_cluster")
    parser.add_argument("--n_clusters",         type=int,   default=500)
    parser.add_argument("--percent",            type=float, default=1.0)
    parser.add_argument("--init",               type=str,   default="k-means++")
    parser.add_argument("--max_iter",           type=int,   default=100)
    parser.add_argument("--batch_size",         type=int,   default=10000)
    parser.add_argument("--tol",                type=float, default=0.0)
    parser.add_argument("--max_no_improvement", type=int,   default=100)
    parser.add_argument("--n_init",             type=int,   default=20)
    parser.add_argument("--reassignment_ratio", type=float, default=0.0)
    args = parser.parse_args()

    feat_dir = args.feat_dir
    km_path = args.km_path
    n_clusters = args.n_clusters
    percent = args.percent
    init = args.init
    max_iter = args.max_iter
    batch_size = args.batch_size
    tol = args.tol
    max_no_improvement = args.max_no_improvement
    n_init = args.n_init
    reassignment_ratio = args.reassignment_ratio

    os.makedirs(km_path, exist_ok=True)

    learn_kmeans(feat_dir, "train", f"{km_path}/train_km", n_clusters, percent, init, max_iter, batch_size, tol, n_init, reassignment_ratio, max_no_improvement)
    learn_kmeans(feat_dir, "valid", f"{km_path}/valid_km", n_clusters, percent, init, max_iter, batch_size, tol, n_init, reassignment_ratio, max_no_improvement)