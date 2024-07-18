import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import torch.multiprocessing as mp
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.checkpoint_utils import load_model_ensemble_and_task

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")

class HubertFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk, device):
        (model, cfg, task) = load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().to(device)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.device = device
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}, device = {self.device}")

    def read_audio(self, path):
        wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=self.task.cfg.sample_rate)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        return wav

    def get_feats(self, path):
        x = self.read_audio(path)
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(self.device)
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(source=x_chunk, padding_mask=None, mask=False, output_layer=self.layer)
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)

def process_chunk(rank, args, paths, feat_dir, split, device):
    reader = HubertFeatureReader(args['ckpt_path'], args['layer'], args['max_chunk'], device)

    os.makedirs(feat_dir, exist_ok=True)
    for i, path in tqdm(enumerate(paths), total=len(paths), desc=f"Process {rank}"):
        feat = reader.get_feats(path).cpu().numpy()
        feat_path = f"{feat_dir}/{split}_{rank:02d}_{i:07d}.npy"
        np.save(feat_path, feat)
    logger.info("Process %d on device %s finished successfully", rank, device)

def main(tsv_dir, split, ckpt_path, layer, feat_dir, max_chunk, num_process):
    tsv_path = f"{tsv_dir}/{split}.tsv"
    with open(tsv_path, "r", encoding='utf-8') as f:
        root = f.readline().rstrip()
        lines = [line.rstrip().split("\t")[0] for line in f]
        paths = [f"{root}/{subpath}" for subpath in lines]

    num_gpus = torch.cuda.device_count()
    chunk_size = len(paths) // num_process
    processes = []
    
    for i in range(num_process):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_process - 1 else len(paths)
        device = f"cuda:{i % num_gpus}"
        process_args = (i, dict(ckpt_path=ckpt_path, layer=layer, max_chunk=max_chunk, num_process=num_process), paths[start:end], feat_dir, split, device)
        p = mp.Process(target=process_chunk, args=process_args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_dir", type=str, default="data/00_filelist")
    parser.add_argument("--ckpt_path", type=str, default="chinese-hubert-large-fairseq-ckpt.pt")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--feat_dir", type=str, default="data/02_metadata_npy")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--num_process", type=int, default=5)  # Number of processes
    args = parser.parse_args()

    main(args.tsv_dir, "valid", args.ckpt_path, args.layer, args.feat_dir, args.max_chunk, args.num_process)
    main(args.tsv_dir, "train", args.ckpt_path, args.layer, args.feat_dir, args.max_chunk, args.num_process)