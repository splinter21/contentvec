# 1.装环境：
```
git clone https://github.com/bfloat16/contentvec
cd contentvec
conda create -n cvec python=3.10
conda activate cvec
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
python setup.py build_ext --inplace
pip install npy_append_array tensorboardX librosa resemblyzer pyreaper praat-parselmouth
cd ..
rsync -a contentvec/ fairseq/fairseq/
```
下载https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr，放入当前文件夹。

# 2.处理数据
一个说话人一个文件夹，全部放在dataset_raw文件夹内。
```
python 00_resampler.py
python 01_train_valid_tsv.py

python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py "data/metadata" "train" "checkpoint_best_legacy_500.pt" 12 1 0 "data/metadata"
python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py "data/metadata" "valid" "checkpoint_best_legacy_500.pt" 12 1 0 "data/metadata"

python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py "data/metadata" "train" 1 "data/label/train_km" 500 --percent -1
python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py "data/metadata" "valid" 1 "data/label/valid_km" 500 --percent -1

mkdir data/label

python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py "data/metadata" "train" 1 "data/label/train_km" 500 --percent -1
python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py "data/metadata" "valid" 1 "data/label/valid_km" 500 --percent -1

python fairseq/examples/hubert/simple_kmeans/dump_km_label.py "data/metadata" "train" "data/label/train_km" 1 0 "data/label"
python fairseq/examples/hubert/simple_kmeans/dump_km_label.py "data/metadata" "valid" "data/label/valid_km" 1 0 "data/label"

python 02_create_contentvec_dict.py
```
# 3.清理数据

data/label内的train_km和valid_km删除，train_0_1.km和valid_0_1.km重命名成train.km和valid.km

data/label/dict.km.txt扩展到500(已经写好了)

data/metadata内的npy和len删除

# 4.单机多卡训练

依据卡数修改run_pretrain_single.sh里面的distributed_training.nprocs_per_node=8
```
./run_pretrain_single.sh
```