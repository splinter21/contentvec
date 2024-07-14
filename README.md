# 1.装环境：
```
git clone --recurse-submodules https://github.com/bfloat16/contentvec
cd contentvec
conda create -n cvec python=3.10
conda activate cvec
# for windows
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# for Linux
pip3 install torch torchaudio

cd fairseq
pip install --editable ./
python setup.py build_ext --inplace
pip install tensorboard tensorboardX librosa resemblyzer pyreaper praat-parselmouth

cd ..
# for windows
robocopy contentvec fairseq\fairseq /MIR
# for Linux
rsync -a contentvec/ fairseq/fairseq/
```
https://ibm.ent.box.com/s/nv35hsry0v2y595etzysgnn2amsxxb0u

https://huggingface.co/TencentGameMate/chinese-hubert-large/blob/main/chinese-hubert-large-fairseq-ckpt.pt

下载链接里面的模型，放入当前文件夹。

legacy系列不能用，砍掉了东西了。

先试试预测cnhubertlarge的13层，炸了再说。

# 2.处理数据
一个说话人一个文件夹，全部放在dataset_raw文件夹内。

参数都在对应的脚本里面，打开改一下就行（注意train和valid都要跑一遍）
```
python 00_resampler.py
python 01_train_valid_tsv.py

python fairseq/examples/hubert/simple_kmeans/dump_hubert_feature.py
python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py

mkdir data/label

python fairseq/examples/hubert/simple_kmeans/learn_kmeans.py
python fairseq/examples/hubert/simple_kmeans/dump_km_label.py

python 02_create_contentvec_dict.py
```
# 3.清理数据

data/label内的train_km和valid_km删除，train_0_1.km和valid_0_1.km重命名成train.km和valid.km

data/label/dict.km.txt扩展到500(已经写好了)

data/metadata内的一堆npy删除

# 4.单机多卡训练

依据卡数修改run_pretrain_single.sh里面的distributed_training.nprocs_per_node=8
```
./run_pretrain_single.sh
```
