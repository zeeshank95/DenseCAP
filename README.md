# Reproducing Vid2Seq, it is built on top of the codebase from "Learning Grounded Vision-Language Representation for Versatile Understanding in Untrimmed Videos". Major changes are done in vid2seq.py, train.py, video_dataset.py, eval_utils.py


## Getting Started
### Prepare Environment
```bash
git clone --recursive https://github.com/zeeshank95/DenseCAP.git
conda create -n gvl python=3.7
conda activate gvl
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
conda install ffmpeg
pip install -r requirements.txt
```

### Download visual features
1. Download TSP features ActivityNet Captions
```bash
cd data/anet/features
bash download_tsp_features.sh
```

## Training
```bash
config_path=cfgs/anet_tsp_msvg_dvc_pc.yml
gpu_id=0
python train.py --cfg ${config_path} --gpu_id ${gpu_id}
# Checkpoints and logs will be saved under "save/" folder. The best Model will be evaluated in the end of the final epoch
```
