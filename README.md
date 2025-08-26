# AMBER: Anchor-based Mamba-driven Dense Video Captioning

---

# Preparation
```bash
# clone Amber repository
git clone https://github.com/rebu1412/AMBER-DVC.git
cd AMBER-DVC

# Set-up Enviroment (DOCKER, NVIDIA CONTAINER TOOLKIT, IMAGE)
# https://hub.docker.com/r/bombom2108/videomamba
make setup VERSION=v5
docker --version
nvidia-ctk --version
docker images
```

## Install Dependencies
```bash
# Install mamba_ssm and causal-conv1d inside docker container
make run
pip install -e amber/ops/
pip install -e causal-conv1d/
pip install -e mamba/
```

## Check Environment
```bash
# Inside docker container
python
from  .modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba_new import Mamba
from baseline.densevid_eval3.SODA.soda import SODA
# pip install pycocoevalcap 
```

# Training
## Download YouCook2 Features
```bash
cd baseline/data/yc2/features/resnet_bn
bash download_yc2_tsn_features.sh
cd ../../..
```
## Training 
```bash
# training Amber-LSTM 1 epoch with learnt proposals
python train.py --cfg_path cfgs/yc2_tsn_amber.yml
                --disable_cudnn 1
                --save_dir ./save/
                --epoch 1
```bash
# Debug Mode
python train.py --cfg_path cfgs/yc2_tsn_amber.yml
                --disable_cudnn 1
                --save_dir ./save/
                --epoch 1
                --debug
```


# Performance

### Dense video captioning on YouCook2

| Model       | Features | Config Path           | URL  | Recall | Precision | B-4 | R   | C    | SODA |
|-------------|----------|-----------------------|------|--------|-----------|-----|-----|------|------|
| Amber-LSTM  | TSN      | cfgs/yc2_tsn_pdvc.yml | todo | 24.63  | 30.38     | 0.2 | 3.18| 14.5 | 4.11 |
| PDVC-Mamba  | TSN      | cfgs/yc2_tsn_pdvc.yml | todo |   -    |    -      |  -  |  -  |  -   |  -   |

**Notes**  
- `B-4`: BLEU-4  
- `M`: METEOR  
- `R`: ROUGE-L  
- `C`: CIDEr  

More details: [PDVC repo](https://github.com/ttengwang/PDVC/tree/main)

---

## Acknowledgement

This codebase is based on [PDVC](https://github.com/ttengwang/PDVC/tree/main).  
We thank the authors for their contributions.
