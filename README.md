# AMBER: Anchor-based Mamba-driven Dense Video Captioning

## Table of Contents
- [Preparation](#preparation)
  - [Install Dependencies](#install-dependencies)
  - [Install Mamba](#install-mamba)
  - [Check Submodules](#check-submodules)
- [Training](#training)
  - [Download YouCook2 Features](#download-youcook2-features)
  - [Training 1 Epoch](#training-1-epoch)
- [Performance](#performance)
  - [Dense Video Captioning on YouCook2](#dense-video-captioning-on-youcook2)
- [Acknowledgement](#acknowledgement)
---

## Preparation

1. **Install PyTorch and dependencies**  
   Tested on:
   - torch==1.13.1+cu117
   - torch==2.1.2+cu118  
   Other versions may also work.

   conda install ffmpeg
   pip install -r requirement.txt

2. **Install Mamba** following the main README.md of [Mamba Suited](https://github.com/OpenGVLab/video-mamba-suite), make sure it can be imported successfully:  
   from mamba_ssm.modules.mamba_simple import Mamba

   from mamba_ssm.modules.mamba_new import Mamba

3. **Check submodules `SODA` and `pycocoevalcap`** (used for evaluation):  
   from densevid_eval3.SODA.soda import SODA

   If not found, ensure the repo is cloned with `--recursive`, or run:  
   cd path/to/your/video-mamba-suite/

   git submodule update --init --recursive

---

## Training

### Download YouCook2 Features

Download yc2 clip feature [baiduyun drive link](https://drive.google.com/file/d/17H_lxSKFve57kHpkAD7pcYS7ijHIy4-M/view). Then put it in data folder like 'CM2/data/yc2/features/clipvitl14.pth'

---

### Training 1 epoch

Example: train Amber-LSTM with learnt proposals on YouCook2 dataset (1 GPU):

python train.py
  --cfg_path cfgs/yc2_clip_pdvc.yml
  --disable_cudnn 1 
  --save_dir ./save/ 
  --epoch 1

---

## Performance

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
