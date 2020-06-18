# ESAFN
Dataset and codes for our paper "Entity-Sensitive Attention and Fusion Network for Entity-Level Multimodal Sentiment Classification".

## Paper:
Entity-Sensitive Attention and Fusion Network for Entity-Level Multimodal Sentiment Classification. Jianfei Yu, Jing Jiang, and Rui Xia. Accepted by IEEE/ACM TASLP 2020.

Author

Jianfei Yu

jfyu.2014@phdis.smu.edu.sg

Dec 2, 2019

## Requirement

* PyTorch 1.0.0
* Python 3.7


## Download tweet images and set up image path
- Step 1: Download each tweet's associated image via this link (https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view)
- Step 2: Change the image path in line 468 and line 472 of the "train.py" file
- Step 3: Download the pre-trained ResNet-152 via this link (https://download.pytorch.org/models/resnet152-b121ed2d.pth)
- Setp 4: Put the pre-trained ResNet-152 model under the folder named "resnet"
- Setp 5: Change the Glove word embedding path in line 30 of the "data_utils.py" file

## Code Usage

### Training for ESAFN
- This is the training code of tuning parameters on the dev set, and testing on the test set. Note that you can change "CUDA_VISIBLE_DEVICES=0" based on your available GPUs.

```sh
sh run.sh
```

- To show the running process, we show our running logs in the "twitter_log.txt" and the "twitter2015_log.txt".
