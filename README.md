# ESAFN
Dataset and codes for our paper "Entity-Sensitive Attention and Fusion Network for Entity-Level Multimodal Sentiment Classification".

## Paper:
Entity-Sensitive Attention and Fusion Network for Entity-Level Multimodal Sentiment Classification. Jianfei Yu, Jing Jiang, and Rui Xia. Accepted by IEEE/ACM TASLP 2020.

Author

Jianfei Yu

jfyu.2014@phdis.smu.edu.sg

Dec 2, 2019

## Code:
- We will clean up our codes and release it soon. Currently, you may first refer to our TomBERT model (https://github.com/jefferyYu/TomBERT), which is based on BERT and has much better performance than ESAFN on the two multimodal datasets.

## Data:
- Download multimodal data via this link (https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view)

### Textual Content for Each Tweet

* Under the folder "twitter2015"
* Under the folder "twitter2017"

### Visual Content for Each Tweet

* Under the folder "twitter2015_images"
* Under the folder "twitter2017_images"

### Description

#### Data Split

* We randomly split our annotated data into training (60%), development (20%), and test sets (20%).

#### Format

* We provide two kinds of format, one is ".txt" for LSTM-based models, and another is ".tsv" for BERT models.
