# Test-time Contrastive Concepts
Official repository for 'Test-time Contrastive Concepts for Open-world Semantic Segmentation with Vision-Language Models' (TMLR 2025)


### This repository contains:
1. Code to extract Contrastive Concepts (CC) based on statistics from CLIP pre-training dataset (Laion). 
2. LLM-based CCs used in our paper.
3. [Soon to be released] Code for our proposed IoU-single metric 


## Installation

```
conda create --name ccs python=3.9
conda activate ccs

conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=[XX]-c pytorch -c nvidia
```
Next, install the rest of necessary requirements:


```
pip install -r requirements.txt
```

## CC extraction

1. Download pre-computed co-occurrence matrix of concepts in Laion dataset from [here](https://drive.google.com/file/d/1Smm-h3cyYoVX0XPwS1_PFSCXnESnnVYt/view?usp=sharing)


