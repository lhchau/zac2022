# Zalo AI Challenge 2022 - Team Life Panther

## Guide

```
!pip install underthesea
!pip install pytorch_pretrained_bert==0.4.0
!pip install -U gensim
```

### Models

Create a folder named `models` containing BERT fine-tuning model. 
[Our checkpoint 10000 model](https://drive.google.com/drive/folders/1-78AQwifgk2WCbvnHOXzuz_pIGZuj5VB?usp=share_link) has parameters:

```
!python -m run_squad  --model_type bert \
                      --model_name_or_path bert-base-multilingual-cased \
                      --do_train \
                      --do_lower_case \
                      --train_file data/train_assemble.json\
                      --predict_file data/dev_assemble.json \
                      --per_gpu_train_batch_size 4 \
                      --learning_rate 2e-5 \
                      --weight_decay 0.05 \
                      --num_train_epochs 3 \
                      --max_seq_length 384 \
                      --doc_stride 128 \
                      --save_steps 10000 \
                      --output_dir models/bert_fine_tuning/
```

We collect data from many sources: 
- `trainZaloAI2022`, `mailong25`, `xquad`, `UITHelper_QAS`, `vi-dev-mlqa`, `vi-train-mlqa`, `UIT-ViQuAD`.
 

### Resources

Create a folder named `resources` containing all resources needed in project
Extracting this link and move all files into `resources` [Resources link](https://drive.google.com/drive/folders/1CtMArUwokk5VXDd1kbDHmNcQ7Kk1l6oE?usp=sharing)

## Running

```
python main.py
```
