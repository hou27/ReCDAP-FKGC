# ReCDAP

**ReCDAP** is a framework for Few-shot Knowledge Graph Completion. This repository utilizes the NELL and FB15K-237 datasets to demonstrate efficient few-shot learning for knowledge graph completion.

---

## Key Features

**Relation-based Conditional Diffusion**:

- The model employs a diffusion process that is conditioned on the task relation, the extended support set (including both positive and negative triples), and explicit label information.
- This conditional diffusion allows the model to accurately estimate separate latent distributions for positive and negative cases, rather than merely using negative examples as a contrastive signal.  

**Attention Pooler**:

- A transformer-inspired Attention Pooler is used to independently extract key features from both the positive and negative representations.
- This selective pooling enhances the model’s ability to capture discriminative features compared to simple mean or max pooling.

---

## Requirements

```bash
torch==1.11.0  
dgl-cu113==0.9.0  
diffusers==0.16.1
huggingface-hub==0.27.1
```

---

## Environment

- **Python:** 3.8.20
- **Operating System:** Ubuntu 20.04
- **GPU:** RTX3090 24GB Memory

---

## Dataset & Checkpoints

### Original Datasets

- **NELL:** [NELL Dataset](https://github.com/xwhan/One-shot-Relational-Learning)
- **FB15K-237:** [FB15K-237 Dataset](https://github.com/SongW-SW/REFORM)

### Processed Dataset

- [Processed Dataset](#)

> **Note:** Download and extract the datasets to the root folder of the project.

### Checkpoints are available for download here:

- [Learned Checkpoint](#)

Note: Not yet available because of anonymous rule.

**Directory Structure Example:**

```bash
ReCDAP
├── NELL
├── FB15K
└── ...
```

---

## Installation & Environment Setup

### Create a Conda Environment

```bash
conda create -n recdap python=3.8
conda activate recdap
```

### Install PyTorch & CUDA

```bash
conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch -c nvidia
```

### Install DGL and Additional Packages

```bash
pip install dgl-cu113==0.9.0 -f https://data.dgl.ai/wheels/repo.html
pip install -r requirement_recdap.txt
```

---

## How to Run

### Training

#### For the NELL Dataset

```bash
python main.py --dataset NELL-One \
               --data_path ./NELL \
               --few 5 \
               --data_form Pre-Train \
               --prefix nell_recdap_iter100_lr1e-3_margin1 \
               --device 0 \
               --batch_size 64 \
               --g_batch 1024 \
               --epoch 100000 \
               --eval_epoch 1000 \
               --checkpoint_epoch 1000 \
               --learning_rate 1e-3 \
               --margin 1.0 \
               --num_diffusion_iters 100
```

#### For the FB15K Dataset

```bash
python main.py --dataset FB15K-One \
               --data_path ./FB15K \
               --few 5 \
               --data_form Pre-Train \
               --prefix fb15k_recdap_iter100_lr1e-4_margin1 \
               --device 0 \
               --batch_size 32 \
               --g_batch 1024 \
               --epoch 100000 \
               --eval_epoch 1000 \
               --checkpoint_epoch 1000 \
               --learning_rate 1e-4 \
               --margin 1.0 \
               --num_diffusion_iters 100
```

### Testing

Use pre-trained checkpoints for evaluation.

**Checkpoint should be downloaded and extracted to the project and located below the `state` directory.**

```bash
ReCDAP
├── state
│   ├── state_dict_nell_best_505
│   └── state_dict_fb15k_best_579
```

#### Testing on the NELL Dataset

```bash
python main.py --dataset NELL-One \
               --data_path ./NELL \
               --few 5 \
               --data_form Pre-Train \
               --prefix state_dict_nell_best_505 \
               --state_dict_filename state_dict_nell_best_505 \
               --device 0 \
               --batch_size 64 \
               --g_batch 1024 \
               --epoch 100000 \
               --eval_epoch 1000 \
               --checkpoint_epoch 1000 \
               --learning_rate 1e-3 \
               --margin 1.0 \
               --num_diffusion_iters 100 \
               --step test
```

#### Testing on the FB15K Dataset

```bash
python main.py --dataset FB15K-One \
               --data_path ./FB15K \
               --few 5 \
               --data_form Pre-Train \
               --prefix state_dict_fb15k_best_579 \
               --state_dict_filename state_dict_fb15k_best_579 \
               --device 0 \
               --batch_size 32 \
               --g_batch 1024 \
               --epoch 100000 \
               --eval_epoch 1000 \
               --checkpoint_epoch 1000 \
               --learning_rate 1e-4 \
               --margin 1.0 \
               --num_diffusion_iters 100 \
               --step test
```

---

## Experimental Results

### 5-shot FKGC Results

| Dataset       | MRR   | Hits@10 | Hits@5 | Hits@1 |
|---------------|-------|---------|--------|--------|
| **NELL**      | 0.505 | 0.528   | 0.506  | 0.493  |
| **FB15K-237** | 0.579 | 0.745   | 0.695  | 0.491  |

---

## Implementation Details

- **Initial Embeddings:** TransE embeddings from GMatching and REFORM  
- **Diffusion Iterations:** 100 steps  
- **Sequence Padding:** Zero-filled padding of length 2  
- **Attention Pooler:** Single-head configuration  
- **Embedding Dimension:** 100 (for both NELL and FB15K-237)  
- **Global Aggregator & Relation Learner:** Settings as in NP-FKGC  
- **Optimizer:** Adam  
  - **Learning Rate:** 1e-3 (NELL) and 1e-4 (FB15K-237)  
- **Batch Size:** 64 (NELL) and 32 (FB15K-237)  
- **Margin:** 1

---

## Acknowledgements

This repository is based on [NP-FKGC](https://github.com/rmanluo/np-fkgc). We appreciate the efforts of the original authors and thank them for their excellent work.