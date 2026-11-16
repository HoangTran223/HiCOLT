# A Framework for Neural Topic Modeling using Hierarchical Clustering and Contrastive Learning with Optimal Transport (Neurocomputing 2026)

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Neurocomputing-Paper-b31b1b.svg)](https://doi.org/10.1016/j.neucom.2025.132530)
[![Code](https://img.shields.io/badge/Code-PyTorch-50a3b2.svg)](.)
[![Journal](https://img.shields.io/badge/Neurocomputing-2026-blue.svg)](https://www.sciencedirect.com/journal/neurocomputing)

<h3>Official PyTorch implementation of the Neurocomputing 2026 paper.</h3>

</div>

## 📖 Abstract

Recent neural topic models have improved topic quality, but document representations still align poorly with topic distributions, global structure is inconsistent, and the topics have no clear hierarchy.

We introduce **HiCOLT** (Hierarchical Clustering and Contrastive Learning with Optimal Transport for Neural Topic Modeling), a modular framework that can be added to existing topic models. Optimal transport aligns topic distributions with document representations from a pretrained language model, which also improves global structural consistency. A hierarchical contrastive module first groups topics and then refines that hierarchy with a triple-level contrastive loss, at the intra-cluster, inter-cluster, and document-topic levels. On benchmark datasets, adding HiCOLT to several neural topic models improves them across the reported metrics.

**Key Contributions:**
*   **Alignment:** Optimal transport aligns topic distributions with pretrained document representations and improves global structural consistency.
*   **Hierarchy:** Topics are grouped dynamically, then refined by contrastive learning within clusters, between clusters, and between documents and topics.
*   **Experiments:** Adding HiCOLT to several neural topic models improves them on multiple benchmark datasets.

## 📌 Table of Contents
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Running Experiments](#-running-experiments)
- [Options](#-options)
- [Citation](#-citation)
- [Acknowledgement](#-acknowledgement)
- [License](#-license)

## 🛠 Installation

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Install Java.

3. The Palmetto jar is already at `evaluations/pametto.jar`, which is the path the code calls. If it is missing, download [palmetto-0.1.0-jar-with-dependencies.jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to that path.

4. Download [Wikipedia_bd.zip](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) and extract it to `datasets/wikipedia/`, so the index is at `datasets/wikipedia/wikipedia_bd`.

## 📂 Project Structure

```text
HiCOLT/
├── CTM/                         # CTM in CombinedTM.py; HiCOLT added in CombinedTM_Plus.py
├── ECRTM/                       # ECRTM in ECRTM.py; HiCOLT added in ECRTM_Plus.py
├── ETM/                         # ETM in ETM.py; HiCOLT added in ETM_Plus.py
├── FASTopic/                    # FASTopic in FASTopic.py; HiCOLT added in FASTopic_Plus.py
├── NeuroMax/                    # NeuroMax in NeuroMax.py; HiCOLT added in NeuroMax_Plus.py
├── datasethandler/              # Dataset loading
├── datasets/
│   ├── 20NG/
│   ├── AGNews/
│   ├── GoogleNews/
│   ├── IMDB/
│   ├── SearchSnippets/
│   ├── StackOverflow/
│   └── YahooAnswers/
├── evaluations/                 # Coherence, diversity, clustering, and classification
│   └── pametto.jar
├── utils/                       # Arguments, logging, and seeds
│   └── config.py
├── basic_trainer.py             # Training loop
├── eval_llm.py                  # LLM-based topic evaluation
├── main.py                      # Entry point
└── requirements.txt
```

`datasets/wikipedia/` is not in the repository. Installation step 4 adds it for topic coherence. Doc2Vec embeddings are in `datasets/<dataset>/doc2vec/`. SBERT embeddings are in `datasets/<dataset>/with_bert/`.

## 🚀 Running Experiments

Run from the repository root. This command trains ECRTM with HiCOLT on 20NG.

```bash
python main.py \
    --dataset 20NG \
    --model ECRTM_Plus \
    --num_topics 50 \
    --dropout 0.2 \
    --use_pretrainWE \
    --weight_ECR 165.0 \
    --alpha_ECR 20 \
    --alpha_TP 2 \
    --weight_loss_TP 2.0 \
    --weight_loss_CLC 0.5 \
    --max_clusters 9 \
    --threshold_epoch 150 \
    --threshold_cluster 30 \
    --weight_loss_CLT 0.4 \
    --method_CL HAC \
    --metric_CL euclidean \
    --beta_temp 0.16 \
    --epochs 445 \
    --batch_size 128 \
    --lr 0.002 \
    --device cuda \
    --seed 2 \
    --lr_scheduler StepLR \
    --lr_step_size 125 \
    --weight_loss_DTC 1.0 \
    --positive_sampling random \
    --negative_sampling random
```

The suffix `_Plus` means that backbone with HiCOLT added. The file without `_Plus` is the backbone alone.

*   To run **ETM + HiCOLT**, set `--model` to `ETM_Plus`.
*   To run **ECRTM + HiCOLT**, set `--model` to `ECRTM_Plus`.
*   To run **CTM + HiCOLT**, set `--model` to `CombinedTM_Plus`.
*   To run **FASTopic + HiCOLT**, set `--model` to `FASTopic_Plus`.
*   To run **NeuroMax + HiCOLT**, set `--model` to `NeuroMax_Plus`.

Results are written under `results/`.

## Options

*   **Datasets.** The runs in this repository use `20NG`, `AGNews`, and `IMDB`. Pass the name with `--dataset`.

*   **Hierarchical clustering.** Set `--method_CL` to one of:

    *   `HAC`: hierarchical agglomerative clustering.
    *   `HDBSCAN`: hierarchical density-based clustering.

*   **Distance metric.** Set `--metric_CL` to `euclidean` or `cosine`. This is the distance used inside contrastive learning.

*   **Sampling.** Both default to `random`.

    *   `--positive_sampling`: `random` or `hard`.
    *   `--negative_sampling`: `random` or `hard`.

*   **Contrastive loss.** Set `--contrastive_loss_type` to `triplet`, `circle`, or `infonce`.

## 📚 Citation

If you find our work or this codebase helpful, please consider citing:

```bibtex
@article{phamkhanh2026hicolt,
  title     = {A framework for neural topic modeling using hierarchical clustering and contrastive learning with optimal transport},
  author    = {Pham Khanh, Chi and Tran Vuong, Hoang and Dinh Viet, Sang and Ngo Van, Linh},
  journal   = {Neurocomputing},
  volume    = {670},
  pages     = {132530},
  year      = {2026},
  issn      = {0925-2312},
  publisher = {Elsevier},
  doi       = {10.1016/j.neucom.2025.132530},
  url       = {https://doi.org/10.1016/j.neucom.2025.132530}
}
```

## Acknowledgement

Some parts of this implementation are based on [TopMost](https://github.com/BobXWu/TopMost). We also utilize [Palmetto](https://github.com/dice-group/Palmetto) for evaluating topic coherence.

## 📜 License

This project is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
