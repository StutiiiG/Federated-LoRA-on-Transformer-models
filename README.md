# Federated LoRA: Parameter-Efficient Federated Fine-Tuning of LLMs

> Federated fine-tuning of large language models (LLMs) using LoRA adapters, with heterogeneous clients, Dirichlet data splits, and centralized aggregation of adapter weights only.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## 🧩 Overview

This repository implements a **federated learning framework for LoRA-based fine-tuning of LLMs**.  
Multiple clients fine-tune **only LoRA adapters** on their private data, while a central server **aggregates the adapter weights**, never seeing any raw data.

The default experiments use:

- **Base model:** BLOOM / BLOOMZ or LLaMA-2 (Hugging Face)
- **Task:** Text classification (e.g., SST-2)
- **Federation:** Simulated clients with **Dirichlet data partitioning**
- **PEFT:** [LoRA](https://arxiv.org/abs/2106.09685) with varying ranks (e.g., `r = 16, 64, 256`)

The goal is to explore:

- How well LoRA works in a **federated, non-IID** setting
- Trade-offs between **LoRA rank vs. communication cost vs. accuracy**
- Practical tricks for running federated LLM experiments on **limited GPU budgets**

---

## 📚 Table of Contents

1. [Repository Structure](#-repository-structure)  
2. [Model & Training Architecture](#-model--training-architecture)  
3. [Installation](#-installation)  
4. [Quickstart](#-quickstart)  
5. [Configuration](#-configuration)  
6. [Running Experiments](#-running-experiments)  
7. [Logging & Monitoring](#-logging--monitoring)  
8. [Results](#-results)  
9. [Extending the Project](#-extending-the-project)  
10. [Limitations & Future Work](#-limitations--future-work)  
11. [Citing](#-citing)  
12. [License](#-license)

---

## 🗂 Repository Structure

```text
federated-lora/
├── configs/
│   ├── bloom_sst2_base.yaml
│   ├── bloom_sst2_fed.yaml
│   └── llama2_sst2_fed.yaml
├── data/
│   └── (downloaded datasets / cached HF datasets)
├── federated/
│   ├── client.py           # Client training loop (local LoRA updates)
│   ├── server.py           # Aggregation logic (FedAvg / weighted avg)
│   ├── partition.py        # Dirichlet data partitioning utilities
│   └── utils.py
├── models/
│   ├── lora_wrapper.py     # PEFT / LoRA integration
│   └── tokenizer_utils.py
├── scripts/
│   ├── run_single_lora.py  # Baseline (non-federated) LoRA fine-tuning
│   ├── run_federated.py    # Full federated LoRA training script
│   └── evaluate.py         # Centralized evaluation script
├── notebooks/
│   └── exploration.ipynb   # EDA & quick checks
├── requirements.txt
├── README.md
└── LICENSE
