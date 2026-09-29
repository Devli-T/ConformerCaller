# ConformerCaller: Direct Motif Calling from Nanopore Signals for DNA Data Storage

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation and pretrained models for **"ConformerCaller: Direct Motif Calling from Nanopore Signals for DNA Data Storage"** (Dev Thakkar and Long Tran-Thanh, Department of Computer Science, University of Warwick).

---

## 📌 Overview

Motif-based combinatorial DNA synthesis enables high-density, low-cost digital data archiving by assembling predefined short sequences (motifs) into structured payload strands. However, decoding digital information from raw Oxford Nanopore Technologies (ONT) ionic current squiggles has traditionally relied on a decoupled, two-stage pipeline:

1. **General-purpose basecalling** (e.g., Guppy, Dorado) mapping ionic current to nucleotide strings ($A, C, G, T$).
2. **Heuristic motif alignment** (e.g., Zero-Error Search, Adaptive Margin Search) aligning basecalled strings against motif dictionaries.

This decoupled paradigm suffers from **compound error propagation**: insertion, deletion, and substitution errors during basecalling degrade subsequent sequence alignment, especially on low-quality reads.

**ConformerCaller** replaces this pipeline with an **end-to-end neural architecture** that directly translates raw, continuous nanopore current traces into discrete payload motif sequences using Connectionist Temporal Classification (CTC) decoding.

<p align="center">
  <img src="results/paper/encoding_schematic.png" width="90%" alt="ConformerCaller Encoding and Decoding Pipeline" />
</p>

---

## 🚀 Key Highlights & Results

- **92.1% Motif Detection Rate**: Directly decodes raw electrical signals on a held-out test split of 66,843 empirical reads.
- **90.4% Mean Per-Cycle Accuracy**: Consistently high classification accuracy across all eight payload cycles ($C_3$–$C_{10}$).
- **Zero Cross-Cycle Confusion**: 100% of all 31,033 observed misclassifications occurred strictly within the true cycle index. The model naturally learns macromolecular physical ordering without explicit positional tokens.
- **$2.8\times$ Reduction in Sequencing Coverage**: Lowers the read coverage required to reach 95% payload information recovery from 14 reads down to 5 reads.
- **Real-Time Portable Inference**: Achieves 129.7 reads/sec with a mean latency of 7.7 ms per read on standard consumer silicon.
- **Trained on Verified Physical Labels**: Supervised against verified physical synthesis catalogs (`master_db.csv`), bypassing the error floor of heuristic aligners.

### Benchmark Comparison

| Method | Decoding Paradigm | Training Supervision | Detection Rate (%) | 95% Recovery (Reads) |
|---|---|---|:---:|:---:|
| **ZE Search** | Two-step (Basecalled) | Heuristic dictionary | ~65% | ~50 (Empirical) |
| **AM Search** | Two-step (Basecalled) | Heuristic aligner | ~85% | ~22 (Empirical) |
| **CNN + BiGRU** | End-to-end | AM Search labels | 91.1%* | 14 (Empirical) |
| **ConformerCaller (Greedy)** | **End-to-end** | **Synthesis ground-truth** | **92.1%** | **~5 (Analytical)** |
| **ConformerCaller ($\tau \ge 0.70$)** | **End-to-end** | **Synthesis ground-truth** | **82.3%** | **~5 (Analytical)** |

*\*Evaluated against AM Search labels. ConformerCaller is evaluated against verified physical ground truth.*

---

## 🧠 Model Architecture

ConformerCaller comprises three core functional stages totaling **11.02M parameters**:

1. **Convolutional Subsampler (1D CNN)**: 5 layers with channel dimensions $1 \to 32 \to 64 \to 128 \to 256 \to 256$, kernel sizes $[11, 9, 7, 5, 5]$, strides $[1, 2, 2, 2, 2]$, Batch Normalization, and SiLU activations. Applies a $16\times$ temporal downsampling factor.
2. **Conformer Encoder Stack**: 6 Conformer blocks ($d_{\text{model}} = 256$, 8 attention heads, depthwise convolution kernel 31, feed-forward expansion factor 4). Employs relative positional encoding to track long-range sequential context across variable-speed translocations.
3. **Linear Classification Head & CTC Decoder**: Maps encoder states to 65 classes (64 payload motifs + 1 CTC blank token $\epsilon$). Decoded via greedy CTC or confidence-filtered greedy decoding ($\tau \ge 0.70$).

<p align="center">
  <img src="results/paper/architecture_diagram.png" width="45%" alt="ConformerCaller Architecture Diagram" />
</p>

---

## 📂 Repository Structure

```
ConformerCaller/
├── config.py                  # Global configuration (paths, constants, parameters)
├── train.py                   # Main CLI entry point for model training
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── README.md                  # Project documentation
│
├── checkpoints/
│   └── conformer/
│       └── best_model.pt      # Pretrained optimal checkpoint (Epoch 48, val_loss=0.7971)
│
├── notebooks/
│   ├── colab_training.ipynb   # Distributed / Google Colab training notebook
│   ├── full_eval.py           # Full test set evaluation script (66,843 reads)
│   ├── paper_analysis.py      # Detailed metrics, confusion analysis, and recovery curves
│   ├── generate_architecture_diagram.py  # Generates Figure 2 (architecture)
│   └── generate_encoding_schematic.py   # Generates Figure 1 (encoding/decoding)
│
├── results/
│   └── paper/
│       ├── full_test_eval.json           # Complete test set metrics
│       ├── top_confusions.json           # Detailed pairwise confusion frequencies
│       ├── architecture_diagram.png      # High-resolution architecture diagram
│       ├── encoding_schematic.png        # Pipeline schematic diagram
│       ├── per_cycle_accuracy_66k.png    # Per-cycle accuracy chart
│       ├── information_recovery_66k.png  # Information recovery vs. read depth
│       ├── confusion_analysis.png        # Within-cycle error breakdown
│       ├── detection_comparison.png      # Method comparison bar chart
│       └── training_curves.png           # 55-epoch training and validation loss curves
│
└── src/
    ├── data/
    │   ├── motif_library.py              # Motif catalog mappings & cycle allocations
    │   ├── fast5_reader.py               # Raw FAST5 signal extraction & MAD normalisation
    │   ├── label_builder.py              # Ground-truth label construction from master DB
    │   ├── dataset.py                    # PyTorch Dataset and DataLoader implementations
    │   ├── build_training_set.py         # One-time dataset index partitioning
    │   └── extract_signals_sharded.py    # Sharded signal extraction utility
    ├── model/
    │   └── motifcaller.py                # Conformer, CNN-BiGRU, and Causal CNN architectures
    ├── training/
    │   ├── trainer.py                    # PyTorch CTC training engine & early stopping
    │   ├── ctc_decoder.py                # Greedy & confidence-thresholded CTC decoders
    │   └── metrics.py                    # Levenshtein distance, exact match, & cycle accuracy
    ├── evaluation/
    │   └── evaluate.py                   # Architecture evaluation and comparison suite
    └── inference/
        └── predict.py                    # End-to-end inference on raw FAST5 files
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/devthakkar/ConformerCaller.git
cd ConformerCaller
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on ONT FAST5 files:** ONT sequencing data uses VBZ compression. The `vbz-h5py-plugin` package included in `requirements.txt` is required to read multi-read FAST5 files.

---

## ⚡ Quickstart: Inference on Raw Reads

Run inference on raw FAST5 files using the provided pretrained model:

```bash
python src/inference/predict.py \
    --checkpoint checkpoints/conformer/best_model.pt \
    --fast5-dir /path/to/fast5_files \
    --output results/predictions.csv \
    --threshold 0.70
```

The output CSV contains per-read predicted payload motif sequences:
```csv
read_id,predicted_motifs,n_motifs,confidence
0001a79f-683a-442f-8703-a4176378a3c5,"ltm8_3x1,ltm8_4x6,ltm8_5x2,ltm8_6x4,ltm8_7x1,ltm8_8x3,ltm8_9x8,ltm8_10x2",8,0.942
```

---

## 📊 Reproducing Paper Results & Figures

### 1. Full Test Set Evaluation (66,843 Reads)
To reproduce the complete test set evaluation and per-cycle statistics:
```bash
PYTHONPATH=. python notebooks/full_eval.py
```
Outputs are written to `results/paper/full_test_eval.json`.

### 2. Regenerate Paper Figures
To regenerate the exact publication-ready figures:
```bash
# Architecture diagram (Figure 2)
PYTHONPATH=. python notebooks/generate_architecture_diagram.py

# Encoding & decoding schematic (Figure 1)
PYTHONPATH=. python notebooks/generate_encoding_schematic.py

# Confusion analysis and information recovery curves
PYTHONPATH=. python notebooks/paper_analysis.py
```

---

## 🏋️ Model Training

### Local Training
To train ConformerCaller from scratch locally:
```bash
python train.py \
    --arch conformer \
    --epochs 100 \
    --batch-size 16 \
    --lr 3e-4 \
    --checkpoint-dir checkpoints/conformer
```

### Distributed / Cloud Training (Google Colab A100)
For large-scale training across all 501,330 training reads:
1. Open [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb) in Google Colab.
2. Select an **A100 GPU (High RAM)** runtime.
3. Execute the cells to stream sharded signals with mixed-precision (FP16) CTC training.

---

## 🔒 Data Availability

The empirical sequencing dataset was provided by **Helixworks Technologies Limited** under a Data Sharing Agreement. In accordance with the agreement:
- Raw FAST5 files and electrical signal squiggles are proprietary and cannot be publicly redistributed.
- Model architecture definitions, trained checkpoint weights (`checkpoints/conformer/best_model.pt`), evaluation scripts, shard partition definitions (`shard_split_ids.json`), and comprehensive evaluation outputs are fully available in this repository.

Researchers wishing to access the underlying raw FAST5 sequencing data should contact Helixworks Technologies Limited directly.

---

## 📜 Citation

If you find this work or codebase useful in your research, please cite:

```bibtex
@article{thakkar2026conformercaller,
  title={ConformerCaller: Direct Motif Calling from Nanopore Signals for DNA Data Storage},
  author={Thakkar, Dev and Tran-Thanh, Long},
  journal={Department of Computer Science, University of Warwick},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
