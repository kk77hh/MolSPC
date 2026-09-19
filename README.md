# MolSPC

**Reshaping Chemical Space with Virtual Intermediates for Continuity-Aware Molecular Foundation Model**

MolSPC is a graph-language molecular foundation model for de novo generation and multi-objective lead optimization. The repository includes a continuity-aware data augmentation stage: nearby molecules are aligned in graph space, their Laplacian eigenvalue representations and atom features are interpolated, and chemically valid virtual intermediates are added before fine-tuning.

The seven properties used for conditioning and evaluation are **QED, LogP, MW, HBA, HBD, TPSA, and RB** (rotatable bonds). A task can request one property or a `+`-separated combination such as `QED+LogP+TPSA`.

All model training and downstream experiments were performed using two NVIDIA A100 GPUs, each equipped with 80 GB of GPU memory.


## Repository layout

```text
MolSPC-main/
├── molecule_augmentation/
│   ├── virtual_intermediates.py  # graph interpolation and property calculation
│   └── generate_pairs.py         # property-dataset augmentation CLI
├── data_finetune_molopt.py       # CSV → MolSPC fine-tuning directories
├── data_pretrain_stage3.py       # stage-3 pretraining data preparation
├── stage2.py                     # pretraining, fine-tuning, and evaluation
├── data_provider/                # PyTorch/PyG datasets and collators
├── model/                        # graph encoder and language-model modules
├── datasets/                     # example optimization train/test CSV files
└── environment.yml
```

## Installation

The supplied environment targets Python 3.8, PyTorch 2.0, and CUDA 11.7.

```bash
conda env create -f environment.yml
conda activate molspc
```

Run the commands below from the `MolSPC-main` repository root. If the
interpreter reports `No module named numpy` (or another scientific package),
the environment is not active; check with `python -c "import numpy, pandas, rdkit, scipy, torch, torch_geometric, ogb"` before starting augmentation.

For a CPU-only setup, install the matching PyTorch and PyTorch Geometric wheels after creating the environment. RDKit, OGB, SciPy, pandas, and scikit-learn are required by the graph and augmentation modules.



## 1. Augment a standalone property dataset

For a CSV with columns `smiles` and one property column, use the public augmentation entry point:

```bash
python -m molecule_augmentation.generate_pairs \
  --input molecule_augmentation/datasets/tpsa/tpsa.csv \
  --property tpsa \
  --output data_aug/augmented_tpsa_pairs.csv \
  --percentage 0.10 --gamma 0.10
```

The output records the source molecule, interpolation partner, generated SMILES, and the interpolated property value. The implementation assigns more samples to sparse property regions and uses property proximity to choose interpolation partners. Invalid RDKit reconstructions are skipped.

## 2. Train and evaluate

Pretraining stage1 and stage2. We achieved chemical self-awareness pretraining and cross-modal semantic grounding through the first two stages of pretraining, with the implementation approach referenced from the [molca](https://github.com/acharkq/MolCA) article. Please visit the following link to download the full dataset and weight files: https://huggingface.co/kk77hh/MolSPC/tree/main

Stage 3 pretraining data:

```bash
python data_pretrain_stage3.py
```

Fine-tuning from the released stage-3 checkpoint:

```bash
python data_finetune_molopt.py 
```

Pre-train

```bash
python stage2.py \
  --root data/compare/train/ --valid_root data/compare/valid/ \
  --devices 0,1 --filename stage2 \
  --stage2_path all_checkpoints/pretrain_stage2.ckpt \
  --init_checkpoint all_checkpoints/pretrain_stage2.ckpt \
  --opt_model facebook/galactica-1.3b --mode pretrain \
  --prompt '[START_I_SMILES]{}[END_I_SMILES]' \
  --tune_gnn --llm_tune lora --double True \
  --max_epochs 20 --batch_size 8 --inference_batch_size 2
```

Fine-tune

```bash
python stage2.py \
  --root data/opt/train/ --valid_root data/opt/valid/ \
  --devices 0,1 --filename stage2 \
  --stage2_path all_checkpoints/pretrain_stage3.ckpt \
  --init_checkpoint all_checkpoints/pretrain_stage3.ckpt \
  --opt_model facebook/galactica-1.3b --mode pretrain \
  --prompt '[START_I_SMILES]{}[END_I_SMILES]' \
  --tune_gnn --llm_tune lora --double True \
  --max_epochs 8 --batch_size 16 --inference_batch_size 2
```

Evaluation uses the same data roots and a saved optimization checkpoint:

```bash
python stage2.py \
  --root data/opt/train/ --valid_root data/opt/valid/ \
  --devices 0, --filename stage2 \
  --stage2_path all_checkpoints/stage2/checkpoint_molopt.ckpt \
  --init_checkpoint all_checkpoints/stage2/checkpoint_molopt.ckpt \
  --opt_model facebook/galactica-1.3b --mode eval \
  --prompt '[START_I_SMILES]{}[END_I_SMILES]' \
  --tune_gnn --llm_tune lora --double True --batch_size 1
```

## Reproducibility notes

- Set `--seed` in `data_finetune_molopt.py` and `--seed` in `stage2.py` when comparing runs.
- Use `--overwrite` when regenerating a data root so stale numbered examples are removed.
- Augmentation should be applied to the training split only. Keep validation molecules untouched to avoid leakage.
- `gamma=0.1` is the default used in the experiments; values closer to 0 or 1 keep the intermediate closer to one endpoint.

## Citation

If you use this code, cite the MolSPC paper with the title above and report whether virtual intermediate augmentation was enabled during fine-tuning.
