# ICPR 2026 TVRID — FastReID

FastReID-based research code for the **ICPR 2026 Privacy-Preserving Person
Re-Identification from Top-View RGB-Depth Camera (TVRID)** competition. This
fork adds TVRID data loaders and training components for RGB, depth, and
RGB↔Depth person re-identification.

The benchmark, dataset, evaluation protocol, and official results are maintained
in the [official TVRID repository](https://github.com/RaphaelDel/ICPR-2026-TVRID).
This repository is a participant implementation, not the official competition
codebase.

## Result

The [official final leaderboard](https://github.com/RaphaelDel/ICPR-2026-TVRID#winners)
lists **Hien Pham Duy et al.** in first place on the RGB track with **100.0%
CMC@1** and **100.0% mAP**.

## What is included

- TVRID dataset adapters for RGB, depth, and cross-modal retrieval.
- Single-frame, multi-frame, dynamic-frame, DB-only, combined, and
  cross-camera validation variants.
- Depth-guided RGB foreground masking and body/background augmentations.
- TransReID, ViT/PCB-style backbones, XBM and class-memory losses.
- Re-ranking and extended distance/evaluation utilities.
- CPU data-loader fallback and Kaggle dataset path detection.

The implementation builds on [FastReID](https://github.com/JDAI-CV/fast-reid).
General framework documentation remains available in [INSTALL.md](INSTALL.md),
[GETTING_STARTED.md](GETTING_STARTED.md), and [MODEL_ZOO.md](MODEL_ZOO.md).

## Installation

The original environment targets Linux or macOS, Python 3.6+, and PyTorch 1.6+.
Follow [INSTALL.md](INSTALL.md) to install PyTorch, then install the remaining
dependencies:

```bash
pip install -r docs/requirements.txt
```

## Data

Download the TVRID data from the source linked by the
[official repository](https://github.com/RaphaelDel/ICPR-2026-TVRID#data-download-and-layout).
For the extracted benchmark, use this layout at the repository root:

```text
data/
└── DB_extracted/
    ├── train_labels.csv
    ├── public_test_labels.csv
    ├── train/
    └── test_public/
```

The loaders also detect optional auxiliary datasets at
`data/TVPR_extracted/` and `data/TVPR_2_extracted/` when present. Each should
contain `train_labels.csv` and a `train/` directory.

## Training

The simplest RGB baseline uses the existing FastReID bag-of-tricks config and
selects the registered `TVRID_RGB` dataset:

```bash
python tools/train_net.py \
  --config-file configs/Base-bagtricks.yml \
  --num-gpus 1 \
  DATASETS.NAMES '("TVRID_RGB",)' \
  DATASETS.TESTS '("TVRID_RGB",)' \
  OUTPUT_DIR logs/tvrid_rgb
```

Replace `TVRID_RGB` with `TVRID_Depth` or `TVRID_Cross` for the other tracks.
Additional registered variants are defined in
[`fastreid/data/datasets/tvrid.py`](fastreid/data/datasets/tvrid.py), including
multi-frame and custom validation splits. Multi-frame selection can be changed
from the command line:

```bash
DATASETS.N_FRAMES 20 DATASETS.FRAME_STRATEGY middle_expand
```

Evaluate a checkpoint with the same dataset and config:

```bash
python tools/train_net.py \
  --config-file configs/Base-bagtricks.yml \
  --eval-only \
  DATASETS.TESTS '("TVRID_RGB",)' \
  MODEL.WEIGHTS /path/to/model.pth
```

Exact competition-result configs and checkpoints are not included in this
repository.

## Citation

If this code or the TVRID benchmark is useful in your research, cite the
[official competition repository](https://github.com/RaphaelDel/ICPR-2026-TVRID)
and its [companion report](https://arxiv.org/abs/2605.04977):

```bibtex
@article{delecluse2026tvrid,
  title   = {ICPR 2026 Competition on Privacy-Preserving Person
             Re-Identification from Top-View RGB-Depth Camera (TVRID)},
  author  = {Del{\'e}cluse, Rapha{\"e}l and Wannous, Hazem and Guimas, Laurent},
  journal = {arXiv preprint arXiv:2605.04977},
  year    = {2026}
}
```

Please also cite FastReID:

```bibtex
@article{he2020fastreid,
  title   = {FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author  = {He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and
             Cheng, Peng and Mei, Tao},
  journal = {arXiv preprint arXiv:2006.02631},
  year    = {2020}
}
```

## License

This repository retains FastReID's [Apache 2.0 license](LICENSE). The TVRID
dataset and official competition materials are subject to their respective
terms.
