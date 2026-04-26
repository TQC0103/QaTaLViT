# QaTaLViT

QaTaLViT is a text-augmented medical image segmentation project built on top of LViT. The repository is organized as a paper-style research repo: core code, reproducible experiment notebooks, LaTeX manuscript source, and presentation material are kept separate.

## Repository Layout

```text
.
|-- src/                 # Core PyTorch training, evaluation, model, and dataset code
|-- scripts/             # Experiment launchers, Kaggle bundle builders, analysis helpers
|-- notebooks/           # Standalone Kaggle/Colab notebooks
|-- experiments/         # Curated Kaggle notebook/script snapshots used by the report
|-- manuscript/          # LaTeX source, figures, tables, bibliography, compiled report PDF
|-- slides/              # Beamer slide source, scripts, and compiled slide PDFs
|-- docs/                # Dataset and Kaggle usage notes
|-- tests/               # Lightweight unit tests for dataset/model/training utilities
```

Large datasets, checkpoints, run logs, generated zip packages, and LaTeX build artefacts are intentionally ignored. Use the public datasets listed in `docs/DATA.md` and keep local data under `data/` or another ignored directory.

## Main Results

The report studies QaTaLViT on QaTa-COV19 at 25%, 50%, and 100% label ratios, and extends the analysis to MosMedData+ at 50% labels. The most recent experimental tables and discussion are in `manuscript/main.pdf`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For Kaggle notebooks, use `requirements-kaggle.txt` and the notebook-specific setup cells.

## Training

Example for QaTa-COV19 50% labels:

```powershell
python scripts/run_qatacov19_050pct.py --dataset-root C:\path\to\QaTa-Covid19 --epochs 120
```

The launcher writes outputs to `runs/`, which is ignored by Git. Use `--resume` to continue from a matching checkpoint in the run directory.

## Notebooks

`notebooks/KAGGLE_QATACOV19_ALL_IN_ONE.ipynb` is a self-contained Kaggle notebook. The curated notebooks under `experiments/kaggle_runs/` correspond to the ablations and MosMedData+ runs discussed in the manuscript.

## Manuscript

The LaTeX report lives in `manuscript/`.

```powershell
cd manuscript
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

## Slides

The presentation sources and compiled PDFs live in `slides/`.

```powershell
cd slides
xelatex -interaction=nonstopmode qatalvit_high_level_slides.tex
xelatex -interaction=nonstopmode qatalvit_report_summary_slides.tex
```

## Acknowledgement

This project builds on the public LViT implementation and paper:

```bibtex
@article{li2023lvit,
  title={LViT: Language Meets Vision Transformer in Medical Image Segmentation},
  author={Li, Zihan and Li, Yunxiang and Li, Qingde and Wang, Puyang and Guo, Dazhou and Lu, Le and Jin, Dakai and Zhang, You and Hong, Qingqi},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
```
