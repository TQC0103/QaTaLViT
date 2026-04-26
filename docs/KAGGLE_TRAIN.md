# Kaggle Training Guide

## 1. Expected dataset layout

Place your dataset so the script sees this structure:

```text
/kaggle/input/<dataset-name>/
  Train_Folder/
    img/
    labelcol/
    Train_text.xlsx
    text_cache/           # optional but recommended
  Val_Folder/
    img/
    labelcol/
    Val_text.xlsx
    text_cache/           # optional but recommended
```

`Train_text.xlsx` and `Val_text.xlsx` should contain `Image` and `Description` columns.

## 2. Install dependencies

Inside the Kaggle notebook:

```python
!pip install -r requirements-kaggle.txt
```

## 3. Recommended training command

This script is set up to use an effective batch size of 16 by default:

- physical batch size: `4`
- gradient accumulation steps: `4`
- AMP: enabled

Run:

```python
!python train_improved_ssl.py ^
  --dataset-root /kaggle/input/<dataset-name> ^
  --task-name MoNuSeg ^
  --epochs 120 ^
  --batch-size 4 ^
  --grad-accum-steps 4 ^
  --amp ^
  --num-workers 2 ^
  --save-dir /kaggle/working/improved_ssl_run
```

For QaTa-COV19 or MosMedData+, keep the same command and only change:

- `--dataset-root`
- `--task-name`

## 3.1. QaTa-COV19 benchmark runs

To run the three benchmark settings you want:

- `100%` labels
- `50%` labels
- `25%` labels

use:

```python
!python run_qatacov19_benchmark.py ^
  --dataset-root /kaggle/input/<qatacov19-dataset> ^
  --save-root /kaggle/working/qatacov19_benchmark ^
  --epochs 120 ^
  --batch-size 4 ^
  --grad-accum-steps 4 ^
  --amp
```

This will create:

```text
/kaggle/working/qatacov19_benchmark/
  100pct/
  050pct/
  025pct/
```

Each run saves its own:

- split manifest
- config
- history
- best and last checkpoint
- validation examples
- training curves

## 3.2. Run the three settings on three Kaggle accounts

If you want three completely separate files for three separate Kaggle jobs, use:

- [run_qatacov19_100pct.py](/C:/Users/ASUS/OneDrive/Documents/GitHub/LViT_improved/run_qatacov19_100pct.py)
- [run_qatacov19_050pct.py](/C:/Users/ASUS/OneDrive/Documents/GitHub/LViT_improved/run_qatacov19_050pct.py)
- [run_qatacov19_025pct.py](/C:/Users/ASUS/OneDrive/Documents/GitHub/LViT_improved/run_qatacov19_025pct.py)

Examples:

```python
!python run_qatacov19_100pct.py --dataset-root /kaggle/input/<qatacov19-dataset> --save-root /kaggle/working/qatacov19_100pct --epochs 120 --batch-size 4 --grad-accum-steps 4 --amp
!python run_qatacov19_050pct.py --dataset-root /kaggle/input/<qatacov19-dataset> --save-root /kaggle/working/qatacov19_050pct --epochs 120 --batch-size 4 --grad-accum-steps 4 --amp
!python run_qatacov19_025pct.py --dataset-root /kaggle/input/<qatacov19-dataset> --save-root /kaggle/working/qatacov19_025pct --epochs 120 --batch-size 4 --grad-accum-steps 4 --amp
```

## 4. Internet on vs off

### Option A: internet enabled

Let the script download the Hugging Face text encoder on first run:

```python
!python train_improved_ssl.py --dataset-root /kaggle/input/<dataset-name> --save-dir /kaggle/working/improved_ssl_run
```

### Option B: internet disabled

Upload precomputed `text_cache` folders inside `Train_Folder/` and `Val_Folder/`, then run:

```python
!python train_improved_ssl.py ^
  --dataset-root /kaggle/input/<dataset-name> ^
  --local-files-only ^
  --save-dir /kaggle/working/improved_ssl_run
```

Important:

- the uploaded cache must match the same text model name
- the uploaded cache must match the same `--max-text-units`

Otherwise the script will invalidate the cache and try to re-encode text.

## 5. Resume training

Resume from the last checkpoint:

```python
!python train_improved_ssl.py ^
  --dataset-root /kaggle/input/<dataset-name> ^
  --save-dir /kaggle/working/improved_ssl_run ^
  --resume /kaggle/working/improved_ssl_run/last_model.pt
```

## 6. Output files

The script saves these artifacts under `--save-dir`:

- `run_config.json`
- `run_summary.json`
- `history.csv`
- `history.json`
- `training_curves.png`
- `best_model.pt`
- `last_model.pt`
- `val_examples/`
- `checkpoints/` when `--save-every > 0`

## 7. Useful variations

### If memory is tight

```python
!python train_improved_ssl.py ^
  --dataset-root /kaggle/input/<dataset-name> ^
  --batch-size 2 ^
  --grad-accum-steps 8 ^
  --amp ^
  --save-dir /kaggle/working/improved_ssl_run
```

### If you want more frequent checkpoints

```python
!python train_improved_ssl.py ^
  --dataset-root /kaggle/input/<dataset-name> ^
  --save-every 10 ^
  --save-dir /kaggle/working/improved_ssl_run
```
