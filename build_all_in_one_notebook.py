import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
NOTEBOOK_PATH = ROOT / "KAGGLE_QATACOV19_ALL_IN_ONE.ipynb"
DIST_NOTEBOOK_PATH = DIST_DIR / NOTEBOOK_PATH.name


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def build_utils_source() -> str:
    return """import numpy as np
import pandas as pd


def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_on_batch(masks, pred):
    dices = []
    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        dices.append(dice_coef(mask_tmp, pred_tmp))
    return np.mean(dices)


def read_text(filename):
    df = pd.read_excel(filename, engine='openpyxl')
    text = {}
    for i in df.index.values:
        count = len(df.Description[i].split())
        if count < 9:
            df.loc[i, 'Description'] = df.Description[i] + ' EOF XXX' * (9 - count)
        text[df.Image[i]] = df.Description[i]
    return text
"""


def build_notebook() -> dict:
    text_encoder_source = read_source("text_encoder.py")
    dataset_source = read_source("Load_Dataset.py")
    improved_ssl_source = read_source("nets/improved_ssl.py")
    improved_model_source = read_source("nets/LViT_improved.py")
    improved_training_source = read_source("nets/improved_training.py")
    train_script_source = read_source("train_improved_ssl.py")

    module_loader_source = f"""import sys
import types


def register_module(module_name: str, source: str, package: str | None = None):
    module = types.ModuleType(module_name)
    module.__file__ = module_name.replace('.', '/') + '.py'
    module.__package__ = package if package is not None else module_name.rpartition('.')[0]
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


nets_pkg = types.ModuleType("nets")
nets_pkg.__path__ = []
sys.modules["nets"] = nets_pkg

UTILS_SOURCE = {build_utils_source()!r}
TEXT_ENCODER_SOURCE = {text_encoder_source!r}
LOAD_DATASET_SOURCE = {dataset_source!r}
IMPROVED_SSL_SOURCE = {improved_ssl_source!r}
IMPROVED_MODEL_SOURCE = {improved_model_source!r}
IMPROVED_TRAINING_SOURCE = {improved_training_source!r}
TRAIN_SCRIPT_SOURCE = {train_script_source!r}

register_module("utils", UTILS_SOURCE)
register_module("text_encoder", TEXT_ENCODER_SOURCE)
register_module("Load_Dataset", LOAD_DATASET_SOURCE)
register_module("nets.improved_ssl", IMPROVED_SSL_SOURCE, package="nets")
register_module("nets.LViT_improved", IMPROVED_MODEL_SOURCE, package="nets")
register_module("nets.improved_training", IMPROVED_TRAINING_SOURCE, package="nets")
register_module("train_improved_ssl", TRAIN_SCRIPT_SOURCE)

print("Inline modules registered successfully.")
"""

    train_cell_source = """import os
import sys

os.makedirs(SAVE_DIR, exist_ok=True)

argv = [
    "train_improved_ssl.py",
    "--dataset-root", DATASET_ROOT,
    "--task-name", TASK_NAME,
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--grad-accum-steps", str(GRAD_ACCUM_STEPS),
    "--label-ratio", str(LABEL_RATIO),
    "--image-size", str(IMAGE_SIZE),
    "--lr", str(LR),
    "--min-lr", str(MIN_LR),
    "--weight-decay", str(WEIGHT_DECAY),
    "--scheduler", SCHEDULER,
    "--seed", str(SEED),
    "--num-workers", str(NUM_WORKERS),
    "--ema-decay", str(EMA_DECAY),
    "--text-model-name", TEXT_MODEL_NAME,
    "--max-text-units", str(MAX_TEXT_UNITS),
    "--save-every", str(SAVE_EVERY),
    "--save-val-examples", str(SAVE_VAL_EXAMPLES),
    "--save-dir", SAVE_DIR,
]

if AMP:
    argv.append("--amp")
else:
    argv.append("--no-amp")

if LOCAL_FILES_ONLY:
    argv.append("--local-files-only")

print("Running command:")
print(" ".join(argv))

sys.argv = argv
import train_improved_ssl
train_improved_ssl.main()
"""

    notebook = {
        "cells": [
            markdown_cell(
                "# QaTa-COV19 All-in-One Kaggle Notebook\n\n"
                "Notebook này đã chứa toàn bộ code cần thiết trong chính file `.ipynb`, "
                "không cần import các file Python khác trong repo.\n\n"
                "Cách dùng:\n"
                "1. Upload đúng 1 notebook này lên Kaggle.\n"
                "2. Add dataset QaTa-COV19 đã đúng format LViT.\n"
                "3. Bật GPU và Internet.\n"
                "4. Nếu dùng đúng dataset của bạn thì có thể để nguyên config mặc định.\n"
                "5. Chỉnh `LABEL_RATIO` ở cell config thành `1.0`, `0.5` hoặc `0.25` khi cần.\n"
                "6. Chạy `Run All`."
            ),
            code_cell(
                "!pip uninstall -y -q torch torchvision torchaudio\n"
                "!pip install -q torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121\n"
                "!pip install -q ml-collections openpyxl protobuf sentencepiece thop timm \"transformers<5\"\n"
                "\n"
                "import torch\n"
                "print('torch version:', torch.__version__)\n"
                "print('torch cuda:', torch.version.cuda)\n"
                "if torch.cuda.is_available():\n"
                "    print('gpu:', torch.cuda.get_device_name(0))\n"
                "    print('supported arch:', torch.cuda.get_arch_list())\n"
                "    x = torch.randn(1, device='cuda')\n"
                "    print('cuda smoke tensor:', x)"
            ),
            code_cell(
                "DATASET_ROOT = \"/kaggle/input/datasets/tqc0103/qata-covid19/QaTa-Covid19\"\n"
                "TASK_NAME = \"QaTa-COV19\"\n"
                "LABEL_RATIO = 1.0  # change to 0.5 or 0.25 on other Kaggle accounts\n"
                "SAVE_DIR = f\"/kaggle/working/qatacov19_{int(LABEL_RATIO * 100):03d}pct\"\n"
                "\n"
                "EPOCHS = 120\n"
                "BATCH_SIZE = 4\n"
                "GRAD_ACCUM_STEPS = 4\n"
                "IMAGE_SIZE = 224\n"
                "LR = 1e-4\n"
                "MIN_LR = 1e-6\n"
                "WEIGHT_DECAY = 1e-4\n"
                "SCHEDULER = \"cosine\"\n"
                "SEED = 666\n"
                "NUM_WORKERS = 2\n"
                "EMA_DECAY = 0.99\n"
                "TEXT_MODEL_NAME = \"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext\"\n"
                "MAX_TEXT_UNITS = 10\n"
                "LOCAL_FILES_ONLY = False\n"
                "AMP = False\n"
                "SAVE_EVERY = 0\n"
                "SAVE_VAL_EXAMPLES = 4\n"
                "\n"
                "print('Effective batch size =', BATCH_SIZE * GRAD_ACCUM_STEPS)\n"
                "print('Outputs will be saved to', SAVE_DIR)"
            ),
            code_cell(module_loader_source),
            code_cell(train_cell_source),
            markdown_cell(
                "## Expected outputs\n\n"
                "Sau khi train xong, thư mục `SAVE_DIR` sẽ có:\n\n"
                "- `best_model.pt`\n"
                "- `last_model.pt`\n"
                "- `run_config.json`\n"
                "- `run_summary.json`\n"
                "- `split_manifest.json`\n"
                "- `history.csv`\n"
                "- `history.json`\n"
                "- `training_curves.png`\n"
                "- `val_examples/`"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def main() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy2(NOTEBOOK_PATH, DIST_NOTEBOOK_PATH)
    print(f"Notebook created at: {NOTEBOOK_PATH}")
    print(f"Notebook copied to: {DIST_NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
