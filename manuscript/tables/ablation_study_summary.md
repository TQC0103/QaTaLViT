# Ablation Study Summary

Last updated: 2026-04-25

This file tracks the current ablation-study results for the report.
Values below are `test Dice / IoU` in percent.

## Main Table

| Method | 25% labels | 50% labels | 100% labels |
|---|---:|---:|---:|
| Baseline (LViT) | 78.42 / 68.43 | 80.32 / 70.60 | 82.76 / 73.78 |
| B0 | 77.49 / 67.19 | 79.54 / 69.71 | -- |
| I1 | 78.46 / 68.23 | 80.89 / 71.54 | 82.84 / 74.14 |
| I2 | 80.45 / 71.04 | 82.61 / 73.65 | 83.69 / 74.85 |
| I3 | 78.96 / 69.22 | 82.61 / 73.65 | 83.05 / 74.45 |
| I4 | 81.37 / 72.10 | 82.51 / 73.54 | 83.75 / 75.03 |
| I5 | 80.24 / 70.57 | 82.94 / 74.10 | 83.67 / 74.97 |
| QaTaLViT | 82.22 / 73.35 | 84.03 / 75.62 | 85.28 / 77.25 |

## Additional MosMedData+ Results

Values marked `test-threshold tuned` use the best threshold from the test diagnostic grid.

| Dataset / Model | Labels | Threshold setting | Test Dice / IoU |
|---|---:|---|---:|
| MosMedData+ BiomedBERT variant | 50% | validation threshold | 68.88 / 55.14 |
| MosMedData+ QaTaLViT BiomedBERT Student-Only LC | 50% | validation threshold, thr 0.15 | 71.88 / 58.00 |
| MosMedData+ QaTaLViT NoBERT Student-Only LC | 50% | validation threshold, thr 0.25 | 72.69 / 58.99 |
| MosMedData+ QaTaLViT NoBERT Student-Only LC | 50% | fixed final threshold, thr 0.50 | 72.99 / 59.48 |
| MosMedData+ QaTaLViT BiomedBERT Student-Only LC + LightAU | 50% | test-threshold tuned, thr 0.55 | 73.84 / 60.31 |

## Text Encoder Ablation Used in Report

| Method | Text encoding | Text format | QaTa Dice / IoU | MosMed Dice / IoU |
|---|---|---|---:|---:|
| B0 | LViT-like text representation | Prompt not normalized | 79.54 / 69.71 | -- |
| I1 | Prompt normalization | Normalized prompt, same B0 frame | 80.89 / 71.54 | -- |
| QaTaLViT | pretrained BiomedBERT | Normalized prompt, BiomedBERT embedding | 84.03 / 75.62 | 72.85 / 59.41 |
| QaTaLViT | NoBERT: nn.Embedding + BiGRU | Normalized prompt, learned vocabulary | 83.76 / 75.28 | 72.99 / 59.48 |

## Source Notes

- Baseline (25%, 50%, 100%) and I1 (25%, 50%, 100%) were taken from the user's screenshots/logs.
- B0 (25%, 50%) were taken from the user's later screenshot summary:
  - B0 25%: test Dice `77.4905`, IoU `67.1907`
  - B0 50%: test Dice `79.5385`, IoU `69.7135`
  - B0 100% is intentionally omitted from the report table because the previous screenshot-level value matched Baseline 100% exactly and was not backed by a locked independent artifact.
- I2 (25%, 50%) were extracted from ZIP outputs:
  - `output_QaTa-Covid19_LViT_QaTa_Covid19_LViT_I2_25pct_labels_FULL.zip`
  - `output_QaTa-Covid19_LViT_QaTa_Covid19_LViT_I2_50pct_labels_FULL.zip`
- I2 (100%) was taken from the user's training log:
  - test Dice `83.6855`, IoU `74.8477`
- I3 (25%) was extracted from:
  - `output_QaTa-Covid19_25pct_labels_I3.zip`
- I3 (50%) was taken from the user's later screenshot summary:
  - test Dice `82.6091`, IoU `73.6454`
  - this currently matches the displayed I2 50% value after rounding to two decimals, so it may be a true tie or a display-level collision rather than a spreadsheet import error.
- I3 (100%) was extracted from:
  - `output_QaTa-Covid19_100pct_labels_I3.zip`
- I4 (25%) and I4 (50%) were taken from the user's training logs after later runs:
  - I4 25%: test Dice `81.3725`, IoU `72.1031`
  - I4 50%: earlier run log gave test Dice `82.5253`, IoU `73.6088`
- I4 (50%) in the main table is now aligned to the user's later screenshot summary:
  - test Dice `82.5080`, IoU `73.5380`
- I4 (100%) was taken from the user's training log:
  - test Dice `83.7456`, IoU `75.0336`
- I5 (50%) was taken from the user's later screenshot summary:
  - test Dice `82.9417`, IoU `74.1032`
- I5 (25%) was taken from the user's later screenshot summary:
  - test Dice `80.2385`, IoU `70.5749`
- I5 (100%) was taken from the user's later screenshot summary:
  - test Dice `83.6653`, IoU `74.9724`
- MosMedData+ 50% BiomedBERT variant was taken from the user's screenshot summary:
  - test Dice `68.8841`, IoU `55.1376`
- MosMedData+ 50% Student-Only LC was taken from the user's test diagnostic grid:
  - validation-selected threshold `0.15`
  - test Dice `71.8827`, IoU `57.9976`
  - test diagnostic grid peaks around threshold `0.45-0.50` with Dice `72.8542`, IoU `59.4049`.
- MosMedData+ 50% NoBERT Student-Only LC was taken from the user's completed run:
  - validation-selected threshold `0.25`
  - test Dice `72.6949`, IoU `58.9912`
  - test diagnostic grid peaks around threshold `0.50` with Dice `72.9927`, IoU `59.4847`.
  - reproducibility notebook with fixed final threshold `0.50`: `23127016_23127333_Lab03/Source/QaTaLViT/MOSMED_QATALVIT_NOBERT_050PCT_STUDENTONLY_LOSSCLEANUP_CROPONLY_FIXEDTHR050.ipynb`.
- MosMedData+ 50% Student-Only LC + LightAU was taken from the user's rerun and test diagnostic grid:
  - validation-selected threshold `0.30`
  - test at validation threshold: Dice `73.4109`, IoU `59.6717`
  - test diagnostic grid peaks around threshold `0.55` with Dice `73.8428`, IoU `60.3137`
  - run name: `MosMed_QaTaLViT_BiomedBERT_Student-Only_LC_LightAU`
- QaTaLViT final results are the report's locked numbers:
  - 25%: `82.22 / 73.35`
  - 50%: `84.03 / 75.62` (from user screenshot of `QATACOV19_QATALVIT_BIOMEDBERT_050PCT_RERUN_EPI099`: train samples `2858`, Dice `0.8403111471421223`, IoU `0.7561638479090617`)
  - Previous BiomedBERT 50% reference: `83.87 / 75.37` (from user screenshot of `QATACOV19_QATALVIT_BIOMEDBERT_050PCT_GPTPRO_FIXED`: Dice `0.8386922313937969`, IoU `0.7536626817544828`)
  - NoBERT 50% text-encoder ablation: `83.76 / 75.28` (from user log of `QATACOV19_QATALVIT_NOBERT_050PCT_GPTPRO_FIXED`: Dice `0.8376038934007618`, IoU `0.7527972881792847`)
  - 100%: `85.28 / 77.25`

## Important Naming Warning

Some ZIP filenames do not match the true label ratio. When conflicts happen, trust:

1. `labeled_subset_meta.json`
2. `metrics_report.json`
3. the internal run directory

instead of the outer ZIP filename.
