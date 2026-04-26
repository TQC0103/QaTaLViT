# Datasets

This repository does not track medical image datasets or trained checkpoints. Keep local data under `data/`, `datasets/`, or another ignored directory.

## QaTa-COV19

Public dataset:

- Kaggle: https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset

Expected layout:

```text
QaTa-Covid19/
|-- Train_Folder/
|   |-- Train_text.xlsx
|   |-- img/
|   `-- labelcol/
|-- Val_Folder/
|   |-- Val_text.xlsx
|   |-- img/
|   `-- labelcol/
`-- Test_Folder/
    |-- Test_text.xlsx
    |-- img/
    `-- labelcol/
```

## MosMedData+

Public sources:

- Medical Segmentation: http://medicalsegmentation.com/covid19/
- Kaggle mirror: https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset

Use the same LViT-style split folders as above after preprocessing.

## Checkpoints

Model checkpoints are ignored by Git. For a public release, upload checkpoints to a release asset, Kaggle model, Hugging Face repository, or another external storage endpoint, then document the download path here.
