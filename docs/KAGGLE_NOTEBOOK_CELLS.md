# Kaggle Notebook Cells

## Cell 1: copy code and install dependencies

```python
!mkdir -p /kaggle/working/lvit_improved
!cp -r /kaggle/input/lvit-improved-kaggle-bundle/* /kaggle/working/lvit_improved/
%cd /kaggle/working/lvit_improved
!pip install -q -r requirements-kaggle.txt
```

## Cell 2: run 100% labels

```python
%cd /kaggle/working/lvit_improved
!python run_qatacov19_100pct.py --dataset-root /kaggle/input/qatacov19-lvit-format --save-root /kaggle/working/qatacov19_100pct --epochs 120 --batch-size 4 --grad-accum-steps 4 --amp
```

## Cell 3: run 50% labels

```python
%cd /kaggle/working/lvit_improved
!python run_qatacov19_050pct.py --dataset-root /kaggle/input/qatacov19-lvit-format --save-root /kaggle/working/qatacov19_050pct --epochs 120 --batch-size 4 --grad-accum-steps 4 --amp
```

## Cell 4: run 25% labels

```python
%cd /kaggle/working/lvit_improved
!python run_qatacov19_025pct.py --dataset-root /kaggle/input/qatacov19-lvit-format --save-root /kaggle/working/qatacov19_025pct --epochs 120 --batch-size 4 --grad-accum-steps 4 --amp
```
