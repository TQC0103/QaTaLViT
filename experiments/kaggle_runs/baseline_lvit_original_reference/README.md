# baseline_lvit_original_reference

## Vai Trò

Package này chứa source baseline LViT gốc để người chấm có thể đối chiếu với QaTaLViT. Đây là baseline tham chiếu, không phải mô hình cải tiến chính.

## File Chính

- `train_model.py`: entry point train baseline LViT.
- `test_model.py`: entry point evaluate baseline LViT.
- `Config.py`: cấu hình task, dataset path, learning rate, batch size, checkpoint path.
- `Load_Dataset.py`, `Train_one_epoch.py`, `utils.py`: pipeline dữ liệu và train/eval loop.
- `nets/`: kiến trúc LViT gốc gồm `LViT.py`, `UNet.py`, `Vit.py`, `pixlevel.py`, `textlevel.py`.
- `BASELINE_LVIT_RUNNER.ipynb`: notebook runner tối giản để mở/chạy baseline trên Kaggle hoặc local.

## Kết quả / vai trò

Mốc LViT trong report: LViT-TW 25% = 79.08 / 69.42, 50% = 80.35 / 70.74, 100% = 81.12 / 71.37; LViT-T 25% = 80.95 / 71.31, 50% = 82.73 / 73.99, 100% = 83.66 / 75.11 trên QaTa-COV19. Trên MosMedData+, LViT-TW 25% = 70.65 / 58.07, 50% = 71.89 / 59.63, 100% = 72.58 / 60.40; LViT-T 25% = 72.48 / 60.31, 50% = 73.56 / 61.05, 100% = 74.57 / 61.33.

## Dataset Cần Thêm Trên Kaggle

Dataset QaTa-COV19 hoặc dataset baseline muốn chạy lại, với cấu trúc folder khớp `Config.py`.

## Cách Chạy

1. Upload cả folder này lên Kaggle nếu muốn chạy lại baseline gốc.
2. Add Input dataset public.
3. Sửa `Config.py` cho đúng `task_name` và đường dẫn dataset.
4. Chạy `BASELINE_LVIT_RUNNER.ipynb`, bỏ comment cell `!python train_model.py` hoặc `!python test_model.py`.

## Ghi Chú

Các mốc LViT-T/LViT-TW trong report là mốc so sánh công bố. Package này được nộp để minh bạch source baseline, còn kết quả chính của bài nằm ở các package QaTaLViT/MosMed trong `Source`.
