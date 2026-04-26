# qatacov19_hp_lr1e4_global32

## Notebook

`QATACOV19_QATALVIT_BIOMEDBERT_050PCT_HP_LR1E4_GLOBAL32.ipynb`

## Python source

`QATACOV19_QATALVIT_BIOMEDBERT_050PCT_HP_LR1E4_GLOBAL32.py`

## Dùng trong report

Hyperparameter ablation: effective batch size 32, lr = 1e-4.

## Kết quả / vai trò

QaTa-COV19 50% HP row: 83.57 / 74.99

## Dataset cần thêm trên Kaggle

QaTa-COV19 public Kaggle input

## Cách chạy nhanh trên Kaggle

1. Tạo Kaggle Notebook mới.
2. Add Input dataset public tương ứng: QaTa-COV19 public Kaggle input.
3. Upload notebook trong folder này lên Kaggle.
4. Chọn accelerator GPU T4 x2 nếu notebook cần train; các cấu hình nhẹ hơn vẫn có thể chạy trên 1 GPU nhưng chậm hơn.
5. Run All. Notebook đã gom code train, validate, test và xuất checkpoint/output cần thiết.

## Ghi chú

File `.py` là bản export đã được làm sạch từ notebook: các lệnh shell và `%%writefile` đã được chuyển sang Python thường để có thể kiểm tra bằng `python -m py_compile`. Khi chạy thí nghiệm đầy đủ trên Kaggle, vẫn ưu tiên file `.ipynb` vì notebook có bố cục cell rõ hơn.
