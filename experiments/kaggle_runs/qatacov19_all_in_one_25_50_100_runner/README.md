# qatacov19_all_in_one_25_50_100_runner

## Notebook

`KAGGLE_QATACOV19_ALL_IN_ONE.ipynb`

## Python source

`KAGGLE_QATACOV19_ALL_IN_ONE.py`

## Dùng trong report

Notebook all-in-one để chạy lại ba mốc QaTa-COV19 25%, 50% và 100% bằng biến LABEL_RATIO.

## Kết quả / vai trò

QaTa-COV19 25%: 82.22 / 73.35; 50%: 84.03 / 75.62; 100%: 85.28 / 77.25.

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
