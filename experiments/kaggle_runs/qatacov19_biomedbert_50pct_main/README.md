# qatacov19_biomedbert_50pct_main

## Notebook

`QATACOV19_QATALVIT_BIOMEDBERT_050PCT_GPTPRO_FIXED.ipynb`

## Python source

`QATACOV19_QATALVIT_BIOMEDBERT_050PCT_GPTPRO_FIXED.py`

## Dùng trong report

Bản QaTa-COV19 50% BiomedBERT trước mốc locked, giữ lại để đối chiếu với notebook locked.

## Kết quả / vai trò

Bản tham chiếu trước mốc locked; mốc chính 84.03 / 75.62 nằm ở package `qatacov19_biomedbert_50pct_rerun_epi099_locked`.

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
