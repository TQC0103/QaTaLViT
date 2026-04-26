# mosmed_biomedbert_preprocess_crop_50pct

## Notebook

`MOSMED_QATALVIT_BIOMEDBERT_050PCT_GPTPRO_FIXED_WIDEGRID_PSEUDOTHR025_CROPONLY.ipynb`

## Python source

`MOSMED_QATALVIT_BIOMEDBERT_050PCT_GPTPRO_FIXED_WIDEGRID_PSEUDOTHR025_CROPONLY.py`

## Dùng trong report

MosMedData+ BiomedBERT với tiền xử lý crop-only và điều chỉnh ngưỡng pseudo-label.

## Kết quả / vai trò

MosMedData+ 50%: 70.36 / 56.97

## Dataset cần thêm trên Kaggle

MosMedData+ public Kaggle input

## Cách chạy nhanh trên Kaggle

1. Tạo Kaggle Notebook mới.
2. Add Input dataset public tương ứng: MosMedData+ public Kaggle input.
3. Upload notebook trong folder này lên Kaggle.
4. Chọn accelerator GPU T4 x2 nếu notebook cần train; các cấu hình nhẹ hơn vẫn có thể chạy trên 1 GPU nhưng chậm hơn.
5. Run All. Notebook đã gom code train, validate, test và xuất checkpoint/output cần thiết.

## Ghi chú

File `.py` là bản export đã được làm sạch từ notebook: các lệnh shell và `%%writefile` đã được chuyển sang Python thường để có thể kiểm tra bằng `python -m py_compile`. Khi chạy thí nghiệm đầy đủ trên Kaggle, vẫn ưu tiên file `.ipynb` vì notebook có bố cục cell rõ hơn.
