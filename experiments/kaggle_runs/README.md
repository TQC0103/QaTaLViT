# Source Submission Packages

Folder này gồm các bộ source cần nộp kèm report. Mỗi folder con là một package độc lập, gồm:

- 1 notebook Kaggle-ready (`.ipynb`).
- 1 file Python export trực tiếp từ notebook (`.py`).
- 1 README ngắn giải thích package đó dùng cho thí nghiệm nào.

## Cách Dùng Chung

1. Chọn package đúng với bảng hoặc thí nghiệm cần chạy.
2. Trên Kaggle, tạo notebook mới hoặc upload trực tiếp file `.ipynb` trong package.
3. Add Input dataset public tương ứng: QaTa-COV19 hoặc MosMedData+.
4. Chọn GPU T4 x2 cho các cấu hình MosMed/LightAU và các lần train đầy đủ.
5. Run All. Notebook đã gom code train, validate, test và xuất checkpoint/output theo cấu hình bên trong.

## Lưu Ý Nộp Bài

- Không đóng gói dataset vào folder này; dataset dùng bản public đã đăng trên Kaggle.
- File `.py` trong mỗi package được export và làm sạch từ chính notebook, giúp người chấm xem hoặc kiểm tra cú pháp source mà không cần mở notebook.
- Các mốc LViT-T/LViT-TW trong report là mốc so sánh công bố hoặc tham chiếu; package baseline được nộp để minh bạch source đối chiếu.
- Mốc recipe chuẩn QaTa-COV19 50% trong report là package `qatacov19_biomedbert_50pct_rerun_epi099_locked`; package `qatacov19_biomedbert_50pct_main` là bản BiomedBERT 50% trước đó được giữ để đối chiếu.

## Danh Sách Package

| Package | Dataset | Kết quả / vai trò |
|---|---|---|
| `baseline_lvit_original_reference` | QaTa-COV19 / MosMedData+ | LViT-TW và LViT-T: xem các mốc Dice/mIoU công bố trong README package baseline |
| `qatacov19_all_in_one_25_50_100_runner` | QaTa-COV19 public Kaggle input | 25%: 82.22 / 73.35; 50%: 84.03 / 75.62; 100%: 85.28 / 77.25 |
| `qatacov19_biomedbert_50pct_rerun_epi099_locked` | QaTa-COV19 public Kaggle input | Mốc QaTa-COV19 50% locked: 84.03 / 75.62 |
| `qatacov19_biomedbert_50pct_main` | QaTa-COV19 public Kaggle input | Bản BiomedBERT 50% trước mốc locked, dùng để đối chiếu |
| `qatacov19_nobert_50pct_text_encoder` | QaTa-COV19 public Kaggle input | QaTa-COV19 50%: 83.76 / 75.28 |
| `qatacov19_i3_i5_clean_off_ablation` | QaTa-COV19 public Kaggle input | B0: 79.54 / 69.71; I1: 80.89 / 71.54 |
| `qatacov19_hp_batch_global16_lr3e4_clean` | QaTa-COV19 public Kaggle input | Batch 16, lr 3e-4: 83.63 / 75.18 |
| `qatacov19_hp_batch_global24_lr3e4_clean` | QaTa-COV19 public Kaggle input | Batch-size ablation, lr 3e-4: 83.66 / 75.09 |
| `qatacov19_hp_lr1e3_global32` | QaTa-COV19 public Kaggle input | QaTa-COV19 50% HP row: 82.55 / 73.76 |
| `qatacov19_hp_lr1e4_global32` | QaTa-COV19 public Kaggle input | QaTa-COV19 50% HP row: 83.57 / 74.99 |
| `qatacov19_hp_globalbs16_lr3e4` | QaTa-COV19 public Kaggle input | QaTa-COV19 50% HP row: 83.63 / 75.18 |
| `mosmed_biomedbert_base_50pct` | MosMedData+ public Kaggle input | MosMedData+ 50%: 68.88 / 55.14 |
| `mosmed_biomedbert_preprocess_crop_50pct` | MosMedData+ public Kaggle input | MosMedData+ 50%: 70.36 / 56.97 |
| `mosmed_biomedbert_studentonly_50pct` | MosMedData+ public Kaggle input | MosMedData+ 50%: 72.85 / 59.41 |
| `mosmed_nobert_studentonly_valthr_50pct` | MosMedData+ public Kaggle input | MosMedData+ NoBERT validation-threshold: 72.69 / 58.99 |
| `mosmed_nobert_studentonly_fixedthr050_50pct` | MosMedData+ public Kaggle input | MosMedData+ 50%: 72.99 / 59.48 |
| `mosmed_biomedbert_lightau_50pct` | MosMedData+ public Kaggle input | MosMedData+ 50%: 73.84 / 60.31 |
| `mosmed_nobert_lightau_fixedthr050_50pct` | MosMedData+ public Kaggle input | MosMedData+ 50%: 72.77 / 59.22 |

## Mapping Với Report

- Mốc baseline LViT-T/LViT-TW: `baseline_lvit_original_reference`. Đây là source baseline gốc; các số trong report vẫn lấy theo mốc công bố/trích dẫn nếu không chạy lại.
- Bảng text encoder / ablation QaTa-COV19: `qatacov19_biomedbert_50pct_rerun_epi099_locked`, `qatacov19_biomedbert_50pct_main`, `qatacov19_nobert_50pct_text_encoder`, `qatacov19_i3_i5_clean_off_ablation`.
- Các mốc QaTa-COV19 25% và 100%: `qatacov19_all_in_one_25_50_100_runner`, đổi `LABEL_RATIO` thành `0.25` hoặc `1.0`.
- Bảng hyperparameter QaTa-COV19: `qatacov19_biomedbert_50pct_rerun_epi099_locked`, `qatacov19_hp_lr1e3_global32`, `qatacov19_hp_lr1e4_global32`, `qatacov19_hp_globalbs16_lr3e4`, kèm hai notebook clean ablation `qatacov19_hp_batch_global16_lr3e4_clean` và `qatacov19_hp_batch_global24_lr3e4_clean`.
- Bảng thiết lập MosMedData+: `mosmed_biomedbert_base_50pct`, `mosmed_biomedbert_preprocess_crop_50pct`, `mosmed_biomedbert_studentonly_50pct`, `mosmed_nobert_studentonly_valthr_50pct`, `mosmed_nobert_studentonly_fixedthr050_50pct`, `mosmed_biomedbert_lightau_50pct`, `mosmed_nobert_lightau_fixedthr050_50pct`.
