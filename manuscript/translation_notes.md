# Ghi chú thuật ngữ đã Việt hóa trong báo cáo

Tài liệu này tổng hợp các từ/cụm từ tiếng Anh đã được chuyển sang tiếng Việt trong báo cáo để giữ cách dùng thống nhất giữa các phần.

## Nguyên tắc chung

- Giữ nguyên tiếng Anh với tên phương pháp, tên mô-đun, tên mô hình, chỉ số và thuật ngữ kỹ thuật cần giữ dạng gốc.
- Ưu tiên dịch sang tiếng Việt với các từ phổ thông trong văn phong học thuật.
- Các cụm thuộc họ `attention` được giữ bằng tiếng Anh theo yêu cầu hiện tại.

## Các thuật ngữ giữ nguyên tiếng Anh

- `LViT`, `QaTaLViT`, `PLAM`, `StructuredTextConditioner`, `CrossModalSkipAdapter`, `BottleneckCrossAttention`
- `Dice`, `mIoU`, `EMA`, `Focal Tversky`, `BCE`
- `Transformer`, `CNN`, `BiomedBERT`, `CLIP`, `ViLT`, `LAVT`, `FCN`, `U-Net`, `UNet++`, `nnU-Net`
- `attention`, `cross-attention`, `self-attention`, `local attention`, `pixel-word attention`
- `teacher`, `student`, `laterality`

## Các thuật ngữ đã dịch sang tiếng Việt

| Tiếng Anh | Cách dùng trong báo cáo |
| --- | --- |
| `prior` | `gợi ý` |
| `spatial prior` | `gợi ý định vị không gian` |
| `text spatial prior` | `gợi ý định vị từ văn bản` |
| `text prior` | `gợi ý văn bản` |
| `teacher-prior fusion` | `hợp nhất teacher với gợi ý văn bản` |
| `template` | `khuôn mẫu` |
| `pseudo-label` | `nhãn giả` |
| `mask` | `mặt nạ` |
| `benchmark` | giữ nguyên `benchmark` |
| `baseline` | `mô hình nền` |
| `ablation` | `phân tích loại bỏ thành phần` |
| `fusion` | `hợp nhất` |
| `alignment` | `căn chỉnh` |
| `text alignment loss` | `hàm mất mát căn chỉnh văn bản` |
| `pipeline` | `quy trình` hoặc `kiến trúc` tùy ngữ cảnh |
| `module` | `mô-đun` |
| `encoder` | `bộ mã hóa` |
| `decoder` | `bộ giải mã` |
| `skip connection` | giữ nguyên `skip connection` |
| `skip feature` | giữ nguyên `skip feature` |
| `bottleneck` | `nút thắt` |
| `checkpoint` | `mốc lưu` |
| `epoch` | giữ nguyên `epoch` |
| `val` | `kiểm định` |
| `holdout` | `tập giữ lại` |
| `test set` | `tập kiểm tra` |
| `train` | `huấn luyện` |
| `feature` | `đặc trưng` |
| `embedding` | `véc-tơ nhúng` hoặc `lớp nhúng` |
| `learning rate` | giữ nguyên `learning rate` |
| `weight decay` | giữ nguyên `weight decay` |
| `batch size` | giữ nguyên `batch size` |
| `gradient accumulation` | `cộng dồn gradient` |
| `overfitting` | `quá khớp` |
| `false negative` | `âm tính giả` |
| `rendered figure` / `rendered image` | `ảnh dựng sẵn` |
| `saliency map` | `bản đồ độ nổi bật` |
| `claim` | `khẳng định` |

## Các cách diễn đạt đã chuẩn hóa

- `vùng cần attention`, không dùng `vùng cần chú ý`.
- `cross-attention ở nút thắt`, không dùng `chú ý chéo ở nút thắt`.
- `attention tại nút thắt`, không dùng `chú ý tại nút thắt`.
- `hợp nhất nhãn giả theo độ tin cậy`, không dùng các cụm quá lạ như `cơ chế hợp nhất pseudo-label có nhận thức bất định`.
- `bộ nhớ mẫu đại diện` và các diễn đạt liên quan `prototype/fallback` đã bị loại khỏi báo cáo vì không phải thành phần thực sự được triển khai.

## Gợi ý khi chỉnh sửa tiếp

- Nếu thêm thuật ngữ mới, ưu tiên kiểm tra xem đó là tên riêng hay từ phổ thông trước khi quyết định giữ tiếng Anh hay dịch.
- Với thuật ngữ thuộc họ `attention`, giữ nguyên tiếng Anh để đồng bộ toàn bài.
- Với các thuật ngữ như `benchmark`, `skip connection`, `skip feature`, `epoch`, `learning rate`, `weight decay`, `batch size`, giữ nguyên tiếng Anh để tránh gượng trong văn phong kỹ thuật.
