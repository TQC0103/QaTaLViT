# Script thuyết trình: phần tổng kết high-level toàn report

Thời lượng gợi ý: 12 đến 15 phút. Deck này là phần tóm tắt toàn report, nên các bảng chính trong report đều được đưa vào slide dưới dạng bảng trình bày lại.

## Slide 1. QaTaLViT

Ở phần này, em tóm tắt toàn bộ report: bài toán, ý tưởng ảnh-văn bản, pipeline mô hình, kết quả thực nghiệm, ablation và giới hạn.

## Slide 2. Luận điểm trung tâm của report

Phân đoạn y khoa ít nhãn không chỉ thiếu dữ liệu mà còn thiếu ngữ cảnh. Ảnh y khoa có biên mờ, vùng tổn thương nhỏ và thay đổi theo nguồn chụp. Trong khi đó, văn bản lâm sàng thường đã đi kèm ảnh và có thể chứa thông tin về vị trí, bên tổn thương, số vùng và mức lan tỏa.

## Slide 3. Bảng định vị phương pháp

Bảng này đặt QaTaLViT trong mạch phát triển từ mô hình ảnh đơn, mô hình thị giác-ngôn ngữ tổng quát, LViT, rồi đến QaTaLViT. Điểm khác của QaTaLViT là văn bản được chuẩn hóa thành prompt và hợp nhất vào nhiều mức đặc trưng ảnh.

## Slide 4. Bảng quy mô dữ liệu

Report dùng hai bộ dữ liệu: QaTa-COV19 là X-quang ngực, MosMedData+ là CT lát cắt. QaTa-COV19 được đánh giá đầy đủ ở 25%, 50% và 100% nhãn; MosMedData+ là thí nghiệm mở rộng ở 50% nhãn để kiểm tra chuyển miền và pseudo-label.

## Slide 5. Từ LViT đến QaTaLViT

QaTaLViT kế thừa trực giác của LViT: văn bản hỗ trợ phân đoạn và hỗ trợ học bán giám sát. Phần nhóm thay đổi là prompt normalization, hợp nhất đa mức trên U-Net, và recipe huấn luyện ổn định hơn.

## Slide 6. Bảng đối chiếu LViT và QaTaLViT

Bảng này tóm tắt bốn khác biệt: biểu diễn văn bản, tương tác ảnh-văn bản, decoder và học bán giám sát. Đây là phần giúp người nghe hiểu đóng góp nằm ở thiết kế toàn pipeline, không chỉ ở một text encoder.

## Slide 7. Pipeline mô hình

Pipeline gồm nhánh văn bản, trục CNN kiểu U-Net, LVFusionStage/ViT, và EMA/EPI. Văn bản không trực tiếp sinh mask mà điều chỉnh đặc trưng ảnh trước decoder.

## Slide 8. Sơ đồ kiến trúc trong report

Hình này cho thấy token văn bản đi vào bốn LVFusionStage dọc encoder. Nhờ vậy, văn bản có thể ảnh hưởng ở nhiều mức đặc trưng, từ chi tiết biên ở tầng nông đến ngữ nghĩa tổn thương ở tầng sâu.

## Slide 9. Nhánh văn bản

Prompt normalization đưa các mô tả lâm sàng ngắn về dạng ổn định hơn. Report khảo sát hai lựa chọn: BiomedBERT và NoBERT. Hai lựa chọn này chỉ thay text encoder, còn phần U-Net, LVFusionStage, decoder và loss giữ cùng nguyên tắc.

## Slide 10. Hợp nhất ảnh-văn bản và học ít nhãn

LVFusionStage token hóa feature CNN, dùng token ảnh làm query và token văn bản làm key/value. Với dữ liệu chưa nhãn, teacher EMA và EPI tạo pseudo-label ổn định hơn, nhưng cơ chế này cần kiểm soát khi chuyển sang CT.

## Slide 11. LVFusionStage và EMA/EPI trong report

Hình này giải thích hai cơ chế chính: phần trên là hợp nhất ảnh-văn bản, phần dưới là cách EMA/EPI xử lý dữ liệu có nhãn và chưa nhãn. Đây là cầu nối giữa kiến trúc và kết quả ablation.

## Slide 12. Thiết lập thực nghiệm

QaTa-COV19 kiểm chứng đầy đủ ở 25%, 50% và 100% nhãn. MosMedData+ kiểm tra khả năng chuyển miền sang CT ở mức 50% nhãn, vì đây là mức cân bằng giữa mask thật và dữ liệu chưa nhãn.

## Slide 13. Bảng kết quả tổng hợp: mô hình ảnh đơn

Bảng này là nhóm baseline không dùng văn bản. Nó cho thấy mô hình ảnh đơn có thể đạt kết quả khá tốt, nhưng còn thiếu ngữ cảnh lâm sàng để xử lý các vùng tổn thương mơ hồ.

## Slide 14. Bảng kết quả tổng hợp: mô hình có văn bản

Các mô hình có văn bản như ConVIRT, GLoRIA, ViLT hay LAVT cho thấy văn bản có ích, nhưng mức tăng phụ thuộc mạnh vào cách hợp nhất ảnh-văn bản.

## Slide 15. Bảng kết quả tổng hợp: LViT và QaTaLViT

Đây là phần chính của bảng kết quả. Trên QaTa-COV19, QaTaLViT vượt LViT-T ở cả ba mức nhãn. Trên MosMedData+ 50%, QaTaLViT nhỉnh hơn LViT-T về Dice nhưng còn thấp hơn về mIoU.

## Slide 16. Kết quả chính trên QaTa-COV19

Slide này rút lại mốc chính trên QaTa-COV19: 82.22/73.35 ở 25%, 84.03/75.62 ở 50%, và 85.28/77.25 ở 100%. Mốc 50% đã tiến rất gần 100%, cho thấy lợi ích trong điều kiện nhãn hạn chế.

## Slide 17. Bảng bán giám sát trên QaTa-COV19

Bảng này tách riêng các thiết lập 25% và 50% nhãn. QaTaLViT vượt các phương pháp không văn bản như DTC, PLCT, MC-Net+, và cũng vượt LViT-T ở cùng tỉ lệ nhãn.

## Slide 18. Pseudo-label trên QaTa-COV19 50%

Pseudo-label holdout có Dice/mIoU cao và precision/recall khá cân bằng. Điều này giải thích vì sao teacher EMA và EPI có thể hỗ trợ student trên QaTa-COV19.

## Slide 19. Động học EMA/EPI trên QaTa-COV19 50%

Sau warm-up, tỉ lệ pseudo-label hợp lệ tăng ổn định; EPI bank phủ gần hết tập chưa nhãn. Nghĩa là nhãn giả được làm mượt qua nhiều lần xuất hiện, không phụ thuộc quá nhiều vào một mini-batch.

## Slide 20. Kết quả trên MosMedData+

MosMedData+ khó hơn vì là CT và có domain shift mạnh. Ở 50% nhãn, cấu hình tốt nhất đạt 73.84 Dice và 60.31 mIoU; kết quả này hơn LViT-T về Dice nhưng chưa hơn về mIoU.

## Slide 21. Bảng mở rộng MosMedData+

Bảng này chỉ giữ thiết lập 50% của MosMedData+. Mốc 50% được chọn vì vừa đủ dữ liệu thật để student ổn định, vừa còn đủ dữ liệu chưa nhãn để pseudo-label/EMA/EPI có vai trò trong đánh giá.

## Slide 22. Bảng ablation thành phần trên QaTa-COV19

Chuỗi ablation cho thấy kết quả cuối là hiệu ứng toàn hệ thống. Prompt, tăng cường dữ liệu, 4-LVFusionStage, Focal Tversky, căn chỉnh ảnh-văn bản và recipe cuối phát huy tốt nhất khi phối hợp.

## Slide 23. Bảng ablation siêu tham số

Bảng này kiểm tra learning rate và effective batch size. Recipe chuẩn với batch 32 và learning rate \(3\times10^{-4}\) là mốc tốt nhất trong nhóm thử nghiệm này.

## Slide 24. Bảng ablation mã hóa văn bản

BiomedBERT nhỉnh hơn trên QaTa-COV19, còn NoBERT rất cạnh tranh trên MosMedData+. Điều này cho thấy prompt có cấu trúc đã mang nhiều tín hiệu quan trọng, không phải cứ text encoder nặng hơn là chắc chắn tốt hơn.

## Slide 25. Bảng ablation trên MosMedData+

MosMedData+ nhạy với pseudo-label. Khi teacher yếu, pseudo-label dễ kéo student về nền. Vì vậy student-only, crop/loss cleanup và LightAU giúp pipeline ổn định hơn, đưa kết quả từ 68.88 lên 73.84 Dice ở cấu hình tốt nhất hiện tại.

## Slide 26. Giới hạn và hướng mở rộng

Giới hạn chính là MosMedData+ mới phân tích sâu ở 50% nhãn, hiệu quả phụ thuộc prompt, và EMA/EPI cần hiệu chuẩn theo miền dữ liệu. Hướng mở rộng là kiểm tra thêm các tập CT độc lập, tự động hóa prompt và hiệu chuẩn pseudo-label theo nguồn ảnh, kích thước tổn thương.

## Slide 27. Kết luận high-level

QaTaLViT là một hướng thực dụng cho phân đoạn y khoa ít nhãn có văn bản đi kèm. Ảnh quyết định hình dạng mask, văn bản bổ sung ngữ cảnh, prompt chuẩn hóa giúp ổn định tín hiệu, và hợp nhất đa mức giúp văn bản tác động đúng vào đặc trưng ảnh.
