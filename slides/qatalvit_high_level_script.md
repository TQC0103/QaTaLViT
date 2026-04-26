# Script thuyết trình ngắn: QaTaLViT

Thời lượng gợi ý: 5 đến 6 phút. Các slide hình dùng để khoanh trực tiếp trên kiến trúc; slide ngay sau đó dùng để nói tóm gọn ý chính.

## Slide 1. QaTaLViT

Chào thầy và các bạn. Nhóm em trình bày QaTaLViT, một mô hình phân đoạn ảnh y khoa có sử dụng thêm mô tả văn bản ngắn để hỗ trợ định vị tổn thương.

Ý tưởng chính là ảnh vẫn quyết định hình dạng mask, còn văn bản đóng vai trò gợi ý vị trí và ngữ cảnh.

## Slide 2. Tổng quan hệ thống

Đây là toàn bộ pipeline của QaTaLViT. Em sẽ chia hệ thống thành bốn phần để trình bày: nhánh văn bản, trục CNN U-Net, LVFusionStage/ViT, và EMA/EPI.

## Slide 3. Tóm tắt tổng quan

QaTaLViT giữ ảnh làm nguồn quyết định hình dạng mặt nạ, còn văn bản đóng vai trò gợi ý ngữ cảnh và vị trí tổn thương.

Về dữ liệu, nhóm dùng hai bộ chính: QaTa-COV19 là ảnh X-quang ngực có mô tả văn bản đi kèm, còn MosMedData+ là CT lát cắt, khó hơn vì có nhiều nguồn ảnh và lệch miền mạnh.

Hệ thống gồm bốn phần chính. Nhánh văn bản tạo token ngôn ngữ. CNN U-Net giữ chi tiết ảnh. LVFusionStage đưa văn bản vào đặc trưng ảnh. EMA/EPI hỗ trợ tạo pseudo-label cho dữ liệu chưa nhãn.

Kết quả chính phải đọc cùng đối sánh. Trên QaTa-COV19 50% nhãn, QaTaLViT đạt 84.03 Dice và 75.62 mIoU, cao hơn LViT-T 50% là 82.73 Dice và 73.99 mIoU. Trên MosMedData+ 50% nhãn, cấu hình tốt nhất hiện tại đạt 73.84 Dice và 60.31 mIoU; so với LViT-T 50%, kết quả này nhỉnh hơn về Dice nhưng còn thấp hơn về mIoU.

## Slide 4. Phần 1: Nhánh văn bản

Ở hình này, em khoanh phần nhánh văn bản. Đầu vào là mô tả lâm sàng ngắn. Mô tả này được chuẩn hóa prompt, rồi đi qua từ vựng, `nn.Embedding` và BiGRU PromptEncoder.

## Slide 5. Tóm tắt nhánh văn bản

Đầu vào văn bản là mô tả lâm sàng ngắn, thường chứa thông tin về bên phổi, số vùng tổn thương và vị trí tương đối.

Prompt normalization đưa các cách viết khác nhau về dạng ổn định hơn, giúp mô hình không phải học quá nhiều biến thể ngôn ngữ không cần thiết.

Prompt sau chuẩn hóa được ánh xạ qua từ vựng, `nn.Embedding` và BiGRU PromptEncoder để tạo token văn bản \(T\). Token này không trực tiếp sinh mask, mà được dùng làm tín hiệu điều hướng cho các tầng LVFusionStage.

## Slide 6. Phần 2: Trục CNN kiểu U-Net

Ở phần CNN, ảnh đi qua Stem, Down1, Down2, Down3 và Bottom để tạo đặc trưng nhiều mức. Sau đó decoder đi ngược từ Up4 đến Up1 để khôi phục mặt nạ.

## Slide 7. Tóm tắt trục CNN

CNN encoder là xương sống chính để giữ chi tiết không gian, biên tổn thương và cấu trúc cục bộ của ảnh.

Ảnh đi qua Stem, Down1, Down2, Down3 và Bottom để tạo các đặc trưng nhiều mức \(e_1..e_4\) và bottleneck \(x_5\). Decoder dùng \(Up4 \rightarrow Up1\) để tăng dần độ phân giải và khôi phục mặt nạ phân đoạn.

Skip connection nối encoder với decoder, nhưng trong QaTaLViT các skip feature đã được tinh chỉnh bởi văn bản trước khi đưa vào decoder.

## Slide 8. Phần 3: LVFusionStage và ViT

Ở hình cơ chế này, phần trên mô tả LVFusionStage. Feature CNN được token hóa, token ảnh chú ý chéo với token văn bản, sau đó đi qua ViT/Transformer block và được chiếu ngược về feature map.

## Slide 9. Tóm tắt LVFusionStage

LVFusionStage là nơi tương tác ảnh--văn bản diễn ra trực tiếp trên đặc trưng encoder ở nhiều mức.

Feature CNN \(D_i\) được token hóa bằng PatchTokenizer; token ảnh đóng vai trò truy vấn, còn token văn bản \(T\) đóng vai trò khóa/giá trị trong CrossLanguageFusion.

Sau chú ý chéo, token đã hợp nhất đi qua ViT/Transformer block để học quan hệ dài hơn, rồi TokenToSpatial chiếu ngược về feature map.

Nhờ vậy, văn bản không thay thế CNN, mà điều chỉnh đặc trưng ảnh trước khi đặc trưng đó đi vào skip connection và decoder.

## Slide 10. Phần 4: EMA và EPI

Ở phần dưới của hình là nhánh bán giám sát. Dữ liệu có nhãn đi theo đường supervised bình thường: student dự đoán mask, rồi so với ground truth để tính loss.

Với dữ liệu chưa nhãn, ta không có ground truth nên dùng EMA teacher để tạo nhãn giả. Teacher này không học bằng backprop trực tiếp, mà được cập nhật bằng trung bình trượt trọng số của student, nên dự đoán thường ổn định hơn student tại từng bước huấn luyện.

Sau đó EPI tiếp tục làm mượt ở mức từng ảnh. Nghĩa là với mỗi ảnh chưa nhãn, mô hình lưu lại xác suất dự đoán qua nhiều lần ảnh đó xuất hiện, rồi trộn dự đoán mới với dự đoán cũ để giảm nhiễu tức thời.

## Slide 11. Tóm tắt EMA và EPI

Với dữ liệu có nhãn, student học trực tiếp từ ground truth; teacher không tạo nhãn cho phần này.

Với dữ liệu chưa nhãn, EMA teacher sinh probability map. Map này được EPI làm mượt theo từng ảnh, rồi mới lọc thành pseudo-label. Chỉ những pixel đủ tin cậy mới được giữ lại: pixel dương là vùng nghi có tổn thương, pixel âm là vùng nền đủ chắc chắn, còn vùng không chắc thì bỏ qua để tránh ép mô hình học từ nhãn sai.

Pseudo-label sau lọc quay lại huấn luyện student bằng pseudo BCE, vì vậy EMA và EPI tạo thành một vòng lặp teacher--student: student học từ nhãn thật và nhãn giả, còn teacher lại là bản làm mượt của student.

Điểm cần nhấn mạnh là cơ chế này chỉ có lợi khi teacher đủ tốt và xác suất được hiệu chuẩn ổn. Trên MosMedData+, tổn thương nhỏ, ảnh CT đến từ nhiều nguồn và domain shift mạnh, nên teacher dễ under-confident ở vùng bệnh nhưng lại tự tin ở nền. Khi đó confidence mask giữ nhiều pixel âm hơn pixel dương, pseudo-label bị lệch về nền và làm student học kém đi. Vì vậy trong thí nghiệm cuối, nhóm dùng EMA/EPI thận trọng và có cả cấu hình student-only để tránh pseudo-label yếu kéo giảm kết quả.

## Slide 12. Kết quả trên QaTa-COV19

Trên QaTa-COV19, QaTaLViT cải thiện ở cả ba mức nhãn khi đặt cạnh LViT-T. Ở 25% nhãn, QaTaLViT đạt 82.22 Dice so với LViT-T là 80.95. Ở 50% nhãn, QaTaLViT đạt 84.03 Dice so với 82.73. Ở 100% nhãn, QaTaLViT đạt 85.28 Dice so với 83.66.

Điểm đáng chú ý là mô hình vẫn tăng tốt trong bối cảnh ít nhãn.

## Slide 13. Kết quả trên MosMedData+

Trên MosMedData+, tập dữ liệu khó hơn do CT slice được ghép từ nhiều nguồn và có domain shift mạnh.

Kết quả tốt nhất hiện tại của nhóm ở 50% nhãn là 73.84 Dice và 60.31 mIoU. Khi đặt cạnh LViT-T 50% là 73.56 Dice và 61.05 mIoU, QaTaLViT nhỉnh hơn về Dice nhưng chưa vượt về mIoU. Recipe ổn định nhất là BiomedBERT, student-only, loss cleanup, crop-only và augmentation nhẹ.

Ở slide này, MosMedData+ chỉ được dùng ở 50% nhãn. Lý do là mức này cân bằng nhất giữa dữ liệu có mask thật và dữ liệu chưa nhãn, nên phù hợp hơn để kiểm tra các cơ chế liên quan đến pseudo-label. Mức 100% không phải trọng tâm vì khi toàn bộ dữ liệu có nhãn, vai trò của nhánh teacher--student và dữ liệu chưa nhãn không còn rõ.

## Slide 14. Tài liệu tham khảo

Tóm lại, QaTaLViT có ba ý chính: ảnh giữ chi tiết, văn bản định hướng, và recipe huấn luyện quyết định độ ổn định.

Em xin kết thúc phần trình bày tại đây. Em cảm ơn thầy và các bạn đã lắng nghe.
