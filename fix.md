CONTEXT: Codebase CFM-based multimodal recommendation (fork từ DiffMM). Log training 
thật (baby dataset, 44 epoch) cho thấy: gc_loss có magnitude lớn hơn diff_loss khoảng 
1000-8000 lần (ví dụ epoch 0: diff_loss≈0.0015, gc_loss≈187-458). Dù e_loss=0.01 được 
dùng để hạ trọng số gc_loss, gc_loss*e_loss vẫn áp đảo hoàn toàn diff_loss trong tổng 
loss (Di image loss = diff_loss + e_loss*gc_loss ≈ gần như bằng e_loss*gc_loss). 
Hệ quả quan sát được: diff_loss gần như không giảm qua 44 epoch (0.00157→0.00142, 
~10%), trong khi gc_loss giảm mạnh (458→7, ~98%) — network chủ yếu học để tối thiểu 
gc_loss, KHÔNG học velocity field đúng nghĩa. Ngoài ra, bucket t∈[0.8,1.0] có gc_loss 
CAO NHẤT trong mọi epoch — bất thường vì hệ số (1-t) nhân với v_pred trong công thức 
reconstruct x_hat1 = x_t + (1-t)*v_pred lẽ ra phải làm gc_loss tại t→1 tiệm cận 0 một 
cách cơ học. Nghi ngờ: v_pred có magnitude bùng nổ ở vùng t gần 1 để bù lại hệ số 
(1-t) nhỏ, do thiếu gradient signal hiệu quả ở vùng này.

=====================================================================
TASK 4: Chuẩn hoá scale giữa diff_loss và gc_loss trước khi combine
=====================================================================

VẤN ĐỀ:
Hai loss có nguồn gốc khác nhau về mặt số học: diff_loss là MSE trên velocity 
(kích thước item_dim), gc_loss là MSE trên embedding sau khi nhân với ma trận 
feature/embedding lớn (usr_model_embeds, usr_id_embeds) — magnitude tự nhiên 
lớn hơn nhiều bậc. Hệ số e_loss cố định (0.01) không đủ để cân bằng vì nó không 
thích ứng với sự thay đổi magnitude qua các epoch.

YÊU CẦU SỬA:
1. Tìm đoạn code combine loss hiện tại (nơi có `diff_loss_image.mean() + 
   gc_loss_image.mean() * args.e_loss`, tương tự cho text và audio nếu tiktok).

2. Thêm cơ chế normalize gc_loss về cùng bậc độ lớn với diff_loss TRƯỚC khi 
   nhân e_loss, dùng EMA (exponential moving average) của tỷ lệ giữa hai loss, 
   theo pattern sau (áp dụng riêng cho từng modality: image, text, và audio 
   nếu có):

   # Khởi tạo (một lần, ngoài vòng lặp training) — ví dụ trong __init__ của 
   # model hoặc như biến global/buffer:
   self.gc_scale_ema_image = None
   self.gc_scale_ema_text = None
   self.ema_decay = 0.99

   # Trong training loop, ngay trước khi combine loss, với mỗi modality:
   with torch.no_grad():
       ratio = diff_loss_image.mean() / (gc_loss_image.mean() + 1e-8)
       if self.gc_scale_ema_image is None:
           self.gc_scale_ema_image = ratio
       else:
           self.gc_scale_ema_image = (self.ema_decay * self.gc_scale_ema_image 
                                       + (1 - self.ema_decay) * ratio)
   
   gc_loss_image_scaled = gc_loss_image.mean() * self.gc_scale_ema_image
   loss_image = diff_loss_image.mean() + gc_loss_image_scaled * args.e_loss

   (Lặp lại tương tự cho text, và audio nếu dataset là tiktok)

3. Sau khi normalize, gc_loss_scaled sẽ ~cùng bậc với diff_loss trước khi nhân 
   e_loss — nghĩa là e_loss giờ mới thực sự đóng vai trò "trọng số tương đối" 
   như thiết kế ban đầu, thay vì phải gánh luôn việc bù chênh lệch scale.

4. QUAN TRỌNG: dùng .detach()/no_grad() khi tính ratio — KHÔNG để gradient chảy 
   qua phép tính EMA ratio, chỉ dùng nó như một hệ số scale cố định tại mỗi step.

VERIFY:
Log ra mỗi epoch: giá trị self.gc_scale_ema_image/text (để theo dõi nó ổn định 
dần theo thời gian, không dao động mạnh), và giá trị 
gc_loss_scaled*e_loss so với diff_loss — hai giá trị này nên cùng bậc độ lớn 
(tỷ lệ trong khoảng 0.1x - 10x), không còn chênh lệch 1000-8000 lần như trước.

=====================================================================
TASK 5: Retune e_loss sau khi normalize
=====================================================================

YÊU CẦU:
Sau khi Task 4 xong, giá trị e_loss=0.01 hiện tại không còn ý nghĩa cũ (nó được 
chọn để bù scale, giờ scale đã được EMA xử lý). Chạy lại training với vài giá 
trị e_loss khác nhau: thử e_loss ∈ {0.1, 0.5, 1.0, 2.0}, giữ nguyên mọi 
hyperparameter khác, so sánh Recall@20/NDCG@20 để chọn giá trị tốt nhất.
KHÔNG cần sửa code cho task này, chỉ cần chạy lại Main.py với --e_loss khác nhau.

=====================================================================
TASK 6: Thêm logging magnitude của v_pred theo bucket t
=====================================================================

VẤN ĐỀ:
Cần xác nhận giả thuyết: v_pred (velocity dự đoán) có magnitude bùng nổ bất 
thường ở vùng t gần 1, do thiếu gradient signal hiệu quả ở vùng này khi hệ số 
(1-t) trong công thức reconstruct quá nhỏ.

YÊU CẦU THÊM:
1. Trong đúng vị trí đã có bucket-t logging cho diff_loss/gc_loss (Task 3 cũ), 
   thêm tracking cho ||v_pred|| (norm theo chiều cuối, tức theo item_dim) mỗi 
   sample, gán vào cùng 5 bucket t: [(0.0,0.2), (0.2,0.4), (0.4,0.6), (0.6,0.8), 
   (0.8,1.0)].

2. Sau mỗi epoch, in thêm cột v_pred_norm (mean, std) cho từng bucket, cùng 
   format với log hiện tại:

   Epoch {e} | t in [0.0,0.2]: n=..., diff_loss(...), gc_loss(...), gc/diff=..., 
   v_pred_norm(mean=...,std=...)

3. Làm việc này cho cả image và text (và audio nếu tiktok).

VERIFY:
Chạy training vài epoch sau khi đã áp Task 4+5. Quan sát: v_pred_norm ở bucket 
[0.8,1.0] có cao bất thường so với các bucket khác không (ví dụ cao hơn 
2-3 lần so với bucket [0.0,0.2])? Nếu sau khi normalize gc_loss (Task 4) mà 
pattern này biến mất hoặc giảm hẳn, điều đó xác nhận nguyên nhân là do gradient 
signal yếu ở vùng t→1 (đã được Task 4 khắc phục gián tiếp). Nếu vẫn còn rõ rệt, 
cần xem xét thêm việc thêm gradient clipping cho v_pred hoặc weighting loss 
theo t (đã đề cập ở lần trước, chưa cần làm ngay).

=====================================================================
LƯU Ý CHUNG:
- Làm Task 4 trước, train lại, kiểm tra xem diff_loss có giảm rõ rệt hơn theo 
  epoch không (so với chỉ giảm ~10% như hiện tại) — đây là tín hiệu quan trọng 
  nhất cho biết velocity field đã thực sự được học.
- Sau đó làm Task 5 (retune e_loss) để tìm giá trị tối ưu.
- Task 6 (logging v_pred) có thể làm song song, không phụ thuộc thứ tự.
- So sánh Recall@20/NDCG@20 cuối cùng với: (a) DiffMM gốc, (b) kết quả trước 
  khi sửa Task 4/5 (Recall≈0.0946 đỉnh, ~0.092-0.094 plateau) — để biết việc 
  rebalance loss có thực sự giúp vượt qua plateau hiện tại hay không.
=====================================================================