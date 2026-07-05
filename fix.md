CONTEXT: Codebase CFM-based multimodal recommendation (fork từ DiffMM), dùng conditional 
flow matching thay cho diffusion. Cần sửa 2 bug và thêm logging cho 1 diagnostic.

=====================================================================
TASK 1: Fix time embedding — sai tần số cho t liên tục trong [0,1]
=====================================================================

VẤN ĐỀ:
Hàm `scale_time(t)` và sinusoidal time embedding hiện tại được kế thừa từ code 
diffusion gốc, nơi timestep là số nguyên chạy 0→1000. Bây giờ t ~ U(0,1) liên tục 
nhưng tần số sinusoid không được điều chỉnh lại, khiến 8/10 chiều embedding gần 
như đóng băng (thay đổi <0.15 trên toàn range t=0→1), chỉ 1 cặp (sin,cos) tần số 
thấp nhất là thực sự biến thiên — network gần như "mù" phần lớn thông tin t.

YÊU CẦU SỬA:
1. Tìm hàm sinh time embedding (search "scale_time" và nơi gọi sinusoidal 
   positional/time embedding trong flow model).
2. Thay tần số hiện tại (được calibrate cho t nguyên 0-1000) bằng tần số phù hợp 
   range [0,1] liên tục. Dùng công thức chuẩn:
   
   freq_i = 2^i  (i = 0, 1, ..., num_freqs-1)
   emb = concat([sin(2*pi*freq_i*t), cos(2*pi*freq_i*t)] for i in range(num_freqs))
   
   (tương đương NeRF-style Fourier features, phù hợp cho input scalar trong [0,1])
3. Đảm bảo num_freqs đủ để tần số cao nhất quét được ít nhất 2-3 chu kỳ đầy đủ 
   trong [0,1] (tức freq cao nhất ~ 4-8), còn tần số thấp nhất vẫn có độ phân giải 
   mượt ở gần t=0 và t=1.
4. KHÔNG thay đổi output dimension của embedding (để không phải sửa lại các layer 
   downstream ăn embedding này) — chỉ thay công thức tính tần số bên trong.

VERIFY SAU KHI SỬA:
Chạy lại đúng script debug_time_embedding.py (hoặc viết lại tương đương) với 
t = [0.1, 0.3, 0.5, 0.7, 0.9], in toàn bộ các chiều embedding + abs diff giữa 
các t liên tiếp. Tiêu chí đạt: KHÔNG có chiều nào có abs diff <0.05 giữa mọi 
cặp t liên tiếp (tức là mọi chiều đều biến thiên rõ rệt, không còn chiều "đóng băng").

=====================================================================
TASK 2: Fix scale mismatch giữa x1 (x_start) và z_prior
=====================================================================

VẤN ĐỀ:
x1.norm().mean() ≈ 2.44, z_prior.norm().mean() ≈ 0.92 → lệch nhau ~2.6 lần. 
Vì x_t = t*x1 + (1-t)*z_prior, lệch scale này làm méo quỹ đạo nội suy — 
đặc biệt ở t nhỏ, x_t bị kéo lệch không tự nhiên về phía z_prior.

YÊU CẦU SỬA:
1. Tìm nơi z_prior được tạo ra (từ popularity hoặc user-history, trước khi 
   đưa vào interpolation x_t).
2. Thêm bước rescale z_prior NGAY SAU khi tạo, TRƯỚC khi dùng trong x_t:
   
   scale_factor = x1.norm(dim=-1).mean() / z_prior.norm(dim=-1).mean()
   z_prior = z_prior * scale_factor
   
   Tính scale_factor một lần trên tập train (hoặc một batch lớn đại diện), 
   lưu lại làm hằng số cố định — KHÔNG tính lại mỗi batch (tránh scale factor 
   dao động giữa các batch gây bất ổn training).
3. Thêm assertion/log ngay sau bước rescale: in ra 
   z_prior.norm(dim=-1).mean() và x1.norm(dim=-1).mean() mỗi vài trăm step 
   đầu tiên, để xác nhận hai giá trị này gần nhau (~cùng bậc độ lớn) trong 
   suốt quá trình train, không chỉ lúc khởi tạo.

VERIFY SAU KHI SỬA:
In lại đúng bảng "[x1 vs z_prior norm]" như debug script cũ. Tiêu chí đạt: 
tỷ lệ x1.norm.mean() / z_prior.norm.mean() nằm trong khoảng [0.9, 1.1] 
(gần 1:1), thay vì ~2.6 như hiện tại.

=====================================================================
TASK 3: Thêm logging diff_loss/gc_loss theo bucket t — TRÊN CHECKPOINT 
ĐÃ TRAIN THẬT, không phải random init
=====================================================================

VẤN ĐỀ:
Cần biết liệu hệ số (1-t) nhân với v_pred trong công thức reconstruct 
x_hat1 = x_t + (1-t)*v_pred có khuếch đại lỗi ở vùng t nhỏ hay không, 
NHƯNG phải đo trên model đã train vài epoch với dữ liệu thật — đo trên 
random-init weight sẽ cho kết quả không phản ánh hành vi thật.

YÊU CẦU THÊM:
1. Trong training loop (Main.py hoặc nơi tính diff_loss/gc_loss mỗi step), 
   thêm một dict tích lũy theo bucket t:
   
   t_buckets = [(0.0,0.2), (0.2,0.4), (0.4,0.6), (0.6,0.8), (0.8,1.0)]
   bucket_stats = {b: {'diff_loss': [], 'gc_loss': [], 'n': 0} for b in t_buckets}
   
2. Mỗi step, với t đã sample cho batch đó, gán từng sample vào đúng bucket 
   (dùng t.item() cho từng sample trong batch, không phải t trung bình cả batch), 
   append giá trị diff_loss và gc_loss (per-sample, TRƯỚC khi .mean() qua batch) 
   vào đúng bucket.
3. Sau mỗi epoch (không phải mỗi step — tránh log quá nhiều), in ra:
   - mean, std của diff_loss và gc_loss từng bucket
   - tỷ lệ gc_loss/diff_loss từng bucket
   - so sánh với epoch trước để thấy xu hướng theo thời gian train
4. Format log giống hệt bảng "[diff_loss vs gc_loss by t bucket]" cũ để 
   dễ so sánh trực tiếp, nhưng thêm cột epoch:
   
   Epoch {e} | t in [0.0,0.2]: n=..., diff_loss(mean=...,std=...), 
   gc_loss(mean=...,std=...), gc/diff=...
   
5. QUAN TRỌNG: log này phải chạy trên model đang train thật (weight cập nhật 
   dần qua epoch), KHÔNG phải script diagnostic riêng với random weight.

VERIFY:
Chạy training thật vài epoch (5-10 epoch đủ để thấy xu hướng), xem log in ra 
mỗi epoch. Điều cần quan sát: gc_loss ở bucket t nhỏ ([0.0,0.2]) có giảm chậm 
hơn / giữ mức cao hơn đáng kể so với bucket t lớn ([0.8,1.0]) hay không, khi 
train tiến triển qua các epoch. Nếu có, đó là bằng chứng cho thấy cần thêm 
weighting theo t vào gc_loss (việc này CHƯA cần làm ngay, chỉ cần log để 
xác nhận trước).

=====================================================================
LƯU Ý CHUNG:
- Làm TASK 1 và TASK 2 trước, train lại từ đầu, RỒI mới bật logging TASK 3 
  để đo trên phiên bản đã fix — không đo TASK 3 trên model còn bug của TASK 1/2.
- Sau khi cả 3 xong, so sánh Recall@20/NDCG@20 với con số DiffMM gốc và với 
  con số 0.02% thấp hơn trước đây, để biết 2 fix này có đóng góp gì không.
=====================================================================