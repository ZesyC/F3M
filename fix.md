# CFM Debugging Playbook — F3M Multimodal Recommendation

> Tài liệu tổng hợp toàn bộ quá trình debug CFM-based graph diffusion (thay thế module diffusion của DiffMM), tình trạng hiện tại, và hướng dẫn cụ thể cho bước tiếp theo. Dùng file này làm brief cho Codex hoặc để tự tra cứu lại tiến độ.

---

## 1. Bối cảnh

Dự án thay module modality-aware graph diffusion của DiffMM bằng Conditional Flow Matching (CFM), với:

- $x_1 = x_{\text{start}}$: interaction vector thật
- $x_0 = z_{\text{prior}}$: prior CF-informed (từ LightGCN/popularity), không phải Gaussian noise
- $x_t = t x_1 + (1-t) x_0$, $t \sim U(0,1)$
- $v_\theta(x_t, t)$: velocity network cần học $v^* = x_1 - x_0$
- Loss tổng: `CFM loss` (MSE velocity) + `e_loss * GC loss` (reconstruction embedding alignment)

Mục tiêu: vượt Recall@20 / NDCG@20 của DiffMM baseline trên Baby/Sports/TikTok.

## 2. Các bug đã tìm và sửa (theo thứ tự thời gian)

| # | Bug | Triệu chứng | Trạng thái |
|---|---|---|---|
| 1 | Time embedding dùng tần số calibrate cho $t$ nguyên 0–1000, áp sai cho $t \in [0,1]$ liên tục | 8/10 chiều embedding gần như "đóng băng" | ✅ Đã sửa — verify bằng diagnostic script, mọi chiều đều biến thiên rõ theo $t$ |
| 2 | Scale mismatch giữa $z_{\text{prior}}$ và $x_1$ (lệch ~2.6 lần) | Quỹ đạo nội suy bị méo | ✅ Đã sửa — rescale $z_{\text{prior}}$ theo `prior_scale` tính trên train loader (~2.9), tỷ lệ norm giờ ~1.0 |
| 3 | `gc_scale_ema` trôi dạt không kiểm soát qua epoch (từ ~0.15 lên ~2.2 lần CFM loss) | Nghi ngờ GC loss áp đảo CFM loss ở cuối train | ✅ Đã thêm clamp (`target=1.00, bounds=[0.50, 1.50]`) — **verify: không ảnh hưởng đáng kể đến Recall** (xem mục 3) |

## 3. Tình trạng hiện tại — Recall đã plateau

Qua nhiều lần train với các fix trên, **Recall@20 dao động trong khoảng 0.090–0.095**, không có xu hướng tăng rõ rệt dù đã sửa nhiều bug:

| Lần chạy | Best Recall | Best NDCG | Ghi chú |
|---|---|---|---|
| Trước khi fix bug 1+2 | ~0.089–0.091 | ~0.038 | Baseline có bug |
| Sau fix 1+2 | 0.0943–0.0950 | 0.0397–0.0402 | Dao động giữa các lần train |
| Sau fix 3 (EMA clamp) | 0.09473 | 0.03961 | Không khác biệt đáng kể so với không clamp (0.09499) |

**Kết luận: các fix thuộc nhóm "sửa bug/tinh chỉnh loss" đã đến giới hạn đóng góp.** Chênh lệch giữa các lần chạy hiện nhỏ hơn cả nhiễu tự nhiên giữa các seed khác nhau.

### Phát hiện quan trọng: velocity collapse tồn tại nhưng KHÔNG phải nguyên nhân chính

So sánh `v_pred_norm` (độ lớn velocity dự đoán) giữa lần train không-clamp và có-clamp:

| Epoch | Không clamp | Có clamp | Target thật (`\|x1-z_prior\|`) |
|---|---|---|---|
| 0 | ~2.45–2.62 | ~2.20–2.32 | ~1.87–1.88 |
| 10 | ~0.41–0.55 | ~0.92–0.97 | ~1.87–1.88 |
| 49 | ~0.24–0.27 | ~0.47–0.60 | ~1.87–1.88 |

→ Việc clamp EMA giúp `v_pred_norm` giữ được **cao gấp ~2 lần** ở cuối train so với không clamp — nghĩa là mức độ velocity collapse giảm rõ rệt.

→ **Nhưng Recall gần như không đổi** (0.09499 vs 0.09473).

**Suy ra: bottleneck không nằm ở độ lớn (magnitude) của velocity, mà nhiều khả năng nằm ở HƯỚNG (direction) của velocity dự đoán.** Model có thể đang dự đoán đúng hướng nhưng sai biên độ (vô hại, vì sau reconstruct vẫn ra đúng chỗ), hoặc đang dự đoán sai hướng nhưng biên độ vừa đủ nhỏ để giảm MSE loss (có hại, nhưng magnitude fix không sửa được).

## 4. Việc cần làm tiếp — theo thứ tự ưu tiên

### Task — Log cosine similarity giữa v_pred và v_target theo bucket t (ưu tiên cao nhất)

**Mục đích:** đo trực tiếp xem velocity có đúng HƯỚNG hay không, tách biệt hoàn toàn khỏi vấn đề độ lớn.

```
CONTEXT: Cùng codebase CFM, cần thêm logging mới — không sửa loss/training logic,
chỉ thêm phép đo chẩn đoán.

YÊU CẦU:
1. Tại nơi tính diff_loss hiện tại (nơi có v_pred và v_target = x_start - z_prior),
   thêm tính cosine similarity PER-SAMPLE:
   
   cos_sim = F.cosine_similarity(v_pred, v_target, dim=-1)  # shape [batch_size]
   
2. Tích lũy cos_sim theo đúng cơ chế bucket t đã có sẵn (5 bucket: [0,0.2), 
   [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0]), tính mean/std mỗi bucket.
3. Log mỗi epoch, format nhất quán với log hiện tại:
   
   Epoch {e} | t in [a,b]: cos_sim(mean=..., std=...), v_pred_norm(mean=...), 
   v_target_norm(mean=...)
   
   (thêm v_target_norm vào cùng dòng luôn — đo trực tiếp thay vì suy ra gián tiếp 
   từ debug script cũ)
4. Áp dụng cho cả 2 modality (image, text), giữ format tách riêng như code hiện tại.
5. QUAN TRỌNG: đây là log thuần túy, không thay đổi bất kỳ giá trị nào dùng để 
   tính loss hay cập nhật gradient.

VERIFY:
Train lại 1 lần (có thể dùng chung config với lần train EMA-clamp gần nhất, 
chỉ thêm log này), theo dõi cos_sim qua các epoch.

CÁCH ĐỌC KẾT QUẢ:
- cos_sim → gần 1.0: velocity gần như đúng hướng hoàn toàn, chỉ sai biên độ.
  → Vấn đề thực sự chỉ là magnitude, quay lại đầu tư vào Task A / guidance scale
    hoặc auxiliary loss ràng buộc magnitude.
- cos_sim → gần 0 hoặc âm: velocity SAI HƯỚNG — đây là vấn đề nghiêm trọng hơn 
  nhiều so với magnitude, và giải thích tại sao magnitude fix không giúp được gì.
  → Cần xem lại kiến trúc v_θ (có đủ capacity/inductive bias để học hướng đúng 
    không), hoặc xem lại target velocity có bị nhiễu/sai công thức không.
- cos_sim khác nhau rõ rệt giữa các bucket t (ví dụ thấp ở t nhỏ, cao ở t lớn):
  → model học hướng tốt hơn ở gần cuối quỹ đạo (t→1, gần data thật) nhưng kém ở 
    gần đầu (t→0, gần prior) — gợi ý cần thêm loss/curriculum tập trung vào 
    vùng t nhỏ.
```

## 5. Nguyên tắc làm việc (nhắc lại, để giữ tính ablation sạch)

- Mỗi lần train lại chỉ nên đổi **một biến ảnh hưởng đến training**. Logging thuần túy (như cosine similarity) an toàn để thêm cùng lúc với bất kỳ thay đổi nào khác.
- Guidance scale test (Task A) không cần train lại — luôn ưu tiên làm trước các thử nghiệm cần train lại vì cho kết quả nhanh và rẻ.
- Khi so sánh giữa các lần train, luôn đối chiếu với biên độ nhiễu tự nhiên đã quan sát được (Recall dao động ~0.003–0.005 giữa các lần chạy cùng config) trước khi kết luận một thay đổi có tác dụng thật hay không.

## 6. Tóm tắt trạng thái để tiếp tục cuộc trò chuyện sau

- Đã fix: time embedding, prior scale, EMA/GC loss clamp — không còn nghi ngờ đây là nguyên nhân chính.
- Recall hiện plateau quanh 0.093–0.095, thấp hơn DiffMM baseline.
- Velocity collapse (magnitude) đã được xác nhận tồn tại nhưng KHÔNG phải nguyên nhân chính giới hạn Recall.
- Đang chuyển hướng nghi vấn sang **hướng (direction)** của velocity dự đoán — cần cosine similarity logging (Task B) để xác nhận trước khi quyết định hướng sửa tiếp theo (kiến trúc $v_\theta$, target velocity, hoặc loss function).