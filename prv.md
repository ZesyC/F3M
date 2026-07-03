# Hướng dẫn: Thay Gaussian Diffusion bằng Conditional Flow Matching (CF-informed prior) trong DiffMM

> Mục tiêu: thay module `GaussianDiffusion` của DiffMM (khởi tạo từ nhiễu Gaussian thuần) bằng
> Conditional Flow Matching (CFM) khởi tạo từ một **CF-informed prior** — tức là điểm xuất phát
> của quá trình sinh không phải noise ngẫu nhiên mà là tín hiệu collaborative filtering đã biết.
> Đây là lever có bằng chứng ablation mạnh nhất trong literature gần đây (FlowCF - KDD 2025,
> FlowRec - 2025) để vượt các baseline dựa trên diffusion.
>
> File này chỉ là hướng dẫn/khung sườn — phần code cụ thể để bạn tự viết và tự debug.

---

## 1. Đọc lại kiến trúc gốc của DiffMM (để biết sửa ở đâu)

Repo gốc: `HKUDS/DiffMM`, file `Model.py`. Hai class cần quan tâm:

### 1.1 `Denoise` (mạng dự đoán)
- Input: `x` (vector tương tác của user, shape `[batch, n_items]`, đã bị nhiễu ở bước `t`) và
  `timesteps` (số nguyên hoặc float).
- Có sẵn sinusoidal time embedding — **đã nhận input float**, nên gần như không cần sửa gì ở đây
  khi chuyển sang `t ∈ [0,1]` liên tục. Đây là điểm may mắn, tận dụng lại được.
- Output: `model_output`, cùng shape với `x`. Lưu ý DiffMM **parameterize theo x0-prediction**
  (model dự đoán trực tiếp vector gốc, không dự đoán noise ε như DDPM chuẩn).

### 1.2 `GaussianDiffusion` (quá trình diffusion)
Các hàm quan trọng và vai trò của chúng:

| Hàm | Vai trò hiện tại | Cần sửa gì |
|---|---|---|
| `get_betas`, `calculate_for_diffusion` | Xây lịch trình beta/alpha cho DDPM | Bỏ hoàn toàn — CFM không cần beta schedule |
| `q_sample(x_start, t, noise)` | Forward: `x_t = sqrt(αcum_t)*x_start + sqrt(1-αcum_t)*noise`, với `noise ~ N(0,I)` | Thay bằng **linear interpolation**: `x_t = t*x_start + (1-t)*z_prior` |
| `training_losses` | MSE(model_output, x_start) có trọng số SNR + `gc_loss` | Đổi target thành **velocity** `(x_start - z_prior)`, bỏ SNR weight |
| `p_sample` | Reverse loop DDPM nhiều bước, có/không noise | Thay bằng **Euler ODE solver** ít bước, deterministic |
| `SNR` | Trọng số loss theo bước | Không cần nữa trong CFM cơ bản |

Điểm mấu chốt: trong `q_sample` gốc, biến `noise = torch.randn_like(x_start)` chính là
**prior distribution** — đây là chỗ cần thay bằng CF-informed prior.

Cũng cần tìm trong `Main.py` đoạn code gọi `GaussianDiffusion.p_sample(...)` để build
`image_adj` / `text_adj` — đó là nơi bạn sẽ đổi sang gọi hàm sample mới.

---

## 2. Toán học: DDPM → Conditional Flow Matching

### 2.1 Ký hiệu
- $x_{\text{data}}$: vector tương tác gốc của user (chính là `x_start` trong code cũ).
- $z_{\text{prior}}$: điểm xuất phát của flow tại $t=0$. **Đây là thứ bạn thay đổi.**
  - DiffMM gốc: $z_{\text{prior}} \sim \mathcal{N}(0, I)$ (noise thuần).
  - Hướng cải tiến: $z_{\text{prior}}$ = tín hiệu CF (xem mục 3).
- $t \in [0, 1]$: liên tục (khác với `ts` nguyên `0..steps-1` trong code cũ).

### 2.2 Forward process (thay `q_sample`)
Linear/OT interpolation path:

```
x_t = t * x_data + (1 - t) * z_prior          # t ~ Uniform(0, 1)
```

Target velocity không đổi dọc đường thẳng này:

```
v_target = x_data - z_prior
```

### 2.3 Training objective (thay `training_losses`)
CFM loss cơ bản (Lipman et al., 2023):

```
v_pred = model(x_t, t)
cfm_loss = mean( (v_pred - v_target) ** 2 )
```

Không cần trọng số SNR như DDPM. Nếu muốn giữ lại `gc_loss` (graph-collaborative consistency)
của DiffMM, phải suy ra $\hat{x}_{\text{data}}$ từ velocity dự đoán trước khi tính:

```
x_data_hat = x_t + (1 - t) * v_pred     # suy ngược từ đường thẳng
gc_loss    = mean( (mm(x_data_hat, model_feats) - mm(x_data, itmEmbeds)) ** 2 )
```

> Lưu ý: đây chính là chỗ liên quan tới bug uninitialized embeddings bạn đang gặp — nếu
> `itmEmbeds`/`model_feats` chưa được khởi tạo/pretrain đúng thứ tự trước khi `gc_loss` này
> chạy lần đầu, `x_data_hat` ở các batch đầu sẽ rất nhiễu → lan sang gradient của cả prior.
> Kiểm tra thứ tự init trước khi debug tiếp phần curriculum.

### 2.4 Sampling (thay `p_sample`)
Không còn reverse SDE nhiều trăm bước — chỉ cần giải ODE bằng Euler (vài bước là đủ):

```
x = z_prior                          # bắt đầu từ CF-informed prior, KHÔNG phải noise
dt = 1.0 / n_steps
for i in range(n_steps):
    t = i * dt
    v = model(x, t)
    x = x + dt * v
# x lúc này là bản sinh cuối cùng, tương đương x_start cũ
```

Vì đường đi gần thẳng, `n_steps` có thể nhỏ hơn nhiều so với `self.steps` gốc của DiffMM
(thử 2–10 bước trước, so với hàng chục/hàng trăm bước của DDPM).

---

## 3. Chọn $z_{\text{prior}}$ cụ thể — đây là phần quan trọng nhất

Không dùng `torch.randn_like(x_start)`. Hai lựa chọn cụ thể, từ đơn giản đến tốt hơn:


### Option — tốt hơn, theo đúng recipe của FlowCF
Dùng **frequency/behavior-based prior**: với mỗi user, prior là phân phối tần suất item toàn
cục (global item popularity), có thể kết hợp thêm few-shot lịch sử tương tác của chính user đó
(nếu có) để cá nhân hoá nhẹ:

```
item_freq = normalize(item_interaction_count)      # [n_items], toàn cục
z_prior = item_freq.expand(batch_size, -1)          # broadcast cho cả batch
# (tuỳ chọn) trộn thêm x_data đã dropout một phần để cá nhân hoá:
z_prior = 0.5 * z_prior + 0.5 * dropout(x_data, p=0.5)
```

**Khuyến nghị**: bắt đầu với Option A (tận dụng code sẵn có, nhanh có kết quả để ablate),
sau đó thử Option B nếu Option A cho gain nhưng bạn muốn đẩy thêm.

---

## 4. Kế hoạch sửa code — thứ tự thực hiện

1. **Viết hàm sinh `z_prior`** (Option A trước) — đặt trong `Model.py` hoặc file mới
   `FlowPrior.py`, độc lập với `GaussianDiffusion` để dễ test riêng.
2. **Tạo class mới** `ConditionalFlowMatching` thay thế `GaussianDiffusion`, copy khung từ
   `GaussianDiffusion` nhưng bỏ toàn bộ phần beta/alpha, thêm 3 hàm: `interpolate`,
   `training_losses` (bản mới theo mục 2.3), `sample` (Euler theo mục 2.4).
3. **Kiểm tra `Denoise.forward`**: input `timesteps` giờ là float trong `[0,1]` thay vì int
   `0..steps-1` — hàm sinusoidal embedding hiện tại dùng `timesteps[:, None].float()` nên
   *nhiều khả năng chạy được ngay không cần sửa*, nhưng nên kiểm tra range của `freqs` có phù
   hợp với t nhỏ (gần 0) hay không — nếu embedding bị "phẳng" ở vùng t nhỏ, có thể cần scale
   lại `t` (ví dụ nhân 1000 trước khi đưa vào embedding, giữ range tương tự bản gốc).
4. **Sửa nơi gọi trong `Main.py`**: tìm đoạn khởi tạo `GaussianDiffusion(...)` và các lệnh gọi
   `.p_sample(...)` để build `image_adj`/`text_adj` — thay bằng `ConditionalFlowMatching(...)`
   và `.sample(...)`, nhớ truyền `z_prior` đã tính ở bước 1 thay vì để hàm tự sinh noise.
5. **Giữ nguyên phần còn lại của DiffMM** (contrastive learning, GCN layers, `forward_MM`,
   `forward_cl_MM`) — không đụng vào, để đảm bảo so sánh công bằng, chỉ đổi đúng module sinh
   graph theo modality.

---

## 5. Kế hoạch ablation (khớp với quy trình 3 bước bạn đang chạy)

Để tách được đúng đâu là nguồn gain (đúng tinh thần bạn đang làm với DiffMM), chạy theo thứ tự:

1. **Control**: CFM nhưng `z_prior` vẫn là Gaussian noise (giữ nguyên training/sampling mới,
   chỉ đổi cơ chế DDPM→CFM). Mục đích: xem riêng việc đổi sang flow matching (đường thẳng,
   ít bước, deterministic) có tự nó cải thiện gì không so với DiffMM gốc.
2. **Treatment**: CFM + CF-informed prior (Option A). So với (1) để cô lập đúng phần đóng góp
   của prior — đây là phép so sánh quan trọng nhất, tương ứng với ablation "w/o Prior" trong
   FlowCF/FlowRec.
3. **Kiểm tra bug false-negative**: chạy lại pipeline hard negative mining hiện tại trên bản
   Treatment — nếu số lượng false negative giảm rõ rệt mà không cần patch curriculum, nghĩa là
   nguồn nhiễu tích luỹ qua nhiều bước reverse-diffusion đúng là nguyên nhân gốc. Nếu vẫn còn,
   bug nằm ở chỗ khác (khả năng cao là thứ tự khởi tạo `itmEmbeds` như đã nói ở mục 2.3).

---

## 6. Checklist / lỗi thường gặp khi implement

- [ ] `t=0` là trường hợp biên: đảm bảo không chia cho 0 ở bất kỳ đâu (không còn `alphas_cumprod`
      nên rủi ro này thấp hơn DDPM, nhưng nếu bạn suy ngược `x_data_hat` từ `v_pred` ở `t` gần 1,
      hệ số `(1-t)` gần 0 có thể làm mất thông tin — kiểm tra numerically).
- [ ] `z_prior` phải cùng device (`.cuda()`) và cùng shape với `x_data` — copy đúng pattern
      `.cuda()` đang dùng trong code gốc.
- [ ] Nếu dùng Option A, đảm bảo đồ thị collaborative-thuần (`adj_plain`) được xây **tách biệt**
      với `image_adj`/`text_adj` — không lẫn tín hiệu modality vào prior, nếu không sẽ mất ý
      nghĩa "warm start từ CF thuần".
  - [ ] So sánh `n_steps` khi sample: thử nhỏ dần (10 → 5 → 2) để xem hiệu năng có giữ được
      không — đây cũng là số liệu tốt để show trong paper (CFM cần ít bước hơn diffusion mà vẫn
      giữ/tăng metric).
- [ ] Log lại giá trị `cfm_loss` và `gc_loss` riêng biệt qua epoch — nếu `gc_loss` không giảm
      trong khi `cfm_loss` giảm tốt, khả năng cao vấn đề nằm ở `itmEmbeds` init, không phải ở
      flow matching.

---

## 7. Tài liệu tham khảo để đọc thêm khi implement

- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 — công thức CFM gốc.
- Liu et al., *Flow Matching for Collaborative Filtering (FlowCF)*, KDD 2025 — behavior-guided
  prior, chính là nguồn cảm hứng cho mục 3.
- Jiang et al., *DiffMM: Multi-Modal Diffusion Model for Recommendation*, ACM MM 2024 —
  code gốc bạn đang sửa (`HKUDS/DiffMM`).