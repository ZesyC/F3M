# Báo cáo chẩn đoán CFM Prior

## Phạm vi kiểm tra

Report này kiểm tra 6 nghi phạm liên quan đến Conditional Flow Matching trong repo hiện tại:

1. Lệch scale/phân phối giữa `X0` và `X1`.
2. Pairing `X0[i]` - `X1[i]` sai do shuffle riêng.
3. Inference vẫn integrate từ `t = 0`.
4. Prior quá gần target, model học gần identity.
5. Prior kéo theo bias/lỗi từ LightGCN hoặc embedding chưa ổn định.
6. Nhiễu cộng dồn với hard-negative-mining/false-negative bug.

Không chạy full training được vì môi trường hiện tại báo `torch.cuda.is_available() == False`, trong khi code đang hard-code `.cuda()` ở `DataHandler.py`, `Main.py`, và `Model.py`.

## Mapping `X0` và `X1` trong code hiện tại

Trong code này, `X0` không phải LightGCN embedding trực tiếp.

- `X1 = x_start`: vector tương tác user-item nhị phân, lấy từ `DiffusionData(torch.FloatTensor(self.trnMat.toarray()))` trong `DataHandler.py:81`.
- `X0 = z_prior`: prior sinh bởi `ConditionalFlowMatching.make_prior(...)` trong `Model.py:325`.
- Nội suy: `x_t = t * x_start + (1.0 - t) * z_prior` trong `Model.py:356-359`.
- Velocity target: `v_target = x_start - z_prior` trong `Model.py:368-369`.

LightGCN/item embedding chỉ đi vào `gc_loss`:

- `itmEmbeds = self.model.getItemEmbeds().detach()` trong `Main.py:145`.
- `usr_id_embeds = torch.mm(x_start, itmEmbeds)` trong `Model.py:379`.

Vì vậy, nếu debug theo công thức FlowCF, cách hiểu đúng trong repo này là:

```text
X0 = z_prior
X1 = x_start
```

## Kết luận nhanh

| Mục | Kết luận | Mức độ nghi ngờ |
|---|---|---|
| 1. Scale `X0/X1` | Với default `--flow_prior popularity --cf_prior_mix 0.5`, scale không lệch nghiêm trọng. Nếu chuyển sang `--flow_prior gaussian`, lệch rất nặng. | Trung bình |
| 2. Pairing minibatch | Không thấy bug pairing. `DiffusionData[index]` trả đúng `(row, index)`, `DataLoader(shuffle=True)` shuffle cả tuple cùng nhau. | Thấp |
| 3. Inference từ `t=0` | Đúng là current code sample từ `x_t = z_prior` và bước đầu tiên tại `t=0`. Chưa có `t_min`. | Cao |
| 4. Prior gần target | Không gần target. Delta trung bình vẫn lớn so với norm của `x_start`. | Thấp |
| 5. LightGCN/prior bias | Có rủi ro ở `gc_loss`: item/model feature embeddings chưa pretrain, epoch đầu gần như random. Chưa đo được Recall riêng do không có CUDA. | Cao |
| 6. Hard-negative-mining | Repo hiện tại không có hard-negative-mining. Chỉ có random negative sampling. False negative do val/test positives tồn tại nhưng rất nhỏ. | Thấp với code hiện tại |

## 1. Scale/phân phối giữa `X0` và `X1`

Code hiện tại:

- Default params trong `Params.py:31-33`:
  - `flow_prior = popularity`
  - `cf_prior_mix = 0.5`
  - `cf_prior_dropout = 0.5`
- `item_prior = item_counts / args.user` trong `Main.py:97-100`.
- `z_prior = 0.5 * popularity_prior + 0.5 * dropout_user_history(x_start)` khi dùng default.

Thống kê trên dataset local:

| Dataset | `x_start` mean | `x_start` std | popularity prior mean | popularity prior std | default prior mean | default prior std |
|---|---:|---:|---:|---:|---:|---:|
| baby | 0.00086479 | 0.02939451 | 0.00086479 | 0.00140433 | 0.00064455 | 0.01034513 |
| tiktok | 0.00095332 | 0.03086111 | 0.00095332 | 0.00173809 | 0.00071745 | 0.01103824 |

Nhận xét:

- Với default popularity/user-history prior, `X0` và `X1` cùng là vector trên không gian item, cùng scale `[0, 1]`.
- `X0` default có std nhỏ hơn `X1` khoảng 3 lần, nhưng không phải mismatch kiểu embedding-vs-adjacency.
- Nếu dùng `--flow_prior gaussian`, `X0 ~ N(0,1)` trong `Model.py:326-327`, trong khi `X1 std ~= 0.03`. Đây là mismatch rất lớn và nên được coi là nghi phạm cao.

Khuyến nghị:

- Nếu cần ablation Gaussian, nên normalize/scale lại hoặc ít nhất report riêng vì nó không cùng scale với binary interaction row.
- Với default popularity prior, chưa cần z-score normalize ngay. Nên log runtime `mean/std/min/max` của `x_start`, `z_prior`, `x_t`, `v_target` để bắt lỗi sớm.

## 2. Pairing `X0[i]` - `X1[i]`

Code hiện tại:

- `DiffusionData.__getitem__(index)` trả `item = self.data[index]` và `index` trong `DataHandler.py:132-134`.
- `diffusionLoader = DataLoader(..., shuffle=True)` trong `DataHandler.py:82`.
- Trong train, `batch_item, batch_index = batch` được truyền cùng nhau vào `training_losses(...)` trong `Main.py:141-159`.

Kiểm tra thực tế trên first shuffled batch:

```text
baby: pairing_ok=True, checked=1024
tiktok: pairing_ok=True, checked=1024
```

Kết luận:

- Không thấy bug shuffle riêng `X0` và `X1`.
- `batch_index` đúng với `batch_item`.
- Lưu ý: `batch_index` hiện không được dùng để tạo `z_prior`; `z_prior` được tạo từ chính `x_start`. Vì vậy pairing user-index chỉ ảnh hưởng `gc_loss`/debug, không phải prior generation.

## 3. Sampling inference vẫn integrate từ `t = 0`

Code hiện tại trong `Model.py:385-395`:

```python
z_prior = self.make_prior(x_start)
x_t = z_prior
dt = 1.0 / n_steps

for i in range(n_steps):
    t = torch.full((x_t.shape[0],), i * dt, device=x_t.device)
    v_pred = model(x_t, self.scale_time(t), False)
    x_t = x_t + dt * v_pred
```

Với default `--steps 5` và `--sampling_steps 0`:

```text
n_steps = 5
t sequence = [0.0, 0.2, 0.4, 0.6, 0.8]
```

Kết luận:

- Đúng là inference bắt đầu từ `t = 0`.
- Chưa có tham số `t_min`.
- Nếu prior informative nhưng model yếu ở vùng đầu trajectory, đây là nghi phạm mạnh.

Khuyến nghị fix nhỏ:

- Thêm tham số `--flow_t_min`, ví dụ default `0.0`.
- Trong `sample(...)`, bắt đầu `x_t` tại `interpolate(x_start, t_min, z_prior)` nếu vẫn truyền `x_start` vào sample.
- Tích phân từ `t_min` đến `1.0`, với `dt = (1.0 - t_min) / n_steps`.
- Thử ablation `t_min = 0.1`, `0.2`, `0.3`.

## 4. Prior quá gần target

Đo delta `||X1 - X0||_2`:

| Dataset | `||x_start||` mean | popularity `||x_start-z||` mean | default `||x_start-z||` mean |
|---|---:|---:|---:|
| baby | 2.3553 | 2.3523 | 1.8287 |
| tiktok | 1.5542 | 1.5543 | 1.2017 |

Thêm thông tin cosine với popularity prior:

| Dataset | Cosine mean giữa `x_start` và popularity prior |
|---|---:|
| baby | 0.0535 |
| tiktok | 0.0628 |

Kết luận:

- Prior không quá gần target.
- Popularity prior gần như chỉ là global item frequency, cosine rất thấp.
- Default prior gần target hơn popularity-only do có `0.5 * dropout_user_history(x_start)`, nhưng delta vẫn không gần 0.

Nghi phạm "model học identity vì prior quá gần target" không phù hợp với dữ liệu hiện tại.

## 5. Bias/lỗi từ LightGCN hoặc embedding chưa pretrain

Đây là rủi ro thật sự, nhưng không nằm ở `X0` trực tiếp.

Code flow mỗi epoch:

1. Tạo `Model(...)` với `uEmbeds`, `iEmbeds` init Xavier trong `Model.py:15-16`.
2. Train denoise/CFM trước trong `Main.py:141-189`.
3. Sau đó mới train recommender BPR trong `Main.py:274-331`.

Trong diffusion/CFM loss:

```python
iEmbeds = self.model.getItemEmbeds().detach()
image_feats = self.model.getImageFeats().detach()
text_feats = self.model.getTextFeats().detach()
```

Epoch đầu, các embedding/projection này chưa được pretrain bởi BPR. Vì vậy:

- `gc_loss` có thể ép denoise model fit theo target embedding random/unstable.
- Nếu `e_loss` lớn, tín hiệu này có thể làm hỏng velocity learning.
- Trên Baby/Sports sparse, rủi ro này cao hơn.

Chưa đo được Recall@K riêng của LightGCN/model vì local không có CUDA và code đang hard-code `.cuda()`.

Khuyến nghị ablation:

- Chạy `--e_loss 0` để tắt `gc_loss`, giữ CFM prior như cũ.
- Hoặc warm-up BPR 1-5 epoch trước khi train CFM.
- Log `CFM loss` và `GC loss` riêng; nếu `GC loss` chi phối tổng loss lúc đầu, đây là dấu hiệu xấu.

## 6. Hard-negative-mining và false negative

Repo hiện tại không có hard-negative-mining.

Chỉ thấy negative sampling random trong `DataHandler.py:91-98`:

```python
while True:
    iNeg = np.random.randint(args.item)
    if (u, iNeg) not in self.dokmat:
        break
self.negs[i] = iNeg
```

Hàm này chỉ loại train positives, không loại val/test positives. Kiểm tra false negative do held-out positives:

| Dataset | Simulated false negatives / train triples | Tỷ lệ |
|---|---:|---:|
| baby | 44 / 118551 | 0.00037115 |
| tiktok | 10 / 59541 | 0.00016795 |

Kết luận:

- Nếu bạn đang nói đến một branch khác có hard-negative-mining, nó không nằm trong working tree hiện tại.
- Với code hiện tại, false negative do random sampler tồn tại nhưng rất nhỏ, không đủ mạnh để giải thích collapse lớn.
- Rủi ro uninitialized embedding trong repo này nằm chủ yếu ở `gc_loss`, không nằm ở hard-negative-mining.

## Ưu tiên debug tiếp theo

Nên làm theo thứ tự:

1. Chạy ablation `--e_loss 0` để cô lập CFM velocity loss khỏi `gc_loss` đang dùng embedding chưa pretrain.
2. Thêm/log `flow_t_min`, thử `0.1`, `0.2`, `0.3` cho sampling.
3. Log `mean/std/min/max` của `x_start`, `z_prior`, `x_t`, `v_target` ở epoch đầu.
4. Nếu dùng `--flow_prior gaussian`, dùng nó như control riêng và ghi rõ scale mismatch; không so sánh trực tiếp với popularity prior nếu chưa normalize.
5. Nếu có branch hard-negative-mining riêng, cần audit branch đó riêng vì working tree hiện tại không có module này.

## Lệnh đã dùng để kiểm tra

Kiểm tra CUDA:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

Kiểm tra code path:

```bash
rg -n "X0|X_0|X1|X_1|flow|Flow|velocity|prior|LightGCN|negative|hard|sample|sampling|tmin|ode|integrat|diffusion" Main.py Model.py DataHandler.py Params.py Utils README.md prv.md
```

Kiểm tra pairing bằng `DataLoader(shuffle=True)`:

```text
baby: first shuffled DataLoader batch pairing_ok=True, checked=1024
tiktok: first shuffled DataLoader batch pairing_ok=True, checked=1024
```
