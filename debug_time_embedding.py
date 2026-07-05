import torch

from Model import ConditionalFlowMatching, Denoise
from Params import args


def tensor_stats(tensor):
	tensor = tensor.detach().float()
	return {
		'mean': tensor.mean().item(),
		'std': tensor.std(unbiased=False).item(),
		'min': tensor.min().item(),
		'max': tensor.max().item()
	}


def print_stats(name, tensor):
	stats = tensor_stats(tensor)
	print('%s: mean = %.8f, std = %.8f, min = %.8f, max = %.8f' % (
		name,
		stats['mean'],
		stats['std'],
		stats['min'],
		stats['max']
	))


def log_time_embedding(flow, denoise):
	t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
	scaled_t = flow.scale_time(t)
	time_emb = denoise.time_embedding(scaled_t).detach()

	print('\n[scale_time + sinusoidal time embedding]')
	print('flow_time_scale = %.6f' % args.flow_time_scale)
	for idx in range(t.shape[0]):
		values = ', '.join(['%.6f' % val for val in time_emb[idx].tolist()])
		print('t = %.1f, scale_time(t) = %.6f, time_emb = [%s]' % (
			t[idx].item(),
			scaled_t[idx].item(),
			values
		))

	for idx in range(1, time_emb.shape[0]):
		diff = (time_emb[idx] - time_emb[idx - 1]).abs()
		values = ', '.join(['%.6f' % val for val in diff.tolist()])
		print('abs diff t %.1f -> %.1f: mean = %.8f, min = %.8f, max = %.8f, diff = [%s]' % (
			t[idx - 1].item(),
			t[idx].item(),
			diff.mean().item(),
			diff.min().item(),
			diff.max().item(),
			values
		))


def log_prior_norms(flow, x_start):
	z_prior = flow.make_prior(x_start)
	x_norm = x_start.reshape(x_start.shape[0], -1).norm(dim=1)
	z_norm = z_prior.reshape(z_prior.shape[0], -1).norm(dim=1)
	delta_norm = (x_start - z_prior).reshape(x_start.shape[0], -1).norm(dim=1)

	print('\n[x1 vs z_prior norm]')
	print_stats('x1 norm', x_norm)
	print_stats('z_prior norm', z_norm)
	print_stats('|x1 - z_prior| norm', delta_norm)
	print_stats('x1 values', x_start)
	print_stats('z_prior values', z_prior)

	return z_prior


def log_loss_buckets(flow, denoise, x_start, z_prior):
	batch_size = x_start.shape[0]
	item_dim = x_start.shape[1]
	latent_dim = 16
	t = torch.rand(batch_size)
	x_t = flow.interpolate(x_start, t, z_prior)

	itm_embeds = torch.randn(item_dim, latent_dim) / (item_dim ** 0.5)
	model_feats = torch.randn(item_dim, latent_dim) / (item_dim ** 0.5)

	with torch.no_grad():
		v_pred = denoise(x_t, flow.scale_time(t), mess_dropout=False)
		v_target = x_start - z_prior
		diff_loss = flow.mean_flat((v_pred - v_target) ** 2)

		t_view = t
		while len(t_view.shape) < len(x_start.shape):
			t_view = t_view[..., None]
		x_data_hat = x_t + (1.0 - t_view) * v_pred

		usr_model_embeds = torch.mm(x_data_hat, model_feats)
		usr_id_embeds = torch.mm(x_start, itm_embeds)
		gc_loss = flow.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

	print('\n[diff_loss vs gc_loss by t bucket]')
	print('batch_size = %d, item_dim = %d, latent_dim = %d' % (batch_size, item_dim, latent_dim))
	for start, end in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
		if end == 1.0:
			mask = (t >= start) & (t <= end)
		else:
			mask = (t >= start) & (t < end)
		if mask.sum().item() == 0:
			print('t in [%.1f, %.1f]: empty' % (start, end))
			continue

		diff_bucket = diff_loss[mask]
		gc_bucket = gc_loss[mask]
		ratio = gc_bucket.mean() / diff_bucket.mean().clamp_min(1e-12)
		print(
			't in [%.1f, %.1f]: n = %d, '
			'diff_loss(mean=%.8f,std=%.8f), '
			'gc_loss(mean=%.8f,std=%.8f), '
			'gc/diff = %.8f' % (
				start,
				end,
				mask.sum().item(),
				diff_bucket.mean().item(),
				diff_bucket.std(unbiased=False).item(),
				gc_bucket.mean().item(),
				gc_bucket.std(unbiased=False).item(),
				ratio.item()
			)
		)


def main():
	torch.manual_seed(args.seed)
	batch_size = 1024
	item_dim = 128
	hidden_dim = 256
	x_start = (torch.rand(batch_size, item_dim) < 0.05).float()
	flow = ConditionalFlowMatching(
		args.steps,
		prior_type=args.flow_prior,
		prior_mix=args.cf_prior_mix,
		prior_dropout=args.cf_prior_dropout,
		time_scale=args.flow_time_scale
	)
	scale_factor = flow.calibrate_prior_scale([(x_start, None)], x_start.device)
	denoise = Denoise([item_dim, hidden_dim], [hidden_dim, item_dim], args.d_emb_size, norm=args.norm)
	denoise.eval()

	print('Synthetic diagnostic only; this does not load the real dataset.')
	print('calibrated prior_scale = %.6f' % scale_factor)
	log_time_embedding(flow, denoise)
	z_prior = log_prior_norms(flow, x_start)
	log_loss_buckets(flow, denoise, x_start, z_prior)


if __name__ == '__main__':
	main()
