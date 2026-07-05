import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, ConditionalFlowMatching, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
import setproctitle
from scipy.sparse import coo_matrix

class Coach:
	def __init__(self, handler):
		self.handler = handler
		self.t_buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
		self.prev_bucket_summary = dict()

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def makeFlowStatsPrint(self, name, stats, diff_loss, gc_loss):
		cfm_loss = diff_loss.mean().detach().item()
		gc_loss_val = gc_loss.mean().detach().item()
		weighted_gc = gc_loss_val * args.e_loss
		gc_ratio = weighted_gc / cfm_loss if abs(cfm_loss) > 1e-12 else float('inf')
		ret = 'Flow Stats %s: cfm = %.6f, gc = %.6f, gc*e_loss = %.6f, gc/cfm = %.6f' % (
			name,
			cfm_loss,
			gc_loss_val,
			weighted_gc,
			gc_ratio
		)
		for stat_name in ['x_start', 'z_prior', 'x_t', 'v_target', 't', 'delta_norm']:
			stat = stats[stat_name]
			ret += ', %s(mean=%.6f,std=%.6f,min=%.6f,max=%.6f)' % (
				stat_name,
				stat['mean'],
				stat['std'],
				stat['min'],
				stat['max']
			)
		return ret

	def logFlowStats(self, name, stats, diff_loss, gc_loss):
		log(self.makeFlowStatsPrint(name, stats, diff_loss, gc_loss), save=False)

	def logPriorNormStats(self, name, stats, step):
		x_norm = stats['x_norm']['mean']
		z_norm = stats['z_norm']['mean']
		ratio = x_norm / max(z_norm, 1e-12)
		log('[x1 vs z_prior norm] %s step %d: x1_norm_mean=%.6f, z_prior_norm_mean=%.6f, ratio=%.6f' % (
			name,
			step,
			x_norm,
			z_norm,
			ratio
		), save=False)

	def initBucketStats(self):
		stats = dict()
		for bucket in self.t_buckets:
			stats[bucket] = {
				'diff_sum': 0.0,
				'diff_sq_sum': 0.0,
				'gc_sum': 0.0,
				'gc_sq_sum': 0.0,
				'n': 0
			}
		return stats

	def updateBucketStats(self, bucket_stats, t, diff_loss, gc_loss):
		t = t.detach()
		diff_loss = diff_loss.detach()
		gc_loss = gc_loss.detach()
		for start, end in self.t_buckets:
			if end == 1.0:
				mask = (t >= start) & (t <= end)
			else:
				mask = (t >= start) & (t < end)
			n = mask.sum().item()
			if n == 0:
				continue
			diff_bucket = diff_loss[mask].float()
			gc_bucket = gc_loss[mask].float()
			stat = bucket_stats[(start, end)]
			stat['diff_sum'] += diff_bucket.sum().item()
			stat['diff_sq_sum'] += (diff_bucket * diff_bucket).sum().item()
			stat['gc_sum'] += gc_bucket.sum().item()
			stat['gc_sq_sum'] += (gc_bucket * gc_bucket).sum().item()
			stat['n'] += n

	def summarizeBucketStats(self, bucket_stats):
		summary = dict()
		for bucket, stat in bucket_stats.items():
			n = stat['n']
			if n == 0:
				summary[bucket] = None
				continue
			diff_mean = stat['diff_sum'] / n
			gc_mean = stat['gc_sum'] / n
			diff_var = max(stat['diff_sq_sum'] / n - diff_mean * diff_mean, 0.0)
			gc_var = max(stat['gc_sq_sum'] / n - gc_mean * gc_mean, 0.0)
			summary[bucket] = {
				'n': n,
				'diff_mean': diff_mean,
				'diff_std': diff_var ** 0.5,
				'gc_mean': gc_mean,
				'gc_std': gc_var ** 0.5,
				'ratio': gc_mean / max(diff_mean, 1e-12)
			}
		return summary

	def logBucketStats(self, epoch, name, bucket_stats):
		summary = self.summarizeBucketStats(bucket_stats)
		prev_summary = self.prev_bucket_summary.get(name)
		log('[diff_loss vs gc_loss by t bucket] %s' % name, save=False)
		for start, end in self.t_buckets:
			stat = summary[(start, end)]
			if stat is None:
				log('Epoch %d | t in [%.1f,%.1f]: empty' % (epoch, start, end), save=False)
				continue
			if prev_summary is not None and prev_summary.get((start, end)) is not None:
				prev = prev_summary[(start, end)]
				diff_delta = stat['diff_mean'] - prev['diff_mean']
				gc_delta = stat['gc_mean'] - prev['gc_mean']
			else:
				diff_delta = 0.0
				gc_delta = 0.0
			log(
				'Epoch %d | t in [%.1f,%.1f]: n=%d, '
				'diff_loss(mean=%.8f,std=%.8f), '
				'gc_loss(mean=%.8f,std=%.8f), '
				'gc/diff=%.8f, '
				'delta_vs_prev(diff=%.8f,gc=%.8f)' % (
					epoch,
					start,
					end,
					stat['n'],
					stat['diff_mean'],
					stat['diff_std'],
					stat['gc_mean'],
					stat['gc_std'],
					stat['ratio'],
					diff_delta,
					gc_delta
				),
				save=False
			)
		self.prev_bucket_summary[name] = summary

	def logTimeEmbeddingDebug(self):
		t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=self.diffusion_model.item_prior.device)
		scaled_t = self.diffusion_model.scale_time(t)
		time_emb = self.denoise_model_image.time_embedding(scaled_t).detach().cpu()
		scaled_t_cpu = scaled_t.detach().cpu()
		log('Time embedding debug: flow_time_scale = %.6f' % args.flow_time_scale, save=False)
		for idx in range(t.shape[0]):
			values = ', '.join(['%.6f' % val for val in time_emb[idx].tolist()])
			log('t = %.1f, scale_time(t) = %.6f, time_emb = [%s]' % (
				t[idx].item(),
				scaled_t_cpu[idx].item(),
				values
			), save=False)
		for idx in range(1, time_emb.shape[0]):
			diff = (time_emb[idx] - time_emb[idx - 1]).abs()
			values = ', '.join(['%.6f' % val for val in diff.tolist()])
			log('abs diff t %.1f -> %.1f: mean = %.8f, min = %.8f, max = %.8f, diff = [%s]' % (
				t[idx - 1].item(),
				t[idx].item(),
				diff.mean().item(),
				diff.min().item(),
				diff.max().item(),
				values
			), save=False)

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.debug_time_embedding:
			self.logTimeEmbeddingDebug()
			return

		recallMax = 0
		ndcgMax = 0
		precisionMax = 0
		bestEpoch = 0

		log('Model Initialized')

		for ep in range(0, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch(ep)
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					precisionMax = reses['Precision']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))
			print()
		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision', precisionMax)

	def prepareModel(self):
		if args.data == 'tiktok':
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()).cuda()
		else:
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		item_prior = self.buildItemPrior()
		self.diffusion_model = ConditionalFlowMatching(
			args.steps,
			item_prior=item_prior,
			prior_type=args.flow_prior,
			prior_mix=args.cf_prior_mix,
			prior_dropout=args.cf_prior_dropout,
			time_scale=args.flow_time_scale
		).cuda()
		prior_scale = self.diffusion_model.calibrate_prior_scale(
			self.handler.diffusionLoader,
			self.diffusion_model.item_prior.device
		)
		log('CFM prior_scale calibrated on train diffusion loader: %.6f' % prior_scale, save=False)
		
		out_dims = eval(args.dims) + [args.item]
		in_dims = out_dims[::-1]
		self.denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

		out_dims = eval(args.dims) + [args.item]
		in_dims = out_dims[::-1]
		self.denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)

		if args.data == 'tiktok':
			out_dims = eval(args.dims) + [args.item]
			in_dims = out_dims[::-1]
			self.denoise_model_audio = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
			self.denoise_opt_audio = torch.optim.Adam(self.denoise_model_audio.parameters(), lr=args.lr, weight_decay=0)

	def buildItemPrior(self):
		item_counts = np.asarray(self.handler.trnMat.sum(axis=0)).reshape(-1).astype(np.float32)
		item_prior = item_counts / max(float(args.user), 1.0)
		return torch.from_numpy(item_prior)

	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def buildUIMatrix(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	def trainEpoch(self, epoch):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epRecLoss, epClLoss = 0, 0, 0
		epDiLoss = 0
		epDiLoss_image, epDiLoss_text = 0, 0
		epCfmLoss_image, epCfmLoss_text = 0, 0
		epGcLoss_image, epGcLoss_text = 0, 0
		if args.data == 'tiktok':
			epDiLoss_audio = 0
			epCfmLoss_audio, epGcLoss_audio = 0, 0
		steps = trnLoader.dataset.__len__() // args.batch

		diffusionLoader = self.handler.diffusionLoader
		diff_steps = max(len(diffusionLoader), 1)
		bucket_stats_image = self.initBucketStats()
		bucket_stats_text = self.initBucketStats()
		if args.data == 'tiktok':
			bucket_stats_audio = self.initBucketStats()

		for i, batch in enumerate(diffusionLoader):
			batch_item, batch_index = batch
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

			iEmbeds = self.model.getItemEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()

			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()
			if args.data == 'tiktok':
				audio_feats = self.model.getAudioFeats().detach()

			self.denoise_opt_image.zero_grad()
			self.denoise_opt_text.zero_grad()
			if args.data == 'tiktok':
				self.denoise_opt_audio.zero_grad()

			return_flow_stats = args.debug_flow_stats and i < args.debug_flow_stats_batches
			return_norm_stats = i < 1000 and i % 200 == 0
			return_stats = return_flow_stats or return_norm_stats

			if return_stats:
				diff_loss_image, gc_loss_image, flow_stats_image, t_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats, return_stats=True, return_t=True)
				diff_loss_text, gc_loss_text, flow_stats_text, t_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats, return_stats=True, return_t=True)
			else:
				diff_loss_image, gc_loss_image, t_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats, return_t=True)
				diff_loss_text, gc_loss_text, t_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats, return_t=True)
			if args.data == 'tiktok':
				if return_stats:
					diff_loss_audio, gc_loss_audio, flow_stats_audio, t_audio = self.diffusion_model.training_losses(self.denoise_model_audio, batch_item, iEmbeds, batch_index, audio_feats, return_stats=True, return_t=True)
				else:
					diff_loss_audio, gc_loss_audio, t_audio = self.diffusion_model.training_losses(self.denoise_model_audio, batch_item, iEmbeds, batch_index, audio_feats, return_t=True)

			self.updateBucketStats(bucket_stats_image, t_image, diff_loss_image, gc_loss_image)
			self.updateBucketStats(bucket_stats_text, t_text, diff_loss_text, gc_loss_text)
			if args.data == 'tiktok':
				self.updateBucketStats(bucket_stats_audio, t_audio, diff_loss_audio, gc_loss_audio)

			loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
			loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss
			if args.data == 'tiktok':
				loss_audio = diff_loss_audio.mean() + gc_loss_audio.mean() * args.e_loss

			if return_flow_stats:
				self.logFlowStats('image', flow_stats_image, diff_loss_image, gc_loss_image)
				self.logFlowStats('text', flow_stats_text, diff_loss_text, gc_loss_text)
				if args.data == 'tiktok':
					self.logFlowStats('audio', flow_stats_audio, diff_loss_audio, gc_loss_audio)
			if return_norm_stats:
				self.logPriorNormStats('image', flow_stats_image, i)
				self.logPriorNormStats('text', flow_stats_text, i)
				if args.data == 'tiktok':
					self.logPriorNormStats('audio', flow_stats_audio, i)

			epDiLoss_image += loss_image.item()
			epDiLoss_text += loss_text.item()
			epCfmLoss_image += diff_loss_image.mean().item()
			epCfmLoss_text += diff_loss_text.mean().item()
			epGcLoss_image += gc_loss_image.mean().item()
			epGcLoss_text += gc_loss_text.mean().item()
			if args.data == 'tiktok':
				epDiLoss_audio += loss_audio.item()
				epCfmLoss_audio += diff_loss_audio.mean().item()
				epGcLoss_audio += gc_loss_audio.mean().item()

			if args.data == 'tiktok':
				loss = loss_image + loss_text + loss_audio
			else:
				loss = loss_image + loss_text

			loss.backward()

			self.denoise_opt_image.step()
			self.denoise_opt_text.step()
			if args.data == 'tiktok':
				self.denoise_opt_audio.step()

			log('Diffusion Step %d/%d' % (i, diff_steps), save=False, oneline=True)

		log('')
		self.logBucketStats(epoch, 'image', bucket_stats_image)
		self.logBucketStats(epoch, 'text', bucket_stats_text)
		if args.data == 'tiktok':
			self.logBucketStats(epoch, 'audio', bucket_stats_audio)
		log('Start to re-build UI matrix')

		with torch.no_grad():

			u_list_image = []
			i_list_image = []
			edge_list_image = []

			u_list_text = []
			i_list_text = []
			edge_list_text = []

			if args.data == 'tiktok':
				u_list_audio = []
				i_list_audio = []
				edge_list_audio = []

			for _, batch in enumerate(diffusionLoader):
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

				# image
				denoised_batch = self.diffusion_model.sample(self.denoise_model_image, batch_item, args.sampling_steps)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_image.append(int(batch_index[i].cpu().numpy()))
						i_list_image.append(int(indices_[i][j].cpu().numpy()))
						edge_list_image.append(1.0)

				# text
				denoised_batch = self.diffusion_model.sample(self.denoise_model_text, batch_item, args.sampling_steps)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_text.append(int(batch_index[i].cpu().numpy()))
						i_list_text.append(int(indices_[i][j].cpu().numpy()))
						edge_list_text.append(1.0)

				if args.data == 'tiktok':
					# audio
					denoised_batch = self.diffusion_model.sample(self.denoise_model_audio, batch_item, args.sampling_steps)
					top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

					for i in range(batch_index.shape[0]):
						for j in range(indices_[i].shape[0]): 
							u_list_audio.append(int(batch_index[i].cpu().numpy()))
							i_list_audio.append(int(indices_[i][j].cpu().numpy()))
							edge_list_audio.append(1.0)

			# image
			u_list_image = np.array(u_list_image)
			i_list_image = np.array(i_list_image)
			edge_list_image = np.array(edge_list_image)
			self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
			self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

			# text
			u_list_text = np.array(u_list_text)
			i_list_text = np.array(i_list_text)
			edge_list_text = np.array(edge_list_text)
			self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
			self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

			if args.data == 'tiktok':
				# audio
				u_list_audio = np.array(u_list_audio)
				i_list_audio = np.array(i_list_audio)
				edge_list_audio = np.array(edge_list_audio)
				self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
				self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

		log('UI matrix built!')

		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			self.opt.zero_grad()

			if args.data == 'tiktok':
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]
			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			regLoss = self.model.reg_loss() * args.reg
			loss = bprLoss + regLoss
			
			epRecLoss += bprLoss.item()
			epLoss += loss.item()

			if args.data == 'tiktok':
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			if args.data == 'tiktok':
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg
				clLoss += (contrastLoss(usrEmbeds1, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds3, poss, args.temp)) * args.ssl_reg
				clLoss += (contrastLoss(usrEmbeds2, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds2, itmEmbeds3, poss, args.temp)) * args.ssl_reg
			else:
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

			clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, args.temp)) * args.ssl_reg
			clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, args.temp)) * args.ssl_reg
			if args.data == 'tiktok':
				clLoss3 = (contrastLoss(usrEmbeds, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds3, poss, args.temp)) * args.ssl_reg
				clLoss_ = clLoss1 + clLoss2 + clLoss3
			else:
				clLoss_ = clLoss1 + clLoss2

			if args.cl_method == 1:
				clLoss = clLoss_

			loss += clLoss

			epClLoss += clLoss.item()

			loss.backward()
			self.opt.step()

			log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
				i, 
				steps,
				bprLoss.item(),
        regLoss.item(),
				clLoss.item()
				), save=False, oneline=True)

		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['BPR Loss'] = epRecLoss / steps
		ret['CL loss'] = epClLoss / steps
		ret['Di image loss'] = epDiLoss_image / diff_steps
		ret['Di text loss'] = epDiLoss_text / diff_steps
		ret['CFM image loss'] = epCfmLoss_image / diff_steps
		ret['CFM text loss'] = epCfmLoss_text / diff_steps
		ret['GC image loss'] = epGcLoss_image / diff_steps
		ret['GC text loss'] = epGcLoss_text / diff_steps
		if args.data == 'tiktok':
			ret['Di audio loss'] = epDiLoss_audio / diff_steps
			ret['CFM audio loss'] = epCfmLoss_audio / diff_steps
			ret['GC audio loss'] = epGcLoss_audio / diff_steps
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg, epPrecision = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat

		if args.data == 'tiktok':
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
		else:
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)

		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = torch.topk(allPreds, args.topk)
			recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			epPrecision += precision
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (i, steps, recall, ndcg, precision), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		ret['Precision'] = epPrecision / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = allPrecision = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = precision = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
					precision += 1
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			precision = precision / args.topk
			allRecall += recall
			allNdcg += ndcg
			allPrecision += precision
		return allRecall, allNdcg, allPrecision

def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	seed_it(args.seed)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()
