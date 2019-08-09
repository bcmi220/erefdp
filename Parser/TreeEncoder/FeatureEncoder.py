import dynet as dy
from Compositor import LSTMState


class EmptyFeatureEncoder(object):
	def __init__(self, model, options, rel_vocab):
		pass


	def refresh(self):
		pass


	def encode(self, pend_encs, pend_ids, head, dep, irel=None):
		return pend_encs[dep]


class FeatureEncoder(EmptyFeatureEncoder):
	def __init__(self, model, options, rel_vocab):
		self.model = model.add_subcollection('feats')
		self.rel_feat = options.rel_feat
		self.dist_feat = options.dist_feat
		if options.rel_feat:
			self.REL_LOOKUP = self.model.add_lookup_parameters(
								(len(rel_vocab), options.rel_dim))
		if options.dist_feat:
			self.dist_min = options.min_distance
			self.dist_max = options.max_distance
			self.dist_range = self.dist_max - self.dist_min
			self.pos_unk = self.dist_range + 1
			self.neg_unk = self.dist_range + 2
			self.DIST_LOOKUP = self.model.add_lookup_parameters(
								(self.dist_range + 3, options.dist_dim))
		rel_dims = options.rel_dim if options.rel_feat else 0
		dist_dims = options.dist_dim if options.dist_feat else 0
		input_dims = rel_dims + dist_dims + options.compos_outdim
		output_dims = options.compos_outdim
		self.trans_Wp = self.model.add_parameters((output_dims, input_dims))


	def refresh(self):
		self.transW = self.trans_Wp.expr()


	def encode(self, pend_encs, pend_ids, head, dep, irel=None):
		dep_enc = pend_encs[dep].output
		rel_enc = dy.lookup(self.REL_LOOKUP, irel) if self.rel_feat else None
		if self.dist_feat:
			dist = pend_ids[head] - pend_ids[dep] - self.dist_min
			index = self.neg_unk if dist < 0 \
								else (self.pos_unk if dist > self.dist_range \
													else dist)
			dist_enc = dy.lookup(self.DIST_LOOKUP, index)
		else:
			dist_enc = None
		feat_emb = dy.tanh(self.transW * \
						dy.concatenate(filter(None, [dep_enc, rel_enc, dist_enc])))
		return LSTMState(feat_emb, pend_encs[dep].memory_cell)


def feature_encoder(model, options, rel_vocab):
	if not (options.rel_feat or options.dist_feat):
		return EmptyFeatureEncoder(model, options, rel_vocab)
	else:
		return FeatureEncoder(model, options, rel_vocab)

