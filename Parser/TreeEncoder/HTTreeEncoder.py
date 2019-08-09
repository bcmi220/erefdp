from Compositor import *
from FeatureEncoder import *


class EmptyTreeEncoder(object):
	def __init__(self, model, options):
		pass


	def input_encodings(self, encodings, sent_len, sentence=None):
		self._states = [LSTMState(hidden=dy.pick(encodings, i, dim=1)) \
						for i in xrange(sent_len)]
		self._ids = range(sent_len)
		if sentence:
			self._words = list(sentence)
			self._unassigned = {head.w_id: sum(1 for dep in sentence 
								if dep.parent_id == head.w_id) for head in sentence}
		else:
			self._words = None
			self._unassigned = None


	def attach(self, head, dep, train=True):
		del self._states[dep], self._ids[dep]
		if self._words:
			self._unassigned[self._words[dep].parent_id] -= 1
			del self._words[dep]


	@property
	def pending_states(self):
		return self._states


	@property
	def pending_encodings(self):
		return None


	@property
	def pending_words(self):
		return self._words


	@property
	def pending_ids(self):
		return self._ids


	@property
	def unassigned_words(self):
		return self._unassigned


	@property
	def pending_length(self):
		return len(self._states)


	def refresh(self):
		pass


class RecursiveEncoder(EmptyTreeEncoder):
	def __init__(self, model, options, rel_vocab=None):
		self.dropout_rate = options.dropout_rate
		self._com = get_compositor(model, options)
		self._feat_enc = feature_encoder(model, options, rel_vocab)
		self.REL_LOOKUP = self._feat_enc.REL_LOOKUP
		self._states = None
		self._children = None

		self.layers = options.layers
		self.input_dims = options.compos_indim
		self.output_dims = options.lstm_dim * 0.5
		self.model = model#.add_subcollection('historical')
		self._hist_enc = [dy.VanillaLSTMBuilder(self.layers, self.input_dims, \
												self.output_dims, self.model), 
						dy.VanillaLSTMBuilder(self.layers, self.input_dims, \
												self.output_dims, self.model)]
		mlp_indim = options.rel_dim + self.output_dims * 2
		self.hist_Wp = self.model.add_parameters((self.input_dims, mlp_indim))
		self.hist_bp = self.model.add_parameters((self.input_dims,))
		

	def input_encodings(self, encodings, sent_len, sentence=None):
		self._children = [LSTMStateSet() for _ in xrange(sent_len)]
		self._states = [self._com.add_input(self._children[i], 
						dy.pick(encodings, i, dim=1)) for i in xrange(sent_len)]
		self._encs = [dy.pick(encodings, i, dim=1) for i in xrange(sent_len)]
		self._hists = [[self._hist_enc[0].initial_state().add_input(enc),
						self._hist_enc[1].initial_state().add_input(enc)] 
						for enc in self._encs]
		self._ids = range(sent_len)
		if sentence:
			self._words = list(sentence)
			self._unassigned = {head.w_id: sum(1 for dep in sentence 
								if dep.parent_id == head.w_id) for head in sentence}
		else:
			self._words = None
			self._unassigned = None

	
	def attach(self, head, dep, irel=None, train=True):
		_tmp_state = self._feat_enc.encode(self._states, self._ids, head, dep, irel)
		if train:
			self._children[head].add_with_dropout(_tmp_state, self.dropout_rate)
		else:
			self._children[head].add_state(_tmp_state)
		self._states[head] = self._com.add_input(self._children[head], self._encs[head])

		hist_idx = 0 if head > dep else 1
		self._hists[head][hist_idx] = self._hists[head][hist_idx].add_input(
			dy.rectify(self.histW * dy.concatenate([self._hists[dep][0].output(), 
					dy.lookup(self.REL_LOOKUP, irel), self._hists[dep][1].output()]))
					+ self.histb)

		del self._states[dep], self._ids[dep], self._children[dep], self._encs[dep]	,self._hists[dep]
		if self._words:
			self._unassigned[self._words[dep].parent_id] -= 1
			del self._words[dep]


	def refresh(self):
		self._com.refresh()
		self._feat_enc.refresh()
		self.histW = self.hist_Wp.expr()
		self.histb = self.hist_bp.expr()


	@property
	def pending_encodings(self):
		return self._encs


	@property
	def pending_history(self):
		return self._hists


