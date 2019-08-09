import dynet as dy


class EmptyTreeEncoder(object):
	def __init__(self, model, options, rel_vocab=None):
		pass


	def input_encodings(self, encodings, sent_len, sentence=None):
		pass


	def attach(self, head, dep, irel=None, train=True):
		pass


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


class HTState(object):
	def __init__(self, states):
		self._states = states


	def __getitem__(self, key):
		return self._states[key]


	def __setitem__(self, key, value):
		self._states[key] = value


	@property
	def output(self):
		return dy.concatenate([self._states[0].output(), 
								self._states[1].output()])


class HTLSTMEncoder(EmptyTreeEncoder):
	def __init__(self, model, options, rel_vocab=None):
		self.model = model.add_subcollection('htencoder')
		self.dropout_rate = options.dropout_rate
		self.REL_LOOKUP = self.model.add_lookup_parameters(
								(len(rel_vocab), options.rel_dim))
		self._states = None
		self.layers = options.layers
		self.input_dims = options.compos_indim
		self.output_dims = options.lstm_dim * 0.5
		
		self._ht_enc = [dy.VanillaLSTMBuilder(self.layers, self.input_dims, \
												self.output_dims, self.model), 
						dy.VanillaLSTMBuilder(self.layers, self.input_dims, \
												self.output_dims, self.model)]
		mlp_indim = options.rel_dim + self.output_dims * 2
		self.ht_Wp = self.model.add_parameters((self.input_dims, mlp_indim))
		self.ht_bp = self.model.add_parameters((self.input_dims,))
		

	def input_encodings(self, encodings, sent_len, sentence=None):
		self._encs = [dy.pick(encodings, i, dim=1) for i in xrange(sent_len)]
		self._states = [HTState([self._ht_enc[0].initial_state().add_input(enc),
								self._ht_enc[1].initial_state().add_input(enc)]) 
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
		ht_idx = 0 if head > dep else 1
		if train:
			self._states[head][ht_idx] = self._states[head][ht_idx].add_input(
											dy.rectify(self.htb + self.htW * 
												dy.dropout(dy.concatenate(
													[self._states[dep][0].output(), 
													dy.lookup(self.REL_LOOKUP, irel), 
													self._states[dep][1].output()]), 
												self.dropout_rate)))
		else:
			self._states[head][ht_idx] = self._states[head][ht_idx].add_input(
											dy.rectify(self.htb + self.htW * 
												dy.concatenate(
													[self._states[dep][0].output(), 
													dy.lookup(self.REL_LOOKUP, irel), 
													self._states[dep][1].output()])))
		del self._states[dep], self._ids[dep], self._encs[dep]
		if self._words:
			self._unassigned[self._words[dep].parent_id] -= 1
			del self._words[dep]


	def refresh(self):
		self.htW = self.ht_Wp.expr()
		self.htb = self.ht_bp.expr()


	@property
	def pending_encodings(self):
		return self._encs

