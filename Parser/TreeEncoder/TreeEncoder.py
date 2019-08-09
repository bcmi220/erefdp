from Compositor import *
from FeatureEncoder import *

class EmptyTreeEncoder(object):
	def __init__(self, model, options, rel_vocab=None):
		pass
	
	def input_encodings(self, encodings, sent_len, sentence=None):
		self._states = [LSTMState(hidden=dy.pick(encodings, i, dim=1)) \
						for i in xrange(sent_len)]
		self._all_states = [item for item in self._states]
		self._ids = range(sent_len)
		if sentence:
			self._words = list(sentence)
			self._unassigned = {head.w_id: sum(1 for dep in sentence 
								if dep.parent_id == head.w_id) for head in sentence}
		else:
			self._words = None
			self._unassigned = None


	def attach(self, head, dep, ihead, idep, irel=None, train=True):
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

	@property
	def all_states(self):
		return self._all_states

	def refresh(self):
		pass


class RecursiveEncoder(EmptyTreeEncoder):
	def __init__(self, model, options, rel_vocab=None):
		self.dropout_rate = options.dropout_rate
		self._com = get_compositor(model, options)
		self._feat_enc = feature_encoder(model, options, rel_vocab)
		self._states = None
		self._children = None
		
		# #####
		# history lstm
		# self.model = model.add_subcollection('history')
		# layers = options.layers
		# input_dims = options.lstm_output_dim
		# hidden_dims = options.lstm_output_dim
		# self.dropout_input = options.dropout_lstm_input
		# self.dropout_hidden = options.dropout_lstm_hidden
		# self._hist_lstm = dy.VanillaLSTMBuilder(layers, input_dims, \
		# 										hidden_dims, self.model)
		# self._hist_Wp = self.model.add_parameters((hidden_dims, hidden_dims * 2))
		# self._hist_bp = self.model.add_parameters((hidden_dims, ), 
		# 											init=dy.ConstInitializer(0))
		# self._table = self.model.add_lookup_parameters((1, hidden_dims * 2))
		# #####


	def input_encodings(self, encodings, sent_len, sentence=None, train=False):
		self._children = [LSTMStateSet() for _ in xrange(sent_len)]
		self._states = [self._com.add_input(self._children[i], 
											dy.pick(encodings, i, dim=1), train) 
						for i in xrange(sent_len)]
		self._all_states = [item for item in self._states]

		self._encs = [dy.pick(encodings, i, dim=1) for i in xrange(sent_len)]
		self._ids = range(sent_len)
		if sentence:
			self._words = list(sentence)
			self._unassigned = {head.w_id: sum(1 for dep in sentence 
													if dep.parent_id == head.w_id) 
											for head in sentence}
		else:
			self._words = None
			self._unassigned = None

		# #####
		# history lstm
		# if train:
		# 	self._hist_lstm.set_dropouts(self.dropout_input, self.dropout_hidden)
		# else:
		# 	self._hist_lstm.disable_dropout()
		# self.history = self._hist_lstm.initial_state()
		# empty_emb = dy.lookup(self._table, 0)
		# empty_act = dy.tanh(dy.affine_transform(
		# 				[self.histb, self.histW, empty_emb]))
		# self.history = self.history.add_input(empty_act)
		# #####

	
	def attach(self, head, dep, ihead, idep, irel=None, train=False):
		_tmp_state = self._feat_enc.encode(self._states, self._ids, head, dep, irel)
		# _tmp_state = self._states[dep]
		# if train:
		# 	self._children[head].add_with_dropout(_tmp_state, self.dropout_rate)
		# else:
		# 	self._children[head].add_state(_tmp_state)

		# #####
		# history lstm
		# dep_enc = self._states[dep].output
		# head_enc = self._states[head].output
		# action = dy.tanh(dy.affine_transform(
		# 			[self.histb, self.histW, dy.concatenate([dep_enc, head_enc])]))
		# self.history = self.history.add_input(action)
		# #####

		self._children[head].add_state(_tmp_state)

		self._states[head] = self._com.add_input(self._children[head], 
												self._encs[head], train)

		# update states
		self._all_states[ihead] = self._states[head]
												
		del self._states[dep], self._ids[dep], self._children[dep], self._encs[dep]	
		if self._words:
			self._unassigned[self._words[dep].parent_id] -= 1
			del self._words[dep]


	def refresh(self):
		self._com.refresh()
		self._feat_enc.refresh()

		# #####
		# history lstm
		# self.histW = self._hist_Wp.expr()
		# self.histb = self._hist_bp.expr()
		# #####


	@property
	def pending_encodings(self):
		return self._encs

