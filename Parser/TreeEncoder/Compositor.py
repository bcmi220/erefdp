import sys
sys.path.append('..')
import dynet as dy
import Attention
from utils import *


class LSTMState(object):
	def __init__(self, hidden=None, memory=None):
		self._hidden_state = hidden
		self._memory_cell = memory


	@property
	def hidden_state(self):
		return self._hidden_state


	@property
	def output(self):
		return self._hidden_state


	@property
	def memory_cell(self):
		return self._memory_cell


class LSTMStateSet(object):
	def __init__(self):
		self._hidden_set = []
		self._memory_set = []


	def add_state(self, state):
		self._hidden_set.append(state.hidden_state)
		self._memory_set.append(state.memory_cell)


	def add_with_dropout(self, state, dropout_rate=0.25):
		if state.hidden_state:
			self._hidden_set.append(dy.dropout(state.hidden_state, dropout_rate))
		if state.memory_cell:
			self._memory_set.append(dy.dropout(state.memory_cell, dropout_rate))


	@property
	def hidden_set(self):
		return self._hidden_set


	@property
	def memory_set(self):
		return self._memory_set


	def unfold(self):
		return self._hidden_set[:], self._memory_set[:]
	

	def __iter__(self):
		return zip(self._hidden_set, self._memory_set)


class TreeLSTMBuilder(object):
	def __init__(self, model, options):
		self.model = model.add_subcollection('treelstm')
		self.input_dim = options.compos_indim
		self.output_dim = options.compos_outdim
		self.dropout_input = options.dropout_lstm_input
		self.dropout_hidden = options.dropout_lstm_hidden
		self._att = Attention.get_attention(model, options)

		W_i = orthonormal_initializer(self.output_dim, self.input_dim)
		W_f = orthonormal_initializer(self.output_dim, self.input_dim)
		W_o = orthonormal_initializer(self.output_dim, self.input_dim)
		W_u = orthonormal_initializer(self.output_dim, self.input_dim)

		U_i = orthonormal_initializer(self.output_dim, self.output_dim)
		U_f = orthonormal_initializer(self.output_dim, self.output_dim)
		U_o = orthonormal_initializer(self.output_dim, self.output_dim)
		U_u = orthonormal_initializer(self.output_dim, self.output_dim)

		self.tree_W_i = self.model.parameters_from_numpy(W_i)
		self.tree_W_f = self.model.parameters_from_numpy(W_f)
		self.tree_W_o = self.model.parameters_from_numpy(W_o)
		self.tree_W_u = self.model.parameters_from_numpy(W_u)

		self.tree_U_i = self.model.parameters_from_numpy(U_i)
		self.tree_U_f = self.model.parameters_from_numpy(U_f)
		self.tree_U_o = self.model.parameters_from_numpy(U_o)
		self.tree_U_u = self.model.parameters_from_numpy(U_u)

		self.tree_b_i = self.model.add_parameters((self.output_dim), 
													init=dy.ConstInitializer(0.))
		self.tree_b_f = self.model.add_parameters((self.output_dim), 
													init=dy.ConstInitializer(0.))
		self.tree_b_o = self.model.add_parameters((self.output_dim), 
													init=dy.ConstInitializer(0.))
		self.tree_b_u = self.model.add_parameters((self.output_dim), 
													init=dy.ConstInitializer(0.))

		# self.tree_W_i = self.model.add_parameters((self.output_dim, self.input_dim))
		# self.tree_W_f = self.model.add_parameters((self.output_dim, self.input_dim))
		# self.tree_W_o = self.model.add_parameters((self.output_dim, self.input_dim))
		# self.tree_W_u = self.model.add_parameters((self.output_dim, self.input_dim))

		# self.tree_U_i = self.model.add_parameters((self.output_dim, self.output_dim))
		# self.tree_U_f = self.model.add_parameters((self.output_dim, self.output_dim))
		# self.tree_U_o = self.model.add_parameters((self.output_dim, self.output_dim))
		# self.tree_U_u = self.model.add_parameters((self.output_dim, self.output_dim))

		# self.tree_b_i = self.model.add_parameters((self.output_dim))
		# self.tree_b_f = self.model.add_parameters((self.output_dim))
		# self.tree_b_o = self.model.add_parameters((self.output_dim))
		# self.tree_b_u = self.model.add_parameters((self.output_dim))


	def refresh(self):
		self.treeWi = self.tree_W_i.expr()
		self.treeWf = self.tree_W_f.expr()
		self.treeWo = self.tree_W_o.expr()
		self.treeWu = self.tree_W_u.expr()

		self.treeUi = self.tree_U_i.expr()
		self.treeUf = self.tree_U_f.expr()
		self.treeUo = self.tree_U_o.expr()
		self.treeUu = self.tree_U_u.expr()

		self.treebi = self.tree_b_i.expr()
		self.treebf = self.tree_b_f.expr()
		self.treebo = self.tree_b_o.expr()
		self.treebu = self.tree_b_u.expr()

		self._att.refresh()


	def _sum_context(self, context, x):
		context_emb, _ = self._att.attend(context, x)
		return context_emb


	def add_input(self, state_set, x_j, train):
		h_k, c_k = state_set.unfold()
		if len(h_k) == 0:
			h_k.append(dy.zeros((self.output_dim, )))
			c_k.append(dy.zeros((self.output_dim, )))
		if train:
			x_j = dy.dropout(x_j, self.dropout_input)
		h_j = self._sum_context(h_k, x_j)
		i_j = dy.logistic(self.treeWi * x_j + self.treeUi * h_j + self.treebi)
		f_jk = [dy.logistic(self.treeWf * x_j + self.treeUf * h + self.treebf) for h in h_k]			
		o_j = dy.logistic(self.treeWo * x_j + self.treeUo * h_j + self.treebo)
		u_j = dy.tanh(self.treeWu * x_j + self.treeUu * h_j + self.treebu)
		c_out = dy.cmult(i_j, u_j) + dy.esum([dy.cmult(f, c) for f, c in zip(f_jk, c_k)])
		if train:
			c_out = dy.dropout(c_out, self.dropout_hidden)
		h_out = dy.cmult(o_j, dy.tanh(c_out))

		return LSTMState(h_out, c_out)


	# def add_input(self, state_set, x_j, train):
	# 	h_k, c_k = state_set.unfold()
	# 	if len(h_k) == 0:
	# 		h_k.append(dy.zeros((self.output_dim, )))
	# 		c_k.append(dy.zeros((self.output_dim, )))
	# 	if train:
	# 		x_j = dy.dropout(x_j, self.dropout_input)
	# 	h_j = self._sum_context(h_k, x_j)
	# 	f_jk = [self.treeWf * x_j + self.treeUf * h + self.treebf for h in h_k]			
	# 	f_jk.append(self.treeWi * x_j + self.treeUi * h_j + self.treebi)
	# 	gate = dy.softmax(dy.concatenate_cols(f_jk), d=1)
	# 	c_k.append(dy.tanh(self.treeWu * x_j + self.treeUu * h_j + self.treebu))
	# 	c_out = dy.cmult(gate, dy.concatenate_cols(c_k))
	# 	c_out = dy.sum_dim(c_out, d=[1])
	# 	if train:
	# 		c_out = dy.dropout(c_out, self.dropout_hidden)
	# 	h_out = c_out
	# 	return LSTMState(h_out, c_out)


class RCNNBuilder(object):
	def __init__(self, model, options):
		self.model = model.add_subcollection('rcnn')
		self.bilstm_dim = options.compos_indim
		self.output_dim = options.compos_outdim
		self.wgt_hid_dim = options.compos_wgt_dim
		self.hidden_dim = options.compos_hid_dim
		mlp_size = self.wgt_hid_dim + self.hidden_dim
		# mlp_size = self.hidden_dim
		if options.compos_norm:
			head_W = orthonormal_initializer(mlp_size, self.bilstm_dim)
			dep_W = orthonormal_initializer(mlp_size, self.output_dim)
			self.mlp_head_W = self.model.parameters_from_numpy(head_W)
			self.mlp_dep_W = self.model.parameters_from_numpy(dep_W)
		else:
			self.mlp_head_W = self.model.add_parameters((mlp_size, self.bilstm_dim))
			self.mlp_dep_W = self.model.add_parameters((mlp_size, self.output_dim))
		self.mlp_head_b = self.model.add_parameters((mlp_size, ), 
													init=dy.ConstInitializer(0.))
		self.mlp_dep_b = self.model.add_parameters((mlp_size, ), 
													init=dy.ConstInitializer(0.))
		self.wgt_Wp = self.model.add_parameters((self.wgt_hid_dim, self.wgt_hid_dim + 1), 
												init=dy.ConstInitializer(0.))
		self.hid_Wp = self.model.add_parameters((self.output_dim * (self.hidden_dim + 1), 
												self.hidden_dim + 1), 
												init=dy.ConstInitializer(0.))


	def refresh(self):
		self.depW = self.mlp_dep_W.expr()
		self.depb = self.mlp_dep_b.expr()
		self.headW = self.mlp_head_W.expr()
		self.headb = self.mlp_head_b.expr()
		self.wgtW = self.wgt_Wp.expr()
		self.hidW = self.hid_Wp.expr()


	def add_input(self, state_set, x_j):
		h_k, _ = state_set.unfold()
		if len(h_k) == 0:
			h_k.append(dy.zeros((self.output_dim, )))
		hidden = dy.concatenate_cols(h_k)
		hidden_size = len(h_k)
		dep = leaky_relu(dy.affine_transform([self.depb, self.depW, hidden]))
		head = leaky_relu(dy.affine_transform([self.headb, self.headW, x_j]))
		dep_wgt, dep_hid = dep[:self.wgt_hid_dim], dep[self.wgt_hid_dim:]
		head_wgt, head_hid = head[:self.wgt_hid_dim], head[self.wgt_hid_dim:]
		# dep_hid, head_hid = dep, head
		weights = bilinear_transform(dep_wgt, self.wgtW, head_wgt, self.wgt_hid_dim, 
									hidden_size, 1, 1, True, False)
		weights = dy.reshape(weights, (hidden_size, 1))
		context = bilinear_transform(dep_hid, self.hidW, head_hid, self.hidden_dim, 
									hidden_size, 1, self.output_dim, True, True)
		context = dy.reshape(context, (hidden_size, self.output_dim))
		hidden_emb = dy.transpose(dy.transpose(weights) * context)

		# hidden_emb = dy.max_dim(context, 0)
		return LSTMState(hidden_emb, None)


def get_compositor(model, options):
	if options.tree_encoder == 'childsum':
		return TreeLSTMBuilder(model, options)

	if options.tree_encoder == 'rcnn':
		return RCNNBuilder(model, options)

