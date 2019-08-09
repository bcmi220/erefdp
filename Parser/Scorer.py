import dynet as dy
from utils import *


__all__ = ['MLPScorer', 'BiaffineScorer']


class MLPScorer(object):
	def __init__(self, model, options, rel_vocab):
		self.model = model.add_subcollection('scorer')
		self.activation = get_activation(options)
		self.dropout_rate = options.dropout_rate
		self.hid_dims = options.hid_dim
		self.hid2_dims = options.hid2_dim
		self.in_dims = options.scorer_indim
		self.u_weight = options.unlabel_weight
		
		self.hid_Wp = self.model.add_parameters((self.hid_dims, self.in_dims))
		self.hid_bp = self.model.add_parameters((self.hid_dims))

		self.lhid_Wp = self.model.add_parameters((self.hid_dims, self.in_dims))
		self.lhid_bp = self.model.add_parameters((self.hid_dims))

		if self.hid2_dims > 0:
			self.hid2_Wp = self.model.add_parameters((self.hid2_dims, self.hid_dims))
			self.hid2_bp = self.model.add_parameters((self.hid2_dims))
			self.lhid2_Wp = self.model.add_parameters((self.hid2_dims, self.hid_dims))
			self.lhid2_bp = self.model.add_parameters((self.hid2_dims))	
			self.out_Wp = self.model.add_parameters((1, self.hid2_dims))
			self.lout_Wp = self.model.add_parameters((len(rel_vocab), self.hid2_dims))
		else:
			self.out_Wp = self.model.add_parameters((1, self.hid_dims))
			self.lout_Wp = self.model.add_parameters((len(rel_vocab), self.hid_dims))

		self.out_bp = self.model.add_parameters((1, ), init=dy.ConstInitializer(0))
		self.lout_bp = self.model.add_parameters((len(rel_vocab), ))


	def refresh(self):
		self.hid_W = self.hid_Wp.expr()
		self.hid_b = self.hid_bp.expr()
		self.lhid_W = self.lhid_Wp.expr()
		self.lhid_b = self.lhid_bp.expr()

		if self.hid2_dims > 0:
			self.hid2_W = self.hid2_Wp.expr()
			self.hid2_b = self.hid2_bp.expr()
			self.lhid2_W = self.lhid2_Wp.expr()
			self.lhid2_b = self.lhid2_bp.expr()
		
		self.out_W = self.out_Wp.expr()
		self.out_b = self.out_bp.expr(False)
		self.lout_W = self.lout_Wp.expr()
		self.lout_b = self.lout_bp.expr()


	def score(self, head, dep, train=True):
		inputx = dy.concatenate([head, dep])
		if train:
			if self.hid2_dims > 0:
				hid_out = self.hid2_b + self.hid2_W * dy.dropout(
							self.activation(self.hid_b + self.hid_W * 
								dy.dropout(inputx, self.dropout_rate)), 
							self.dropout_rate)
				lhid_out = self.lhid2_b + self.lhid2_W * dy.dropout(
							self.activation(self.lhid_b + self.lhid_W * 
								dy.dropout(inputx, self.dropout_rate)), 
							self.dropout_rate)
			else:
				hid_out = self.hid_b + self.hid_W * dy.dropout(
										inputx, self.dropout_rate)
				lhid_out = self.lhid_b + self.lhid_W * dy.dropout(
										inputx, self.dropout_rate)
		else:
			if self.hid2_dims > 0:
				hid_out = self.hid2_b + self.hid2_W * self.activation(
							self.hid_b + self.hid_W * inputx)
				lhid_out = self.lhid2_b + self.lhid2_W * self.activation(
							self.lhid_b + self.lhid_W * inputx)
			else:
				hid_out = self.hid_b + self.hid_W * inputx
				lhid_out = self.lhid_b + self.lhid_W * inputx

		u_scr = self.out_b + self.out_W * self.activation(hid_out)
		l_scr = self.lout_b + self.lout_W * self.activation(lhid_out)
		return (self.u_weight * u_scr + (1 - self.u_weight) * l_scr)


class BiaffineScorer(object):
	def __init__(self, model, options, rel_vocab):
		self.model = model.add_subcollection('scorer')
		self.activation = get_activation(options)
		self.dropout_rate = options.dropout_rate
		self.arc_dims = options.scr_arc_dim
		self.rel_dims = options.scr_rel_dim
		self.in_dims = options.scorer_indim
		self.u_weight = options.unlabel_weight
		self.rel_size = len(rel_vocab)
		mlp_size = self.arc_dims + self.rel_dims
		# head_W = orthonormal_initializer(mlp_size, self.in_dims)
		# dep_W = orthonormal_initializer(mlp_size, self.in_dims)
		W = orthonormal_initializer(mlp_size, self.in_dims)
		self.mlp_head_W = self.model.parameters_from_numpy(W)
		self.mlp_dep_W = self.model.parameters_from_numpy(W)

		# self.mlp_head_W = self.model.add_parameters((mlp_size, self.in_dims))
		# self.mlp_dep_W = self.model.add_parameters((mlp_size, self.in_dims))
		self.mlp_head_b = self.model.add_parameters((mlp_size, ), 
													init=dy.ConstInitializer(0.))
		self.mlp_dep_b = self.model.add_parameters((mlp_size, ), 
													init=dy.ConstInitializer(0.))
		self.arc_Wp = self.model.add_parameters((self.arc_dims, self.arc_dims + 1))
		self.rel_Wp = self.model.add_parameters((self.rel_size * (self.rel_dims + 1), 
												self.rel_dims + 1))


	def refresh(self):
		self.depW = self.mlp_dep_W.expr()
		self.depb = self.mlp_dep_b.expr()
		self.headW = self.mlp_head_W.expr()
		self.headb = self.mlp_head_b.expr()
		self.arcW = self.arc_Wp.expr()
		self.relW = self.rel_Wp.expr()


	def score(self, head, dep, train=True):
		_dep = self.activation(dy.affine_transform([self.depb, self.depW, dep])) 
		_head = self.activation(dy.affine_transform([self.headb, self.headW, head]))
		if train:
			_dep = dy.dropout(_dep, self.dropout_rate)
			_head = dy.dropout(_head, self.dropout_rate)
		dep_arc, dep_rel = _dep[:self.arc_dims], _dep[self.arc_dims:]
		head_arc, head_rel = _head[:self.arc_dims], _head[self.arc_dims:]
		u_scr = self.bilinear_transform(dep_arc, self.arcW, head_arc, self.arc_dims, 
										1, True, False)
		l_scr = self.bilinear_transform(dep_rel, self.relW, head_rel, self.rel_dims, 
										self.rel_size, True, True)
		return (self.u_weight * u_scr + (1 - self.u_weight) * l_scr)


	def bilinear_transform(self, x, W, y, input_size, num_outputs=1, 
							bias_x=False, bias_y=False):
		if bias_x:
			x = dy.concatenate([x, dy.inputTensor(np.ones((1, ), dtype=np.float32))])
		if bias_y:
			y = dy.concatenate([y, dy.inputTensor(np.ones((1, ), dtype=np.float32))])		
		nx, ny = input_size + bias_x, input_size + bias_y
		lin = W * x
		if num_outputs > 1:
			lin = dy.reshape(lin, (ny, num_outputs))
		blin = dy.transpose(lin) * y
		return blin

