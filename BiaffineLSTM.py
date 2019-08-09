import dynet as dy
import numpy as np


class LSTMState(object):
	def __init__(self, builder, state_idx=-1, prev_state=None, out=None, hidden=None, memory=None):
		self.builder = builder
		self._hidden = hidden
		self._memory = memory
		self.state_idx = state_idx
		self._prev = prev_state
		self._out = out


	def add_input(self, x):
		return self.builder.add_input(self, x)


	def transduce(self, xs):
		cur = self
		res = []
		for x in xs:
			cur = cur.add_input(x)
			res.append(cur._out)
		return res


	def output(self):
		return self._out


	@property
	def hidden_state(self):
		return self._hidden


	@property
	def memory_cell(self):
		return self._memory


class BiaffineLSTMBuilder(object):
	def __init__(self, layers, input_dim, hidden_dim, model):
		self.model = model
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.W_i = self.model.add_parameters((hidden_dim * (hidden_dim + 1), (input_dim + 1)))
		self.W_f = self.model.add_parameters((hidden_dim * (hidden_dim + 1), (input_dim + 1)))
		self.W_o = self.model.add_parameters((hidden_dim * (hidden_dim + 1), (input_dim + 1)))
		self.W_u = self.model.add_parameters((hidden_dim * (hidden_dim + 1), (input_dim + 1)))


	def set_dropout(self, dropout):
		pass


	def disable_dropout(self):
		pass


	def initial_state(self):
		_init_h = dy.zeros((self.hidden_dim, ))
		_init_m = dy.zeros((self.hidden_dim, ))
		self.Wi = self.W_i.expr()
		self.Wf = self.W_f.expr()
		self.Wo = self.W_o.expr()
		self.Wu = self.W_u.expr()
		_init_s = LSTMState(self, -1, hidden=_init_h, memory=_init_m)
		return _init_s


	def _biaffine(self, x, W, y):
		x = dy.concatenate([x, dy.inputTensor(np.ones((1, ), dtype=np.float32))])
		y = dy.concatenate([y, dy.inputTensor(np.ones((1, ), dtype=np.float32))])
		nx, ny = self.input_dim + 1, self.input_dim + 1
		lin = dy.reshape(W * x, (ny, self.hidden_dim))
		blin = dy.transpose(dy.transpose(y) * lin)
		return blin


	def add_input(self, cur, x):
		h = cur.hidden_state
		c = cur.memory_cell
		i = dy.logistic(self._biaffine(x, self.Wi, h))
		f = dy.logistic(self._biaffine(x, self.Wf, h))			
		o = dy.logistic(self._biaffine(x, self.Wo, h))
		u = dy.tanh(self._biaffine(x, self.Wu, h))
		c_out = dy.cmult(i, u) + dy.cmult(f, c)
		h_out = dy.cmult(o, dy.tanh(c_out))
		_cur = LSTMState(self, cur.state_idx+1, prev_state=cur, out=h_out, hidden=h_out, memory=c_out)
		return _cur

