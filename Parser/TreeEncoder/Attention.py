import dynet as dy


class Attention(object):
	def __init__(self, model, options):
		pass


	def refresh(self):
		pass


	def attend(self, context, x):
		pass


class MaxAttention(Attention):
	def attend(self, context, x):
		context_cols = dy.concatenate_cols(context)
		context_emb = dy.max_dim(context_cols, 1)
		return context_emb, None


class SumAttention(Attention):
	def attend(self, context, x):
		context_emb = dy.esum(context)
		weights = dy.softmax(dy.ones((len(context), )))
		return context_emb, weights


class DotAttention(Attention):
	def attend(self, context, x):
		context_cols = dy.concatenate_cols(context)
		weights = dy.softmax(dy.transpose(context_cols) * x)
		context_emb = context_cols * weights
		return context_emb, weights


class MLPAttention(Attention):
	def __init__(self, model, options):
		self.model = model.add_subcollection('attention')
		self.input_dims = options.compos_indim
		self.hidden_dims = options.compos_outdim
		self.atten_dims = options.atten_dim

		self.Va_p = self.model.add_parameters((self.atten_dims, ))
		self.Wia_p = self.model.add_parameters((self.atten_dims, self.input_dims))
		self.Wha_p = self.model.add_parameters((self.atten_dims, self.hidden_dims))


	def refresh(self):
		self.Va = self.Va_p.expr()
		self.Wia = self.Wia_p.expr()
		self.Wha = self.Wha_p.expr()


	def attend(self, context, x):
		context_cols = dy.concatenate_cols(context)
		hidden = dy.tanh(dy.colwise_add(self.Wha * context_cols, self.Wia * x))
		weights = dy.softmax(dy.transpose(hidden) * self.Va)
		context_emb = context_cols * weights
		return context_emb, weights


class BilinearAttention(Attention):
	def __init__(self, model, options):
		self.model = model.add_subcollection('attention')
		self.input_dims = options.compos_indim
		self.hidden_dims = options.compos_outdim
		self.W_p = self.model.add_parameters((self.hidden_dims, self.input_dims))


	def refresh(self):
		self.W = self.W_p.expr()


	def attend(self, context, x):
		context_cols = dy.concatenate_cols(context)
		weights = dy.softmax(dy.transpose(context_cols) * self.W * x)
		context_emb = context_cols * weights
		return context_emb, weights


def get_attention(model, options):
	if options.attention == 'max':
		return MaxAttention(model, options)

	if options.attention == 'sum':
		return SumAttention(model, options)

	if options.attention == 'dot':
		return DotAttention(model, options)

	if options.attention == 'mlp':
		return MLPAttention(model, options)

	if options.attention == 'bilinear':
		return BilinearAttention(model, options)

