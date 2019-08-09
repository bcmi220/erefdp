import dynet as dy
from utils import *
import random


class BaseEncoder(object):
	def __init__(self, model, options, word_count, word_vocab, pos_vocab):
		self.model = model.add_subcollection('base')
		self.word_dims = options.word_dim
		self.pos_dims = options.pos_dim
		self.word_count = word_count
		self.word_vocab = word_vocab
		self.pos_vocab = pos_vocab
		self.word_dropout_rate = options.word_dropout_rate
		self.pos_dropout_rate = options.pos_dropout_rate
		self.WORD_LOOKUP = self.model.add_lookup_parameters((len(word_vocab), 
															self.word_dims))
		self.POS_LOOKUP = self.model.add_lookup_parameters((len(pos_vocab), 
															self.pos_dims))
		
		extrn_emb = None
		if options.extrn_file is not None:
			with open(options.extrn_file, 'r') as efp:
				efp.readline()
				extrn_emb = {line.strip().split(' ')[0]: [float(emb) 
								for emb in line.strip().split(' ')[1:]] 
									for line in efp}
			for word, idx in word_vocab.iteritems():
				if word in extrn_emb:
					self.WORD_LOOKUP.init_row(idx, extrn_emb[word])


	def init_state(self, train_flag):
		self._train_flag = train_flag
		
		
	def encode(self, sentence):
		freqs = [float(self.word_count.get(root.norm, 0)) for root in sentence]
		wembs = [dy.lookup(self.WORD_LOOKUP, self.word_vocab.get(root.norm, 0) 
					if not self._train_flag or (random.random() < \
						(c / (self.word_dropout_rate + c))) 
					else 0) for (root, c) in zip(sentence, freqs)]
		pembs = [dy.lookup(self.POS_LOOKUP, self.pos_vocab[root.pos]) 
					if (not self._train_flag or (random.random() > self.pos_dropout_rate)) 
					else wembs[root.w_id] for root in sentence]
		encode_states = [dy.concatenate([wi, pi]) for wi, pi in zip(wembs, pembs)]
		return dy.concatenate_cols(encode_states)


class MLPEncoder(object):
	def __init__(self, model, options):
		self.model = model.add_subcollection('mlpenc')
		self.activation = get_activation(options)
		self.input_dims = options.word_dim + options.pos_dim
		self.output_dim = options.lstm_input_dim
		self.dropout_rate = options.dropout_rate
		self.W_p = self.model.add_parameters((self.output_dim, self.input_dims))
		self.b_p = self.model.add_parameters((self.output_dim), 
											init=dy.ConstInitializer(0))


	def init_state(self, train_flag):
		self._train_flag = train_flag
		self.W = self.W_p.expr()
		self.b = self.b_p.expr()


	def encode(self, encodings):
		encode_states = self.activation(dy.affine_transform([self.b, self.W, encodings]))
		return dy.dropout(encode_states, self.dropout_rate)


# class BiLSTMEncoder(object):
# 	def __init__(self, model, options):
# 		self.model = model.add_subcollection('bilstm')
# 		self.layers = options.layers
# 		self.input_dims = options.lstm_input_dim
# 		self.hidden_dims = options.lstm_output_dim
# 		self.lstmbuilders = [dy.VanillaLSTMBuilder(self.layers, self.input_dims, \
# 													self.hidden_dims, self.model), 
# 							dy.VanillaLSTMBuilder(self.layers, self.input_dims, \
# 													self.hidden_dims, self.model)]
# 		self.dropout_input = options.dropout_lstm_input
# 		self.dropout_hidden = options.dropout_lstm_hidden
# 		self.output_dim = self.hidden_dims * 2


# 	def init_state(self, train_flag):
# 		if train_flag:
# 			self.lstmbuilders[0].set_dropouts(self.dropout_input, self.dropout_hidden)
# 			self.lstmbuilders[1].set_dropouts(self.dropout_input, self.dropout_hidden)
# 		else:
# 			self.lstmbuilders[0].disable_dropout()
# 			self.lstmbuilders[1].disable_dropout()
# 		self.init_states = [self.lstmbuilders[0].initial_state(), 
# 							self.lstmbuilders[1].initial_state()]


# 	def encode(self, encodings, sent_len):
# 		inputx = [dy.pick(encodings, i, dim=1) for i in xrange(sent_len)]
# 		encode_states = [dy.concatenate_cols(self.init_states[0].\
# 											transduce(inputx)), \
# 						dy.concatenate_cols(self.init_states[1].\
# 											transduce(inputx[::-1])[::-1])]
# 		return dy.concatenate(encode_states)


class BiLSTMEncoder(object):
	def __init__(self, model, options):
		self.model = model.add_subcollection('bilstm')
		self.layers = options.layers
		self.input_dims = options.lstm_input_dim
		self.hidden_dims = options.lstm_output_dim
		self.lstmbuilders = []
		f = orthonormal_VanillaLSTMBuilder(1, self.input_dims, 
											self.hidden_dims, self.model)
		b = orthonormal_VanillaLSTMBuilder(1, self.input_dims, 
											self.hidden_dims, self.model)
		self.lstmbuilders.append((f,b))
		for _ in xrange(self.layers - 1):
			f = orthonormal_VanillaLSTMBuilder(1, 2 * self.hidden_dims, 
												self.hidden_dims, self.model)
			b = orthonormal_VanillaLSTMBuilder(1, 2 * self.hidden_dims, 
												self.hidden_dims, self.model)
			self.lstmbuilders.append((f,b))

		self.dropout_input = options.dropout_lstm_input
		self.dropout_hidden = options.dropout_lstm_hidden
		self.output_dim = self.hidden_dims * 2


	def init_state(self, train_flag):
		dropout_x = self.dropout_input if train_flag else 0.
		dropout_h = self.dropout_hidden if train_flag else 0.
		for fb, bb in self.lstmbuilders:
			fb.set_dropouts(dropout_x, dropout_h)
			bb.set_dropouts(dropout_x, dropout_h)
	
		
	def encode(self, encodings, sent_len):
		inputx = [dy.pick(encodings, i, dim=1) for i in xrange(sent_len)]
		for fb, bb in self.lstmbuilders:
			f, b = fb.initial_state(), bb.initial_state()
			fs, bs = f.transduce(inputx), b.transduce(reversed(inputx))
			inputx = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
		return dy.concatenate_cols(inputx)


class BiLSTMWrapper(object):
	def __init__(self, model, options, word_count, word_vocab, pos_vocab):
		self._base = BaseEncoder(model, options, word_count, \
										word_vocab, pos_vocab)
		self._bilstm = BiLSTMEncoder(model, options)
		self.dropout_rate = options.dropout_rate


	def init_state(self, train_flag):
		self._train_flag = train_flag
		self._base.init_state(train_flag)
		self._bilstm.init_state(train_flag)


	def encode(self, sentence):
		if self._train_flag:
			return dy.dropout_dim(
						self._bilstm.encode(\
							self._base.encode(sentence),\
							len(sentence)), 
						1, self.dropout_rate)
		else:
			return self._bilstm.encode(\
						self._base.encode(sentence),\
						len(sentence))


class BOTSPAWrapper(object):
	def __init__(self, model, options, word_count, word_vocab, pos_vocab):
		self._base = BaseEncoder(model, options, word_count, \
										word_vocab, pos_vocab)
		self._bilstm = BiLSTMEncoder(model, options)
		self._botspa = BOTSPAEncoder(model, options)


	def init_state(self, train_flag):
		self._base.init_state(train_flag)
		self._bilstm.init_state(train_flag)
		self._botspa.init_state(train_flag)


	def encode(self, sentence):
		return self._botspa.encode(\
					self._bilstm.encode(\
						self._base.encode(sentence), \
						len(sentence)), \
					len(sentence))

