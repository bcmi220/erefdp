from Corpus import *
import random, helpers, TreeEncoder
import numpy as np
import dynet as dy


def bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs = 1, bias_x = False, bias_y = False):
	# x,y: (input_size x seq_len) x batch_size
	if bias_x:
		x = dy.concatenate([x, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
	if bias_y:
		y = dy.concatenate([y, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
	
	nx, ny = input_size + bias_x, input_size + bias_y
	# W: (num_outputs x ny) x nx
	lin = W * x
	if num_outputs > 1:
		lin = dy.reshape(lin, (ny, num_outputs*seq_len), batch_size = batch_size)
	blin = dy.transpose(y) * lin
	if num_outputs > 1:
		blin = dy.reshape(blin, (seq_len, num_outputs, seq_len), batch_size = batch_size)
	# seq_len_y x seq_len_x if output_size == 1
	# seq_len_y x num_outputs x seq_len_x else
	return blin

def orthonormal_initializer(output_size, input_size):
	"""
	adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
	"""
	print (output_size, input_size)
	I = np.eye(output_size)
	lr = .1
	eps = .05/(output_size + input_size)
	success = False
	tries = 0
	while not success and tries < 10:
		Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
		for i in xrange(100):
			QTQmI = Q.T.dot(Q) - I
			loss = np.sum(QTQmI**2 / 2)
			Q2 = Q**2
			Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
			if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
				tries += 1
				lr /= 2
				break
		success = True
	if success:
		print('Orthogonal pretrainer loss: %.2e' % loss)
	else:
		print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
		Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
	return np.transpose(Q.astype(np.float32))

def leaky_relu(x):
	return dy.bmax(.1*x, x)

class EasyFirst(object):
	def __init__(self, vocab, options):
		random.seed(1)
		self.model = dy.ParameterCollection()
		self.trainer = helpers.get_trainer(options, self.model)
		self.get_violation = helpers.update_method(options)
		
		word_count = vocab.word_freq
		word_vocab = vocab.wordlookup_tbl
		pos_vocab = vocab.poslookup_tbl
		rel_vocab = vocab.rellookup_tbl
		self.rels = rel_vocab

		self._enc = helpers.get_encoder(self.model, options, word_count, 
										word_vocab, pos_vocab)

		self._tree_enc = TreeEncoder.get_tree_encoder(self.model, 
												options, rel_vocab)


		self.mlp_rel_size = options.mlp_rel_size
		self.hidden_dim = options.compos_outdim
		W = orthonormal_initializer(self.mlp_rel_size, self.hidden_dim)
		self.mlp_dep_W = self.model.parameters_from_numpy(W)
		self.mlp_head_W = self.model.parameters_from_numpy(W)
		self.mlp_dep_b = self.model.add_parameters((self.mlp_rel_size,), init = dy.ConstInitializer(0.))
		self.mlp_head_b = self.model.add_parameters((self.mlp_rel_size,), init = dy.ConstInitializer(0.))
		
		# self.dropout_mlp = options.dropout_mlp
		self.rel_W = self.model.add_parameters((len(rel_vocab)*(self.mlp_rel_size +1) , self.mlp_rel_size + 1), init = dy.ConstInitializer(0.))

		self._train_flag = True
		self.oracle = options.oracle
		self.exploration_rate = options.exploration_rate

	def get_score(self, tree_encoder):

		seq_len = len(tree_encoder.all_states)

		encode_repr = dy.concatenate_cols([item.output for item in tree_encoder.all_states]) #[dy.inputTensor(np.zeros(self.hidden_dim, 1), batch=False)] + 

		W_dep, b_dep = dy.parameter(self.mlp_dep_W), dy.parameter(self.mlp_dep_b)
		W_head, b_head = dy.parameter(self.mlp_head_W), dy.parameter(self.mlp_head_b)
		dep_rel, head_rel = leaky_relu(dy.affine_transform([b_dep, W_dep, encode_repr])),leaky_relu(dy.affine_transform([b_head, W_head, encode_repr]))
		# if self._train_flag:
		# 	dep_rel, head_rel= dy.dropout_dim(dep_rel, 1, self.dropout_mlp), dy.dropout_dim(head_rel, 1, self.dropout_mlp)

		W_rel = dy.parameter(self.rel_W)

		rel_logits = bilinear(dep_rel, W_rel, head_rel, self.mlp_rel_size, seq_len, 1, num_outputs = len(self.rels), bias_x = True, bias_y = True)
		# (#head x rel_size x #dep) x batch_size

		rel_logits = dy.transpose(rel_logits, dims=[0, 2, 1])

		return rel_logits, rel_logits.value()


	def encode(self, sentence):
		self._enc.init_state(self._train_flag)
		return self._enc.encode(sentence)


	def refresh(self):
		dy.renew_cg()
		# self._scr.refresh()
		self._tree_enc.refresh()


	def update(self):
		self.trainer.update()


	def parse_loss(self, sentence):

		sent_len = len(sentence)

		encodings = self.encode(sentence)

		tree_encoder = self._tree_enc

		tree_encoder.input_encodings(encodings, sent_len, sentence, True)

		# 
		expr_matrix, scr_matrix = self.get_score(tree_encoder)

		losses, violations = [], []	

		while tree_encoder.pending_length > 1:

			valid_head, valid_dep, valid_ihead,  valid_idep = None, None, None, None

			valid_expr, valid_score = None, float('-inf')

			wrong_head, wrong_dep, wrong_ihead, wrong_idep = None, None, None, None

			wrong_expr, wrong_score = None, float('-inf')

			pending_ids = set(tree_encoder.pending_ids)

			pending_len = tree_encoder.pending_length

			pending_words = tree_encoder.pending_words

			unassigned = tree_encoder.unassigned_words

			
			# for every pending node
			for idx in xrange(pending_len):#- 1

				# for every operation: AttachLeft AttachRight
				# we ignore the projective constraint and search in the whole space
				#for op in xrange(2):
				for jdx in xrange(pending_len): # - 1

					#hi, di = idx + (1 - op), idx + op
					hi = idx
					di = jdx
					head_id = pending_words[hi].w_id
					dep_id = pending_words[di].w_id

					# for every relation
					for rel, irel in self.rels.iteritems():

						oracle_cost = unassigned[dep_id] + (0 
							if pending_words[di].parent_id not in pending_ids \
							or pending_words[di].parent_id == head_id else 1) 

						if oracle_cost == 0 and (
							pending_words[di].parent_id != head_id \
								or pending_words[di].relation == rel):
														
							if valid_score < scr_matrix[head_id][dep_id][irel]:
								valid_score = scr_matrix[head_id][dep_id][irel]
								valid_expr = expr_matrix[head_id][dep_id][irel]
								valid_head, valid_dep = hi, di
								valid_ihead, valid_idep = head_id, dep_id
								valid_rel, valid_irel = rel, irel

						elif wrong_score < scr_matrix[head_id][dep_id][irel]:
							wrong_score = scr_matrix[head_id][dep_id][irel]
							wrong_expr = expr_matrix[head_id][dep_id][irel]
							wrong_head, wrong_dep = hi, di
							wrong_ihead, wrong_idep = head_id, dep_id
							wrong_rel, wrong_irel = rel, irel

			if valid_expr and valid_score < wrong_score + 1.0:
				violation = wrong_score + 1.0 - valid_score
				loss = wrong_expr + 1.0 - valid_expr
				violations.append(violation)
				losses.append(loss)
				
			if self.oracle or valid_score - wrong_score > 1.0 or \
				(valid_score > wrong_score and 
					random.random() > self.exploration_rate):
				selected_head = valid_head
				selected_dep = valid_dep
				selected_ihead = valid_ihead
				selected_idep = valid_idep
				selected_rel = valid_rel
				selected_irel = valid_irel
			else:
				selected_head = wrong_head
				selected_dep = wrong_dep
				selected_ihead = wrong_ihead
				selected_idep = wrong_idep
				selected_rel = wrong_rel
				selected_irel = wrong_irel
			
			tree_encoder.attach(selected_head, selected_dep, selected_ihead, selected_idep, 
								selected_irel, True)
			
			# 
			expr_matrix, scr_matrix = self.get_score(tree_encoder)

		ret_loss = self.get_violation(violations, losses) \
					if len(losses) > 0 else None
		return ret_loss


	def parse(self, sentence):
		sent_len = len(sentence)
		encodings = self.encode(sentence)
		tree_encoder = self._tree_enc
		tree_encoder.input_encodings(encodings, sent_len, sentence, False)
		
		# 
		expr_matrix, scr_matrix = self.get_score(tree_encoder)

		while tree_encoder.pending_length > 1:
			best_head, best_dep, best_ihead, best_idep, best_score = None, None, None, None, float('-inf')
			pending_len = tree_encoder.pending_length
			pending_words = tree_encoder.pending_words

			for idx in xrange(pending_len - 1):
				for op in xrange(2):
					hi, di = idx + (1 - op), idx + op
					if di == 0:
						continue
					head_id = pending_words[hi].w_id
					dep_id = pending_words[di].w_id
					for rel, irel in self.rels.iteritems():
						if best_score < scr_matrix[head_id][dep_id][irel]:
							
							best_score = scr_matrix[head_id][dep_id][irel]
							best_head, best_dep = hi, di
							best_ihead, best_idep = head_id, dep_id
							best_rel, best_irel = rel, irel

			pending_words[best_dep].pred_pid = pending_words[best_head].w_id
			pending_words[best_dep].pred_rel = best_rel

			tree_encoder.attach(best_head, best_dep, best_ihead, best_idep, best_irel, False)
			
			# 
			expr_matrix, scr_matrix = self.get_score(tree_encoder)


	def set_train_flag(self):
		self._train_flag = True


	def set_test_flag(self):
		self._train_flag = False


	def save(self, filename):
		self.model.save(filename)


	def load(self, filename):
		self.model.populate(filename)

