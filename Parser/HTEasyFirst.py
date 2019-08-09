from Corpus import *
import random, helpers, TreeEncoder
import numpy as np
import dynet as dy


class ScoreMatrix(object):
	def __init__(self, pending, scr_method, rel_size):
		length = pending.pending_length
		self.scr_matrix = [[None for _ in xrange(length)] for _ in xrange(length)]
		self.expr_matrix = [[None for _ in xrange(length)] for _ in xrange(length)]
		self.__states = pending.pending_states
		self.__ids = pending.pending_ids
		self.__encs = pending.pending_history
		self.__score = scr_method
		expr_matrix, scr_matrix = self.expr_matrix, self.scr_matrix
		for i in xrange(length - 1):
			cur_enc = dy.concatenate(filter(None, [self.__encs[i][0].output(), 
							self.__encs[i][1].output(), self.__states[i].output]))
			next_enc = dy.concatenate(filter(None, [self.__encs[i + 1][0].output(), 
							self.__encs[i + 1][1].output(), self.__states[i + 1].output]))
			expr_matrix[i][i + 1] = self.__score(cur_enc, next_enc)
			expr_matrix[i + 1][i] = self.__score(next_enc, cur_enc)
			scr_matrix[i][i + 1] = expr_matrix[i][i + 1].value()
			scr_matrix[i + 1][i] = expr_matrix[i + 1][i].value()


	def update_score(self, i):
		length = len(self.__ids)
		cur_id = self.__ids[i]
		cur_enc = dy.concatenate(filter(None, [self.__encs[i][0].output(), 
						self.__encs[i][1].output(), self.__states[i].output]))
		expr_matrix, scr_matrix = self.expr_matrix, self.scr_matrix
		if i > 0:
			pre_id = self.__ids[i - 1]
			pre_enc = dy.concatenate(filter(None, [self.__encs[i - 1][0].output(), 
						self.__encs[i - 1][1].output(), self.__states[i - 1].output]))
			expr_matrix[cur_id][pre_id] = self.__score(cur_enc, pre_enc)
			expr_matrix[pre_id][cur_id] = self.__score(pre_enc, cur_enc)
			scr_matrix[cur_id][pre_id] = expr_matrix[cur_id][pre_id].value()
			scr_matrix[pre_id][cur_id] = expr_matrix[pre_id][cur_id].value()
			
		if i < (length - 1):
			next_id = self.__ids[i + 1]
			next_enc = dy.concatenate(filter(None, [self.__encs[i + 1][0].output(), 
						self.__encs[i + 1][1].output(), self.__states[i + 1].output]))
			expr_matrix[cur_id][next_id] = self.__score(cur_enc, next_enc)
			expr_matrix[next_id][cur_id] = self.__score(next_enc, cur_enc)
			scr_matrix[cur_id][next_id] = expr_matrix[cur_id][next_id].value()
			scr_matrix[next_id][cur_id] = expr_matrix[next_id][cur_id].value()


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
		self.__enc = helpers.get_encoder(self.model, options, word_count, word_vocab, pos_vocab)
		self.__scr = helpers.get_scorer(self.model, options, rel_vocab)
		self.__tree_enc = TreeEncoder.get_tree_encoder(self.model, options, rel_vocab)		
		self.__train_flag = True
		self.oracle = options.oracle
		self.exploration_rate = options.exploration_rate


	def encode(self, sentence):
		self.__enc.init_state(self.__train_flag)
		return self.__enc.encode(sentence, self.__train_flag)


	def refresh(self):
		dy.renew_cg()
		self.__scr.refresh()
		self.__tree_enc.refresh()


	def update(self):
		self.trainer.update()


	def parse_loss(self, sentence):
		sent_len = len(sentence)
		encodings = self.encode(sentence)
		tree_encoder = self.__tree_enc
		tree_encoder.input_encodings(encodings, sent_len, sentence)
		matrix = ScoreMatrix(tree_encoder, self.__scr.score, len(self.rels))
		scr_matrix, expr_matrix = matrix.scr_matrix, matrix.expr_matrix

		losses, violations = [], []		
		while tree_encoder.pending_length > 1:
			valid_head, valid_dep = None, None
			valid_expr, valid_score = None, float('-inf')
			wrong_head, wrong_dep = None, None
			wrong_expr, wrong_score = None, float('-inf')

			pending_ids = set(tree_encoder.pending_ids)
			pending_len = tree_encoder.pending_length
			pending_words = tree_encoder.pending_words
			unassigned = tree_encoder.unassigned_words

			for idx in xrange(pending_len - 1):
				for op in xrange(2):
					hi, di = idx + (1 - op), idx + op
					head_id = pending_words[hi].w_id
					dep_id = pending_words[di].w_id
					for irel, rel in enumerate(self.rels):
						oracle_cost = unassigned[dep_id] + \
										(0 if pending_words[di].parent_id not in pending_ids \
										or pending_words[di].parent_id == head_id else 1)

						if oracle_cost == 0 and (pending_words[di].parent_id != head_id \
							or pending_words[di].relation == rel):							
							if valid_score < scr_matrix[head_id][dep_id][irel]:
								valid_score = scr_matrix[head_id][dep_id][irel]
								valid_expr = expr_matrix[head_id][dep_id][irel]
								valid_head, valid_dep = hi, di
								valid_rel, valid_irel = rel, irel

						elif wrong_score < scr_matrix[head_id][dep_id][irel]:
							wrong_score = scr_matrix[head_id][dep_id][irel]
							wrong_expr = expr_matrix[head_id][dep_id][irel]
							wrong_head, wrong_dep = hi, di
							wrong_rel, wrong_irel = rel, irel

			if valid_expr and valid_score < wrong_score + 1.0:
				violation = wrong_score + 1.0 - valid_score
				loss = wrong_expr + 1.0 - valid_expr
				violations.append(violation)
				losses.append(loss)
				
			if self.oracle or valid_score - wrong_score > 1.0 or \
				(valid_score > wrong_score and random.random() > self.exploration_rate):
				selected_head = valid_head
				selected_dep = valid_dep
				selected_rel = valid_rel
				selected_irel = valid_irel
			else:
				selected_head = wrong_head
				selected_dep = wrong_dep
				selected_rel = wrong_rel
				selected_irel = wrong_irel
			tree_encoder.attach(selected_head, selected_dep, selected_irel, True)
			head_idx = selected_head if selected_head < selected_dep else selected_head - 1
			matrix.update_score(head_idx)
			
		ret_loss = self.get_violation(violations, losses) if len(losses) > 0 else None
		return ret_loss


	def parse(self, sentence):
		sent_len = len(sentence)
		encodings = self.encode(sentence)
		tree_encoder = self.__tree_enc
		tree_encoder.input_encodings(encodings, sent_len, sentence)
		matrix = ScoreMatrix(tree_encoder, self.__scr.score, len(self.rels))
		scr_matrix, expr_matrix = matrix.scr_matrix, matrix.expr_matrix
				
		while tree_encoder.pending_length > 1:
			best_head, best_dep, best_score = None, None, float('-inf')
			pending_len = tree_encoder.pending_length
			pending_words = tree_encoder.pending_words

			for idx in xrange(pending_len - 1):
				for op in xrange(2):
					hi, di = idx + (1 - op), idx + op
					if di == 0:
						continue
					head_id = pending_words[hi].w_id
					dep_id = pending_words[di].w_id
					for irel, rel in enumerate(self.rels):
						if best_score < scr_matrix[head_id][dep_id][irel]:
							best_score = scr_matrix[head_id][dep_id][irel]
							best_head, best_dep = hi, di
							best_rel, best_irel = rel, irel

			pending_words[best_dep].pred_pid = pending_words[best_head].w_id
			pending_words[best_dep].pred_rel = best_rel
			tree_encoder.attach(best_head, best_dep, best_irel, False)
			head_idx = best_head if best_head < best_dep else best_head - 1
			matrix.update_score(head_idx)


	def set_train_flag(self):
		self.__train_flag = True


	def set_test_flag(self):
		self.__train_flag = False


	def save(self, filename):
		self.model.save(filename)


	def load(self, filename):
		self.model.populate(filename)

