from Scorer import *
from Encoder import *
import dynet as dy
import numpy as np


def get_trainer(options, model):
	if options.trainer == 'adam':
		trainer = dy.AdamTrainer(model, alpha=options.learning_rate, \
					beta_1=options.beta_1, beta_2=options.beta_2)
	else:
		print('Trainer name invalid or not provided, using SGD')
		trainer = dy.SimpleSGDTrainer(model, e0=options.learning_rate, \
					edecay=options.learning_rate_decay)
	trainer.set_clip_threshold(options.gradient_clip)
	return trainer


def get_scorer(model, options, rel_vocab):
	if options.scorer == 'mlp':
		scorer = MLPScorer(model, options, rel_vocab)
	elif options.scorer == 'biaffine':
		scorer = BiaffineScorer(model, options, rel_vocab)
	else:
		print('Unknown type of scorer, using mlp scorer')
		scorer = MLPScorer(model, options, rel_vocab)
	return scorer


def get_encoder(model, options, word_count, word_vocab, pos_vocab):
	if options.encoder == 'empty':
		encoder = BaseEncoder(model, options, word_count, word_vocab, pos_vocab)
	elif options.encoder == 'bilstm':
		encoder = BiLSTMWrapper(model, options, word_count, word_vocab, pos_vocab)
	elif options.encoder == 'botspa':
		encoder = BOTSPAWrapper(model, options, word_count, word_vocab, pos_vocab)
	return encoder


def update_method(opt):
	if opt.violation == 'full':
		return lambda v, l: dy.esum(l) / float(len(l))
	elif opt.violation == 'max':
		return lambda v, l: l[np.array(v).argmax()]
	elif opt.violation == 'greedy':
		return lambda v, l: l[0]

