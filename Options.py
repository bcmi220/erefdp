import numpy as np
import argparse, yaml, sys

__all__ = ['print_config', 'get_options']

parser = argparse.ArgumentParser()
# parser.add_argument("--dynet-seed", type=int, default=0)
parser.add_argument('--dynet-mem', type=int, default=512)
parser.add_argument('--dynet-gpu', help='Use dynet with GPU', action='store_true')
parser.add_argument('--dynet-autobatch', type=int, default=0, 
					help='Use dynet autobatching')
parser.add_argument('--config_file', '-c', type=str, default=None)
parser.add_argument('--env', '-e', type=str, default='train', 
					help='Environment in the config file')
parser.add_argument('--trainer', type=str, default='adam', 
					help='Optimizer. Choose from "sgd, clr, momentum, adam, rmsprop"')
parser.add_argument('--dropout_rate', type=float, default=0.25, help='Dropout rate')
parser.add_argument('--word_dropout_rate', type=float, default=0.25, 
					help='Word dropout rate')
parser.add_argument('--pos_dropout_rate', type=float, default=0.25, 
					help='POS tag dropout rate')
parser.add_argument('--gradient_clip', type=float, default=1.0, 
					help='Gradient clipping. Negative value means no clipping')
parser.add_argument('--learning_rate_decay', type=float, default=0.0, 
					help='learning rate decay')
parser.add_argument('--beta_1', type=float, default=0.9, help='Beta_1 for Adam')
parser.add_argument('--beta_2', type=float, default=0.9, help='beta_2 for Adam')
parser.add_argument('--algorithm', type=str, default='globaleasyfirst', 
					help='Parsing algorithm')
parser.add_argument('--non_proj', action='store_true', default=True, 
					help='Train non-projective sentences')
parser.add_argument('--language', type=str, default='en', help='Corpus language')
parser.add_argument('--violation', type=str, default='max', help='Update method, \
					"full" for full violation, "max" for max violation, "greedy" \
					for greedy violation')
parser.add_argument('--attention', type=str, default='mlp', help='Attention method, \
					choose from "max, sum, dot, mlp, bilinear"')
parser.add_argument('--tree_encoder', type=str, default='empty', help='Partial tree \
					encoding method')

parser.add_argument('--train_file', type=str, help='File path of train set')
parser.add_argument('--dev_file', type=str, help='File path of development set')
parser.add_argument('--test_file', type=str, help='File path of test set')
parser.add_argument('--extrn_file', type=str, help='External embeddings')
parser.add_argument('--outdir', type=str, default='output', 
					help='Output path of the model, parameters and result')
parser.add_argument('--epochs', type=int, default=30, 
					help='Epochs of training, default value is 30')
parser.add_argument('--parse', action='store_true', default=False, 
					help='Parse the test set with the given model')
parser.add_argument('--load_dir', type=str, help='Path to find the saved model \
					and vocabulary file')
parser.add_argument('--model', type=str, default='parser.model', 
					help='File name of the learned model')
parser.add_argument('--vocab', type=str, default='vocab.pkl', 
					help='File name of the vocabulary')
parser.add_argument('--run_dev_every', type=int, default=1, help='Run testing on \
					the development every [run_dev_every] epoch')
parser.add_argument('--run_dev_after', type=int, default=20, help='Run testing on \
					the development after [run_dev_every] epoch')


parser.add_argument('--unlabel_weight', type=float, default=0.5, 
					help='Smoothing between label and unlabel score')
parser.add_argument('--scorer_indim', type=int, default=400, 
					help='Dimension of scorer input')
parser.add_argument('--scr_arc_dim', type=int, default=500)
parser.add_argument('--scr_rel_dim', type=int, default=100)
parser.add_argument('--encoder', type=str, default='bilstm')
parser.add_argument('--scorer', type=str, default='mlp')
parser.add_argument('--lstm_input_dim', type=int, default=200, 
					help='Dimension of lstm cell')
parser.add_argument('--lstm_output_dim', type=int, default=200, 
					help='Dimension of lstm cell')
parser.add_argument('--dropout_lstm_input', type=float, default=0.25)
parser.add_argument('--dropout_lstm_hidden', type=float, default=0.25)

parser.add_argument('--layers', type=int, default=2, 
					help='Number of layers of lstm')
parser.add_argument('--hid_dim', type=int, default=100, 
					help='Size of the hidden layer in mlp scorer')
parser.add_argument('--hid2_dim', type=int, default=0, help='Size of the second hidden \
					layer in mlp scorer, set to 0 to disable the second hidden layer')
parser.add_argument('--word_dim', type=int, default=100, 
					help='Dimension of word embeddings')
parser.add_argument('--pos_dim', type=int, default=100, 
					help='Dimension of POS tag embeddings')

parser.add_argument('--compos_indim', type=int, default=400, help='Input dimension of \
					the tree encoder')
parser.add_argument('--compos_outdim', type=int, default=200, help='Output dimension of \
					the tree encoder')
parser.add_argument('--compos_wgt_dim', type=int, default=500, help='')
parser.add_argument('--compos_hid_dim', type=int, default=100, help='')
parser.add_argument('--compos_norm', action='store_true', default=False, help='')
parser.add_argument('--atten_dim', type=int, default=100, help='Attention dimension, \
					used in attention based tree encoder')
parser.add_argument('--rel_feat', action='store_true', help='Whether to use the relation \
					feature when encoding a subtree')
parser.add_argument('--rel_dim', type=int, default=50, 
					help='Dimension of relation label embeddings')
parser.add_argument('--dist_feat', action='store_true', help='Whether to use the distance \
					feature when encoding a subtree')
parser.add_argument('--max_distance', type=int, default=20, help='Maximun of the distance')
parser.add_argument('--min_distance', type=int, default=-20, help='Minimun of the distance')
parser.add_argument('--dist_dim', type=int, default=50, 
					help='Dimension of the distance embeddings')

parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--learning_rate', type=str, default=2e-3)
parser.add_argument('--oracle', action='store_true', default=False, 
					help='Always follows the oracle action in the training process')
parser.add_argument('--exploration_rate', type=float, default=0.1, 
					help='Probability to explorate the error decision')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--update_errors_num', type=int, default=1, help='Perform updating \
					when the accumulation errors is up to [update_errors_num]')

parser.add_argument('--mlp_rel_size', type=int, default=200, 
					help='Relation Biaffine Size')
# parser.add_argument('--dropout_mlp', type=float, default=0.33, 
# 					help='Dropout for biaffine MLP')


def parse_options():
	options = parser.parse_args()
	arg_dict = vars(options)
	if options.config_file:
		with open(options.config_file, 'r') as rfp:
			cfg_settings = yaml.load(rfp)
			delattr(options, 'config_file')
			for key, value in cfg_settings.items():
				if isinstance(value, dict):
					if key == options.env:
						for k, v in value.items():
							arg_dict[k] = v
					else:
						continue
				else:
					arg_dict[key] = value
	
	#if options.dynet_gpu and '--dynet-gpus' not in sys.argv:
	#	sys.argv.append('--dynet-gpus')
	#	sys.argv.append('1')
	if 'dynet_autobatch' in arg_dict and '--dynet-autobatch' not in sys.argv:
		sys.argv.append('--dynet-autobatch')
		sys.argv.append(str(arg_dict['dynet_autobatch']))
	if 'dynet_mem' in arg_dict and '--dynet-mem' not in sys.argv:
		sys.argv.append('--dynet-mem')
		sys.argv.append(str(arg_dict['dynet_mem']))
	if 'dynet_seed' in arg_dict and '--dynet-seed' not in sys.argv:
		sys.argv.append('--dynet-seed')
		sys.argv.append(str(arg_dict['dynet_seed']))
		if arg_dict['dynet_seed'] > 0:
			np.random.seed(arg_dict['dynet_seed'])
	if options.pos_dropout_rate > 0 and options.word_dim != options.pos_dim:
		print('POS tag dimension must be equal to word dimension when using \
			POS tag dropout. Use word dimension {%d} instead.' % options.word_dim)
		options.word_dim == options.pos_dim
	if options.env == 'test':
		options.parse = True
	return options


def print_config(options, **kwargs):
	print('======= CONFIG =======')
	for k, v in vars(options).items():
		print('%s: %s' % (k, v))
	for k, v in kwargs.items():
		print('%s: %s' % (k, v))
	print('======================')
	sys.stdout.flush()


_options = parse_options()

def get_options():
	return _options

