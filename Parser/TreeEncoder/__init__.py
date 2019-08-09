import TreeEncoder
import HTLSTMEncoder

def get_tree_encoder(model, options, rel_vocab):
	if options.tree_encoder == 'empty':
		return TreeEncoder.EmptyTreeEncoder(model, options, rel_vocab)
	elif options.tree_encoder == 'ht' or options.tree_encoder == 'htlstm':
		return HTLSTMEncoder.HTLSTMEncoder(model, options, rel_vocab)
	else:
		return TreeEncoder.RecursiveEncoder(model, options, rel_vocab)

