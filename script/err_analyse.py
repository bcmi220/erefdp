from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# against length
# def compare(goldfile, predfile):
# 	len_cor = defaultdict(int)
# 	len_all = defaultdict(int)
# 	sent_len = 0
# 	correct = 0
# 	max_idx = 12
# 	with open(goldfile, 'r') as gfp, open(predfile, 'r') as pfp:
# 		for gline, pline in zip(gfp, pfp):
# 			if gline.strip() == '':
# 				index = int(sent_len - 1) / 5
# 				index = index if index < max_idx else max_idx
# 				if index < 10:
# 					len_all[index] += sent_len
# 					len_cor[index] += correct
# 				else:
# 					len_all[10] += sent_len
# 					len_cor[10] += correct
# 				sent_len = 0
# 				correct = 0
# 				continue
# 			gtoks = gline.strip().split('\t')
# 			ptoks = pline.strip().split('\t')
# 			sent_len += 1
# 			w_id = int(gtoks[0])
# 			g_pid = int(gtoks[6])
# 			p_pid = int(ptoks[6])
# 			if g_pid == p_pid:
# 				correct += 1
# 	ret_len = {}
# 	# print len_all
# 	# print len_cor
# 	for key in len_cor:
# 		ret_len[key] = float(len_cor[key]) / len_all[key]
# 		# if key < 10:
# 		# 	ret_len[key] = 1. - float(len_cor[key]) / len_all[key]
# 		# else:
# 		# 	ret_len[10] = 1. - float(len_cor[key]) / len_all[key]
# 			# if ret_len[12]
# 	# print ret_len
# 	return ret_len, None, None

# against distance
# def compare(goldfile, predfile):
# 	len_abs = defaultdict(int)
# 	len_all = defaultdict(int)
# 	dist_list = []
# 	with open(goldfile, 'r') as gfp, open(predfile, 'r') as pfp:
# 		for gline, pline in zip(gfp, pfp):
# 			if gline.strip() == '':
# 				continue
# 			gtoks = gline.strip().split('\t')
# 			ptoks = pline.strip().split('\t')
# 			w_id = int(gtoks[0])
# 			g_pid = int(gtoks[6])
# 			p_pid = int(ptoks[6])

# 			dist = int(abs(g_pid - w_id) - 1) / 5
# 			len_all[dist] += 1
# 			if g_pid == p_pid:
# 				len_abs[dist] += 1
# 	ret_len = {}
# 	for key in len_abs:
# 		if key < 10:
# 			ret_len[key] = 1. - float(len_abs[key]) / len_all[key]
# 	return ret_len, None, None

# pos tag
def compare(goldfile, predfile):
	maptmp = {'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], 
	'noun': ['NN', 'NNS', 'NNP', 'NNPS'], 
	'pron': ['PRP', 'PRP$', 'WP', 'WP$'], 
	'adj': ['JJ', 'JJR', 'JJS'], 
	'adv': ['RB', 'RBR', 'RBS', 'WRB'], 
	'conj': ['CC', 'IN']}
	mapping = {}
	for mapk in maptmp:
		keylist = maptmp[mapk]
		for newk in keylist:
			mapping[newk] = mapk
	pos_dict = defaultdict(int)
	pos_full = defaultdict(int)
	with open(goldfile, 'r') as gfp, open(predfile, 'r') as pfp:
		for gline, pline in zip(gfp, pfp):
			if gline.strip() == '':
				continue
			gtoks = gline.strip().split('\t')
			ptoks = pline.strip().split('\t')
			g_pid = int(gtoks[6])
 			p_pid = int(ptoks[6])
			pos_full[mapping.get(gtoks[3], 'other')] += 1
			if g_pid == p_pid:
				pos_dict[mapping.get(gtoks[3], 'other')] += 1
	for key in pos_dict:
		pos_dict[key] = 1. - float(pos_dict[key]) / pos_full[key]
		print key, pos_full[key]
	return None, None, pos_dict


if __name__ == '__main__':
	# htfile = 'htparser.conll'
	rcnnfile = 'rcnn.conll'
	treefile = 'ext_chsum.conll'
	htfile = 'ht.conll'
	goldfile = 'dev_pro.conll'
	ht_len_abs, ht_rel_dict, ht_pos_dict = compare(goldfile, htfile)
	rcnn_len_abs, rcnn_rel_dict, rcnn_pos_dict = compare(goldfile, rcnnfile)
	tree_len_abs, tree_rel_dict, tree_pos_dict = compare(goldfile, treefile)

	# plt.grid(linestyle='--', linewidth=1)
	# plt.plot(ht_len_abs.keys(), ht_len_abs.values(), 
	# 		linestyle='-.', label='HT-LSTM', )
	# plt.plot(rcnn_len_abs.keys(), rcnn_len_abs.values(), 
	# 		linestyle='--', label='RCNN')
	# plt.plot(tree_len_abs.keys(), tree_len_abs.values(), 
	# 		linestyle='--', label='our model')
	# # print ht_len_abs.values()
	# # print rcnn_len_abs.values()
	# # print tree_len_abs.values()
	# ax = plt.gca()
	# xlabel = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', 
	# 			'36-40', '41-45', '45-60', '61-inf']
	# plt.xticks(fontname='Times New Roman', fontsize=14)
	# plt.yticks(fontname='Times New Roman', fontsize=14)
	# ax.set_xlim((0, 10))
	# # ax.set_ylim((0.8, 1.0))
	# plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
	# ax.set_xticks(range(len(xlabel)))
	# ax.set_xticklabels(xlabel)
	# # plt.plot(*zip(*ht_len_abs.items()))
	# plt.legend(loc='best', prop={'family':'Times New Roman', 'size':14})
	# plt.tight_layout()
	# # # plt.show()
	# plt.savefig('against_length.pdf')

	# against pos tag
	unflod = lambda x, y: (x, y)
	pos_sets = rcnn_pos_dict.keys()
	
	ht_value = [ht_pos_dict[k] for k in pos_sets]
	rcnn_value = [rcnn_pos_dict[k] for k in pos_sets]
	tree_value = [tree_pos_dict[k] for k in pos_sets]

	abstact = np.array([htv - treev for htv, treev in zip(ht_value, tree_value)])
	pos_keys = (-abstact).argsort()

	pos_sets = [pos_sets[k] for k in pos_keys]
	rcnn_value = [rcnn_value[k] for k in pos_keys]
	ht_value = [ht_value[k] for k in pos_keys]
	tree_value = [tree_value[k] for k in pos_keys]

	index = np.arange(len(pos_sets))
	bar_width = 0.25
	opacity = 0.5
	error_config = {'ecolor': '0.3'}

	ax = plt.subplot()
	rects1 = ax.bar(index, ht_value, bar_width,
				alpha=opacity,
				color='b',
				label='HT LSTM')
	rects2 = ax.bar(index + bar_width, rcnn_value, bar_width,
				alpha=opacity,
				color='g',
				label='RCNN')
	rects3 = ax.bar(index + bar_width * 2, tree_value, bar_width,
				alpha=opacity,
				color='r',
				label='our model')

	plt.yticks(fontsize=14, fontname='Times New Roman')
	plt.setp(ax.xaxis.get_majorticklabels())
	plt.xticks(index + bar_width, pos_sets, fontsize=18, fontname='Times New Roman', fontstyle='italic')
	legend = ax.legend(loc='upper left',bbox_to_anchor=(0, -0.08), prop={'family':'Times New Roman', 'size':15})
	plt.tight_layout()
	# plt.savefig('postags.pdf')
	plt.show()
