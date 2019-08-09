from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from Corpus import *


def compare(goldfile, predfile):
	correct_len = defaultdict(int)
	full_len = defaultdict(int)
	correct_pos = defaultdict(int)
	full_pos = defaultdict(int)
	maptmp = {'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB'], 
				'noun': ['NN', 'NNS', 'NNP', 'NNPS'], 
				'pron': ['PRP', 'PRP$', 'WP', 'WP$', 'WDT'], 
				'adj': ['JJ', 'JJR', 'JJS'], 
				'adv': ['RB', 'RBR', 'RBS'], 
				'conj': ['CC', 'IN']}
	mapping = {}
	for mapk in maptmp:
		keylist = maptmp[mapk]
		for newk in keylist:
			mapping[newk] = mapk
	with open(predfile, 'r') as predfp, open(goldfile, 'r') as goldfp:
		pred_set = read_conll(predfp)
		gold_set = read_conll(goldfp)
		for predsent, goldsent in zip(pred_set, gold_set):
			predsent = predsent[1:]
			goldsent = goldsent[1:]
			sent_len = len(predsent)
			correct = 0
			for ptok, gtok in zip(predsent, goldsent):
				full_pos[mapping.get(ptok.gpos, 'unk')] += 1
				if ptok.parent_id == gtok.parent_id:
					correct += 1
					correct_pos[mapping.get(ptok.gpos, 'unk')] += 1
			index = (sent_len - 1) / 5
			correct_len[index] += correct
			full_len[index] += sent_len
		for k in correct_len:
			correct_len[k] = float(correct_len[k]) / full_len[k]
		for k in correct_pos:
			correct_pos[k] = float(correct_pos[k]) / full_pos[k]
		del correct_pos['unk']
	return correct_len, correct_pos

basefile = 'base.conll'
chsumfile = 'chsum.conll'
chextfile = 'ext_chsum.conll'
goldfile = 'dev_pro.conll'
base_len, base_pos = compare(goldfile, basefile)
chsum_len, chsum_pos = compare(goldfile, chsumfile)
chext_len, chext_pos = compare(goldfile, chextfile)

# plt.grid(linestyle='--', linewidth=1)
# plt.plot(*zip(*base_len.items()), linestyle='-.', label='baseline model')
# plt.plot(*zip(*chsum_len.items()), linestyle='--', label='our model')
# plt.plot(*zip(*chext_len.items()), linestyle='-', label='our model + bilstm + pretrain')
# ax = plt.gca()
# xlabel = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', 
# 		'36-40', '41-45', '45-60', '61-65', '66-70', '71-75']
# plt.xticks(fontname='Times New Roman', fontsize=14)
# plt.yticks(fontname='Times New Roman', fontsize=14)
# ax.set_xlim((0, 12))
# ax.set_ylim((0.6, 1.0))
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
# ax.set_xticks(range(len(xlabel)))
# ax.set_xticklabels(xlabel)
# plt.legend(loc='best', prop={'family':'Times New Roman', 'size':14})
# plt.tight_layout()
# plt.show()
# plt.savefig('against_length.pdf')

unflod = lambda x, y: (x, y)
pos_sets = chsum_pos.keys()
b_value = [base_pos[k] for k in pos_sets]
ch_value = [chsum_pos[k] for k in pos_sets]
ex_value = [chext_pos[k] for k in pos_sets]
abstact = np.array([chv - bv for chv, bv in zip(ch_value, b_value)])
pos_keys = (-abstact).argsort()
pos_sets = [pos_sets[k] for k in pos_keys]
ch_value = [ch_value[k] for k in pos_keys]
b_value = [b_value[k] for k in pos_keys]
ex_value = [ex_value[k] for k in pos_keys]
index = np.arange(len(pos_sets))
bar_width = 0.25
opacity = 0.4
error_config = {'ecolor': '0.3'}
ax = plt.subplot()
rects1 = ax.bar(index, b_value, bar_width,
				alpha=opacity,
				color='b',
				label='baseline model')
rects2 = ax.bar(index + bar_width, ch_value, bar_width,
				alpha=opacity,
				color='r',
				label='our model')
rects3 = ax.bar(index + bar_width * 2, ex_value, bar_width,
				alpha=opacity,
				color='g',
				label='our model + bilstm + pretrain')
plt.yticks(fontsize=14, fontname='Times New Roman')
plt.setp(ax.xaxis.get_majorticklabels())
plt.xticks(index + bar_width * 1, pos_sets, fontsize=18, fontname='Times New Roman', fontstyle='italic')
legend = ax.legend(loc='upper left',bbox_to_anchor=(0, -0.08), prop={'family':'Times New Roman', 'size':15})
plt.tight_layout()
plt.show()