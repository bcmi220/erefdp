from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def compare(goldfile, predfile):
	tmp_order_errs = defaultdict(int)
	margin_errs = defaultdict(int)
	margin_full = defaultdict(int)
	sent_len = 0
	order_errs = defaultdict(int)
	with open(goldfile, 'r') as gfp, open(predfile, 'r') as pfp:
		for gline, pline in zip(gfp, pfp):
			if gline.strip() == '':
				for k, v in tmp_order_errs.items():
					order_errs[int(float(k)/sent_len * 10)] += v
				sent_len = 0
				tmp_order_errs = defaultdict(int)
				continue
			gtoks = gline.strip().split('\t')
			ptoks = pline.strip().split('\t')
			g_pid = int(gtoks[6])
			p_pid = int(ptoks[6])
			order = int(ptoks[8])
			margin = float(ptoks[9]) * 10
			sent_len += 1
			margin_full[int(margin)] += 1
			if g_pid != p_pid:
				tmp_order_errs[order] += 1
				margin_errs[int(margin)] += 1
	return order_errs, margin_errs, margin_full


if __name__ == '__main__':
	predfile = 'dev_pro.conll.pred'
	goldfile = 'dev_pro.conll'
	order_errs, margin_errs, margin_full = compare(goldfile, predfile)
	plt.grid(linestyle='--', linewidth=1)
	# plt.plot(order_errs.keys(), order_errs.values(), 
	# 		linestyle='-.', label='order', )
	plt.plot(margin_errs.keys(), margin_errs.values(), 
			linestyle='--', label='margin')
	plt.plot(margin_full.keys(), margin_full.values(), 
			linestyle='-.', label='margin')
	ax = plt.gca()
	plt.xticks(fontname='Times New Roman', fontsize=14)
	plt.yticks(fontname='Times New Roman', fontsize=14)
	# ax.set_xlim((0, 5))
	# ax.set_ylim((0.6, 1.0))
	plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
	# ax.set_xticks(range(len(xlabel)))
	# ax.set_xticklabels(xlabel)
	plt.legend(loc='best', prop={'family':'Times New Roman', 'size':14})
	plt.tight_layout()
	plt.show()
