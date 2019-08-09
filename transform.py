from Corpus import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-f', type=str)
parser.add_argument('--src_file', '-s', type=str)
parser.add_argument('--trg_file', '-t', type=str)
opts = parser.parse_args()

def generate(in_fp, src_fp, trg_fp):
	for sentence in read_conll(in_fp, lang='eng', proj_cons=True):
		for tok in sentence:
			src_fp.write('%s ' % tok.norm)
			trg_fp.write('%s ' % tok.relation)
		src_fp.write('\n')
		trg_fp.write('\n')


if __name__ == '__main__':
	with open(opts.input, 'r') as ifp, open(opts.src_file, 'w') as sfp, \
		open(opts.trg_file, 'w') as tfp:
		generate(ifp, sfp, tfp)
