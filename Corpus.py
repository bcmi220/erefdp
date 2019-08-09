from collections import Counter
import re, random, pickle

__all__ = ['ConllEntry', 'Vocabulary', 'read_conll', 'write_conll']

class ConllEntry(object):
	def __init__(self, w_id, word, gpos, ppos, pos, 
				parent_id=None, relation=None):
		self.w_id = w_id
		self.form = word
		self.norm = _normalize(word)
		self.gpos = gpos.upper()
		self.ppos = ppos.upper()
		self.pos = pos.upper()
		self.parent_id = parent_id
		self.relation = relation
		self.pred_pid = None
		self.pred_rel = None


class ConllUEntry(object):
	def __init__(self, id, form, lemma, pos, cpos, feats=None, 
				parent_id=None, relation=None, deps=None, misc=None):
		self.id = id
		self.form = form
		self.norm = normalize(form)
		self.cpos = cpos.upper()
		self.pos = pos.upper()
		self.parent_id = parent_id
		self.relation = relation
		self.lemma = lemma
		self.feats = feats
		self.deps = deps
		self.misc = misc
		self.pred_parent_id = None
		self.pred_relation = None

	def __str__(self):
		values = [str(self.id), self.form, self.lemma, self.cpos, 
					self.pos, self.feats, str(self.pred_parent_id) 
					if self.pred_parent_id is not None else None, 
					self.pred_relation, self.deps, self.misc]
		return '\t'.join(['_' if v is None else v for v in values])


def read_conll(fh, lang='en', proj_cons=False):
	read_in_sent = 0
	drop_sent = 0
	root = ConllEntry(0, '<ROOT>', 'ROOT_POS', 'ROOT_POS', 
						'ROOT_POS', 0, 'rroot')
	assert type(lang) == str, "Language type must be <type 'str'>, \
								but %s found" % type(lang)
	get_pos = lambda x: x[4] if lang == 'en' else x[3]
	tokens = [root]
	for line in fh:
		tok = line.strip().split()
		if not tok:
			if len(tokens) > 1:
				if not proj_cons or _projective(tokens):
					yield tokens
				else:
					drop_sent += 1
				read_in_sent += 1
			tokens = [root]
		else:
			tokens.append(ConllEntry(int(tok[0]), tok[1], tok[3], 
							tok[4], get_pos(tok), int(tok[6]) \
							if tok[6] != '_' else -1, tok[7]))
	if len(tokens) > 1:
		yield tokens
		read_in_sent += 1
	print read_in_sent, 'sentences read.'
	if proj_cons:
		print drop_sent, 'non-projective sentences drop.'


def read_conllu(fh, lang='en', proj_cons=False):
	drop_sent = 0
	read_in_sent = 0
	root = ConllEntry(0, '<ROOT>', '<ROOT>', 'ROOT-POS', 'ROOT_POS', 
						'_', -1, 'rroot', '_', '_')
	tokens = [root]
	for line in fh:
		tok = line.strip().split('\t')
		if not tok or line.strip() == '':
			if len(tokens)>1:
				if not proj_cons or _projective(tokens):
					yield tokens
				else:
					#print 'Non-projective sentence dropped'
					drop_sent += 1
				read_in_sent += 1
			tokens = [root]
		else:
			if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
				tokens.append(line.strip())
			else:
				tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], 
								tok[4], tok[3], tok[5], int(tok[6]) 
								if tok[6] != '_' else -1, tok[7], 
								tok[8], tok[9]))
	if len(tokens) > 1:
		yield tokens
		read_in_sent += 1
	print read_in_sent, 'sentences read.'
	if proj_cons:
		print drop_sent, 'non-projective sentences drop.'


def write_conll(filename, conll_gen):
	with open(filename, 'w') as fh:
		for sentence in conll_gen:
			for entry in sentence[1:]:
				fh.write('\t'.join([str(entry.w_id), entry.form, '_', \
					entry.gpos, entry.ppos, '_', str(entry.pred_pid), \
					entry.pred_rel, '_', '_']))
				fh.write('\n')
			fh.write('\n')


def write_conllu(filename, conll_gen):
	with open(filename, 'w') as fh:
		for sentence in conll_gen:
			for entry in sentence[1:]:
				fh.write(str(entry) + '\n')
			fh.write('\n')


def _projective(sentence):
	unassigned = [0] * len(sentence)
	for tok in sentence[1: ]:
		unassigned[tok.parent_id] += 1
	pending = sentence[: ]
	for _ in xrange(len(sentence)):
		for tok_1, tok_2 in zip(pending, pending[1: ]):
			if tok_1.parent_id == tok_2.w_id and unassigned[tok_1.w_id] == 0:
				unassigned[tok_2.w_id] -= 1
				pending.remove(tok_1)
				break
			if tok_2.parent_id == tok_1.w_id and unassigned[tok_2.w_id] == 0:
				unassigned[tok_1.w_id] -= 1
				pending.remove(tok_2)
				break 
	return len(pending) == 1


class Vocabulary(object):
	def __init__(self, words_count, w2i, p2i, r2i):
		self.__wcnt = words_count
		self.__w2i = w2i
		self.__p2i = p2i
		self.__r2i = r2i


	@classmethod
	def build_from_file(cls, filename):
		with open(filename, 'r') as fh:
			cls.build_from_list(read_conll(fh))
			

	@classmethod
	def build_from_list(cls, sentlist):
		words_count = Counter()
		posset = set()
		relset = set()

		for sentence in sentlist:
			words_count.update([tok.norm for tok in sentence])
			posset.update(set([tok.pos for tok in sentence]))
			relset.update(set([tok.relation for tok in sentence]))
		
		w2i = {w: i + 2 for i, w in enumerate(words_count)}
		p2i = {p: i + 2 for i, p in enumerate(posset)}
		r2i = {r: i for i, r in enumerate(relset)}
		w2i['<UNK>'], p2i['<UNK>'] = 0, 0
		w2i['<PAD>'], p2i['<PAD>'] = 1, 1

		return cls(words_count, w2i, p2i, r2i)


	def save(self, filename):
		with open(filename, 'w') as vfp:
			pickle.dump(self.__wcnt, vfp)
			pickle.dump(self.__w2i, vfp)
			pickle.dump(self.__p2i, vfp)
			pickle.dump(self.__r2i, vfp)


	@classmethod
	def load(cls, filename):
		with open(filename, 'r') as vfp:
			wcnt = pickle.load(vfp)
			w2i = pickle.load(vfp)
			p2i = pickle.load(vfp)
			r2i = pickle.load(vfp)
		return cls(wcnt, w2i, p2i, r2i)


	@property
	def wordlookup_tbl(self):
		return self.__w2i


	@property
	def poslookup_tbl(self):
		return self.__p2i


	@property
	def rellookup_tbl(self):
		return self.__r2i


	@property
	def word_freq(self):
		return self.__wcnt


_numberRegex = re.compile('[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+')
def _normalize(word):
	return 'NUM' if _numberRegex.match(word) else word.lower()

