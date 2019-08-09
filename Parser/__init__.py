from EasyFirst import EasyFirst


def get_parser(vocab, opt):
	if opt.algorithm == 'ef' or opt.algorithm == 'easyfirst':
		return EasyFirst(vocab, opt)

	