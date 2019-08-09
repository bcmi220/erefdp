from Options import *
from Corpus import *
import numpy as np
import dynet as dy
import Parser, os, sys, time


def _info(string):
	print(string)
	sys.stdout.flush()


def do_train(opt):
	_info('Run training on process %d (pid)' % os.getpid())
	print_config(opt)
	if opt.dev_file:
		with open(opt.dev_file, 'r') as devfp:
			dev_set = read_conll(devfp, lang=opt.language)
	proj_con = not opt.non_proj
	with open(opt.train_file, 'r') as trfp:
		train_set = list(read_conll(trfp, lang=opt.language, proj_cons=proj_con))
	
	start_point = opt.start_epoch
	load_path = opt.load_dir if opt.load_dir else opt.outdir
	if start_point > 1:
		_info('Start training from the %d epoch model' % start_point)
		vocab = Vocabulary.load(os.path.join(load_path, opt.vocab))
		parser = Parser.get_parser(vocab, opt)
		parser.load(os.path.join(load_path, opt.model))
		
	else:
		vocab = Vocabulary.build_from_list(train_set)
		vocab.save(os.path.join(opt.outdir, opt.vocab))
		parser = Parser.get_parser(vocab, opt)

	losses, num_loss = [], 0.
	update_len = opt.update_errors_num
	parser.set_train_flag()
	parser.refresh()
	for epoch in xrange(start_point, opt.epochs + 1):
		_info('\n[epoch %d]' % epoch)
		np.random.shuffle(train_set)
		for sentence in train_set:
			loss_expr = parser.parse_loss(sentence)
			if loss_expr:
				losses.append(loss_expr)
				num_loss += 1
				if num_loss > update_len:
					mean_loss = dy.esum(losses) / num_loss
					# mean_loss.forward()
					sys.stdout.write("%.4f"%mean_loss.value())
					sys.stdout.write("\r")
					sys.stdout.flush()
					mean_loss.backward()
					losses, num_loss = [], 0.
					parser.update()
					parser.refresh()
		if num_loss > 0:
			mean_loss = dy.esum(losses) / num_loss
			# mean_loss.forward()
			# print mean_loss.value(),
			sys.stdout.write("%.4f"%mean_loss.value())
			sys.stdout.write("\r")
			sys.stdout.flush()

			mean_loss.backward()
			losses, num_loss = [], 0.
			parser.update()
			parser.refresh()
		
		if opt.dev_file and epoch > opt.run_dev_after and epoch % opt.run_dev_every == 0:
			parser.set_test_flag()
			dev_output = os.path.join(opt.outdir, 'dev_epoch_%d.pred' % epoch)
			write_conll(dev_output, _parse_file(parser, opt.dev_file))
			# print 'perl script/eval.pl -g %s -s %s > %s.txt' % \
			# 			(opt.dev_file, dev_output, dev_output)
			os.system('perl script/eval.pl -q -g %s -s %s > %s.txt' % \
						(opt.dev_file, dev_output, dev_output))
			with open('%s.txt'%dev_output, 'r') as f:
				print f.readline().strip()
				print f.readline().strip()
			parser.save(os.path.join(opt.outdir, '%s%s' % (opt.model, epoch)))
			parser.set_train_flag()
	if not (opt.dev_file and epoch > opt.run_dev_after and epoch % opt.run_dev_every == 0):
		parser.save(os.path.join(opt.outdir, '%s%s' % (opt.model, epoch)))


def _parse_file(parser, filename):
	with open(filename, 'r') as cfp:
		for sentence in read_conll(cfp, lang=opt.language):
			parser.refresh()
			parser.parse(sentence)
			yield sentence


def do_parse(opt):
	_info('Run parsing on process %d (pid)' % os.getpid())
	print_config(opt)
	load_path = opt.load_dir if opt.load_dir else opt.outdir
	vocab = Vocabulary.load(os.path.join(load_path, opt.vocab))
	parser = Parser.get_parser(vocab, opt)
	parser.set_test_flag()
	_info('Loading model: %s' % opt.model)
	parser.load(os.path.join(load_path, opt.model))
	output_file = os.path.join(opt.outdir, '%s.pred' % \
								os.path.basename(opt.test_file))
	start = time.time()
	write_conll(output_file, _parse_file(parser, opt.test_file))
	end = time.time()
	print 'Parsing time elapse: %f' % (end - start)
	os.system('perl script/eval.pl -q -g %s -s %s > %s.txt' % \
						(opt.test_file, output_file, output_file))
	

if __name__ == '__main__':
	opt = get_options()
	if opt.parse:
		do_parse(opt)
	else:
		do_train(opt)

