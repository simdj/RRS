import time
import os
from preprocess import preprocess
from attack_model_bandwagon import attack_model_bandwagon
from user2vec import user2vec
from helpful import helpful_measure
# WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.

from parameter_controller import *


class Timer(object):
	def __init__(self, name=None):
		self.name = name

	def __enter__(self):
		self.tstart = time.time()
		print('[%s]' % self.name, 'Started')

	def __exit__(self, type, value, traceback):
		print('[%s]' % self.name, "Elapsed: %.2g sec" % (time.time() - self.tstart))
		print('')

def whole_process(exp_title):
	#######################################################################################
	print('#######################################################################################')
	print('Experiment Title', exp_title)
	# 'emb_16_rank_50_bandwagon_3%_3%_3%'
	# ('learning sentence # :', 774994)
	# ('learning sentence # :', 944319)
	params = parse_exp_title(exp_title)

	#######################################################################################

	with Timer("1. preprocess"):
		if not os.path.exists(params.review_origin_numpy_path):
			pp = preprocess(params=params)
			pp.whole_process()
		else:
			print("Preprocess is already done")

	with Timer("2. attack"):
		if not os.path.exists(params.review_fake_numpy_path):
			if params.attack_model == 'bandwagon':
				am = attack_model_bandwagon(params=params)
				am.whole_process()
			else:
				print("No bandwagon attack")
		else:
			print("Attack is already done")

	with Timer("3. User2Vec"):
		if not os.path.exists(params.embedding_attacked_path):
			u2v_attacked = user2vec(params=params, fake_flag=True, camo_flag=True,
			                        embedding_output_path=params.embedding_attacked_path)
			u2v_attacked.whole_process()
		else:
			print("User embedding on the attacked dataset is already done")
		if not os.path.exists(params.embedding_clean_path):
			u2v_clean = user2vec(params=params, fake_flag=False, camo_flag=False,
			                     embedding_output_path=params.embedding_clean_path)
			u2v_clean.whole_process()
		else:
			print("User embedding on the clean dataset is already done")

	with Timer("4. Compute helpfulness"):
		params.doubt_weight = 10
		print("Clean and naive")
		hm_clean_naive = helpful_measure(params=params, fake_flag=False, camo_flag=False, robust_flag=False)
		hm_clean_naive.whole_process()
		hm_clean_naive.helpful_test()
		print("Clean and robust")
		hm_clean_robust = helpful_measure(params=params, fake_flag=False, camo_flag=False, robust_flag=True)
		hm_clean_robust.whole_process()
		hm_clean_robust.helpful_test()
		print("Attacked and naive")
		hm_attacked_naive = helpful_measure(params=params, fake_flag=True, camo_flag=True, robust_flag=False)
		hm_attacked_naive.whole_process()
		hm_attacked_naive.helpful_test()
		print("Attacked and robust")
		hm_attacked_robust = helpful_measure(params=params, fake_flag=True, camo_flag=True, robust_flag=True)
		hm_attacked_robust.whole_process()
		hm_attacked_robust.helpful_test()


# with Timer("5. Matrix factorization"):
# 	from matrix_factorization import matrix_factorization
# 	mf = matrix_factorization(rank=50, lda=1, max_iter=5001)
# 	mf.whole_process()
import numpy as np

whole_process('bandwagon_1%_1%_1%_emb_32')

import winsound

winsound.Beep(300, 1000)
