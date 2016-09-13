import time

from preprocess import preprocess
from attack_model_bandwagon import attack_model_bandwagon
from user2vec import user2vec
from helpful import helpful_measure, helpful_test
# WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.

from parameter_controller import *

class Timer(object):
	def __init__(self, name=None):	
		self.name = name
	def __enter__(self):
		self.tstart = time.time()
	def __exit__(self, type, value, traceback):
		print('[%s]' % self.name, "Elapsed: %.2g sec" % (time.time() - self.tstart))

def whole_process(exp_title):
	#######################################################################################
	print('#######################################################################################')
	print('#######################################################################################')
	print('Experiment Title', exp_title)
	# 'emb_16_rank_50_bandwagon_3%_3%_3%'
		# ('learning sentence # :', 774994)
		# ('learning sentence # :', 944319)
	params = parse_exp_title(exp_title)

	#######################################################################################

	# with Timer("1. preprocess"):
	# 	pp=preprocess(params=params)
	# 	pp.whole_process()

	with Timer("2. attack"):
		if params.attack_model == 'bandwagon':
			am = attack_model_bandwagon(params=params)
			am.whole_process()
		else:
			print("No attack")

	with Timer("3. User2Vec"):
		u2v = user2vec(params=params)
		u2v.new_whole_process()
	
	with Timer("4. Compute helpfulness"):
		params.doubt_weight = 10
		hm = helpful_measure(params=params)
		hm.whole_process()
		helpful_test() 



	# with Timer("5. Matrix factorization"):
	# 	from matrix_factorization import matrix_factorization
	# 	mf = matrix_factorization(rank=50, lda=1, max_iter=5001)
	# 	mf.whole_process()
import numpy as np
# np.random.seed(1)
# whole_process('emb_32_rank_50_bandwagon_1%_1%_1%')

# # whole_process('emb_32_rank_50_bandwagon_1%_1%_3%')

# whole_process('emb_64_rank_50_bandwagon_1%_1%_1%')
# whole_process('emb_64_rank_50_bandwagon_3%_1%_1%')
whole_process('emb_64_rank_50_bandwagon_1%_1%_5%')
# whole_process('emb_64_rank_50_bandwagon_3%_1%_5%')

# # exp_title = 'emb_64_rank_50_None'
# # exp_title = 'emb_64_rank_50_bandwagon_3%_3%_3%'
# # exp_title = 'emb_64_rank_50_bandwagon_1%_1%_10%'
# # exp_title = 'emb_64_rank_50_average_1%_1%_1%'
# # exp_title = 'emb_64_rank_50_bandwagon_1%_1%_1%'
	
import winsound
winsound.Beep(300,1000)
