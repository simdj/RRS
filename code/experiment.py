import time
import os
from preprocess import preprocess
from attack_model_bandwagon import attack_model_bandwagon
from user2vec import user2vec
from helpful import helpful_measure
# WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.

from WMF import *
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
	print('#######################################################################################')
	print('Experiment Title', exp_title)
	params = parse_exp_title(exp_title)
	with Timer("1. preprocess"):
		if not os.path.exists(params.review_origin_path):
			pp = preprocess(params=params)
			pp.whole_process()
		else:
			print("Preprocess is already done")
	with Timer("2. attack"):
		if not os.path.exists(params.review_fake_path):
			if params.attack_model == 'bandwagon':
				am = attack_model_bandwagon(params=params)
				am.whole_process()
			else:
				print("No bandwagon attack")
		else:
			print("Attack is already done")
	with Timer("3. User2Vec"):
		# attacked embedding
		if not os.path.exists(params.embedding_attacked_path):
			u2v_attacked = user2vec(params=params, fake_flag=True, camo_flag=True,
			                        embedding_output_path=params.embedding_attacked_path)
			u2v_attacked.whole_process()
		else:
			print("User embedding on the attacked dataset is already done")
		# clean embedding
		if not os.path.exists(params.embedding_clean_path):
			u2v_clean = user2vec(params=params, fake_flag=False, camo_flag=False,
			                     embedding_output_path=params.embedding_clean_path)
			u2v_clean.whole_process()
		else:
			print("User embedding on the clean dataset is already done")
	with Timer("4. Compute helpfulness"):
		params.doubt_weight = 100

		attack_flag_list=[False,True]
		robust_flag_list=[False,True]

		for attack_flag in attack_flag_list:
			for robust_flag in robust_flag_list:
				hm_name = ['Clean', 'Naive']
				if attack_flag:	hm_name[0]='Attacked'
				if robust_flag:	hm_name[1]='Robust'
				print (' and '.join(hm_name))
				# helpful computation start
				hm = helpful_measure(params=params, fake_flag=attack_flag, camo_flag=attack_flag, robust_flag=robust_flag)
				hm.whole_process()
				hm.helpful_test()
	with Timer("5. Matrix factorization"):

		algorithm_model_list = ['base','naive','robust']
		rank_list = [10,20,30,40,50,60,70,80,100]
		# params = parse_exp_title(exp_title)
		for rank in rank_list:
			for am in algorithm_model_list:
				print('----------------------', am, rank,'----------------------')
				wp = WMF_params(params=params, algorithm_model=am, attack_flag=True)
				wp.rank = rank
				wp.max_iter=30001
				wmf_instance = WMF(params=wp)
				wmf_instance.whole_process()
				# performance recording
				performance = metric(params=wp)
				print('++++++++++ Important value (honest) ', performance.mean_prediction_rating_on_target(honest=True))
				print('(fake)', performance.mean_prediction_rating_on_target(honest=False))
				# # helpful test again..
				# try:
				# 	origin_help = np.load(wp.helpful_origin_path)[:,-1]
				# 	fake_help = np.load(wp.helpful_fake_path)[:,-1]
				# 	print ('Helpful distribution',np.percentile(origin_help,10),np.percentile(origin_help,50),np.percentile(origin_help,90),np.mean(fake_help))
				# except:
				# 	pass

if __name__ == "__main__":
	exp_title_list = ['bandwagon_1%_1%_1%_emb_32','bandwagon_2%_1%_2%_emb_32','bandwagon_3%_1%_3%_emb_32','bandwagon_4%_1%_4%_emb_32','bandwagon_5%_1%_5%_emb_32'   ]
	for exp_title in exp_title_list:
		whole_process(exp_title)
