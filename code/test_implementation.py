import time
import os
import numpy as np
# from preprocess import preprocess
from new_preprocess import preprocess
from attack_model_bandwagon import attack_model_bandwagon
from user2vec import user2vec
from helpful import helpful_measure
# WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
try:
	from WMF import *
except:
	pass
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
		params.user_threshold=5
		params.item_threshold=5
		pp = preprocess(params=params)
		# start preprocess if needed
		if not os.path.exists(params.review_origin_path):
			pp.whole_process()
		else:
			print("Preprocess is already done")
	
	with Timer("2. attack"):
		if params.attack_model == 'bandwagon':
			am = attack_model_bandwagon(params=params)
		else:
			print("No bandwagon attack")
		# start attack if needed	
		if not os.path.exists(params.review_fake_path):
				am.whole_process()
		else:
			print("Attack is already done")
	
	with Timer("3. User2Vec"):
		# attacked embedding
		u2v_attacked = user2vec(params=params, fake_flag=True, camo_flag=True, embedding_output_path=params.embedding_attacked_path)
		if not os.path.exists(params.embedding_attacked_path):
			u2v_attacked.whole_process()
		else:
			print("User embedding on the attacked dataset is already done")
		u2v_attacked.similarity_test()
		# clean embedding
		u2v_clean = user2vec(params=params, fake_flag=False, camo_flag=False, embedding_output_path=params.embedding_clean_path)
		if not os.path.exists(params.embedding_clean_path):
			u2v_clean.whole_process()
		else:
			print("User embedding on the clean dataset is already done")
		u2v_clean.similarity_test()

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
				if not os.path.exists(hm.helpful_origin_path):
					hm.whole_process()
				else:
					print("Helpfulness computing is already done")
				hm.helpful_test()
	with Timer("5. Matrix factorization"):
		# lda_list = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
		# rank_list = [10,20,30,40,50,60,70,80,100]
		# rank_list = [15,20,25,30]
		rank_list = [20,25,30]
		# lda_list = [0.001, 0.01, 0.1]
		lda_list=[0.01, 0.1]
		max_iter_list = [5001, 50001]

		# algorithm_model_list = ['base','base','naive','robust']
		# attack_flag_list = [False, True, True, True]
		# algorithm_model_list = ['naive','robust']
		# algorithm_model_list=['base']
		# attack_flag_list=[False]
		algorithm_model_list = ['base', 'base','naive','robust']
		attack_flag_list = [False, True, True, True]
		for rank in rank_list:
			for lda in lda_list:
				for max_iter in max_iter_list:
					important_value_list = []
					for am, af in zip(algorithm_model_list, attack_flag_list):
						print('-----------------------',am,'attack',af, 'rank',rank, 'lda', lda, '---------------------')
						wp = WMF_params(params=params, algorithm_model=am, attack_flag=af)
						wp.rank = rank
						wp.lda = lda
						# print("small test")
						wp.max_iter=max_iter

						wmf_instance = WMF(params=wp)
						wmf_instance.whole_process()

						performance = metric(params=wp)

						try:
							origin_help = np.load(wp.helpful_origin_path)[:,-1]
							fake_help = np.load(wp.helpful_fake_path)[:,-1]
							print (np.percentile(origin_help,25),np.percentile(origin_help,50),np.percentile(origin_help,75),np.mean(fake_help))
						except:
							pass

						important_value = []
						important_value.append(performance.mean_prediction_rating_on_target(honest=True))
						important_value.append(performance.rmse_rating_on_target(honest=True))
						important_value.append(performance.rmse_rating_on_target(honest=False))

						important_value_list.append(important_value)
						print('')

					np.set_printoptions(precision=4)
					print('')
					print('')
					print(exp_title, am, af, rank, lda)
					print('[[[[Important_value_list]]]]')
					print('(expected rating on target, RMSE(honest rating on target), RMSE(fake rating on target)')
					print(np.array(important_value_list))
					print('')
					print('')

if __name__ == "__main__":
	# exp_title_list = ['bandwagon_0.25%_0.25%_0.25%_emb_32', 'bandwagon_0.5%_0.5%_0.5%_emb_32', 'bandwagon_1%_1%_1%_emb_32']
	exp_title_list = ['bandwagon_1%_1%_1%_emb_32']
	for exp_title in exp_title_list:
		whole_process(exp_title)