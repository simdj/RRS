import time
import os
import numpy as np
# from preprocess import preprocess
from new_preprocess import preprocess
from new_attack_model_bandwagon import attack_model_bandwagon
from new_user2vec import user2vec
from helpful import helpful_measure
# WARNING (theano.configdefaults): g++ not detected ! Theano will be unable to execute optimized C-implementations (for both CPU and GPU) and will default to Python implementations. Performance will be severely degraded. To remove this warning, set Theano flags cxx to an empty string.
# try:
# 	from WMF import *
# except:
# 	pass
try:
	from WMF_validation import *
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


def prediction_result_testset(U,V, test_data):
	ret = []
	for review_row in test_data:
		user = int(review_row[0])
		item = int(review_row[1])
		# rating = int(review_row[2])
		ret.append(np.dot(U[user,:],V[:,item]))
	return np.array(ret)

def prediction_result_target(U,V, fake_user_start, target_item_list):
	# 
	assert(type(fake_user_start)==type(1))
	assert(type(target_item_list[0])==type(1))
	return np.dot( U[:fake_user_start,:],V[:,target_item_list] )
	

def evaluation(params):
	# test_data is sparse format np.array([user, item, rating, helpful])
	test_data = np.load(params.test_data_path)

	UV_path_list = []
	UV_path_list.append([params.base_U_clean_path, params.base_V_clean_path])
	UV_path_list.append([params.base_U_attacked_path, params.base_V_attacked_path])
	UV_path_list.append([params.naive_U_clean_path, params.naive_V_clean_path])
	UV_path_list.append([params.naive_U_attacked_path, params.naive_V_attacked_path])
	UV_path_list.append([params.robust_U_clean_path, params.robust_V_clean_path])
	UV_path_list.append([params.robust_U_attacked_path, params.robust_V_attacked_path])
	
	# overall shift for test data set
	prediction_list = [prediction_result_testset(np.load(U_path),np.load(V_path), test_data) for (U_path,V_path) in UV_path_list]

	prediction_shift_base = np.mean(np.abs(prediction_list[0]-prediction_list[1]))
	prediction_shift_naive = np.mean(np.abs(prediction_list[2]-prediction_list[3]))
	prediction_shift_robust = np.mean(np.abs(prediction_list[4]-prediction_list[5]))
	print 'Overall'
	print 'Base\t', prediction_shift_base
	print 'Naive\t', prediction_shift_naive
	print 'Robust\t', prediction_shift_robust

	# target prediction shift
	fake_user_start = int(np.min(np.load(params.fake_user_id_list_path)))
	target_item_list = map(int,np.load(params.target_item_list_path))

	target_prediction_list = [prediction_result_target(np.load(U_path),np.load(V_path), fake_user_start, target_item_list) for (U_path,V_path) in UV_path_list]

	target_prediction_shift_base = np.mean(target_prediction_list[1]-target_prediction_list[0])
	target_prediction_shift_naive = np.mean(target_prediction_list[3]-target_prediction_list[2])
	target_prediction_shift_robust = np.mean(target_prediction_list[5]-target_prediction_list[4])
	print 'Target'
	print 'Base  \t', target_prediction_shift_base, '\tMean at (before,after)', np.mean(target_prediction_list[0]), np.mean(target_prediction_list[1])
	print 'Naive \t', target_prediction_shift_naive, '\tMean at (before,after)', np.mean(target_prediction_list[2]), np.mean(target_prediction_list[3])
	print 'Robust\t', target_prediction_shift_robust, '\tMean at (before,after)', np.mean(target_prediction_list[4]), np.mean(target_prediction_list[5])


def print_distribution(l, end_flag=True):
	percentile_list = [10,25,50,75,90]
	print percentile_list, 
	for p in percentile_list:
		print np.percentile(l,p),
	if end_flag:
		print ''
	

def whole_process(params):
		
	refresh_flag = True
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
		if not os.path.exists(params.review_fake_path) or refresh_flag:
				am.whole_process()
		else:
			print("Attack is already done")
	
	with Timer("3. User2Vec"):
		# attacked embedding
		u2v_attacked = user2vec(params=params, fake_flag=True, camo_flag=True, embedding_output_path=params.embedding_attacked_path)
		if not os.path.exists(params.embedding_attacked_path) or refresh_flag:
			u2v_attacked.whole_process()
		else:
			print("User embedding on the attacked dataset is already done")
		u2v_attacked.similarity_test()
		# clean embedding
		u2v_clean = user2vec(params=params, fake_flag=False, camo_flag=False, embedding_output_path=params.embedding_clean_path)
		if not os.path.exists(params.embedding_clean_path) or refresh_flag:
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
				# print (' and '.join(hm_name))

				# helpful computation start
				hm = helpful_measure(params=params, fake_flag=attack_flag, camo_flag=attack_flag, robust_flag=robust_flag)
				if not os.path.exists(hm.helpful_origin_path) or refresh_flag:
					hm.whole_process()
				else:
					print("Helpfulness computing is already done")
				# hm.helpful_test()
	
	with Timer("5. Matrix factorization"):
		# real
		rank_list = [20, 30, 40] 
		# lda_list = [1e-5, 1e-4, 5e-4] 
		lda_list = [1e-4] 
		# rank 10 is not stable, rank 50 is stable but lda should be low (0.0005)
		# rank 20, lda 0.0005 -> base:naive:robust=0.96:1.36:0.21 //good
		# rank 30, lda 0.0005 -> base:naive:robust=0.93:1.02:0.18 //good
		# rank 40, lda 0.0005 -> base:naive:robust=1.09:1.18:0.18 //good

		# rank 20, lda 0.005 -> base:naive:robust=1.18:1.35:0.77
		# rank 30, lda 0.005 -> base:naive:robust=0.95:1.03:0.64
		# rank 40, lda 0.005 -> base:naive:robust=0.95:1.05:0.70
		max_iter_list = [25001]

		algorithm_model_list = ['base','base','naive','naive', 'robust','robust']
		attack_flag_list = [False, True, False, True, False, True]

		# # helpfulness test varying embedding method
		# rank_list = [20]
		# lda_list = [0.0005]
		# ####################
		# max_iter_list = [25001]

		# algorithm_model_list = ['base','base','naive','naive', 'robust','robust']
		# attack_flag_list = [False, True, False, True, False, True]
		
		
		for rank in rank_list:
			for lda in lda_list:
				for max_iter in max_iter_list:
					important_value_list = []
					for am, af in zip(algorithm_model_list, attack_flag_list):
						# prepare parameters for matrix factorization
						print'-----------------------',am,'attack',af, 'rank',rank, 'lda', lda, '---------------------'
						wp = WMF_params(params=params, algorithm_model=am, attack_flag=af)
						wp.rank = rank
						wp.lda = lda
						wp.max_iter=max_iter

						try:
							origin_help = np.load(wp.helpful_origin_path)[:,-1]
							fake_help = np.load(wp.helpful_fake_path)[:,-1]
							print "Helpfulness distribution"
							print '[ Fake target] Mean', np.mean(fake_help),
							print_distribution(fake_help)
							print '[Honest  all ] Mean', np.mean(origin_help), 
							print_distribution(origin_help)
							
							target_item_list = map(int,np.load(params.target_item_list_path))
							for target_item in target_item_list:
								origin_help_target = np.load(wp.helpful_origin_path)
								origin_help_target = origin_help_target[origin_help_target[:,1]==target_item,-1]
								print '[Honest target] Mean', np.mean(origin_help_target), 
								print_distribution(origin_help_target,end_flag=True)
								# honest rating size=15 1% attack
						except:
							pass

					# 	# do matrix factorization
					# 	wmf_instance = WMF(params=wp)
					# 	wmf_instance.whole_process()


					# with Timer("6. Evaluation"):
					# 	evaluation(params)
					print('')
					print('')

if __name__ == "__main__":
	exp_title_list = []

	# # 1026
	# exp_title_list += ['bandwagon_0.5%_0.5%_0%_emb_32']
	# exp_title_list += ['bandwagon_1%_0.5%_0%_emb_32']
	# exp_title_list += ['bandwagon_3%_0.5%_0%_emb_32']

	# # 1026
	# exp_title_list += ['bandwagon_0.5%_1%_0%_emb_32']
	# exp_title_list += ['bandwagon_1%_1%_0%_emb_32']
	# exp_title_list += ['bandwagon_3%_1%_0%_emb_32']


	# # 1027
	# exp_title_list += ['bandwagon_0.5%_2%_0%_emb_32']
	# exp_title_list += ['bandwagon_1%_2%_0%_emb_32']
	# exp_title_list += ['bandwagon_3%_2%_0%_emb_32']

	# no experiment
	# exp_title_list += ['bandwagon_0.5%_0.5%_1.1_emb_32']
	# exp_title_list += ['bandwagon_1%_0.5%_1.1_emb_32']
	# exp_title_list += ['bandwagon_3%_0.5%_1.1_emb_32']
	
	# # 1027
	# exp_title_list += ['bandwagon_0.5%_1%_1.1_emb_32']
	# exp_title_list += ['bandwagon_1%_1%_1.1_emb_32']
	# exp_title_list += ['bandwagon_3%_1%_1.1_emb_32']

	# # 1027
	# exp_title_list += ['bandwagon_0.5%_2%_1.1_emb_32']
	# exp_title_list += ['bandwagon_1%_2%_1.1_emb_32']
	# exp_title_list += ['bandwagon_3%_2%_1.1_emb_32']


	# # 1028
	# exp_title_list += ['bandwagon_0.5%_0%_1%_emb_32']
	# exp_title_list += ['bandwagon_1%_0%_1%_emb_32']
	# exp_title_list += ['bandwagon_3%_0%_1%_emb_32']

	# # 1028
	# exp_title_list += ['bandwagon_0.5%_1%_1%_emb_32']
	# exp_title_list += ['bandwagon_1%_1%_1%_emb_32']
	# exp_title_list += ['bandwagon_3%_1%_1%_emb_32']

	# # 1028
	# exp_title_list += ['bandwagon_0.5%_0%_2%_emb_32']
	# exp_title_list += ['bandwagon_1%_0%_2%_emb_32']
	# exp_title_list += ['bandwagon_3%_0%_2%_emb_32']


	# 1029
	exp_title_list += ['bandwagon_1%_1%_0%_emb_32'] 
	# exp_title_list += ['bandwagon_1%_1%_1.1_emb_32']
	exp_title_list += ['bandwagon_1%_1%_1%_emb_32']

	# exp_title_list += ['bandwagon_1%_3%_0%_emb_32']
	# # exp_title_list += ['bandwagon_1%_3%_1.1_emb_32']
	# exp_title_list += ['bandwagon_1%_3%_1%_emb_32']

	# exp_title_list += ['bandwagon_3%_1%_0%_emb_32']
	# # exp_title_list += ['bandwagon_3%_1%_1.1_emb_32']
	# exp_title_list += ['bandwagon_3%_1%_1%_emb_32']

	# exp_title_list += ['bandwagon_3%_3%_0%_emb_32']
	# # exp_title_list += ['bandwagon_3%_3%_1.1_emb_32']
	# exp_title_list += ['bandwagon_3%_3%_1%_emb_32']


	# for uu in [1,5,10]:
	for uu in [1]:
		for exp_title in exp_title_list:
			params = parse_exp_title(exp_title)
			# for camo_vote_size_multiple in [0, 1, 5, 10]:
			# for camo_vote_size_multiple in [1, 10]:
			for camo_vote_size_multiple in [1]:
				print '#######################################################################################'
				print 'Experiment Title', exp_title
				print "FAKE NUM ITEM", uu
				print "camo_vote_size_multiple", camo_vote_size_multiple
				params.num_fake_item = uu
				params.camo_vote_size_multiple = camo_vote_size_multiple
				whole_process(params)
				# evaluation(params)
				