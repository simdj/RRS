
import numpy as np
# from sklearn.cross_validation import train_test_split
from collections import Counter
try:
	import tensorflow as tf
except:
	print 'fail to import tensorflow'

# input: review data R(user,item,rating,helpful)
# output: U,V

class WMF():
	def __init__(self, params):
		self.params = params
		# self.log_path = params.intermediate_dir_path+'_'.join(map(str,[params.algorithm_model,params.attack_flag,params.rank,params.lda,params.max_iter]))
		self.log_path = '/tmp/recommender_log'
		# print(self.log_path)

		self.rank = params.rank
		self.lda = params.lda
		self.max_iter = params.max_iter

		self.U_path = params.U_path
		self.V_path = params.V_path

		self.train_data = params.train_data

	def do_mf(self, train_data, mean_rating=0, max_iter=20001, lr=0.01, decay_lr=True):
		# input : rating data [user, item, rating, helpful]
		# output : final_cost, final_rmse_tr, final_U, final_V

		# tf.reset_default_graph()

		# get stats of train data
		num_users = len(np.unique(train_data[:,0]))
		num_items = len(np.unique(train_data[:,1]))
		global_review_mean = np.mean(train_data[:,2])
		good_mean=np.sqrt(global_review_mean / self.rank)

		# initialize latent factor matrix
		W = tf.Variable(tf.truncated_normal([num_users, self.rank], stddev=0.005, mean=good_mean, seed=7), name="users")
		H = tf.Variable(tf.truncated_normal([self.rank, num_items], stddev=0.005, mean=good_mean, seed=7), name="items")
		regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), self.lda, name="regularize")
		
		user_indices = train_data[:, 0]
		item_indices = train_data[:, 1]
		rating_observed = train_data[:, 2]
		num_review = len(item_indices)
		
		helpful_values = train_data[:, 3]
		HELPFUL = tf.to_float(tf.reshape(tf.constant(helpful_values),[-1]))
		# print("helpful value square!!")
		# HELPFUL = tf.to_float(tf.reshape(tf.constant(np.square(helpful_values)),[-1]))
		
		
		# prediction = W*H	
		result = tf.matmul(W, H)
		# sparse format
		rating_prediction = tf.gather(tf.reshape(result, [-1]), user_indices * tf.shape(result)[1] + item_indices)

		# using global_review_mean
		# diff_op = tf.mul(tf.sub(tf.add(rating_prediction, mean_rating, name="add_mean"), rating_observed, name="raw_training_error"), helpful_values, name="weighted_training_error")
		# diff_op_val = tf.sub(tf.add(result_values_val, mean_rating, name="add_mean_val"), rating_observed_test, name="raw_validation_error")

		# using helpful info
		diff_op = tf.square(tf.sub(rating_prediction,rating_observed), name="squared_training_error")

		with tf.name_scope("training_cost") as scope:
			# print("yes helpful")
			weighted_diff_op = tf.mul(diff_op, HELPFUL, name='square_training_error_multiplied_by_helpful')
			base_cost = tf.reduce_sum(weighted_diff_op, name="sum_squared_error")
			# print("no helpful")
			# base_cost = tf.reduce_sum(diff_op)


			# (prediction error + regulaization) / num_review????
			# cost = tf.div(tf.add(base_cost, regularizer), num_review * 2, name="average_error")
			
			cost = tf.add(base_cost, regularizer, name="total_cost")
			# cost_summ = tf.scalar_summary("cost_summary", cost)

		with tf.name_scope("train") as scope:
			if decay_lr:
				global_step = tf.Variable(0, trainable=False)
				learning_rate = tf.train.exponential_decay(lr, global_step, 5000, 0.96, staircase=True)
				optimizer = tf.train.AdamOptimizer(learning_rate)
				# optimizer = tf.train.FtrlOptimizer(learning_rate)
				# optimizer = tf.train.RMSPropOptimizer(learning_rate)
				# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
				train_step = optimizer.minimize(cost, global_step=global_step)
			else:
				optimizer = tf.train.AdamOptimizer(lr)
				# optimizer = tf.train.FtrlOptimizer(lr)
				# optimizer = tf.train.RMSPropOptimizer(lr)
				# optimizer = tf.train.GradientDescentOptimizer(lr)
				train_step = optimizer.minimize(cost)

		with tf.name_scope("training_rmse") as scope:
			rmse_tr = tf.sqrt(tf.div(tf.reduce_sum(tf.square(diff_op)), num_review))
		
		# # Make sure summaries get written to the logs.
		# summary_op = tf.merge_all_summaries()

		# ===========================================
		# ==============Session started==============
		sess = tf.Session()
		sess.run(tf.initialize_all_variables())

		# writer = tf.train.SummaryWriter(self.log_path, sess.graph)

		last_cost = 0
		diff = 1

		for i in range(max_iter):
			sess.run(train_step)
			if i > 0 and i % 1000 == 0:
				res = sess.run([cost, rmse_tr])
				cost_ev = res[0]
				rmse_tr_ev = res[1]
				# summary_str = res[2]
				diff = abs(cost_ev - last_cost)
				last_cost = cost_ev

				# writer.add_summary(summary_str,i)
				if i % 10000 == 0:
					print("At step %s) Cost %s / RMSE %s " % (i, cost_ev, rmse_tr_ev))
				
				if diff < 0.00001:
					print("Converged at iteration %s" % (i))
					break
		final_cost = cost.eval(session=sess)
		final_rmse_tr = rmse_tr.eval(session=sess)
		final_U = W.eval(session=sess)
		final_V = H.eval(session=sess)
		sess.close()
		# ==============Session finished==============
		# ============================================

		print("Final) Cost %s / RMSE %s" % (final_cost, final_rmse_tr))

		return final_rmse_tr, final_U, final_V

	def whole_process(self):
		# do WMF
		final_error, final_U, final_V = self.do_mf(self.train_data, max_iter=self.max_iter, lr=0.01, decay_lr=True)
		np.save(self.U_path, final_U)
		np.save(self.V_path, final_V)
		print("U path: ", self.U_path)
		return final_U, final_V


class WMF_params():
	def __init__(self, params, algorithm_model='base', attack_flag=False):
		# rank, lda, max_iter
		# U_path, V_path
		# input_review_path_list, input_helpful_path_list
		self.params = params

		self.algorithm_model = algorithm_model
		self.attack_flag = attack_flag
		
		self.rank = params.rank
		self.lda = params.lda
		self.max_iter = params.max_iter

		self.target_item_list_path = params.target_item_list_path
		self.fake_user_id_list_path = params.fake_user_id_list_path

		self.intermediate_dir_path = params.intermediate_dir_path

		self.review_origin_path = ''
		self.review_fake_path = ''
		self.review_camo_path = ''

		self.helpful_origin_path = ''
		self.helpful_fake_path = ''
		self.helpful_camo_path = ''

		# set review, helpful and output path 
		self.set_input_output_path(params, algorithm_model, attack_flag)
		# loading and generate rating dataset [user item rating helpful]
		
		self.train_data = self.loading_data(algorithm_model)
		# self.train_data = self.loading_data_test(algorithm_model)
	def get_robust_weight(self, robust_scale=2):
		naive_list = [self.params.helpful_origin_attacked_naive, self.params.helpful_fake_attacked_naive, self.params.helpful_camo_attacked_naive]
		robust_list = [self.params.helpful_origin_attacked_robust, self.params.helpful_fake_attacked_robust, self.params.helpful_camo_attacked_robust]

		naive_helpful_vector = np.concatenate([np.load(data_path) for data_path in naive_list])[:,-1]
		robust_helpful_vector = np.concatenate([np.load(data_path) for data_path in robust_list])[:,-1]
		# user,item,helpful
		# print("naive_helpful_vector and robust_helpful_vector shape", naive_helpful_vector.shape, robust_helpful_vector.shape)

		# helpful vector
		diff_helpful = robust_helpful_vector-naive_helpful_vector
		# if robust helpful< naive helpful -> penalty!!: *exp(-a)
		robust_weight = naive_helpful_vector*np.exp(diff_helpful*robust_scale)
		return robust_weight


	def loading_data(self, algorithm_model='base'):
		review_data = [np.load(data_path) for data_path in self.input_review_path_list]
		fake_start = len(review_data[0])
		review_data = np.concatenate(review_data)
		
		if algorithm_model=='base':
			self.train_data = self.merge_review_helpful(review_data, 1)
		elif algorithm_model=='naive':
			helpful_matrix = [np.load(data_path) for data_path in self.input_helpful_path_list]
			# honest_num = len(helpful_matrix[0])
			# fake_num = len(helpful_matrix[1])
			# print '!!!! Worst in naive'
			# helpful_matrix[1]*=10
			# print self.input_helpful_path_list[1], ' np.mean(helpful_matrix[1][:,-1])', np.mean(helpful_matrix[1][:,-1])
			helpful_vector = np.concatenate(helpful_matrix)[:,-1]


			# 0~1
			# helpful_vector /= 5
			self.train_data = self.merge_review_helpful(review_data, helpful_vector)
			
		elif algorithm_model=='robust':
			# helpful_vector = [np.load(data_path) for data_path in self.input_helpful_path_list]
			# # honest_num = len(helpful_vector[0])
			# # fake_num = len(helpful_vector[1])
			# # print '!!!! Best in robust'
			# # helpful_vector[1]*=0.1
			# # # # helpful_vector[1]*=0
			# # print self.input_helpful_path_list[1], ' np.mean(helpful_vector[1][:,-1])', np.mean(helpful_vector[1][:,-1])
			# helpful_vector = np.concatenate(helpful_vector)[:,-1]
			# # 0~1
			# # helpful_vector /= 5

			helpful_vector = self.get_robust_weight()
			# print np.mean(helpful_vector[:fake_start-10]), np.percentile(helpful_vector[:fake_start-10],25), np.percentile(helpful_vector[:fake_start-10],50), np.percentile(helpful_vector[:fake_start-10], 75)
			# print np.mean(helpful_vector[fake_start:fake_start+50]), np.percentile(helpful_vector[fake_start:fake_start+50],25), np.percentile(helpful_vector[fake_start:fake_start+50],50), np.percentile(helpful_vector[fake_start:fake_start+50], 75)
			self.train_data = self.merge_review_helpful(review_data, helpful_vector)
		
		return self.train_data


	def merge_review_helpful(self, review, helpful):
		# review : user,item,rating,reveiw_id), 
		# helpful : scalar -> np.ones() / vector([helpfulness values])
		ret = []
		if type(helpful)==type(1):
			helpful = np.ones((len(review),1))
		assert (len(review) == len(helpful))
		num_review = len(review)
		for i in xrange(num_review):
			tmp_review_with_helpful = list(review[i][:-1])
			tmp_review_with_helpful.append(helpful[i])
			ret.append(tmp_review_with_helpful)

		ret = np.array(ret)
		# print('merge result', ret.shape)
		return ret

	def set_input_output_path(self, params, algorithm_model, attack_flag):
		self.review_origin_path = params.review_origin_path
		
		if attack_flag == True:
			self.review_fake_path = params.review_fake_path
			self.review_camo_path = params.review_camo_path

		if algorithm_model == 'base' and attack_flag == False:  # base/clean
			# output
			self.U_path = params.base_U_clean_path
			self.V_path = params.base_V_clean_path
			# no helpful use
		elif algorithm_model == 'base' and attack_flag == True:  # base/attacked
			# output
			self.U_path = params.base_U_attacked_path
			self.V_path = params.base_V_attacked_path
			# no helpful use
		elif algorithm_model == 'naive' and attack_flag == False:  # naive/clean
			# output
			self.U_path = params.naive_U_clean_path
			self.V_path = params.naive_V_clean_path
			# review and helpful
			self.helpful_origin_path = params.helpful_origin_clean_naive_path
		elif algorithm_model == 'naive' and attack_flag == True:  # naive/attacked
			# output
			self.U_path = params.naive_U_attacked_path
			self.V_path = params.naive_V_attacked_path
			# review and helpful
			self.helpful_origin_path = params.helpful_origin_attacked_naive
			self.helpful_fake_path = params.helpful_fake_attacked_naive
			self.helpful_camo_path = params.helpful_camo_attacked_naive
		elif algorithm_model == 'robust' and attack_flag == False:  # robust/clean
			# output
			self.U_path = params.robust_U_clean_path
			self.V_path = params.robust_V_clean_path
			# review and helpful
			self.helpful_origin_path = params.helpful_origin_clean_robust_path
		elif algorithm_model == 'robust' and attack_flag == True:  # robust/attacked
			# output
			self.U_path = params.robust_U_attacked_path
			self.V_path = params.robust_V_attacked_path
			# review and helpful
			self.helpful_origin_path = params.helpful_origin_attacked_robust
			self.helpful_fake_path = params.helpful_fake_attacked_robust
			self.helpful_camo_path = params.helpful_camo_attacked_robust

		self.input_review_path_list = [self.review_origin_path, self.review_fake_path, self.review_camo_path]
		self.input_review_path_list = filter(lambda x: x!='', self.input_review_path_list)
		
		self.input_helpful_path_list = [self.helpful_origin_path, self.helpful_fake_path, self.helpful_camo_path]
		self.input_helpful_path_list = filter(lambda x: x!='', self.input_helpful_path_list)

class metric():
	def __init__(self, params):
		self.review_origin_path = params.review_origin_path
		self.review_origin = np.load(self.review_origin_path)
		try:
			self.review_fake_path = params.review_fake_path
			self.review_fake = np.load(self.review_fake_path)
		except:
			# print("no attack")
			pass
		# self.review_fake_path = params.default_review_fake_path


		self.target_item_list = np.load(params.target_item_list_path)
		self.fake_user_id_list = np.load(params.fake_user_id_list_path)
		# self.get_honest_user_list = np.array(list(range(np.min(self.fake_user_id_list))))

		self.U_path = params.U_path
		self.V_path = params.V_path

	def get_honest_user_list(self):
		return np.unique(self.review_origin[:,0])

	def get_honest_high_degree_user_list(self, num_user=100):
		review_data = np.load(self.review_origin_path)
		user_counter = Counter(review_data[:,0])
		return map(lambda x:x[0],user_counter.most_common(num_user))

	def rmse_rating_on_target(self, honest=True):
		selected_review = self.review_origin
		if not honest:
			try:
				selected_review = self.review_fake
			except:
				return -1.0

		rating_on_target = []
		for target_item in self.target_item_list:
			tmp=selected_review[selected_review[:,1]==target_item,:]
			rating_on_target.append(tmp)
		rating_on_target = np.concatenate(rating_on_target)

		U_matrix = np.load(self.U_path)
		V_matrix = np.load(self.V_path)  # V_matrix.shape = (rank,I)

		rmse = 0.0
		for rating_row in rating_on_target:
			user_id = int(rating_row[0])
			target_item = int(rating_row[1])

			rating_observed = rating_row[2]
			rating_prediction = np.matmul(U_matrix[user_id,:], V_matrix[:,target_item])

			rmse+=np.square(rating_observed-rating_prediction)
		rmse=np.sqrt(rmse/float(len(rating_on_target)))
		return rmse


	def mean_prediction_rating_on_target(self, honest=True):
		U_matrix = np.load(self.U_path)
		V_matrix = np.load(self.V_path)  # V_matrix.shape = (rank,I)
		
		# (user,target_item)
		if honest:
			focus_user_list = self.get_honest_user_list()
			# focus_user_list = self.get_honest_high_degree_user_list(100)
		else:
			focus_user_list = self.fake_user_id_list
		target_item_list = self.target_item_list

		target_U_matrix = U_matrix[map(int,focus_user_list),:]
		target_V_matrix = V_matrix[:,map(int, target_item_list)]
		matmul_result = np.matmul(target_U_matrix, target_V_matrix)
		# average
		# each_overall_rating = np.mean(matmul_result, axis=0)
		total_overall_rating = np.mean(matmul_result)
		print('[RAW Rating Distribution]',np.min(matmul_result), np.percentile(matmul_result,25),np.median(matmul_result),np.percentile(matmul_result,75), np.max(matmul_result), total_overall_rating)

		# limit
		matmul_result[matmul_result<1]=1
		matmul_result[matmul_result>5]=5

		total_overall_rating = np.mean(matmul_result)
		print('[Limit Rating Distribution]',np.min(matmul_result), np.percentile(matmul_result,25),np.median(matmul_result),np.percentile(matmul_result,75), np.max(matmul_result), total_overall_rating)

		return total_overall_rating

# if __name__=="__main__":
# 	from parameter_controller import *
# 	# exp_title_list = ['bandwagon_1%_1%_1%_emb_32','bandwagon_2%_1%_2%_emb_32','bandwagon_3%_1%_3%_emb_32','bandwagon_4%_1%_4%_emb_32','bandwagon_5%_1%_5%_emb_32'   ]
# 	exp_title_list = ['bandwagon_1%_1%_1%_emb_32']
	
# 	# algorithm_model_list = ['base','naive','robust']
# 	algorithm_model_list = ['robust']
# 	# lda_list = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
# 	# rank_list = [10,20,30,40,50,60,70,80,100]
# 	lda_list = [0.001, 0.01, ]
# 	rank_list = [30,40]


# 	for exp_title in exp_title_list:
# 		print('Experiment Title', exp_title)
# 		params = parse_exp_title(exp_title)

# 		for rank in rank_list:
# 			for lda in lda_list:
# 				for am in algorithm_model_list:
# 					for af in [False,True]:
# 						wp = WMF_params(params=params, algorithm_model=am, attack_flag=af)
# 						# wp = WMF_params(params=params, algorithm_model=am, attack_flag=True)
# 						wp.lda = lda
# 						wp.rank = rank
# 						wp.max_iter=10001
# 						wmf_instance = WMF(params=wp)
# 						wmf_instance.whole_process()
# 						# In wp, U and V path are specified.
# 						performance = metric(params=wp)
# 						print('-----------------------','algorithm:',am, 'rank',rank, 'lda', lda, '---------------------')

# 						try:
# 							origin_help = np.load(wp.helpful_origin_path)[:,-1]
# 							fake_help = np.load(wp.helpful_fake_path)[:,-1]
# 							print ('[Helpfulness distribution]',np.percentile(origin_help,10),np.percentile(origin_help,50),np.percentile(origin_help,90),np.mean(fake_help))
# 						except:
# 							pass

# 						# print('(fake)', performance.mean_prediction_rating_on_target(honest=False))
# 						print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Important value [honest] ', performance.mean_prediction_rating_on_target(honest=True),'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# 						print('')
# 	print("++++++++++++++++++++++++++++++++++++++++++++")
# 	print("++++++++++++++++++++++++++++++++++++++++++++")
# 	print("end")
# 	print("++++++++++++++++++++++++++++++++++++++++++++")
# 	print("++++++++++++++++++++++++++++++++++++++++++++")
