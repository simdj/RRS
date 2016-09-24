
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from collections import Counter

class matrix_factorization():
	def __init__(self, params, algorithm_model='robust', attack_flag=True):
		self.rank = params.rank
		self.lda = params.lda
		self.max_iter = params.max_iter

		self.attack_flag = attack_flag
		self.algorithm_model = algorithm_model

		# review
		self.review_origin_path = params.review_origin_path
		self.review_fake_path = params.review_fake_path
		self.review_camo_path = params.review_camo_path

		# helpfulness
		self.helpful_origin_path = ''
		self.helpful_fake_path = ''
		self.helpful_camo_path = ''

		# output
		self.U_path = ''
		self.V_path = ''

		if algorithm_model == 'base' and attack_flag == False:  # base/clean
			# no helpful use
			# output
			self.U_path = params.base_U_clean_path
			self.V_path = params.base_V_clean_path
		elif algorithm_model == 'base' and attack_flag == True:  # base/attacked
			# no helpful use
			# output
			self.U_path = params.base_U_attacked_path
			self.V_path = params.base_V_attacked_path
		elif algorithm_model == 'naive' and attack_flag == False:  # naive/clean
			self.helpful_origin_path = params.helpful_origin_clean_naive_path
			# output
			self.U_path = params.naive_U_clean_path
			self.V_path = params.naive_V_clean_path
		elif algorithm_model == 'naive' and attack_flag == True:  # naive/attacked
			self.helpful_origin_path = params.helpful_origin_attacked_naive
			self.helpful_fake_path = params.helpful_fake_attacked_naive
			self.helpful_camo_path = params.helpful_camo_attacked_naive
			# output
			self.U_path = params.naive_U_attacked_path
			self.V_path = params.naive_V_attacked_path
		elif algorithm_model == 'robust' and attack_flag == False:  # robust/clean
			self.helpful_origin_path = params.helpful_origin_clean_robust_path
			# output
			self.U_path = params.robust_U_clean_path
			self.V_path = params.robust_V_clean_path
		elif algorithm_model == 'robust' and attack_flag == True:  # robust/attacked
			self.helpful_origin_path = params.helpful_origin_attacked_robust
			self.helpful_fake_path = params.helpful_fake_attacked_robust
			self.helpful_camo_path = params.helpful_camo_attacked_robust
			# output
			self.U_path = params.robust_U_attacked_path
			self.V_path = params.robust_V_attacked_path

		self.train_data_path = params.train_data_path
		self.test_target_data_path = params.test_target_data_path
		self.test_overall_data_path = params.test_overall_data_path

	# Extracts user indices, item indices, rating values, helpfulness score and number of ratings from the ratings row
	def extract_rating_info(self, ratings):
		user_indices = ratings[:, 0]
		item_indices = ratings[:, 1]
		rating_values = ratings[:, 2]
		helpful_values = ratings[:, 3]
		num_ratings = len(item_indices)
		return user_indices, item_indices, rating_values, helpful_values, num_ratings

	# Given a set of ratings, 2 matrix factors that include one or more
	# trainable variables, and a regularizer, uses gradient descent to
	# learn the best values of the trainable variables.
	def do_mf(self, review_train, review_test, target_item_review_test, W, H, regularizer, mean_rating, max_iter, lr=0.01, decay_lr=False,log_summaries=False):
		# Extract info from training and validation data
		user_indices_train, item_indices_train, rating_values_train, helpful_values_train, num_review_train = self.extract_rating_info(review_train)
		user_indices_test, item_indices_test, rating_values_test, helpful_values_test, num_review_test = self.extract_rating_info(review_test)

		user_indices_target_test, item_indices_target_test, rating_values_target_test, helpful_values_target_test, num_review_target_test = self.extract_rating_info(target_item_review_test)

		# Multiply the factors to get our result as a dense matrix
		result = tf.matmul(W, H)

		# Now we just want the values represented by the pairs of user and item
		# indices for which we had known ratings.
		result_values_tr = tf.gather(tf.reshape(result, [-1]),
		                             user_indices_train * tf.shape(result)[1] + item_indices_train,
		                             name="extract_training_ratings")
		result_values_val = tf.gather(tf.reshape(result, [-1]),
		                              user_indices_test * tf.shape(result)[1] + item_indices_test,
		                              name="extract_validation_ratings")

		result_values_target_test = tf.gather(tf.reshape(result, [-1]),
		                              user_indices_target_test * tf.shape(result)[1] + item_indices_target_test,
		                              name="extract_validation_ratings")

		# Calculate the difference between the predicted ratings and the actual
		# ratings. The predicted ratings are the values obtained form the matrix
		# multiplication with the mean rating added on.

		# using global_review_mean
		# diff_op = tf.mul(tf.sub(tf.add(result_values_tr, mean_rating, name="add_mean"), rating_values_train, name="raw_training_error"), helpful_values_train, name="weighted_training_error")
		# diff_op_val = tf.sub(tf.add(result_values_val, mean_rating, name="add_mean_val"), rating_values_test, name="raw_validation_error")

		# using helpful
		# diff_op = tf.mul(tf.sub(result_values_tr, rating_values_train), helpful_values_train, name="weighted_training_error")
		diff_op = tf.sub(result_values_tr,rating_values_train, name="weighted_training_error")

		diff_op_val = tf.sub(result_values_val, rating_values_test, name="raw_validation_error")

		

		with tf.name_scope("training_cost") as scope:
			base_cost = tf.reduce_sum(tf.square(diff_op, name="squared_difference"), name="sum_squared_error")
			# (prediction error + regulaization) / num_review????
			# cost = tf.div(tf.add(base_cost, regularizer), num_review_train * 2, name="average_error")
			cost = tf.add(base_cost, regularizer, name="total_cost")

		with tf.name_scope("train") as scope:
			if decay_lr:
				# Use an exponentially decaying learning rate.
				global_step = tf.Variable(0, trainable=False)
				learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)
				# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
				optimizer = tf.train.AdamOptimizer(learning_rate)
				# Passing global_step to minimize() will increment it at each step
				# so that the learning rate will be decayed at the specified
				# intervals.
				train_step = optimizer.minimize(cost, global_step=global_step)
			else:
				# optimizer = tf.train.GradientDescentOptimizer(lr)
				optimizer = tf.train.AdamOptimizer(lr)
				train_step = optimizer.minimize(cost)

		with tf.name_scope("training_rmse") as scope:
			rmse_tr = tf.sqrt(tf.div(tf.reduce_sum(tf.square(tf.sub(result_values_tr, rating_values_train))), num_review_train))

		with tf.name_scope("validation_rmse") as scope:
			# Validation set rmse:
			rmse_val = tf.sqrt(tf.div(tf.reduce_sum(tf.square(tf.sub(result_values_val, rating_values_test))), num_review_test))

		with tf.name_scope("target_test_rmse") as scope:
			# Validation set rmse:
			rmse_target_test = tf.sqrt(tf.div(tf.reduce_sum(tf.square(tf.sub(result_values_target_test, rating_values_target_test))), num_review_target_test))

		with tf.name_scope("mean_rmse") as scope:
			diff_op_val_mean = tf.sub(mean_rating, rating_values_test, name="raw_validation_error")
			rmse_val_mean = tf.sqrt(tf.div(tf.reduce_sum(tf.square(diff_op_val_mean)), num_review_test))

		# Create a TensorFlow session and initialize variables.
		sess = tf.Session()
		sess.run(tf.initialize_all_variables())

		# if log_summaries:
		# 	# Make sure summaries get written to the logs.
		# 	# accuracy_val_summary = tf.scalar_summary("accuracy_val", accuracy_val)
		# 	# accuracy_tr_summary = tf.scalar_summary("accuracy_tr", accuracy_tr)
		# 	summary_op = tf.merge_all_summaries()
		# 	writer = tf.train.SummaryWriter("/tmp/recommender_logs", sess.graph_def)
		# 	pass
		# Keep track of cost difference.
		last_cost = 0
		diff = 1
		# Run the graph and see how we're doing on every 1000th iteration.
		for i in range(max_iter):
			if i > 0 and i % 100 == 0:
				if diff < 0.000001:
					print("Converged at iteration %s" % (i))
					break
				# if log_summaries:
				# 	res = sess.run([rmse_tr, rmse_val, cost, summary_op])
				# 	summary_str = res[3]
				# 	writer.add_summary(summary_str, i)
				# else:
				# 	res = sess.run([rmse_tr, rmse_val, cost])
				res = sess.run([rmse_tr, rmse_val, rmse_target_test, cost])
				acc_tr = res[0]
				acc_val = res[1]
				acc_target_test = res[2]
				cost_ev = res[3]
				# print("Training RMSE at step %s: %s" % (i, acc_tr), "Validation RMSE at step %s: %s" % (i, acc_val), "Target test RMSE at step %s: %s" % (i, acc_target_test))
				if i % 1000 == 0:
					print("At step %s) Training , Overall test, Target test: (%s, %s, %s) " % (i, acc_tr, acc_val, acc_target_test))
				if i>1000 and acc_val>10:
					print("may divergence")
					# break
				diff = abs(cost_ev - last_cost)
				last_cost = cost_ev
			else:
				sess.run(train_step)

		finalTrain = rmse_tr.eval(session=sess)
		finalVal = rmse_val.eval(session=sess)
		final_target_test = rmse_target_test.eval(session=sess)
		finalW = W.eval(session=sess)
		finalH = H.eval(session=sess)
		
		print("RMSE of mean rating %s" % sess.run(rmse_val_mean))
		# 1.07341...

		sess.close()
		return finalTrain, finalVal, final_target_test, finalW, finalH

	# Learns factors of the given rank with specified regularization parameter.
	def initialize_latent_factor_matrix(self, num_users, num_items, rank, lda, good_mean=0.28):
		# Initialize the matrix factors from random normals with mean 0. W will
		# represent users and H will represent items.
		# np.sqrt(np.mean(ratings[:,2])/rank)
		W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.02, mean=good_mean), name="users")
		H = tf.Variable(tf.truncated_normal([rank, num_items], stddev=0.02, mean=good_mean), name="items")
		regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), lda, name="regularize")
		return W, H, regularizer

	def get_target_item_list(self):
		return np.unique(np.load(self.review_fake_path)[:, 1])

	def get_heavy_user_list(self, review_data, threshold =20):
		# # [(user_id, occurence)*]
		# # [(944, 212), (912, 156), (1436, 135), (2277, 126), (2577, 124), (2797, 115)]
		user_counter = Counter(review_data[:,0])
		ret = []
		for user_id, occurence in user_counter.iteritems():
			if occurence >= threshold:
				ret.append(user_id)
		return ret
	def split_heavy_user_data(self, review_data):
		heavy_user_list = self.get_heavy_user_list(review_data)
		print('heavy user number', len(heavy_user_list))
		heavy_user_data = []
		etc_user_data = []
		for row in review_data:
			if row[0] in heavy_user_list:
				heavy_user_data.append(row)
			else:
				etc_user_data.append(row)
		return np.array(heavy_user_data), np.array(etc_user_data)

	def split_target_split(self, review_origin, helpful_origin):
		target_item_review = []
		etc_review = []
		# [user, item, rating, helpfulness]
		data = self.merge_review_helpful(review_origin, helpful_origin)

		target_item_list = self.get_target_item_list()

		num_review = len(data)
		for i in xrange(num_review):
			if data[i][1] in target_item_list:
				target_item_review.append(data[i])
			else:
				etc_review.append(data[i])
		target_item_review = np.array(target_item_review)
		etc_review = np.array(etc_review)

		target_item_review_train, target_item_review_test = train_test_split(target_item_review, train_size=.9)

		# etc_review_train, etc_review_test = train_test_split(etc_review, train_size=int(0.9*len(etc_review)))
		# return np.concatenate((target_item_review_train, etc_review_train)), etc_review_test, target_item_review_test

		heavy_user_data, etc_user_data = self.split_heavy_user_data(etc_review)
		train_size = int(min(len(heavy_user_data), 0.9*len(etc_review)))
		heavy_review_train, heavy_review_test = train_test_split(heavy_user_data, train_size=.9)
		if train_size>=len(heavy_user_data):
			print(" heavy user data threshold is too high")
		print('data size')
		print len(target_item_review_train), len(heavy_review_train), len(etc_user_data), len(heavy_review_test), len(target_item_review_test)

		return np.concatenate((target_item_review_train, heavy_review_train, etc_user_data)), heavy_review_test, target_item_review_test


	def merge_review_helpful(self, review, helpful):
		ret = []
		assert (len(review) == len(helpful))
		num_review = len(review)
		for i in xrange(num_review):
			tmp_review_with_helpful = list(review[i][:-1])
			tmp_review_with_helpful.append(helpful[i])
			ret.append(tmp_review_with_helpful)
		ret = np.array(ret)
		# print('merge result', ret.shape)
		return ret

	def whole_process(self):
		# np.random.seed(1)
		# loading original review
		review_origin = np.load(self.review_origin_path)
		# loading original helpful
		if self.algorithm_model == 'base':
			helpful_origin = np.ones((len(review_origin),))
		else:
			helpful_origin = np.load(self.helpful_origin_path)[:, 2]
		# loading fake/camo review
		if self.attack_flag:
			review_fake = np.load(self.review_fake_path)
			review_camo = np.load(self.review_camo_path)
		# loading fake/camo helpful
		if self.algorithm_model == 'base' and self.attack_flag:
			helpful_fake = np.ones((len(review_fake),))
			helpful_camo = np.ones((len(review_camo),))
		if self.algorithm_model != 'base' and self.attack_flag:
			helpful_fake = np.load(self.helpful_fake_path)[:, 2]
			helpful_camo = np.load(self.helpful_camo_path)[:, 2]

		# each one has a row such that [user, item, rating, helpfulness]
		overall_review_train, overall_review_test, target_item_review_test = self.split_target_split(review_origin, helpful_origin)

		# append overall_review_train
		if self.attack_flag:
			overall_review_train = np.concatenate((overall_review_train,
			                                       self.merge_review_helpful(review_fake, helpful_fake),
			                                       self.merge_review_helpful(review_camo, helpful_camo)))

		np.save(self.train_data_path, overall_review_train)
		np.save(self.test_overall_data_path, overall_review_test)
		np.save(self.test_target_data_path, target_item_review_test)

		every_review = np.concatenate((overall_review_train, overall_review_test, target_item_review_test))
		num_users = len(np.unique(every_review[:, 0]))
		num_items = len(np.unique(every_review[:, 1]))
		num_reviews = len(every_review)

		global_review_mean = np.mean(overall_review_train[:, 2])
		print('rating matrix size', num_users, num_items, '# of reviews', num_reviews)
		
		W, H, reg = self.initialize_latent_factor_matrix(num_users=num_users, num_items=num_items, rank=self.rank, lda=self.lda, good_mean=np.sqrt(global_review_mean / self.rank))

		# do_mf(self, review_train, review_test, target_item_review_test, W, H, regularizer, mean_rating, max_iter, lr=0.01, decay_lr=False, log_summaries=False):
		tr, val,final_target_test, finalw, finalh = self.do_mf(overall_review_train, overall_review_test, target_item_review_test, W, H, reg, global_review_mean, max_iter=self.max_iter, lr=0.01, decay_lr=True)

		print("Final training RMSE %s" % (tr), "\tFinal validation RMSE %s" % (val))
		np.save(self.U_path, finalw)
		np.save(self.V_path, finalh)
		print(self.U_path, "and", self.V_path, "saved")
		return tr, val,final_target_test

	def small_test(self, num=10, which_test='overall'):
		print("Test : ", which_test)
		if which_test=='target_test':
			test_data = np.load(self.test_target_data_path)
		else:
			test_data = np.load(self.test_overall_data_path)

		for i in xrange(num):
			user = test_data[i][0]
			item = test_data[i][1]
			U=np.load(self.U_path)
			V=np.load(self.V_path)

			real_value = test_data[i][2]
			prediction = np.dot(U[user,:], V[:,item])

			print("real , prediction, diff: (%s, %s, %s) " % (real_value, prediction, real_value-prediction))
			if np.abs(prediction)>5:
				print test_data[i]

	def overall_rating_of_target(self):
		# target_list, U_matrix, V_matrix
		target_list = self.get_target_item_list()


		U_matrix=np.load(self.U_path)
		V_matrix=np.load(self.V_path)

		# V_matrix.shape = (rank,I)
		# print('??????',target_list)
		target_V_matrix = V_matrix[:,map(int, target_list)]
		# (user,target_item)
		matmul_result = np.matmul(U_matrix, target_V_matrix)
		# average
		each_overall_rating = np.mean(matmul_result, axis=0)
		total_overall_rating = np.mean(matmul_result)
		return each_overall_rating, total_overall_rating

	def measure_performance(self, iteration=10):
		train_err_list = []
		test_err_list = []
		target_test_err_list = []
		overall_rating_list = []
		for i in xrange(iteration):
			train_err, test_err, target_test_err = self.whole_process()
			overall_rating = self.overall_rating_of_target()
			train_err_list.append(train_err)
			test_err_list.append(test_err)
			target_test_err_list.append(target_test_err)
			overall_rating_list.append(overall_rating)
		print("overall test error", np.mean([train_err_list, test_err_list,target_test_err_list, overall_rating_list], axis=1))
		return [train_err_list, test_err_list,target_test_err_list, overall_rating_list]



if __name__ == "__main__":
	from parameter_controller import *
	exp_title = 'bandwagon_1%_1%_1%_emb_32'
	params = parse_exp_title(exp_title)

	params.max_iter=3001
	print("=========================================")
	print("1. base / clean")
	mf = matrix_factorization(params=params, algorithm_model='base', attack_flag=False)
	mf.whole_process()
	mf.small_test(num=10, which_test='target_test')
	mf.small_test(num=10, which_test='overall')

	print("=========================================")
	print("2. base / attacked")
	mf = matrix_factorization(params=params, algorithm_model='base', attack_flag=True)
	mf.whole_process()
	mf.small_test(num=10, which_test='target_test')
	mf.small_test(num=10, which_test='overall')

	print("=========================================")
	print("3. naive / clean")
	mf = matrix_factorization(params=params, algorithm_model='naive', attack_flag=False)
	mf.whole_process()
	mf.small_test(num=10, which_test='target_test')
	mf.small_test(num=10, which_test='overall')

	print("=========================================")
	print("4. naive / attacked")
	mf = matrix_factorization(params=params, algorithm_model='naive', attack_flag=True)
	mf.whole_process()
	mf.small_test(num=10, which_test='target_test')
	mf.small_test(num=10, which_test='overall')

	print("=========================================")
	print("5. robust / clean")
	mf = matrix_factorization(params=params, algorithm_model='robust', attack_flag=False)
	mf.whole_process()
	mf.small_test(num=10, which_test='target_test')
	mf.small_test(num=10, which_test='overall')


	print("=========================================")
	print("6. robust / attacked")
	mf = matrix_factorization(params=params, algorithm_model='robust', attack_flag=True)
	mf.whole_process()
	mf.small_test(num=10, which_test='target_test')
	mf.small_test(num=10, which_test='overall')
