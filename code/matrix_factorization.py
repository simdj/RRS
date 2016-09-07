# 5. MF(helpfulness*ratings)
# 	Input
# 		origin review 	<-- './intermediate/review.npy'
# 		fake review 	<-- './intermediate/fake_review_[average|bandwagon].npy'
# 		camo review 	<-- './intermediate/camo_review_[average|bandwagon].npy'

# 		origin helpful 	<-- './intermediate/helpful.npy'
# 		fake helpful 	<-- './intermediate/fake_helpful.npy'
# 		camo helpful 	<-- './intermediate/camo_helpful.npy'
		

# 	Output
# 		user_latent --> './output/user_latent.npy'
# 		item_latent --> './output/item_latent.npy'
from time import time
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split

class matrix_factorization():
	def __init__(self, rank=50, lda=1, max_iter=5001
		,review_numpy_path = './intermediate/review.npy'
		,fake_review_numpy_path = './intermediate/fake_review_bandwagon.npy'
		,camo_review_numpy_path = './intermediate/camo_review_bandwagon.npy'
		,helpful_numpy_path='./intermediate/helpful.npy'
		,fake_helpful_numpy_path='./intermediate/fake_helpful.npy'
		,camo_helpful_numpy_path='./intermediate/camo_helpful.npy'):
			
		self.rank = rank
		self.lda = lda
		self.max_iter = max_iter

		self.review_numpy_path = './intermediate/review.npy'
		self.fake_review_numpy_path = './intermediate/fake_review_bandwagon.npy'
		self.camo_review_numpy_path = './intermediate/camo_review_bandwagon.npy'

		self.helpful_numpy_path='./intermediate/helpful.npy'
		self.fake_helpful_numpy_path='./intermediate/fake_helpful.npy'
		self.camo_helpful_numpy_path='./intermediate/camo_helpful.npy'

	# Given a set of ratings, 2 matrix factors that include one or more
	# trainable variables, and a regularizer, uses gradient descent to
	# learn the best values of the trainable variables.
	def do_mf(review_train, review_test, helpful_train, helpful_test, W, H, regularizer, mean_rating, max_iter, lr = 0.01, decay_lr = False, log_summaries = False):
		# Extract info from training and validation data
		rating_values_train, num_review_train, user_indices_train, item_indices_train = self.extract_rating_info(review_train)
		rating_values_test, num_review_test, user_indices_test, item_indices_test = self.extract_rating_info(review_test)

		# sim add!!
		helpful_values_train = helpful_train[:,2]

		# Multiply the factors to get our result as a dense matrix
		result = tf.matmul(W, H)



		# Now we just want the values represented by the pairs of user and item
		# indices for which we had known ratings.
		result_values_tr = tf.gather(tf.reshape(result, [-1]), user_indices_train * tf.shape(result)[1] + item_indices_train, name="extract_training_ratings")
		result_values_val = tf.gather(tf.reshape(result, [-1]), user_indices_test * tf.shape(result)[1] + item_indices_test, name="extract_validation_ratings")

		# Calculate the difference between the predicted ratings and the actual
		# ratings. The predicted ratings are the values obtained form the matrix
		# multiplication with the mean rating added on.

		# using global_review_mean
		# diff_op = tf.mul(tf.sub(tf.add(result_values_tr, mean_rating, name="add_mean"), rating_values_train, name="raw_training_error"), helpful_values_train, name="weighted_training_error")
		# diff_op_val = tf.sub(tf.add(result_values_val, mean_rating, name="add_mean_val"), rating_values_test, name="raw_validation_error")

		# using helpful
		diff_op = tf.mul(tf.sub(result_values_tr,rating_values_train), helpful_values_train, name="weighted_training_error")

		# diff_op = tf.sub(result_values_tr,rating_values_train, name="weighted_training_error")
		diff_op_val = tf.sub(result_values_val, rating_values_test, name="raw_validation_error")

		with tf.name_scope("training_cost") as scope:
			base_cost = tf.reduce_sum(tf.square(diff_op, name="squared_difference"), name="sum_squared_error")

			cost = tf.div(tf.add(base_cost, regularizer), num_review_train * 2, name="average_error")

		with tf.name_scope("validation_cost") as scope:
			cost_val = tf.div(tf.reduce_sum(tf.square(diff_op_val, name="squared_difference_val"), name="sum_squared_error_val"), num_review_test * 2, name="average_error")

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
		  rmse_tr = tf.sqrt(tf.div(tf.reduce_sum(tf.square(diff_op)), num_review_train))

		with tf.name_scope("validation_rmse") as scope:
		  # Validation set rmse:
		  rmse_val = tf.sqrt(tf.div(tf.reduce_sum(tf.square(diff_op_val)), num_review_test))

		with tf.name_scope("mean_rmse") as scope:
			diff_op_val_mean = tf.sub(mean_rating, rating_values_test, name="raw_validation_error")
			rmse_val_mean = tf.sqrt(tf.div(tf.reduce_sum(tf.square(diff_op_val_mean)),num_review_test))


		# Create a TensorFlow session and initialize variables.
		sess = tf.Session()
		sess.run(tf.initialize_all_variables())

		if log_summaries:
			# Make sure summaries get written to the logs.
			# accuracy_val_summary = tf.scalar_summary("accuracy_val", accuracy_val)
			# accuracy_tr_summary = tf.scalar_summary("accuracy_tr", accuracy_tr)
			summary_op = tf.merge_all_summaries()
			writer = tf.train.SummaryWriter("/tmp/recommender_logs", sess.graph_def)
			pass
		# Keep track of cost difference.
		last_cost = 0
		diff = 1
		# Run the graph and see how we're doing on every 1000th iteration.
		for i in range(max_iter):
			# if i > 0 and i % 1000 == 0:
			if i > 0 and i % 100 == 0:
			# if i%10 == 0:
				if diff < 0.000001:
					print("Converged at iteration %s" % (i))
					break;
				if log_summaries:
					res = sess.run([rmse_tr, rmse_val, cost, summary_op])
					summary_str = res[3]
					writer.add_summary(summary_str, i)
				else:
					res = sess.run([rmse_tr, rmse_val, cost])
				acc_tr = res[0]
				acc_val = res[1]
				cost_ev = res[2]
				print("Training RMSE at step %s: %s" % (i, acc_tr), "\tValidation RMSE at step %s: %s" % (i, acc_val))
				diff = abs(cost_ev - last_cost)
				last_cost = cost_ev
			else:
				sess.run(train_step)

		finalTrain = rmse_tr.eval(session=sess)
		finalVal = rmse_val.eval(session=sess)
		finalW = W.eval(session=sess)
		finalH = H.eval(session=sess)
		print("****")
		print("\tRMSE of mean rating %s" % sess.run(rmse_val_mean))
		# 1.07341...

		sess.close()
		return finalTrain, finalVal, finalW, finalH

	# Extracts user indices, item indices, rating values and number
	# of ratings from the ratings triplets.
	def extract_rating_info(ratings):
		rating_values = ratings[:,2]
		user_indices = ratings[:,0]
		item_indices = ratings[:,1]
		num_ratings = len(item_indices)
		return rating_values, num_ratings, user_indices, item_indices


	# Learns factors of the given rank with specified regularization parameter.
	def initialize_latent_factor_matrix(num_users, num_items, rank, lda, good_mean=0.28):
		# Initialize the matrix factors from random normals with mean 0. W will
		# represent users and H will represent items.
		#np.sqrt(np.mean(ratings[:,2])/rank)
		W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.02, mean=good_mean), name="users")
		H = tf.Variable(tf.truncated_normal([rank, num_items], stddev=0.02, mean=good_mean), name="items")
		regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), lda, name="regularize")
		return W, H, regularizer

	
	def whole_process(self):

		np.random.seed(1)

		origin_review = np.load(self.review_numpy_path)
		fake_review = np.load(self.fake_review_numpy_path)
		camo_review = np.load(self.camo_review_numpy_path)

		origin_helpful = np.load(self.helpful_numpy_path)
		fake_helpful = np.load(self.fake_helpful_numpy_path)
		camo_helpful = np.load(self.camo_helpful_numpy_path)

		overall_review = np.concatenate((origin_review, fake_review, camo_review))
		# overall_helpful = np.concatenate((origin_helpful, fake_helpful, camo_helpful))

		num_users = len(np.unique(overall_review[:,0]))
		num_items = len(np.unique(overall_review[:,1]))
		num_reviews = len(overall_review)

		global_review_mean = np.mean(overall_review[:,2])

		print("=======================================================================")
		print('rating matrix size', num_users, num_items, '# of reviews', num_reviews)
		print("=======================================================================")



		# max_iter = 10001
		# lda = 1
		# rank = 50

		# train/test split
		review_train, review_test, helpful_train, helpful_test = train_test_split(origin_review,origin_helpful, train_size=.9)
		review_train = np.concatenate((review_train,fake_review, camo_review))
		helpful_train = np.concatenate((helpful_train,fake_helpful, camo_helpful))

		# t0 = time()
		W, H, reg = self.initialize_latent_factor_matrix(num_users, num_items, self.rank, self.lda, good_mean = np.sqrt(global_review_mean/rank) )
		tr, val, finalw, finalh = self.do_mf(review_train, review_test, helpful_train, helpful_test, W, H, reg, global_review_mean, self.max_iter, 1.0, True)
		# t1 = time()
		# print("=========================================")
		# print("Elasped Time for MF : %.2g sec" % (t1 - t0))

		print("Final training RMSE %s" % (tr),"\tFinal validation RMSE %s" % (val))
		print("=========================================")
		np.save("./intermediate/final_w",finalw)
		np.save("./intermediate/final_h",finalh)

if __name__=="__main__":
	mf = matrix_factorization(rank=50, lda=1, max_iter=5001)
	mf.whole_process()