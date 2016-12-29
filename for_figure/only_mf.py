
import numpy as np

try:
	import tensorflow as tf
except:
	print 'fail to import tensorflow'

# input: review data R(user,item,rating,helpful)
# output: U,V

def sparse_to_dense(coo):
	num_row=int(max(coo[:,0]))+1
	num_col=int(max(coo[:,1]))+1
	ret = np.zeros((num_row,num_col))*np.inf
	for c in coo:
		ret[c[0]][c[1]]=c[2]
	return ret
	# sparse_to_dense(train_data)
def limit_prediction(prediction_result):
	ret = prediction_result
	ret[ret>5]=5
	ret[ret<1]=1
	return ret
def do_mf(train_data, rank=2, lda=0.000001, max_iter=1001, lr=0.01, decay_lr=True):
	# input : rating data [user, item, rating, helpful]

	# total data stats
	num_users = len(np.unique(train_data[:,0]))
	num_items = len(np.unique(train_data[:,1]))

	# train data stats
	global_review_mean = np.mean(train_data[:,2])
	good_mean=np.sqrt(global_review_mean / rank)

	# initialize latent factor matrix
	W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.005, mean=good_mean, seed=1), name="users")
	H = tf.Variable(tf.truncated_normal([rank, num_items], stddev=0.005, mean=good_mean, seed=1), name="items")
	
	# regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(tf.sub(W,baseW))), tf.reduce_sum(tf.square(tf.sub(H,baseH)))), lda, name="regularize")
	regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), lda, name="regularize")
	
	user_indices = train_data[:, 0]
	item_indices = train_data[:, 1]
	rating_observed = train_data[:, 2]
	num_review = len(item_indices)
	
	# Helpfulness score
	helpful_values = train_data[:, 3]
	HELPFUL = tf.to_float(tf.reshape(tf.constant(helpful_values),[-1]))
	
	# prediction = W*H	
	result = tf.matmul(W, H)

	with tf.name_scope("training_cost") as scope:
		# sparse format
		rating_prediction = tf.gather(tf.reshape(result, [-1]), user_indices * tf.shape(result)[1] + item_indices)
		diff_op = tf.sub(rating_prediction,rating_observed)
		
		# # without helpful
		# base_cost = tf.reduce_sum(diff_op)

		# multiply helpful
		weighted_square_diff_op = tf.mul(tf.square(diff_op), HELPFUL)
		base_cost = tf.reduce_sum(weighted_square_diff_op, name="sum_squared_error")
		
		#######################################################################			
		# cost = tf.add(base_cost, regularizer, name="total_cost")
		# (prediction error + regulaization) / num_review????
		cost = tf.div(tf.add(base_cost, regularizer), num_review * 2, name="average_error")
		
	with tf.name_scope("training_rmse") as scope:
		rmse_tr = tf.sqrt(tf.reduce_mean(tf.square(diff_op)))

	with tf.name_scope("train") as scope:
		if decay_lr:
			global_step = tf.Variable(0, trainable=False)
			learning_rate = tf.train.exponential_decay(lr, global_step, 5000, 0.96, staircase=True)
			optimizer = tf.train.AdamOptimizer(learning_rate)
			train_step = optimizer.minimize(cost, global_step=global_step)
		else:
			optimizer = tf.train.AdamOptimizer(lr)
			train_step = optimizer.minimize(cost)

	# ===========================================
	# ==============Session started==============
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	last_cost = 0
	diff = 1

	for i in range(max_iter):
		sess.run(train_step)
		if i > 0 and i % 1000 == 0:
			res = sess.run([cost, rmse_tr])
			cost_ev = res[0]
			rmse_tr_ev = res[1]
			
			diff = abs(cost_ev - last_cost)

			# if diff < 0.00001:
			# if last_cost>0 and (diff/last_cost) <= 0.0001:
			if last_cost>0 and (diff/last_cost) <= 0.0005:
				print("Converged at iteration %s" % (i))
				break
			
			# update the last cost 
			last_cost = cost_ev

	final_U = W.eval(session=sess)
	final_V = H.eval(session=sess)
	final_prediction = result.eval(session=sess)

	np.set_printoptions(precision=1)
	print 'U'
	print final_U
	print 'V'
	print final_V
	print 'observed'
	print sparse_to_dense(train_data)
	print 'final_prediction'
	print limit_prediction(final_prediction)
	sess.close()
	# ==============Session finished==============
	# ============================================


	return final_prediction

def normal_data():
	train_data = []
	train_data.append([0,0,5])
	train_data.append([0,1,5])
	train_data.append([0,3,2])

	train_data.append([1,0,4])
	train_data.append([1,2,2])
	train_data.append([1,4,1])

	train_data.append([2,0,2])
	train_data.append([2,2,4])
	train_data.append([2,3,4])
	train_data.append([2,4,1])

	train_data.append([3,1,2])
	train_data.append([3,2,5])
	train_data.append([3,3,5])
	return train_data

def attacked_data():
	train_data=normal_data()
	train_data.append([4,1,3])
	train_data.append([4,3,3])
	train_data.append([4,4,5])

	train_data.append([5,0,3])
	train_data.append([5,2,3])
	train_data.append([5,4,5])
	return train_data


def merge(train_data,helpful_data=1):
	# add column	
	if type(helpful_data)==type(1):
		helpful_data=np.ones((len(train_data),1))
	train_data=np.array(train_data)
	train_data=np.concatenate((train_data,helpful_data), axis=1)
	return train_data
		
print 'normal and 1'
do_mf(merge(normal_data(),1))

print 'attacked and 1'
do_mf(merge(attacked_data(),1))

print 'attacked and normal:helpful=10:1'
good_helpful = np.ones((len(attacked_data()),1))*0.1
good_helpful[:len(normal_data())]*=9
print merge(attacked_data(),good_helpful)
do_mf(merge(attacked_data(),good_helpful))

print 'attacked and normal:helpful=1:10'
bad_helpful = np.ones((len(attacked_data()),1))*0.9
bad_helpful[:len(normal_data())]/=9
# print merge(attacked_data(),bad_helpful)
do_mf(merge(attacked_data(),bad_helpful))


