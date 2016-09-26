from collections import Counter
import numpy as np
from preprocess import preprocess 

exp_title = 'bandwagon_1%_1%_1%_emb_32'
print('Experiment Title', exp_title)
from parameter_controller import *
params = parse_exp_title(exp_title)

# params.user_threshold = 10
# params.item_threshold = 10
# pp = preprocess(params=params)
# pp.whole_process()

class pp():
	def __init__(self, params, threshold=10):
		self.review_origin_path = params.review_origin_path
		self.threshold = threshold

	def removing_not_enough_obj(self, review_data):
		# user_list = self.get_rating_counter(review_data, user_or_item='user',threshold=threshold)
		# item_list = self.get_rating_counter(review_data, user_or_item='item',threshold=threshold)
		ret_review_data = []
		user_counter = Counter(review_data[:,0])
		item_counter = Counter(review_data[:,1])
		for review_row in review_data:
			reviewer = int(review_row[0])
			item = int(review_row[1])
			if user_counter[reviewer]>=self.threshold and item_counter[item]>=self.threshold:
				ret_review_data.append(review_row)
		
		return np.array(ret_review_data)

	def get_enough_rating_review(self):
		raw_review_data = np.load(self.review_origin_path)

		dummy = np.zeros((self.threshold*self.threshold, 4))
		for i in xrange(self.threshold):
			for j in xrange(self.threshold):
				dummy[i*self.threshold+j][0]=i+10000
				dummy[i*self.threshold+j][1]=j+10000

		raw_review_data = np.concatenate([raw_review_data, dummy])

		review_data = raw_review_data
		last_num_data = len(review_data)
		while True:
			review_data = self.removing_not_enough_obj(review_data)
			current_num_data = len(review_data)
			print current_num_data
			if last_num_data==current_num_data:
				break
			else:
				last_num_data = current_num_data
				pass
		return review_data

pp_instance = pp(params, threshold=10)
# review_data = np.load(params.review_origin_path)
review_data = pp_instance.get_enough_rating_review()


num_users = len(np.unique(review_data[:,0]))
num_items = len(np.unique(review_data[:,1]))
num_reviews = len(review_data)

user_counter = Counter(review_data[:,0])
user_degree_list = user_counter.values()

item_counter = Counter(review_data[:,1])
item_degree_list = item_counter.values()



print num_users, num_items, num_reviews
num_users_fake = [0.01, 0.03, 0.05, 0.1]
num_items_camo = [0.01, 0.03, 0.05, 0.1]
for i in xrange(len(num_users_fake)):
	print num_users_fake[i], num_users*num_users_fake[i], num_items*num_items_camo[i]

print '-------------------------------------------------------'
print 'np.percentile(user_degree_list,10)', np.percentile(user_degree_list,10)
print 'np.percentile(user_degree_list,50)', np.percentile(user_degree_list,50)
print 'np.percentile(user_degree_list,90)', np.percentile(user_degree_list,90)
print 'np.percentile(user_degree_list,99)', np.percentile(user_degree_list,99)
print 'np.mean(user_degree_list)', np.mean(user_degree_list)

print '-------------------------------------------------------'


print 'np.percentile(item_degree_list,10)', np.percentile(item_degree_list,10)
print 'np.percentile(item_degree_list,50)', np.percentile(item_degree_list,50)
print 'np.percentile(item_degree_list,90)', np.percentile(item_degree_list,90)
print 'np.percentile(item_degree_list,99)', np.percentile(item_degree_list,99)
print 'np.mean(item_degree_list)', np.mean(item_degree_list)

