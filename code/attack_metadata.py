from collections import Counter
import numpy as np
from preprocess import preprocess 

from parameter_controller import *


class new_preprocess():
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

exp_title = 'bandwagon_1%_1%_1%_emb_32'
print('Experiment Title', exp_title)
params = parse_exp_title(exp_title)

params.user_threshold = 5
params.item_threshold = 5
# pp = preprocess(params=params)
# pp.whole_process()

# print params.__dict__
# params.user_threshold=5
# params.item_threshold=5
# pp = preprocess(params=params)
# pp.whole_process()


# pp_instance = new_preprocess(params=params, threshold=5)
# review_data = pp_instance.get_enough_rating_review()
review_data = np.load(params.review_origin_path)
vote_data = np.load(params.vote_origin_path)

num_total_users = len(np.unique(np.concatenate((review_data[:,0],vote_data[:,0]))))
num_users = len(np.unique(review_data[:,0]))
num_items = len(np.unique(review_data[:,1]))
num_reviews = len(review_data)

user_counter = Counter(review_data[:,0])
user_degree_list = user_counter.values()

item_counter = Counter(review_data[:,1])
item_degree_list = item_counter.values()

# rank_list = [10,50,100,500]
# ucm = user_counter.most_common(501)
# icm = item_counter.most_common(501)
# for rank in rank_list:
# 	print rank, 'user deg', ucm[rank][1], 'item deg', icm[rank][1]


print 'User #', num_users, 'Item #', num_items, 'Review #', num_reviews
print 'Total user #', num_total_users, 'Vote #', len(vote_data)
percent_list = [0.005, 0.01, 0.02, 0.03, 0.05]
for i in xrange(len(percent_list)):
	print percent_list[i]*100,'%', '-->', int(num_users*percent_list[i]), 'users', int(num_items*percent_list[i]), 'items'

rating_percent_list = [ 0.5, 1, 2, 3, 5, 10,20,30,40,50,60,70,80, 90, 95, 99]

print '-------------------------------------------------------'
for p in rating_percent_list:
	print 'top', p, '% item ratings by', np.percentile(item_degree_list,100-p), 'users'
print 'np.mean(item degree)', np.mean(item_degree_list)

print '-------------------------------------------------------'
for p in rating_percent_list:
	print 'top', p, '% user rates', np.percentile(user_degree_list,100-p)
print 'np.mean(user degree)', np.mean(user_degree_list)

# threshold (10,10)
# 	info:	[1100 x 1500], 20K ratings
# 	target item requirement: 	(number of ratings >= 10 && mean(rating)<3.0)
# 	attacker
# 		1% attacker(11 users) --> 11(+10) ~> top 45%(17%) item
# 		2% attacker(22 users) --> 22(+10) ~> top 17%(5%) item
# 		3% attacker(33 users) --> 33(+10) ~> top 5%(2%) item
# 	filler
# 		1% filler (15)~> top 35% user
# 		2% filler (30) ~> top 15% user
# 		3% filler (45) ~> top 7% user
# 		4% filler (60) ~> top 4% user
# 	mean 
# 		user rate 18.3 item / item get 13.6 reviews
# 	top rank
# 		10 user deg (153) item deg (70)
# 		50 user deg (57) item deg (38)
# 		100 user deg (37) item deg (31)
# 		500 user deg (12) item deg (14)
# 	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 	tradeoff - 1% attacker(inject 11 fake profiles), 3% filler (45 popular item)
# 	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# threshold (5,5)
# 	[2500 x 3000] : 33K ratings
# 	target item requirement: (number of ratings >= ? && mean(rating)<3.0)
# 	attacker
# 		0.5% attacker(12 users) --> 12(+?) ~> top 25%(?%) item
# 		1% attacker(25 users) --> 25(+?) ~> top 12%(?%) item
# 		2% attacker(51 users) --> 51(+?) ~> top 2%(?%) item
# 		3% attacker(77 users) --> 77(+?) ~> top 0.7%(?%) item
# 	filler
# 		0.5% filler (15)~> top 20% user
# 		1% filler (30)~> top 7% user
# 		2% filler (61) ~> top 2.5% user
# 		3% filler (91) ~> top 1.5% user
# 	mean
# 		user rate 13.1 item / item get 11.1 reviews
# 	top rank
# 		10 user deg 216 item deg 103
# 		50 user deg 70 item deg 50
# 		100 user deg 47 item deg 40
# 		500 user deg 15 item deg 17
# 	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 	tradeoff - 1% attacker(inject 25 fake profiles), 1% filler (30 popular item)
# 	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# threshold (1,1)
# 	[17000 x 16000] : 72K ratings
# 	target item requirement: (number of ratings >= ? && mean(rating)<3.0)
# 	attacker
# 		0.1% attacker(17 users) --> 17(+?) ~> top 5%(?%) item
# 		0.5% attacker(88 users) --> 88(+?) ~> top less 0.5%(?%) item
# 		1% attacker(176 users) --> 176(+?) ~> top less 0.5%(?%) item
# 		2% attacker(352 users) --> 352(+?) ~> top less 0.5%(?%) item
# 		3% attacker(528 users) --> 528(+?) ~> top less 0.5%(?%) item
# 	filler
# 		0.1% filler (16)~> top 4% user
# 		0.5% filler (80)~> top less 0.5 % user
# 		1% filler (160)~> top less 0.5 % user
# 		2% filler (320) ~> top less 0.5 % user
# 		3% filler (480) ~> top less 0.5 % user
# 	mean
# 		user rate 4.1 item / item get 4.5 reviews
# 	top rank
# 		10 user deg 322 item deg 198
# 		50 user deg 119 item deg 93
# 		100 user deg 64 item deg 65
# 		500 user deg 19 item deg 26
# 	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 	tradeoff - 0.1<% attacker(inject 17 fake profiles), 0.1% filler (16 popular item)
# 	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!








# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# real threshold (5,5)!!!!
# 	[1822 x 2069] : 28374 ratings, 
# 	14972 user including voter, 661040 voting
# 	target item requirement: (number of ratings >= ? && mean(rating)<3.0)
# 	attacker
# 		0.5% attacker(9 users) --> 9(+?) ~> top 50%(?%) item
# 		1% attacker(18 users) --> 18(+?) ~> top 20%(?%) item
# 		2% attacker(36 users) --> 36(+?) ~> top 5%(?%) item
# 		3% attacker(54 users) --> 54(+?) ~> top 2%(?%) item
# 	filler
# 		0.5% filler (10)~> top 40% user
# 		1% filler (20)~> top 15% user
# 		2% filler (41) ~> top 6% user
# 		3% filler (62) ~> top 3% user
# 	mean
# 		user rate 15.5 item / item get 13.69 reviews
# 	top rank
# 		10 user deg 188 item deg 99
# 		50 user deg 65 item deg 50
# 		100 user deg 43 item deg 38
# 		500 user deg 13 item deg 16
# 	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 	tradeoff - 1% attacker(inject 18 fake profiles), 1% filler (20 popular item)
# 	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# real threshold (10, 10)
# 	[468 x 560] : 10358 ratings
# 	target item requirement: (number of ratings >= ? && mean(rating)<3.0)
# 	attacker
# 		0.5% attacker(2 users) --> 2(+?) ~> top 100%(?%) item
# 		1% attacker(4 users) --> 4(+?) ~> top 100%(?%) item
# 		2% attacker(9 users) --> 9(+?) ~> top 100%(?%) item
# 		3% attacker(14 users) --> 14(+?) ~> top 60%(?%) item
# 	filler
# 		0.5% filler (2)~> top 100% user
# 		1% filler (5)~> top 100% user
# 		2% filler (11) ~> top 80% user
# 		3% filler (16) ~> top 50% user
# 	mean
# 		user rate ? item / item get ? reviews
# 	top rank
# 		10 user deg 188 item deg 99
# 		50 user deg 65 item deg 50
# 		100 user deg 43 item deg 38
# 		500 user deg 13 item deg 16
# 	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 	tradeoff - ?% attacker(inject ? fake profiles), ?% filler (? popular item)
# 	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
