# 2. inject fake review/vote
# 	Input
# 		review 	<-- './intermediate/review.npy'
# 		vote 	<-- './intermediate/vote.npy'
# 	Output
# 		fake review 	--> './intermediate/fake_review_[average|bandwagon].npy'
# 		camo review 	--> './intermediate/camo_review_[average|bandwagon].npy'
# 		fake vote 		--> './intermediate/fake_vote_[average|bandwagon].npy'

import numpy as np
import csv
print("this is attack model bandwagon new!!!!!!!!!!!!!!!!!!")
class attack_model_bandwagon():
	def __init__(self, params=None):

		# num_fake_user=0.01, num_fake_item=0.01, num_camo_item=0.01 , num_fake_review=1, num_fake_vote=1, fake_rating_value=5, fake_helpful_value=5, badly_rated_item_flag=True
		# input
		self.review_origin_path = params.review_origin_path  #'./intermediate/review.npy'
		self.vote_origin_path = params.vote_origin_path  #'./intermediate/vote.npy'
		# output
		self.review_fake_path = params.review_fake_path  #'./intermediate/fake_review_bandwagon.npy'
		self.review_camo_path = params.review_camo_path  #'./intermediate/camo_review_bandwagon.npy'
		self.vote_fake_path = params.vote_fake_path  #'./intermediate/fake_vote_bandwagon.npy'
		
		self.target_item_list_path = params.target_item_list_path
		self.fake_user_id_list_path = params.fake_user_id_list_path
		#######################################
		# origin review stats
		self.origin_review_matrix = np.load(self.review_origin_path)
		self.origin_vote_matrix = np.load(self.vote_origin_path)
		
		self.num_origin_user = len(np.unique(self.origin_review_matrix[:,0]))
		
		self.num_origin_item = len(np.unique(self.origin_review_matrix[:,1]))
		self.num_origin_review = len(self.origin_review_matrix)
		self.num_origin_vote = len(self.origin_vote_matrix)
		
		self.construct_avg_rating()
		#######################################
		# fake review stats
		self.fake_user_start = self.num_origin_user
		self.fake_item_start = self.num_origin_item
		self.fake_review_id_start = self.num_origin_review

		self.num_fake_user = int(self.num_origin_user*params.num_fake_user) if params.num_fake_user<=1 else int(params.num_fake_user)
		self.fake_user_id_list = list(range(self.fake_user_start, self.fake_user_start+self.num_fake_user))

		# self.num_fake_item = int(self.num_origin_item*params.num_fake_item) if params.num_fake_item<=1 else int(params.num_fake_item)
		print("num fake item is one!!!!!!!!!!!!!!")
		self.num_fake_item = params.num_fake_item
		self.fake_item_id_list = list(range(self.fake_item_start, self.fake_item_start+self.num_fake_item))
		
		# #(fake reviews) cannot be larger than #(fake_user)*#(fake_item)
		self.num_fake_review = int(self.num_fake_user*self.num_fake_item*params.num_fake_review) if params.num_fake_review<=1 else int(params.num_fake_review)
		self.num_fake_review = min(self.num_fake_review, self.num_fake_user*self.num_fake_item)
		self.fake_review_id_list = list(range(self.fake_review_id_start,self.fake_review_id_start+self.num_fake_review))

		# #(fake votes) cannot be larger than #(fake users)*#(fake reviews)
		self.num_fake_vote = int(self.num_fake_review*self.num_fake_user*params.num_fake_vote) if params.num_fake_vote<=1 else int(params.num_fake_vote)
		self.num_fake_vote = min(self.num_fake_vote, self.num_fake_review*self.num_fake_user)
		
		# fake_rating value is usually extream value
		self.fake_rating_value = params.fake_rating_value
		self.fake_helpful_value = params.fake_helpful_value

		# camofoulage
		self.filler_size = int(self.num_origin_item*params.filler_size) if params.filler_size <=1 else int(params.filler_size)
		self.selected_size = int(self.num_origin_item*params.selected_size) if params.selected_size <=1 else int(params.selected_size)
		# self.num_camo_item = int(self.num_origin_item*params.num_camo_item) if params.num_camo_item <=1 else params.num_camo_item
		self.num_camo_item = self.filler_size+self.selected_size
		self.camo_item_id_list = []
		self.camo_review_id_start = self.fake_review_id_start+self.num_fake_review

		# importatnt data
		self.fake_review_matrix = []
		self.fake_vote_matrix = []
		self.camo_review_matrix = []
		self.badly_rated_item_flag=params.badly_rated_item_flag
		self.bad_item_threshold = params.bad_item_threshold

		print('[Origin] Users x Items :', self.num_origin_user, self.num_origin_item, 'Reviews, Votes : ', self.num_origin_review, self.num_origin_vote)
		print('[Fake] Users x Items :', self.num_fake_user, self.num_fake_item, 'Reviews, Votes : ',self.num_fake_review, self.num_fake_vote)
		print('[Bandwagon] Filler size :', self.filler_size, 'Selected size : ', self.selected_size)

	def construct_avg_rating(self):
		# self.avg_rating = list()
		sum_rating_ist = np.zeros((self.num_origin_item,))
		num_rating_list = np.zeros((self.num_origin_item,))
		for row in self.origin_review_matrix:
			item_id = int(row[1])
			rating = float(row[2])
			sum_rating_ist[item_id]+=rating
			num_rating_list[item_id]+=1.0
		self.avg_rating_list = sum_rating_ist/num_rating_list

	def get_popular_item(self, topk=10):
		from collections import Counter
		a = self.origin_review_matrix[:,1]
		b = Counter(a)
		# [(item_id, occurence)*]
		# [(944, 212), (912, 156), (1436, 135), (2277, 126), (2577, 124), (2797, 115)]
		return b.most_common(topk)

	def get_bad_items_info(self,topk=10, threshold=0, random_flag=False):
		# many bad rating
		item_score = dict()
		for row in self.origin_review_matrix:
			item_id = row[1]
			rating = float(row[2])
			if item_id not in item_score:
				item_score[item_id]=[0,0]
			item_score[item_id][0]+=rating
			item_score[item_id][1]+=1

		item_score_list = []
		for k in item_score.keys():
			# item_id, overall rating, the number of rating
			item_score_list.append([k, item_score[k][0]/float(item_score[k][1]), item_score[k][1]])

		item_score_list = sorted(item_score_list, key=lambda x:x[1])
		item_score_list = filter(lambda x: x[1]<3 and x[2]>=threshold, item_score_list)
		# print 'item_score_list  2', item_score_list
		if random_flag:
			random_index_list = np.random.choice(a=len(item_score_list), size=topk, replace=False)
			return [item_score_list[idx] for idx in random_index_list]
		else:
			return item_score_list[0:topk]

	def generate_fake_reviews(self, target_item_id_list):
		fake_review_coo = np.random.choice(self.num_fake_user*self.num_fake_item, self.num_fake_review, replace=False)

		fake_review_id = self.fake_review_id_start
		for coo in fake_review_coo:
			fake_u = self.fake_user_id_list[int(coo/self.num_fake_item)]
			fake_i = target_item_id_list[coo%self.num_fake_item]

			self.fake_review_matrix.append([fake_u, fake_i, self.fake_rating_value, fake_review_id])
			fake_review_id+=1

	def generate_fake_reviews_bad_item(self):
		# target items are badly rated items
		bad_item_info_list = self.get_bad_items_info(self.num_fake_item, threshold=self.bad_item_threshold)
		# print 'bad_item_info_list', bad_item_info_list
		self.fake_item_id_list = [x[0] for x in bad_item_info_list]
		# print ('target items', self.fake_item_id_list)
		self.generate_fake_reviews(self.fake_item_id_list)
	
	def generate_camo_reviews(self):
		camo_review_id = self.camo_review_id_start
		# random attack
		for u in self.fake_user_id_list:
			random_item_list = np.random.choice(a=self.num_origin_item, size=self.filler_size)
			for i in random_item_list:
				tmp_random_review = [u, i, int(np.round(self.avg_rating_list[i])), camo_review_id]
				self.camo_review_matrix.append(tmp_random_review)
				camo_review_id+=1

		# popular attack (bandwagon attack)
		popular_item_list = self.get_popular_item(self.selected_size)
		self.camo_item_id_list = [x[0] for x in popular_item_list]
		# print('camo item list', popular_item_list)
		for u in self.fake_user_id_list:
			for i in self.camo_item_id_list:
				self.camo_review_matrix.append([u,i,5,camo_review_id])
				camo_review_id+=1
	
	##################
	###### VOTE ######
	##################

	def generate_fake_votes(self):
		fake_vote_coo = np.random.choice(self.num_fake_user*self.num_fake_review, self.num_fake_vote, replace=False)
		for coo in fake_vote_coo:
			fake_u = self.fake_user_id_list[int(coo/self.num_fake_review)]
			fake_r = self.fake_review_id_list[coo%self.num_fake_review]
			self.fake_vote_matrix.append([fake_u,fake_r, self.fake_helpful_value])

	def save_attack_review_matrix(self):
		np.save(self.review_fake_path, np.array(self.fake_review_matrix))
		np.save(self.review_camo_path, np.array(self.camo_review_matrix))

	def save_attack_vote_matrix(self):
		np.save(self.vote_fake_path, np.array(self.fake_vote_matrix))

	def save_fake_user_id_list(self):
		fake_review_matrix_ = np.array(self.fake_review_matrix)
		np.save(self.fake_user_id_list_path, np.unique(fake_review_matrix_[:,0]))

	def save_target_item_list(self):
		fake_review_matrix_ = np.array(self.fake_review_matrix)
		np.save(self.target_item_list_path, np.unique(fake_review_matrix_[:,1]))

	def whole_process(self):
		# review
		if self.badly_rated_item_flag ==True:
			self.generate_fake_reviews_bad_item()
		else:
			print("any other??")
		self.generate_camo_reviews()
		self.save_attack_review_matrix()

		# vote
		self.generate_fake_votes()
		self.save_attack_vote_matrix()

		# save the metadata of attack
		self.save_fake_user_id_list()
		self.save_target_item_list()

if __name__=="__main__":
	# am = attack_model_bandwagon(num_fake_user=100, num_fake_item=10, num_camo_item=0.01, badly_rated_item_flag=True)
	# am.whole_process()
	from parameter_controller import *
	exp_title = 'bandwagon_1%_1%_1%_emb_32'
	print('Experiment Title', exp_title)
	params = parse_exp_title(exp_title)
	am = attack_model_bandwagon(params=params)
	am.whole_process()

	# candi = am.get_bad_items_info(5, threshold=5)
	# print candi
	# print filter(lambda x: x[2]>=5, candi)
	# print len(filter(lambda x: x[2]>=5, candi))
