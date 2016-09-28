# 3. embedding
#     Input
#         review_numpy <-- './intermediate/review_numpy.npy'
#         vote_numpy <-- './intermediate/vote_numpy.npy'
#         fake review     --> './intermediate/fake_review_[average|bandwagon].npy'
#         camo review     --> './intermediate/camo_review_[average|bandwagon].npy'
#         fake vote       --> './intermediate/fake_vote_[average|bandwagon].npy'
#     Output
#         user_embedding --> './intermediate/user2vec.emb'
#     Intermediate
#         review_writer : review_writer(review)=reviewer
#         similar_reviewer : similar_reviewer(item,rating)= set of reviewers
#         reviewer_follower : reviewer_follower(reviewer) = list of followers (allowing duplicated)
#         // similar_voter : similar_voter(review,vote_value) = set of voters

import numpy as np
import csv
import itertools
from time import time
from gensim.models import Word2Vec

# import logging
# logging.basicConfig(filename='./test.log',level=logging.DEBUG)
def cosine_distance(v1, v2):
	return 1 - np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))


class reviewer_tensor_entry():
	def __init__(self):
		self.len_rating_list=0
		self.rating_list =[]
		self.rating_reviewer_degree_list=[]
		self.rating_reviewer_weight_list=[]
		self.reviewer_2dlist=[]
		

class follower_tensor_entry():
	def __init__(self):
		self.len_reviewer_list=0
		self.reviewer_list=[]
		self.follower_num_list=[]
		self.reviewer_weight_list=[]
		self.follower_2dlist=[]

class user2vec():
	def __init__(self, params=None, fake_flag=True, camo_flag=True, embedding_output_path=''):

		# input
		self.review_origin_path = params.review_origin_path
		self.review_fake_path = params.review_fake_path
		self.review_camo_path = params.review_camo_path
		
		self.vote_origin_path = params.vote_origin_path
		self.vote_fake_path = params.vote_fake_path
		# self.vote_camo_path = params.vote_camo_path

		# output
		# self.embedding_clean_path = params.embedding_clean_path
		# self.embedding_attacked_path = params.embedding_attacked_path
		self.embedding_output_path = embedding_output_path
		
		# embedding
		self.fake_flag = fake_flag
		self.camo_flag = camo_flag
		self.embedding_dim = params.embedding_dim
		self.word2vec_iter = params.word2vec_iter

		# embedding model (Word2Vec)
		self.user_embedding_model = None


		# generate sentence
		self.reviewer_tensor = dict()
		self.review_id_info = dict()
		self.follower_tensor = dict()
		
		self.num_item = 0
		self.item_degree_list = []
		self.item_weight_list = []

		self.observed_user_list = set()

	def load_overall_review_matrix(self):
		overall_review_matrix = np.load(self.review_origin_path)
		if self.fake_flag:
			overall_review_matrix = np.concatenate((overall_review_matrix, np.load(self.review_fake_path)))
			if self.camo_flag:
				overall_review_matrix = np.concatenate((overall_review_matrix, np.load(self.review_camo_path)))
		return overall_review_matrix

	def load_overall_vote_matrix(self):
		overall_vote_matrix = np.load(self.vote_origin_path)
		if self.fake_flag:
			overall_vote_matrix = np.concatenate((overall_vote_matrix, np.load(self.vote_fake_path)))
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# if self.camo_flag:
		# 	overall_vote_matrix = np.concatenate((overall_vote_matrix, np.load(self.vote_camo_path)))
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		return overall_vote_matrix

	def construct_item_weight(self):
		# 1. #(reviewer)
		# 2. #(reviewer U voter) <-- good
		item_related_user_set = dict()
		review_id_item_id_mapping = dict()

		# 1) add reviewer

		overall_review_matrix = self.load_overall_review_matrix()
		for row in overall_review_matrix:
			reviewer = int(row[0])
			item_id = int(row[1])
			# rating_value = row[2]
			review_id = int(row[3])
			if item_id not in item_related_user_set:
				item_related_user_set[item_id] = set()
			item_related_user_set[item_id].add(reviewer)

			review_id_item_id_mapping[review_id] = item_id

		# 2) add reviewer

		overall_vote_matrix = self.load_overall_vote_matrix()
		for row in overall_vote_matrix:
			voter = int(row[0])
			review_id = int(row[1])
			# helpful_vote = row[2]

			item_id = review_id_item_id_mapping[review_id]
			if item_id not in item_related_user_set:
				item_related_user_set[item_id] = set()
			item_related_user_set[item_id].add(voter)

		# mainly compute weight of items
		self.num_item = len(np.unique(overall_review_matrix[:, 1]))
		self.item_degree_list = np.zeros((self.num_item, 1)).reshape([-1])
		self.item_weight_list = np.zeros((self.num_item, 1)).reshape([-1])
		for item_id, user_set in item_related_user_set.iteritems():
			self.item_degree_list[item_id] = len(user_set)
		# self.item_weight_list = 1.0 / np.log2(self.item_degree_list + 5)
		self.item_weight_list = 1.0 / (self.item_degree_list + 5)
		normalization = 1.0 * np.sum(self.item_weight_list)
		self.item_weight_list = self.item_weight_list/normalization
		
		# print np.percentile(self.item_degree_list, 99), np.percentile(self.item_weight_list, 1)
		# print np.percentile(self.item_degree_list, 95), np.percentile(self.item_weight_list, 5)
		# print np.percentile(self.item_degree_list, 90), np.percentile(self.item_weight_list, 10)
		
		# print np.percentile(self.item_degree_list, 10), np.percentile(self.item_weight_list, 90)
		# print np.percentile(self.item_degree_list, 5), np.percentile(self.item_weight_list, 95)
		# print np.percentile(self.item_degree_list, 1), np.percentile(self.item_weight_list, 99)
		# target = np.argmax(self.item_degree_list>707)
		# print self.item_degree_list[target], self.item_weight_list[target]
		# # drawing degree plot
		# item_degree_list = np.array(item_degree_list)
		# hist, bins = np.histogram(item_degree_list, bins=max(item_degree_list)+1)
		# print hist
		# print bins
		# %matplotlib inline

		# import matplotlib
		# import matplotlib.pyplot as plt
		# width = 1.0 * (bins[1] - bins[0])
		# center = (bins[:-1] + bins[1:]) / 2
		# plt.bar(center, hist, align='center', width=width)
		# plt.show()



	def construct_reviewer_tensor(self):
		# data loading
		overall_review_matrix = self.load_overall_review_matrix()
		for row in overall_review_matrix:
			reviewer = int(row[0])
			item_id = int(row[1])
			rating_value = row[2]
			review_id = int(row[3])

			# needed in follower_tensor
			self.review_id_info[review_id] = (reviewer, item_id)

			# Fill reviewer_tensor
			if item_id not in self.reviewer_tensor:
				self.reviewer_tensor[item_id] = reviewer_tensor_entry()

			reviewer_info = self.reviewer_tensor[item_id]
			if rating_value not in reviewer_info.rating_list:
				reviewer_info.rating_list.append(rating_value)
				reviewer_info.rating_reviewer_degree_list.append([0])
				reviewer_info.rating_reviewer_weight_list.append([0])
				reviewer_info.reviewer_2dlist.append([])

			target_index = reviewer_info.rating_list.index(rating_value)
			reviewer_info.reviewer_2dlist[target_index].append(reviewer)

		# num_list, weight list update
		for item_id, reviewer_info in self.reviewer_tensor.iteritems():
			num_rating = len(reviewer_info.rating_list)
			reviewer_info.len_rating_list = num_rating
			for i in xrange(num_rating):
				reviewer_info.rating_reviewer_degree_list[i]=len(reviewer_info.reviewer_2dlist[i])
				reviewer_info.rating_reviewer_weight_list[i] = 1.0/np.log2(reviewer_info.rating_reviewer_degree_list[i]+5)
			reviewer_info.rating_reviewer_weight_list /= sum(reviewer_info.rating_reviewer_weight_list)*1.0

	def construct_follower_tensor(self):
		overall_vote_matrix = self.load_overall_vote_matrix()
		for row in overall_vote_matrix:
			voter = int(row[0])
			review_id = int(row[1])
			helpful_vote = row[2]

			reviewer, item_id = self.review_id_info[review_id]

			# fill observed user list
			self.observed_user_list.add(reviewer)
			self.observed_user_list.add(voter)

			# note that helpfulness vote threshold is important!!!!
			if helpful_vote <=3:
				continue

			if item_id not in self.follower_tensor:
				self.follower_tensor[item_id] = follower_tensor_entry()

			follower_info = self.follower_tensor[item_id]
			if reviewer not in follower_info.reviewer_list:
				follower_info.reviewer_list.append(reviewer)
				follower_info.follower_num_list.append([0])
				follower_info.reviewer_weight_list.append([0])
				follower_info.follower_2dlist.append([])

			target_index = follower_info.reviewer_list.index(reviewer)
			follower_info.follower_2dlist[target_index].append(voter)

		for item_id,follower_info in self.follower_tensor.iteritems():
			num_reviewer = len(follower_info.reviewer_list)
			follower_info.len_reviewer_list=num_reviewer
			for i in xrange(num_reviewer):
				follower_info.follower_num_list[i]=len(follower_info.follower_2dlist[i])
				follower_info.reviewer_weight_list[i] = 1.0/np.log2(follower_info.follower_num_list[i]+5)
			follower_info.reviewer_weight_list /= sum(follower_info.reviewer_weight_list)*1.0

	##########################################################################
	# def learn_embedding(self, train_data=None):
	# 	#  Learn embedding by optimizing the Skipgram objective using negative sampling.
	# 	train_data = [map(str, x) for x in train_data]
	# 	print("learning sentence # :", len(train_data))
	# 	if not self.user_embedding_model:
	# 		# first train data
	# 		self.user_embedding_model = Word2Vec(train_data, size=self.embedding_dim
	# 		                                     , sg=1, negative=20, window=2, min_count=1, workers=8,
	# 		                                     iter=self.word2vec_iter)
	# 	else:
	# 		# new sentences
	# 		self.user_embedding_model.train(train_data)
	# 	return


	def generate_reviewer_reviewer(self, num_sample):
		ret=[]
		# follower_tensor
		# sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample/2, p=self.item_weight_list)
		sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample/2)
		for item_id in sampled_item_id_list:
			sample_reviewer_info = self.reviewer_tensor[item_id]
			# sample_rating_index_list = np.random.choice(a=sample_reviewer_info.len_rating_list, size=2,p=sample_reviewer_info.rating_reviewer_weight_list)
			sample_rating_index_list = np.random.choice(a=sample_reviewer_info.len_rating_list, size=2)

			for idx in sample_rating_index_list:
				try:
					sample_reviewer_id_list = np.random.choice(a=sample_reviewer_info.reviewer_2dlist[idx], size=2, replace=False)
					ret.append(sample_reviewer_id_list)
				except Exception, ex:
					pass
		return ret

	def generate_reviewer_follower(self, num_sample):
		ret=[]
		# follower_tensor
		# sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample/5, p=self.item_weight_list)
		sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample/5)
		for item_id in sampled_item_id_list:
			try:
				sample_follower_info = self.follower_tensor[item_id]
			except Exception, ex:
				continue
			sample_reviewer_index_list = np.random.choice(a=sample_follower_info.len_reviewer_list, size=5, p=sample_follower_info.reviewer_weight_list)
			# sample_reviewer_index_list = np.random.choice(a=sample_follower_info.len_reviewer_list, size=5)
			
			for idx in sample_reviewer_index_list:
				sample_follower_id_list = np.random.choice(a=sample_follower_info.follower_2dlist[idx],size=1)
				reviewer_id = sample_follower_info.reviewer_list[idx]
				# reviewer-follower
				for follower_id in sample_follower_id_list:
					ret.append([reviewer_id,follower_id])
		return ret

	def generate_follower_follower(self, num_sample):
		ret=[]
		# follower_tensor
		sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample/5, p=self.item_weight_list)
		# sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample/5)
		for item_id in sampled_item_id_list:
			try:
				sample_follower_info = self.follower_tensor[item_id]
			except Exception, ex:
				continue
			sample_reviewer_index_list = np.random.choice(a=sample_follower_info.len_reviewer_list, size=5, p=sample_follower_info.reviewer_weight_list)
			# sample_reviewer_index_list = np.random.choice(a=sample_follower_info.len_reviewer_list, size=5)
			
			for idx in sample_reviewer_index_list:
				try:
					sample_follower_id_list = np.random.choice(a=sample_follower_info.follower_2dlist[idx], size=2, replace=False)
					ret.append(sample_follower_id_list)
				except Exception, ex:
					pass
		return ret

	def get_string_sentence(self,train_data):
		num_iter = len(train_data)
		for i in xrange(num_iter):
			row = train_data[i]
			train_data[i]=[int(x) for x in train_data[i]]
			train_data[i]=map(str,train_data[i])
		return train_data

	def build_user_vocab(self):
		# self.user_embedding_model = Word2Vec(size=self.embedding_dim, sample=0, sg=1, hs=0, negative=100, window=2, min_count=0, workers=8, iter=self.word2vec_iter)
		self.user_embedding_model = Word2Vec(size=self.embedding_dim, sample=0, sg=0, hs=1, negative=0, window=2, min_count=0, workers=8, iter=self.word2vec_iter)
		user_vocab = [[str(int(x)) for x in self.observed_user_list]]
		self.user_embedding_model.build_vocab(user_vocab)

	def train_embedding_model(self, train_data=None):
		#  Learn embedding by optimizing the Skipgram objective using negative sampling.
		# train_data = [map(str, int(x)) for x in train_data]
		train_data = self.get_string_sentence(train_data)
		# print("learning sentence # :", len(train_data))
		self.user_embedding_model.train(train_data)


	def save_embedding(self):
		self.user_embedding_model.save(self.embedding_output_path)
		# self.user_embedding_model.save_word2vec_format(self.user_embedding_csv_path)


	def load_embedding(self):
		return Word2Vec.load(self.embedding_output_path)

	def similarity_test(self, origin_user_id_list=list(range(0,1000)), fake_user_id_list=list(range(1823,1830))):
		our_model = self.load_embedding()
		try:
			print 'fake-fake',
			x_list = np.random.choice(fake_user_id_list,7, replace=False)
			y_list = np.random.choice(fake_user_id_list,7, replace=False)
			sim_list = []
			for i in xrange(7):
				sim_list.append( our_model.similarity(str(x_list[i]),str(y_list[i])) )
			print sim_list

			print 'fake-origin',
			x_list = np.random.choice(origin_user_id_list,7, replace=False)
			y_list = np.random.choice(fake_user_id_list,7, replace=False)
			sim_list = []
			for i in xrange(7):
				sim_list.append( our_model.similarity(str(x_list[i]),str(y_list[i])) )
			print sim_list
		except:
			print ("sorry")
		print 'origin-origin',
		x_list = np.random.choice(origin_user_id_list,7, replace=False)
		y_list = np.random.choice(origin_user_id_list,7, replace=False)
		sim_list = []
		for i in xrange(7):
			sim_list.append( our_model.similarity(str(x_list[i]),str(y_list[i])) )
		print sim_list
		print ''


	
	def whole_process(self,iteration=10, type0_ratio=1, type1_ratio=1, type2_ratio=1):
		self.construct_reviewer_tensor()
		self.construct_follower_tensor()
		self.construct_item_weight()

		# Word2Vec init
		self.build_user_vocab()
		
		batch_size=2e+5
		for it in xrange(iteration):
			for i in xrange(type0_ratio):
				self.train_embedding_model(self.generate_reviewer_reviewer(batch_size))
			for i in xrange(type1_ratio):
				self.train_embedding_model(self.generate_reviewer_follower(batch_size))
			for i in xrange(type2_ratio):
				self.train_embedding_model(self.generate_follower_follower(batch_size))

		# if self.fake_flag:
		# 	self.similarity_test()

		self.save_embedding()

if __name__ == "__main__":
	from parameter_controller import *

	exp_title = 'bandwagon_1%_1%_1%_emb_32'
	print('Experiment Title', exp_title)
	params = parse_exp_title(exp_title)
	
	u2v_attacked = user2vec(params=params, fake_flag=True, camo_flag=True, embedding_output_path=params.embedding_attacked_path)
	u2v_attacked.whole_process()

	u2v_clean = user2vec(params=params, fake_flag=False, camo_flag=False, embedding_output_path=params.embedding_clean_path)
	u2v_clean.whole_process()
	


