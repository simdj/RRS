# 3. embedding
#     Input
#         review_numpy <-- './intermediate/review_numpy.npy'
#         vote_numpy <-- './intermediate/vote_numpy.npy'
#         fake review     --> './intermediate/fake_review_[average|bandwagon].npy'
#         camo review     --> './intermediate/camo_review_[average|bandwagon].npy'
#         fake vote       --> './intermediate/fake_vote_[average|bandwagon].npy'
#     Output
#         user_embedding --> './intermediate/user2vec.emb'

import numpy as np
import csv
import itertools
from collections import Counter

from time import time
from gensim.models import Word2Vec

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
		self.vote_camo_path = params.vote_camo_path

		self.fake_user_id_list_path = params.fake_user_id_list_path

		# output
		# self.embedding_clean_path = params.embedding_clean_path
		# self.embedding_attacked_path = params.embedding_attacked_path
		self.embedding_output_path = embedding_output_path
		
		# embedding
		self.fake_flag = fake_flag
		self.camo_flag = camo_flag
		self.embedding_dim = params.embedding_dim
		self.word2vec_iter = params.word2vec_iter

		# data loaded
		self.item_rating_data = None
		self.helpfulness_rating_data = None

		self.positive_helpful_threshold = 5
		self.positive_item_rating_data = None # positive_item_rating_data is subset of review matrix
		self.positive_helpfulness_rating_data = None # positive_helpfulness_rating_data is subset of vote matrix
		
		# generate sentence
		self.review_id_info = dict() # review_id_info[review_id]=(reviewer, item)
		self.observed_user_list = set() # needed in Word2Vec initialization
		
		# embedding model (Word2Vec)
		self.user_embedding_model = None

		## data structure needed in sampling
		self.item_list = []
		self.item_degree_list = []
		self.item_prob_list = []
		self.reviewer_set_of_item = dict()

		self.review_list = []
		self.review_degree_list = []
		self.review_prob_list = []
		self.follower_set_of_item = dict()


	def load_overall_review_matrix(self):
		overall_review_matrix = np.load(self.review_origin_path)
		print '1 len(overall_review_matrix)', len(overall_review_matrix)
		if self.fake_flag:
			overall_review_matrix = np.concatenate((overall_review_matrix, np.load(self.review_fake_path)))
			print '2 len(overall_review_matrix)', len(overall_review_matrix)
			if self.camo_flag:
				overall_review_matrix = np.concatenate((overall_review_matrix, np.load(self.review_camo_path)))
				print '3 len(overall_review_matrix)', len(overall_review_matrix)

		self.item_rating_data = overall_review_matrix
		self.positive_item_rating_data = overall_review_matrix[overall_review_matrix[:,2]==5,:]

	def load_overall_vote_matrix(self):
		overall_vote_matrix = np.load(self.vote_origin_path)
		print '1 len(overall_vote_matrix)', len(overall_vote_matrix)
		if self.fake_flag:
			overall_vote_matrix = np.concatenate((overall_vote_matrix, np.load(self.vote_fake_path)))
			print '2 len(overall_vote_matrix)', len(overall_vote_matrix)
			if self.camo_flag:
				try:
					overall_vote_matrix = np.concatenate((overall_vote_matrix, np.load(self.vote_camo_path)))
					print '3 len(overall_vote_matrix)', len(overall_vote_matrix)
				except:
					print 'no camo vote in new_user2vec.py'
		
		self.helpfulness_rating_data = overall_vote_matrix
		self.positive_helpfulness_rating_data = overall_vote_matrix[overall_vote_matrix[:,2]>=self.positive_helpful_threshold,:]

	def fill_review_id_info(self):
		assert(len(self.item_rating_data)>0)

		for row in self.item_rating_data:
			reviewer = int(row[0])
			item_id = int(row[1])
			# rating_value = row[2]
			review_id = int(row[3])

			self.review_id_info[review_id] = (reviewer, item_id)

	def fill_observed_user_list(self):
		assert(len(self.helpfulness_rating_data)>0)
		assert(len(self.review_id_info)>0)

		for row in self.helpfulness_rating_data:
			voter = int(row[0])
			review_id = int(row[1])
			# helpful_vote = row[2]

			reviewer, item_id = self.review_id_info[review_id]

			# fill observed user list
			self.observed_user_list.add(reviewer)
			self.observed_user_list.add(voter)


	###################################################################################
	############## sample reviews  and randomly choose one pair per review ############
	###################################################################################
	def prepare_item_sampling(self):
		item_counter = Counter(self.positive_item_rating_data[:,1])
		ici = item_counter.items()
		ici = filter(lambda x: x[1]>=2, ici)

		self.item_list = map(lambda x:int(x[0]), ici)
		self.item_degree_list = map(lambda x: x[1], ici)
		self.item_prob_list = np.array(self.item_degree_list)/(1.0*sum(self.item_degree_list))

		self.reviewer_set_of_item = dict()
		for row in self.positive_item_rating_data:
			reviewer_id = int(row[0])
			item_id = int(row[1])

			if item_counter[item_id]<2:
				continue

			if item_id not in self.reviewer_set_of_item:
				self.reviewer_set_of_item[item_id]=[]
			self.reviewer_set_of_item[item_id].append(reviewer_id)

	def get_sample_reviewer_reviewer_pair(self,sample_size):
		sampled_item_list = np.random.choice(a=self.item_list, size=sample_size, p=self.item_prob_list)
		return map(lambda item_id: map(int, np.random.choice(a=self.reviewer_set_of_item[item_id], size=2)), sampled_item_list)

	def prepare_review_sampling(self):
		review_helpfulness_counter = Counter(self.positive_helpfulness_rating_data[:,1])
		vci = review_helpfulness_counter.items()
		vci = filter(lambda x: x[1]>=2, vci)

		self.review_list = map(lambda x:int(x[0]), vci)
		self.review_degree_list = map(lambda x: x[1], vci)
		# self.review_degree_list = map(lambda x: 1/(1.0*np.log(5+x[1])), vci)
		
		self.review_prob_list = np.array(self.review_degree_list)/(1.0*sum(self.review_degree_list))

		self.follower_set_of_review = dict()
		for row in self.positive_helpfulness_rating_data:
			follower_id = int(row[0])
			review_id = int(row[1])

			# filter out reviews having less than 2 follower
			if review_helpfulness_counter[review_id]<2:
				continue

			if review_id not in self.follower_set_of_review:
				self.follower_set_of_review[review_id]=[]
			self.follower_set_of_review[review_id].append(follower_id)
	
	def get_sample_reviewer_follower_pair(self,sample_size):
		sampled_review_list = np.random.choice(a=self.review_list, size=sample_size, p=self.review_prob_list)
		return map(
			lambda review_id: 
			[
			int(self.review_id_info[review_id][0])
			,int(np.random.choice(a=self.follower_set_of_review[review_id], size=1)[0])
			]
			, sampled_review_list
			)
	
	def get_sample_follower_follower_pair(self,sample_size):
		sampled_review_list = np.random.choice(a=self.review_list, size=sample_size, p=self.review_prob_list)
		return map(lambda review_id: map(int, np.random.choice(a=self.follower_set_of_review[review_id], size=2)), sampled_review_list)


	##########################################################################################
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
		#  Learn embedding by optimizing the Hierarchical Softmax objective.
		train_data = self.get_string_sentence(train_data)
		# print("learning sentence # :", len(train_data))
		self.user_embedding_model.train(train_data)

	def save_embedding(self):
		self.user_embedding_model.save(self.embedding_output_path)
		
	def load_embedding(self):
		return Word2Vec.load(self.embedding_output_path)

	############################################################
	############ User embedding test function ##################
	############################################################
	def similarity_test(self, model=None):
		if model==None:
			our_model = self.load_embedding()
		else:
			our_model = model
		# print("len model", len(our_model.vocab))
		self.similarity_test_on_real_vote(our_model, 5000, maximum_rating_flag=False)
		self.similarity_test_on_real_vote(our_model, 5000, maximum_rating_flag=True)
			
		try:
			# self.target_item_list = np.load(params.target_item_list_path)
			fake_user_id_list = map(int,list(np.load(self.fake_user_id_list_path)))
			# print 'fake_user_id_list', fake_user_id_list
			origin_user_id_list = list(range(int(np.min(fake_user_id_list)-1)))
			# print('fake_user_id_list', fake_user_id_list)
			
			x_list = fake_user_id_list
			
			print 'Pair(fake,fake)',
			sim_list = []
			for k in xrange(1,len(x_list)-1):
				for i in xrange(len(x_list)-k):
					sim_list.append( our_model.similarity(str(x_list[i]),str(x_list[i+k])) )
			print np.min(sim_list), np.percentile(sim_list, 25), np.median(sim_list), np.percentile(sim_list,75), np.percentile(sim_list, 95)
			if np.min(sim_list)<0.9:
				if np.min(sim_list)<0.8:
					print 'Not good seriously: min(Pair(fake,fake))<0.8'
				else:
					print 'Not good: min(Pair(fake,fake))<0.9'

			# print 'Pair(fake,origin)',
			# x_list = np.random.choice(a=origin_user_id_list,size=300, replace=True)
			# y_list = np.random.choice(a=fake_user_id_list,size=300, replace=True)
			# sim_list = []
			# for i in xrange(300):
			# 	sim_list.append( our_model.similarity(str(x_list[i]),str(y_list[i])) )
			# print np.min(sim_list), np.percentile(sim_list, 25), np.median(sim_list), np.percentile(sim_list,75), np.percentile(sim_list, 95)
			
			self.test_camo_vote(our_model, 300)
		except:
			import sys
			print (sys.exc_info())
			pass
		
		print ''

	def similarity_test_on_real_vote(self, our_model, num_sample, maximum_rating_flag=True):
		sim_list=[]
		data = np.load(self.vote_origin_path)

		if maximum_rating_flag:
			data = data[data[:,2]==5,:]
		
		sample_index_list = np.random.choice(a=len(data), size=num_sample)
		for idx in sample_index_list:
			row = data[idx]
			follower_id = int(row[0])
			review_id = int(row[1])
			# helpful_vote = row[2]

			reviewer_id, item_id = self.review_id_info[review_id]
			sim_list.append(our_model.similarity(str(reviewer_id),str(follower_id)))

		if maximum_rating_flag:
			print 'Pair(origin,origin) (max, observed)',
		else:
			print 'Pair(origin,origin) (any, observed)',
		print np.min(sim_list), np.percentile(sim_list, 25), np.median(sim_list), np.percentile(sim_list,75), np.percentile(sim_list, 95)

	def test_camo_vote(self, our_model, num_sample=10):
		data = np.load(self.vote_camo_path)
		data = data[data[:,2]==5,:]
		
		sim_list=[]
		min_fake = min(map(int,list(np.load(self.fake_user_id_list_path))))
		max_fake = max(map(int,list(np.load(self.fake_user_id_list_path))))
		sample_index_list = np.random.choice(a=len(data), size=num_sample)
		for idx in sample_index_list:
			row = data[idx]
			follower_id = int(row[0])
			review_id = int(row[1])
			# helpful_vote = row[2]

			reviewer_id, item_id = self.review_id_info[review_id]
			try:
				sim_list.append(our_model.similarity(str(reviewer_id),str(follower_id)))
			except Exception, ex:
				continue
		try:
			print 'Pair(origin writer,fake follower) (max, camo helpful)',
			print np.min(sim_list), np.percentile(sim_list, 25), np.median(sim_list), np.percentile(sim_list,75), np.percentile(sim_list, 95)
		except Exception, ex:
			pass
	
	########################################################

	def whole_process(self):
		# data loading 
		self.load_overall_review_matrix()
		self.load_overall_vote_matrix()
		
		self.fill_review_id_info()
		self.fill_observed_user_list()
		
		self.prepare_item_sampling()
		self.prepare_review_sampling()
	
		# Word2Vec init
		self.build_user_vocab()

		# Train on sample
		# iteration = int(100/self.word2vec_iter)
		iteration = 50
		for it in xrange(iteration):
			self.train_embedding_model(self.get_sample_reviewer_follower_pair(10000))
			self.train_embedding_model(self.get_sample_follower_follower_pair(10000))
			self.train_embedding_model(self.get_sample_reviewer_reviewer_pair(10000))
			
			if it%(iteration/5)==0 and it>0:
				print it
				self.similarity_test(self.user_embedding_model)	
		
		self.save_embedding()

# if __name__ == "__main__":
# 	from parameter_controller import *

# 	exp_title = 'bandwagon_1%_1%_1%_emb_32'
# 	print('Experiment Title', exp_title)
# 	params = parse_exp_title(exp_title)
	
# 	u2v_attacked = user2vec(params=params, fake_flag=True, camo_flag=True, embedding_output_path=params.embedding_attacked_path)
# 	u2v_attacked.whole_process()

# 	u2v_clean = user2vec(params=params, fake_flag=False, camo_flag=False, embedding_output_path=params.embedding_clean_path)
# 	u2v_clean.whole_process()
	


