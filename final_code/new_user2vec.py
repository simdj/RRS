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

		# embedding model (Word2Vec)
		self.user_embedding_model = None


		# generate sentence
		self.review_id_info = dict() # review_id_info[review_id]=(reviewer, item)
		self.observed_user_list = set() # needed in Word2Vec initialization


		self.reviewer_tensor = dict()
		self.follower_tensor = dict()
		
		self.num_item = 0
		self.item_degree_list = []
		self.item_weight_list = []


		self.test_vote_matrix =[]

		self.item_rating_data = None
		self.helpfulness_rating_data = None
		self.maximum_item_rating_data = None # maximum_item_rating_data is subset of review matrix
		self.positive_helpfulness_rating_data = None #positive_helpfulness_rating_data is subset of vote matrix

		self.positive_helpful_threshold = 5

		## sampling!!!
		self.review_list = []
		self.review_degree_list = []

	def load_overall_review_matrix(self):
		overall_review_matrix = np.load(self.review_origin_path)
		if self.fake_flag:
			overall_review_matrix = np.concatenate((overall_review_matrix, np.load(self.review_fake_path)))
			if self.camo_flag:
				overall_review_matrix = np.concatenate((overall_review_matrix, np.load(self.review_camo_path)))

		self.item_rating_data = overall_review_matrix
		self.maximum_item_rating_data = overall_review_matrix[overall_review_matrix[:,2]==5,:]
		# return overall_review_matrix

	def load_overall_vote_matrix(self):
		overall_vote_matrix = np.load(self.vote_origin_path)
		if self.fake_flag:
			overall_vote_matrix = np.concatenate((overall_vote_matrix, np.load(self.vote_fake_path)))
			if self.camo_flag:
				try:
					overall_vote_matrix = np.concatenate((overall_vote_matrix, np.load(self.vote_camo_path)))
				except:
					print 'no camo vote in new_user2vec.py'
		
		self.helpfulness_rating_data = overall_vote_matrix
		self.positive_helpfulness_rating_data = overall_vote_matrix[overall_vote_matrix[:,2]>=self.positive_helpful_threshold,:]
		# return overall_vote_matrix

	def fill_review_id_info(self):
		assert(len(self.item_rating_data)>0)

		for row in self.item_rating_data:
			reviewer = int(row[0])
			item_id = int(row[1])
			rating_value = row[2]
			review_id = int(row[3])

			# needed in follower_tensor
			self.review_id_info[review_id] = (reviewer, item_id)

	def fill_observed_user_list(self):
		assert(len(self.helpfulness_rating_data)>0)
		assert(len(self.review_id_info)>0)

		for row in self.helpfulness_rating_data:
			voter = int(row[0])
			review_id = int(row[1])
			helpful_vote = row[2]

			reviewer, item_id = self.review_id_info[review_id]

			# fill observed user list
			self.observed_user_list.add(reviewer)
			self.observed_user_list.add(voter)


	# def construct_item_weight(self):
	# 	# 1. #(reviewer)
	# 	# 2. #(reviewer U voter) <-- good
	# 	item_related_user_set = dict()
	# 	review_id_item_id_mapping = dict()

	# 	# 1) add reviewer

	# 	overall_review_matrix = self.load_overall_review_matrix()
	# 	for row in overall_review_matrix:
	# 		reviewer = int(row[0])
	# 		item_id = int(row[1])
	# 		# rating_value = row[2]
	# 		review_id = int(row[3])
	# 		if item_id not in item_related_user_set:
	# 			item_related_user_set[item_id] = set()
	# 		item_related_user_set[item_id].add(reviewer)

	# 		review_id_item_id_mapping[review_id] = item_id

	# 	# 2) add reviewer

	# 	overall_vote_matrix = self.load_overall_vote_matrix()
	# 	for row in overall_vote_matrix:
	# 		voter = int(row[0])
	# 		review_id = int(row[1])
	# 		# helpful_vote = row[2]

	# 		item_id = review_id_item_id_mapping[review_id]
	# 		if item_id not in item_related_user_set:
	# 			item_related_user_set[item_id] = set()
	# 		item_related_user_set[item_id].add(voter)

	# 	# mainly compute weight of items
	# 	self.num_item = len(np.unique(overall_review_matrix[:, 1]))
	# 	self.item_degree_list = np.zeros((self.num_item, 1)).reshape([-1])
	# 	self.item_weight_list = np.zeros((self.num_item, 1)).reshape([-1])
	# 	for item_id, user_set in item_related_user_set.iteritems():
	# 		self.item_degree_list[item_id] = len(user_set)
	# 	self.item_weight_list = 1.0 / np.log2(self.item_degree_list + 5)
	# 	# self.item_weight_list = 1.0 / (self.item_degree_list + 5)
	# 	normalization = 1.0 * np.sum(self.item_weight_list)
	# 	self.item_weight_list = self.item_weight_list/normalization
		


	# 	# print 'degree min', np.min(self.item_degree_list), np.max(self.item_weight_list)
	# 	# percentile_value = [1,5,10,25,50,75,90,95,99]
	# 	# for pv in percentile_value:
	# 	# 	print pv, '%%', np.percentile(self.item_degree_list, pv), np.percentile(self.item_weight_list, 100-pv)
	# 	# print 'degree max', np.max(self.item_degree_list), np.min(self.item_weight_list)

	# 	# print 'TARGET ITEM', self.item_degree_list[616], self.item_weight_list[616]

	# 	# target = np.argmax(self.item_degree_list>707)
	# 	# print self.item_degree_list[target], self.item_weight_list[target]
	# 	# # drawing degree plot
	# 	# item_degree_list = np.array(item_degree_list)
	# 	# hist, bins = np.histogram(item_degree_list, bins=max(item_degree_list)+1)
	# 	# print hist
	# 	# print bins
	# 	# %matplotlib inline

	# 	# import matplotlib
	# 	# import matplotlib.pyplot as plt
	# 	# width = 1.0 * (bins[1] - bins[0])
	# 	# center = (bins[:-1] + bins[1:]) / 2
	# 	# plt.bar(center, hist, align='center', width=width)
	# 	# plt.show()

	# def construct_reviewer_tensor(self, max_item_rating=5):
	# 	# data loading
	# 	overall_review_matrix = self.load_overall_review_matrix()
		
	# 	self.reviewer_tensor_degree = np.zeros((self.num_item,))
	# 	self.reviewer_tensor_weight = np.zeros((self.num_item,))
		
	# 	for row in overall_review_matrix:
	# 		reviewer = int(row[0])
	# 		item_id = int(row[1])
	# 		rating_value = row[2]
	# 		review_id = int(row[3])

	# 		# needed in follower_tensor
	# 		self.review_id_info[review_id] = (reviewer, item_id)

	# 		# only maximum value!!!!!!!!!!!!!!!!!!!!!!!
	# 		if rating_value < max_item_rating:
	# 			continue
			

	# 		if item_id not in self.reviewer_tensor:
	# 			self.reviewer_tensor[item_id]=[]
	# 		self.reviewer_tensor[item_id].append(reviewer)
	# 		self.reviewer_tensor_degree[item_id]+=1
	# 	normalization = sum(self.reviewer_tensor_degree)*1.0
	# 	self.reviewer_tensor_weight=np.array(self.reviewer_tensor_degree)/normalization

	# def construct_follower_tensor(self, max_helpful_rating=5):
	# 	overall_vote_matrix = self.load_overall_vote_matrix()
	# 	for row in overall_vote_matrix:
	# 		voter = int(row[0])
	# 		review_id = int(row[1])
	# 		helpful_vote = row[2]

	# 		reviewer, item_id = self.review_id_info[review_id]

	# 		# fill observed user list
	# 		self.observed_user_list.add(reviewer)
	# 		self.observed_user_list.add(voter)

	# 		# note that helpfulness vote threshold is important!!!!
	# 		# usually max_helpful_rating 5
	# 		# only maximum value!!!!!!!!!!!!!!!!!!!!!!!
	# 		if helpful_vote < max_helpful_rating:
	# 			continue

	# 		# fill follower_tensor 
	# 		# concept: follower tensor[item][user] indicates follower list
	# 		# follower_tensor[item_id] entry has
	# 		# reviewer_list : [reviewers of item]
	# 		# follower_num_list : [num of followers reviewers[i] of item]
	# 		# follower_2dlist : [list of followers reviewers[i] of item]

	# 		if item_id not in self.follower_tensor:
	# 			self.follower_tensor[item_id] = follower_tensor_entry()
	# 		follower_info = self.follower_tensor[item_id]
	# 		if reviewer not in follower_info.reviewer_list:
	# 			follower_info.reviewer_list.append(reviewer)
	# 			follower_info.follower_num_list.append([0])
	# 			follower_info.reviewer_weight_list.append([0])
	# 			follower_info.follower_2dlist.append([])

	# 		target_index = follower_info.reviewer_list.index(reviewer)
	# 		follower_info.follower_2dlist[target_index].append(voter)

	# 	for item_id,follower_info in self.follower_tensor.iteritems():
	# 		num_reviewer = len(follower_info.reviewer_list)
	# 		follower_info.len_reviewer_list=num_reviewer
	# 		for i in xrange(num_reviewer):
	# 			follower_info.follower_num_list[i]=len(follower_info.follower_2dlist[i])
	# 			follower_info.reviewer_weight_list[i] = 1.0/np.log2(follower_info.follower_num_list[i]+5)
	# 		follower_info.reviewer_weight_list = np.array(follower_info.reviewer_weight_list)/sum(follower_info.reviewer_weight_list)*1.0

	# ##################################################
	# ############## enumerating all pair ##############
	# ##################################################
	
	# def enumerate_reviewer_pair(self):
	# 	ret = []
	# 	return ret
	
	# def enumerate_follower_pair(self):
	# 	ret = []

	# 	good_cnt = 0
	# 	bad_cnt = 0
	# 	etc_cnt =0
	# 	min_fake = min(map(int,list(np.load(self.fake_user_id_list_path))))
	# 	max_fake = max(map(int,list(np.load(self.fake_user_id_list_path))))

	# 	supporter_counter = Counter(self.positive_helpfulness_rating_data[:,1])
	# 	# review_at_least_two_supporters = Counter(np.unique(self.positive_helpfulness_rating_data[:,1])).most_common()
	# 	# review_at_least_two_supporters = map(lambda x:int(x[0]), filter(lambda x:x[1]>1,review_at_least_two_supporters))
	# 	supporters_union = dict()
	# 	for row in self.positive_helpfulness_rating_data:
	# 		follower_id = int(row[0])
	# 		review_id = int(row[1])
	# 		# helpful_vote = row[2]
	# 		if supporter_counter[review_id]<2:
	# 			continue

	# 		if review_id not in supporters_union:
	# 			supporters_union[review_id]=[]
	# 		supporters_union[review_id].append(follower_id)

	# 		# reviewer_id, item_id = self.review_id_info[review_id]
	# 	for item_id, user_set in supporters_union.iteritems():
	# 		user_set_len = len(user_set)
	# 		for i in xrange(user_set_len):
	# 			for k in xrange(1, user_set_len-i):
	# 				ret.append([int(user_set[i]),int(user_set[i+k])])
	# 				# ret.append([str(user_set[i+k]),str(user_set[i])])

	# 				is_follower_fake = (int(user_set[i])>=min_fake and int(user_set[i])<=max_fake)
	# 				is_reviewer_fake = (int(user_set[i+k])>=min_fake and int(user_set[i+k])<=max_fake)

	# 				if is_follower_fake and is_reviewer_fake:
	# 					good_cnt+=1
	# 				elif is_follower_fake and not is_reviewer_fake:
	# 					bad_cnt+=1
	# 				elif not is_follower_fake and is_reviewer_fake:
	# 					bad_cnt+=1
	# 				elif int(user_set[i])>max_fake or int(user_set[i+k])>max_fake:
	# 					etc_cnt+=1
	# 	total_evidence = len(ret)*1.0
	# 	print 'good', good_cnt, 'total', total_evidence
	# 	print 'enumerate_follower_pair', total_evidence, 'good', good_cnt/total_evidence*100, 'bad', bad_cnt/total_evidence*100, 'only voter', etc_cnt/total_evidence*100, 'normal', (total_evidence-good_cnt-bad_cnt-etc_cnt)/total_evidence*100
	# 	# result
	# 	# ('enumerate_follower_pair', 311254.0, good 0.8857717491180837, bad 0.1664235640345184, only voter 84.99649803697302, normal voter 13.951306649874379)
	# 	return ret

	# def enumerate_reviewer_follower(self):
	# 	ret = []
	# 	# for stats
	# 	good_cnt = 0
	# 	bad_cnt = 0
	# 	etc_cnt =0
	# 	min_fake = min(map(int,list(np.load(self.fake_user_id_list_path))))
	# 	max_fake = max(map(int,list(np.load(self.fake_user_id_list_path))))

	# 	for row in self.positive_helpfulness_rating_data:
	# 		follower_id = int(row[0])
	# 		review_id = int(row[1])
	# 		# helpful_vote = row[2]

	# 		reviewer_id, item_id = self.review_id_info[review_id]
	# 		ret.append([follower_id,reviewer_id])
	# 		# ret.append([reviewer_id,follower_id])

	# 		is_follower_fake = (int(follower_id)>=min_fake and int(follower_id)<=max_fake)
	# 		is_reviewer_fake = (int(reviewer_id)>=min_fake and int(reviewer_id)<=max_fake)

	# 		if is_follower_fake and is_reviewer_fake:
	# 			good_cnt+=1
	# 		elif is_follower_fake and not is_reviewer_fake:
	# 			bad_cnt+=1
	# 		elif not is_follower_fake and is_reviewer_fake:
	# 			bad_cnt+=1
	# 		elif int(reviewer_id)>max_fake or int(follower_id)>max_fake:
	# 			etc_cnt+=1
	# 	total_evidence = len(ret)*1.0
	# 	print 'good', good_cnt, 'total', total_evidence
	# 	print 'enumerate_reviewer_follower', total_evidence, 'good', good_cnt/total_evidence*100, 'bad', bad_cnt/total_evidence*100, 'only voter', etc_cnt/total_evidence*100, 'normal', (total_evidence-good_cnt-bad_cnt-etc_cnt)/total_evidence*100
	# 	# result
	# 	# 	('generate_Reviewer_Follower', 59212.0, good 0.5471863811389583, bad 0.0, only voter 31.128825238127405, normal voter 68.32398838073364)
	# 	# 	after 20~ actual word2vec iteration is 100
	# 	# 	fake-fake 0.998925349849 0.999381386031 0.999535296844 0.999648611035 0.99978642086
	# 	# 	fake-origin -0.625554980127 -0.25086028799 -0.124479815067 0.0644907673583 0.286812664282
	# 	# 	real vote (origin-helper) -0.703560157279 0.274187644612 0.461382565885 0.619609605674 0.774940525212
	# 	return ret

	###################################################################################
	############## sample reviews  and randomly choose one pair per review ############
	###################################################################################
	def data_structure_review(self):
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
			if review_helpfulness_counter[review_id]<2:
				continue

			if review_id not in self.follower_set_of_review:
				self.follower_set_of_review[review_id]=[]
			self.follower_set_of_review[review_id].append(follower_id)
	

	def get_sample_reviewer_follower_pair(self,sample_size):
		sampled_review_list = np.random.choice(a=self.review_list, size=sample_size, p=self.review_prob_list)

		# reviewer_id, item_id = self.review_id_info[review_id]
		# sampled_reviewer_list = map(lambda review_id: int(self.review_id_info[review_id][0]), sampled_review_list)
		# sampled_follower_list = map(lambda review_id: int(np.random.choice(a=self.follower_set_of_review[review_id], size=1)[0]), sampled_review_list)

		# return zip(sampled_reviewer_list,sampled_follower_list)
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

		# reviewer_id, item_id = self.review_id_info[review_id]
		# sampled_reviewer_list = map(lambda review_id: int(self.review_id_info[review_id][0]), sampled_review_list)
		# sampled_follower_list = map(lambda review_id: int(np.random.choice(a=self.follower_set_of_review[review_id], size=1)[0]), sampled_review_list)

		# return zip(sampled_reviewer_list,sampled_follower_list)
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
		# self.target_item_list = np.load(params.target_item_list_path)
		fake_user_id_list = map(int,list(np.load(self.fake_user_id_list_path)))
		# print 'fake_user_id_list', fake_user_id_list
		origin_user_id_list = list(range(int(np.min(fake_user_id_list)-1)))

		if model==None:
			our_model = self.load_embedding()
		else:
			our_model = model
		# print("len model", len(our_model.vocab))
		# print('fake_user_id_list', fake_user_id_list)
		self.similarity_test_on_real_vote(our_model, 5000, maximum_rating_flag=False)
		self.similarity_test_on_real_vote(our_model, 5000, maximum_rating_flag=True)
			
		try:
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
	# def old_whole_process(self):

	# 	self.construct_item_weight()

	# 	self.construct_reviewer_tensor()
	# 	self.construct_follower_tensor()

	# 	# Word2Vec init
	# 	self.build_user_vocab()
		
		
	# 	all_follower_pair = self.enumerate_follower_pair()
	# 	all_reviewer_follower = self.enumerate_reviewer_follower()

	# 	iteration = int(100/self.word2vec_iter)
	# 	if not self.fake_flag:
	# 		iteration/=5
	# 	for it in xrange(iteration):
	# 		self.train_embedding_model(all_follower_pair)
	# 		self.train_embedding_model(all_reviewer_follower)
	# 		# if it%10==0 and it>0:
	# 		# 	print it
	# 	self.similarity_test(self.user_embedding_model)	
		
	# 	self.save_embedding()

	# def amplify_cooccurence(self,corpus):
	# 	# stats
	# 	min_fake = min(map(int,list(np.load(self.fake_user_id_list_path))))
	# 	max_fake = max(map(int,list(np.load(self.fake_user_id_list_path))))
	# 	good_cnt=0
	# 	bad_cnt=0
	# 	etc_cnt=0
	# 	total_evidence=0

	# 	self.pair_counter = dict()


	# 	for x,y in corpus:
	# 		key = str(int(x))+','+str(int(y))
	# 		if key not in self.pair_counter:
	# 			self.pair_counter[key]=0
	# 		self.pair_counter[key]+=1
		

	# 	print 'len(self.pair_counter)', len(self.pair_counter)
	# 	ret = []
		

	# 	test_cnt = Counter()
	# 	for key, co_occur in self.pair_counter.iteritems():
	# 		test_cnt[co_occur]+=1
	# 		# if co_occur>1:
	# 		x,y = key.split(',')
	# 		ret.extend([[x,y]]*(co_occur))
	# 		ret.extend([[y,x]]*(co_occur))
			
	# 		is_follower_fake = (int(x)>=min_fake and int(x)<=max_fake)
	# 		is_reviewer_fake = (int(y)>=min_fake and int(y)<=max_fake)

	# 		if is_follower_fake and is_reviewer_fake:
	# 			good_cnt+=co_occur
	# 		elif is_follower_fake and not is_reviewer_fake:
	# 			bad_cnt+=co_occur
	# 		elif not is_follower_fake and is_reviewer_fake:
	# 			bad_cnt+=co_occur
	# 		elif int(x)>max_fake or int(y)>max_fake:
	# 			etc_cnt+=co_occur
	# 		total_evidence+=1
				
	# 	total_evidence = len(ret)*1.0
	# 	print 'good', good_cnt, 'train data length', len(ret)
	# 	print 'STATS', total_evidence, 'good', good_cnt/total_evidence*100, 'bad', bad_cnt/total_evidence*100, 'only voter', etc_cnt/total_evidence*100, 'normal', (total_evidence-good_cnt-bad_cnt-etc_cnt)/total_evidence*100
	# 	print 'coocur counter', test_cnt
	# 	return ret

	def whole_process(self):
		self.load_overall_review_matrix()
		self.load_overall_vote_matrix()
		self.fill_review_id_info()
		self.fill_observed_user_list()

		# self.construct_item_weight()

		# self.construct_reviewer_tensor()
		# self.construct_follower_tensor()

		# Word2Vec init
		self.build_user_vocab()
		

		#################
		# all_follower_pair = self.enumerate_follower_pair()
		# all_reviewer_follower = self.enumerate_reviewer_follower()

		# corpus = np.concatenate((np.array(all_follower_pair), np.array(all_reviewer_follower)))
		# # print 'before', corpus[0]
		# # print 'corpus.shape', corpus.shape
		# self.amplify_cooccurence(corpus)


		################# sample
		self.data_structure_review()
		# iteration = int(100/self.word2vec_iter)
		iteration = 100
		for it in xrange(iteration):
			if it%2==0:
				train_corpus = self.get_sample_reviewer_follower_pair(10000)
			else:
				train_corpus = self.get_sample_follower_follower_pair(10000)
			# print 'after', train_corpus[0]
			self.train_embedding_model(train_corpus)
			# self.train_embedding_model(train_corpus[:10000])
			# self.train_embedding_model(all_follower_pair)
			# self.train_embedding_model(all_reviewer_follower)
			if it%10==0 and it>0:
				print it
				self.similarity_test(self.user_embedding_model)	
		
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
	


