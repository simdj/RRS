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

		self.reviewer_tensor = dict()
		self.follower_tensor = dict()
		
		self.num_item = 0
		self.item_degree_list = []
		self.item_weight_list = []

		self.observed_user_list = set()

		self.test_vote_matrix =[]

		self.item_rating_data = None
		self.helpfulness_rating_data = None
		self.maximum_item_rating_data = None # maximum_item_rating_data is subset of review matrix
		self.maximum_helpfulness_rating_data = None #maximum_helpfulness_rating_data is subset of vote matrix

	def load_overall_review_matrix(self):
		overall_review_matrix = np.load(self.review_origin_path)
		if self.fake_flag:
			overall_review_matrix = np.concatenate((overall_review_matrix, np.load(self.review_fake_path)))
			if self.camo_flag:
				overall_review_matrix = np.concatenate((overall_review_matrix, np.load(self.review_camo_path)))

		self.item_rating_data = overall_review_matrix
		self.maximum_item_rating_data = overall_review_matrix[overall_review_matrix[:,2]==5,:]
		return overall_review_matrix

	def load_overall_vote_matrix(self):
		overall_vote_matrix = np.load(self.vote_origin_path)
		if self.fake_flag:
			overall_vote_matrix = np.concatenate((overall_vote_matrix, np.load(self.vote_fake_path)))
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# if self.camo_flag:
		# 	overall_vote_matrix = np.concatenate((overall_vote_matrix, np.load(self.vote_camo_path)))
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		self.helpfulness_rating_data = overall_vote_matrix
		self.maximum_helpfulness_rating_data = overall_vote_matrix[overall_vote_matrix[:,2]==5,:]
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
		self.item_weight_list = 1.0 / np.log2(self.item_degree_list + 5)
		# self.item_weight_list = 1.0 / (self.item_degree_list + 5)
		normalization = 1.0 * np.sum(self.item_weight_list)
		self.item_weight_list = self.item_weight_list/normalization
		


		print 'degree min', np.min(self.item_degree_list), np.max(self.item_weight_list)
		percentile_value = [1,5,10,25,50,75,90,95,99]
		for pv in percentile_value:
			print pv, '%%', np.percentile(self.item_degree_list, pv), np.percentile(self.item_weight_list, 100-pv)
		print 'degree max', np.max(self.item_degree_list), np.min(self.item_weight_list)

		print 'TARGET ITEM', self.item_degree_list[616], self.item_weight_list[616]
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

	def construct_reviewer_tensor(self, max_item_rating=5):
		# data loading
		overall_review_matrix = self.load_overall_review_matrix()
		
		self.reviewer_tensor_degree = np.zeros((self.num_item,))
		self.reviewer_tensor_weight = np.zeros((self.num_item,))
		
		for row in overall_review_matrix:
			reviewer = int(row[0])
			item_id = int(row[1])
			rating_value = row[2]
			review_id = int(row[3])

			# needed in follower_tensor
			self.review_id_info[review_id] = (reviewer, item_id)

			# only maximum value!!!!!!!!!!!!!!!!!!!!!!!
			if rating_value < max_item_rating:
				continue
			

			if item_id not in self.reviewer_tensor:
				self.reviewer_tensor[item_id]=[]
			self.reviewer_tensor[item_id].append(reviewer)
			self.reviewer_tensor_degree[item_id]+=1
		normalization = sum(self.reviewer_tensor_degree)*1.0
		self.reviewer_tensor_weight=np.array(self.reviewer_tensor_degree)/normalization

	def construct_follower_tensor(self, max_helpful_rating=5):
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
			# usually max_helpful_rating 5
			# only maximum value!!!!!!!!!!!!!!!!!!!!!!!
			if helpful_vote < max_helpful_rating:
				continue

			# fill follower_tensor 
			# concept: follower tensor[item][user] indicates follower list
			# follower_tensor[item_id] entry has
			# reviewer_list : [reviewers of item]
			# follower_num_list : [num of followers reviewers[i] of item]
			# follower_2dlist : [list of followers reviewers[i] of item]

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
			follower_info.reviewer_weight_list = np.array(follower_info.reviewer_weight_list)/sum(follower_info.reviewer_weight_list)*1.0

	def generate_reviewer_reviewer(self, num_sample):
		ret=[]
		good_cnt = 0
		bad_cnt = 0
		etc_cnt =0
		min_fake = min(map(int,list(np.load(self.fake_user_id_list_path))))
		max_fake = max(map(int,list(np.load(self.fake_user_id_list_path))))
		
		# sample probability is propotional to the inverse of popularity
		# sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample, p=self.reviewer_tensor_weight)

		# uniform sample probability
		sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample)
		
		for item_id in sampled_item_id_list:
			try:
				sample_reviewer_id_list = np.random.choice(a=self.reviewer_tensor[item_id], size=2)
			except Exception, ex:
				continue
			ret.append(sample_reviewer_id_list)

			is_follower_fake = (int(sample_reviewer_id_list[0])>=min_fake and int(sample_reviewer_id_list[0])<=max_fake)
			is_reviewer_fake = (int(sample_reviewer_id_list[1])>=min_fake and int(sample_reviewer_id_list[1])<=max_fake)

			if is_follower_fake and is_reviewer_fake:
				good_cnt+=1
			elif is_follower_fake and not is_reviewer_fake:
				bad_cnt+=1
			elif not is_follower_fake and is_reviewer_fake:
				bad_cnt+=1
			elif int(sample_reviewer_id_list[0])>max_fake or int(sample_reviewer_id_list[1])>max_fake:
				etc_cnt+=1
				# ret.pop()
				
		# # sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample/2)
		# for item_id in sampled_item_id_list:
		# 	sample_reviewer_info = self.reviewer_tensor[item_id]
		# 	# sample_rating_index_list = np.random.choice(a=sample_reviewer_info.len_rating_list, size=2,p=sample_reviewer_info.rating_reviewer_weight_list)
		# 	sample_rating_index_list = np.random.choice(a=sample_reviewer_info.len_rating_list, size=2)

		# 	for idx in sample_rating_index_list:
		# 		try:
		# 			sample_reviewer_id_list = np.random.choice(a=sample_reviewer_info.reviewer_2dlist[idx], size=2, replace=False)
		# 			ret.append(sample_reviewer_id_list)
		# 			### stats
		# 			is_follower_fake = (int(sample_reviewer_id_list[0])>=min_fake and int(sample_reviewer_id_list[0])<=max_fake)
		# 			is_reviewer_fake = (int(sample_reviewer_id_list[1])>=min_fake and int(sample_reviewer_id_list[1])<=max_fake)

		# 			if is_follower_fake and is_reviewer_fake:
		# 				good_cnt+=1
		# 			elif is_follower_fake and not is_reviewer_fake:
		# 				bad_cnt+=1
		# 			elif not is_follower_fake and is_reviewer_fake:
		# 				bad_cnt+=1
		# 			elif int(sample_reviewer_id_list[0])>max_fake or int(sample_reviewer_id_list[1])>max_fake:
		# 				etc_cnt+=1
		# 				# ret.pop()
		# 		except Exception, ex:
		# 			pass
		total_evidence = len(ret)*1.0
		print('generate_Reviewer pair', total_evidence, good_cnt/total_evidence*100, bad_cnt/total_evidence*100, etc_cnt/total_evidence*100, (total_evidence-good_cnt-bad_cnt-etc_cnt)/total_evidence*100)
		# print('generate_reviewer_follower', total_evidence, good_cnt/total_evidence*100, bad_cnt/total_evidence*100,  (total_evidence-good_cnt-bad_cnt)/total_evidence*100)
		
		return ret

	def generate_reviewer_follower(self, num_sample):
		ret=[]
		good_cnt = 0
		bad_cnt = 0
		etc_cnt =0
		min_fake = min(map(int,list(np.load(self.fake_user_id_list_path))))
		max_fake = max(map(int,list(np.load(self.fake_user_id_list_path))))
		
		# # follower_tensor
		# # sample probability is propotional to the inverse of popularity
		# sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample, p=self.item_weight_list)
		
		# uniform sample probability
		sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample)

		for item_id in sampled_item_id_list:
			try:
				sample_follower_info = self.follower_tensor[item_id]
			except Exception, ex:
				continue
			# sample_reviewer_index_list = np.random.choice(a=sample_follower_info.len_reviewer_list, size=1, p=sample_follower_info.reviewer_weight_list)
			# uniform sample probability
			sample_reviewer_index_list = np.random.choice(a=sample_follower_info.len_reviewer_list, size=1)
			
			for idx in sample_reviewer_index_list:
				sample_follower_id_list = np.random.choice(a=sample_follower_info.follower_2dlist[idx],size=1)
				reviewer_id = sample_follower_info.reviewer_list[idx]
				# reviewer-follower
				for follower_id in sample_follower_id_list:
					ret.append([reviewer_id,follower_id])

					### stats
					is_follower_fake = (int(follower_id)>=min_fake and int(follower_id)<=max_fake)
					is_reviewer_fake = (int(reviewer_id)>=min_fake and int(reviewer_id)<=max_fake)

					if is_follower_fake and is_reviewer_fake:
						good_cnt+=1
					elif is_follower_fake and not is_reviewer_fake:
						bad_cnt+=1
					elif not is_follower_fake and is_reviewer_fake:
						bad_cnt+=1
					elif int(reviewer_id)>max_fake or int(follower_id)>max_fake:
						etc_cnt+=1
						# ret.pop()
		total_evidence = len(ret)*1.0
		print('generate_Reviewer_Follower', total_evidence, good_cnt/total_evidence*100, bad_cnt/total_evidence*100, etc_cnt/total_evidence*100, (total_evidence-good_cnt-bad_cnt-etc_cnt)/total_evidence*100)
		# print('generate_reviewer_follower', total_evidence, good_cnt/total_evidence*100, bad_cnt/total_evidence*100,  (total_evidence-good_cnt-bad_cnt)/total_evidence*100)
		return ret

	def generate_follower_follower(self, num_sample):
		ret=[]
		good_cnt = 0
		bad_cnt = 0
		etc_cnt =0
		min_fake = min(map(int,list(np.load(self.fake_user_id_list_path))))
		max_fake = max(map(int,list(np.load(self.fake_user_id_list_path))))

		# # follower_tensor
		# # sample probability is propotional to the inverse of popularity
		# sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample/5, p=self.item_weight_list)
		
		# uniform sample probability
		sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample)
		
		for item_id in sampled_item_id_list:
			try:
				sample_follower_info = self.follower_tensor[item_id]
			except Exception, ex:
				continue
			# sample_reviewer_index_list = np.random.choice(a=sample_follower_info.len_reviewer_list, size=1, p=sample_follower_info.reviewer_weight_list)
			# uniform sample probability
			sample_reviewer_index_list = np.random.choice(a=sample_follower_info.len_reviewer_list, size=1)
			
			for idx in sample_reviewer_index_list:
				try:
					sample_follower_id_list = np.random.choice(a=sample_follower_info.follower_2dlist[idx], size=2, replace=False)
					ret.append(sample_follower_id_list)
					### stats
					is_follower_fake = (int(sample_follower_id_list[0])>=min_fake and int(sample_follower_id_list[0])<=max_fake)
					is_reviewer_fake = (int(sample_follower_id_list[1])>=min_fake and int(sample_follower_id_list[1])<=max_fake)

					if is_follower_fake and is_reviewer_fake:
						good_cnt+=1
					elif is_follower_fake and not is_reviewer_fake:
						bad_cnt+=1
					elif not is_follower_fake and is_reviewer_fake:
						bad_cnt+=1
					elif int(sample_follower_id_list[0])>max_fake or int(sample_follower_id_list[1])>max_fake:
						etc_cnt+=1
						# ret.pop()
				except Exception, ex:
					pass
		total_evidence = len(ret)*1.0
		print('generate_Follower pair', total_evidence, good_cnt/total_evidence*100, bad_cnt/total_evidence*100, etc_cnt/total_evidence*100, (total_evidence-good_cnt-bad_cnt-etc_cnt)/total_evidence*100)
		# print('generate_reviewer_follower', total_evidence, good_cnt/total_evidence*100, bad_cnt/total_evidence*100,  (total_evidence-good_cnt-bad_cnt)/total_evidence*100)
		return ret

	##################################################
	############## enumerating all pair ##############
	##################################################
	def new_generate_reviewer_reviewer(self, num_sample):
		ret=[]
		good_cnt = 0
		bad_cnt = 0
		etc_cnt =0
		min_fake = min(map(int,list(np.load(self.fake_user_id_list_path))))
		max_fake = max(map(int,list(np.load(self.fake_user_id_list_path))))
		
		# sample probability is propotional to the inverse of popularity
		sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample, p=self.reviewer_tensor_weight)

		# # uniform sample probability
		# sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample)
		
		for item_id in sampled_item_id_list:
			try:
				sample_reviewer_id_list = np.random.choice(a=self.reviewer_tensor[item_id], size=2)
			except Exception, ex:
				continue
			ret.append(sample_reviewer_id_list)

			is_follower_fake = (int(sample_reviewer_id_list[0])>=min_fake and int(sample_reviewer_id_list[0])<=max_fake)
			is_reviewer_fake = (int(sample_reviewer_id_list[1])>=min_fake and int(sample_reviewer_id_list[1])<=max_fake)

			if is_follower_fake and is_reviewer_fake:
				good_cnt+=1
			elif is_follower_fake and not is_reviewer_fake:
				bad_cnt+=1
			elif not is_follower_fake and is_reviewer_fake:
				bad_cnt+=1
			elif int(sample_reviewer_id_list[0])>max_fake or int(sample_reviewer_id_list[1])>max_fake:
				etc_cnt+=1
				# ret.pop()
				
		total_evidence = len(ret)*1.0
		print('generate_Reviewer pair', total_evidence, good_cnt/total_evidence*100, bad_cnt/total_evidence*100, etc_cnt/total_evidence*100, (total_evidence-good_cnt-bad_cnt-etc_cnt)/total_evidence*100)
		# print('generate_reviewer_follower', total_evidence, good_cnt/total_evidence*100, bad_cnt/total_evidence*100,  (total_evidence-good_cnt-bad_cnt)/total_evidence*100)
		
		return ret

	def enumerate_reviewer_pair(self):
		ret = []
		return ret
	def enumerate_follower_pair(self):
		ret = []
		return ret

	def enumerate_reviewer_follower(self):
		ret = []
		# for stats
		good_cnt = 0
		bad_cnt = 0
		etc_cnt =0
		min_fake = min(map(int,list(np.load(self.fake_user_id_list_path))))
		max_fake = max(map(int,list(np.load(self.fake_user_id_list_path))))

		for row in self.maximum_helpfulness_rating_data:
			follower_id = int(row[0])
			review_id = int(row[1])
			# helpful_vote = row[2]

			reviewer_id, item_id = self.review_id_info[review_id]
			ret.append([follower_id,reviewer_id])
			ret.append([reviewer_id,follower_id])

			is_follower_fake = (int(follower_id)>=min_fake and int(follower_id)<=max_fake)
			is_reviewer_fake = (int(reviewer_id)>=min_fake and int(reviewer_id)<=max_fake)

			if is_follower_fake and is_reviewer_fake:
				good_cnt+=1
			elif is_follower_fake and not is_reviewer_fake:
				bad_cnt+=1
			elif not is_follower_fake and is_reviewer_fake:
				bad_cnt+=1
			elif int(reviewer_id)>max_fake or int(follower_id)>max_fake:
				etc_cnt+=1
		total_evidence = len(ret)*1.0
		print('generate_Reviewer_Follower', total_evidence, good_cnt/total_evidence*100, bad_cnt/total_evidence*100, etc_cnt/total_evidence*100, (total_evidence-good_cnt-bad_cnt-etc_cnt)/total_evidence*100)
		# result
		# 	('generate_Reviewer_Follower', 59212.0, good 0.5471863811389583, bad 0.0, only voter 31.128825238127405, normal voter 68.32398838073364)
		# 	after 20~ actual word2vec iteration is 100
		# 	fake-fake 0.998925349849 0.999381386031 0.999535296844 0.999648611035 0.99978642086
		# 	fake-origin -0.625554980127 -0.25086028799 -0.124479815067 0.0644907673583 0.286812664282
		# 	real vote (origin-helper) -0.703560157279 0.274187644612 0.461382565885 0.619609605674 0.774940525212
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
		#  Learn embedding by optimizing the Hierarchical Softmax objective.
		train_data = self.get_string_sentence(train_data)
		# print("learning sentence # :", len(train_data))
		self.user_embedding_model.train(train_data)


	def save_embedding(self):
		self.user_embedding_model.save(self.embedding_output_path)
		

	def load_embedding(self):
		return Word2Vec.load(self.embedding_output_path)

	def similarity_test(self, model=None):
		# self.target_item_list = np.load(params.target_item_list_path)
		fake_user_id_list = map(int,list(np.load(self.fake_user_id_list_path)))
		origin_user_id_list = list(range(int(np.min(fake_user_id_list)-1)))

		# print(fake_user_id_list)
		# print(origin_user_id_list)
		if model==None:
			our_model = self.load_embedding()
		else:
			our_model = model
		# print("len model", len(our_model.vocab))
		# print('fake_user_id_list', fake_user_id_list)

		try:
			# print("test", fake_user_id_list[0], our_mode[str(fake_user_id_list[0])])
			x_list = fake_user_id_list
			sim_list = []
			for k in xrange(1,len(x_list)-1):
				for i in xrange(len(x_list)-k):
					sim_list.append( our_model.similarity(str(x_list[i]),str(x_list[i+k])) )
			
			print 'fake-fake',
			print np.min(sim_list), np.percentile(sim_list, 25), np.median(sim_list), np.percentile(sim_list,75), np.percentile(sim_list, 95)

			x_list = np.random.choice(a=origin_user_id_list,size=300, replace=True)
			y_list = np.random.choice(a=fake_user_id_list,size=300, replace=True)
			sim_list = []
			for i in xrange(300):
				sim_list.append( our_model.similarity(str(x_list[i]),str(y_list[i])) )
			print 'fake-origin',
			print np.min(sim_list), np.percentile(sim_list, 25), np.median(sim_list), np.percentile(sim_list,75), np.percentile(sim_list, 95)
		except:
			# print ("sorry")
			import sys
			print (sys.exc_info())
			pass
		
		self.similarity_test_on_real_vote(our_model, 5000)
		self.new_similarity_test_on_real_vote(our_model,5000)

	def similarity_test_on_real_vote(self, our_model, num_sample):
		# for test, uniform !!!
		sim_list=[]
		min_fake = min(map(int,list(np.load(self.fake_user_id_list_path))))
		max_fake = max(map(int,list(np.load(self.fake_user_id_list_path))))
		
		# follower_tensor
		sampled_item_id_list = np.random.choice(a=self.num_item, size=num_sample/5)
		for item_id in sampled_item_id_list:
			try:
				sample_follower_info = self.follower_tensor[item_id]
			except Exception, ex:
				continue
			sample_reviewer_index_list = np.random.choice(a=sample_follower_info.len_reviewer_list, size=5)
			
			for idx in sample_reviewer_index_list:
				sample_follower_id_list = np.random.choice(a=sample_follower_info.follower_2dlist[idx],size=1)
				reviewer_id = sample_follower_info.reviewer_list[idx]
				# reviewer-follower
				for follower_id in sample_follower_id_list:
					if follower_id<=max_fake and follower_id>=min_fake:
						continue
					sim_list.append(our_model.similarity(str(reviewer_id),str(follower_id)))

		print 'real vote (origin-helper)',
		print np.min(sim_list), np.percentile(sim_list, 25), np.median(sim_list), np.percentile(sim_list,75), np.percentile(sim_list, 95)
		print ''

	def new_similarity_test_on_real_vote(self, our_model, num_sample):
		sim_list=[]
		min_fake = min(map(int,list(np.load(self.fake_user_id_list_path))))
		max_fake = max(map(int,list(np.load(self.fake_user_id_list_path))))
		sample_index_list = np.random.choice(a=len(self.maximum_item_rating_data), size=num_sample)
		for idx in sample_index_list:
			row = self.maximum_helpfulness_rating_data[idx]
			follower_id = int(row[0])
			review_id = int(row[1])
			# helpful_vote = row[2]

			reviewer_id, item_id = self.review_id_info[review_id]
			if follower_id<=max_fake and follower_id>=min_fake:
				continue
			sim_list.append(our_model.similarity(str(reviewer_id),str(follower_id)))
		print 'real vote (origin-helper)',
		print np.min(sim_list), np.percentile(sim_list, 25), np.median(sim_list), np.percentile(sim_list,75), np.percentile(sim_list, 95)
		print ''


	
	def whole_process(self,iteration=10, type0_ratio=1, type1_ratio=1, type2_ratio=2):
		self.construct_item_weight()

		self.construct_reviewer_tensor()
		self.construct_follower_tensor()

		# Word2Vec init
		self.build_user_vocab()
		
		# batch_size=1e+5
		# 1M is good
		batch_size=1e+6
		# print("small test!!!!")
		# batch_size=2e+3

		# for it in xrange(iteration):
		# 	for i in xrange(type0_ratio):
		# 		self.train_embedding_model(self.generate_reviewer_reviewer(batch_size))
		# 	for i in xrange(type1_ratio):
		# 		self.train_embedding_model(self.generate_reviewer_follower(batch_size))
		# 	for i in xrange(type2_ratio):
		# 		self.train_embedding_model(self.generate_follower_follower(batch_size))
		# 	self.similarity_test(self.user_embedding_model)	
		for it in xrange(50):
			self.train_embedding_model(self.enumerate_reviewer_follower())
			if it>0 and it%10==0:
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
	


