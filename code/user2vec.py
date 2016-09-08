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

class user2vec():
	def __init__(self, params=None):

		# input
		self.review_numpy_path = params.review_numpy_path
		self.vote_numpy_path = params.vote_numpy_path
		self.fake_review_numpy_path = params.fake_review_numpy_path
		self.camo_review_numpy_path = params.camo_review_numpy_path
		self.fake_vote_numpy_path = params.fake_vote_numpy_path

		# output
		self.user_embedding_path = params.user_embedding_path
		self.user_embedding_csv_path = params.user_embedding_csv_path

		# embedding
		self.fake_flag = params.fake_flag
		self.camo_flag = params.camo_flag
		self.embedding_dim = params.embedding_dim
		self.word2vec_iter = params.word2vec_iter

		# embedding model (Word2Vec)
		self.user_embedding_model = None

		# intermediate
		self.review_writer = dict()
		self.similar_reviewer = dict()
		self.reviewer_follower = dict()

		self.num_item = 0
		self.item_weight_list = []

	def load_overall_review_matrix(self):
		overall_review_matrix = np.load(self.review_numpy_path)
		if self.fake_flag:
			overall_review_matrix = np.concatenate((overall_review_matrix, np.load(self.fake_review_numpy_path)))
			if self.camo_flag:
				overall_review_matrix = np.concatenate((overall_review_matrix, np.load(self.camo_review_numpy_path)))
		return overall_review_matrix

	def load_overall_vote_matrix(self):
		overall_vote_matrix = np.load(self.vote_numpy_path)
		if self.fake_flag:
			overall_vote_matrix = np.concatenate((overall_vote_matrix, np.load(self.fake_vote_numpy_path)))
			# if self.camo_flag:
			# 	overall_vote_matrix = np.concatenate((overall_vote_matrix, np.load(self.camo_vote_numpy_path)))
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
				item_related_user_set[item_id]=set()
			item_related_user_set[item_id].add(reviewer)

			review_id_item_id_mapping[review_id]=item_id

		# 2) add reviewer

		overall_vote_matrix = self.load_overall_vote_matrix()
		for row in overall_vote_matrix:
			voter = int(row[0])
			review_id = int(row[1])
			# helpful_vote = row[2]

			item_id = review_id_item_id_mapping[review_id]
			if item_id not in item_related_user_set:
				item_related_user_set[item_id]=set()
			item_related_user_set[item_id].add(voter)

		# mainly compute weight of items
		self.num_item = len(np.unique(overall_review_matrix[:,1]))
		self.item_weight_list = np.zeros((self.num_item,1)).reshape([-1])
		for item_id, user_set in item_related_user_set.iteritems():
			self.item_weight_list[item_id]=1.0/np.log2(len(user_set)+5)
		normalization = 1.0*np.sum(self.item_weight_list)
		self.item_weight_list/=normalization

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



	def construct_review_writer_and_similar_reviewer(self):
		# data loading
		overall_review_matrix = self.load_overall_review_matrix()
		for row in overall_review_matrix:
			reviewer = int(row[0])
			item_id = int(row[1])
			rating_value = row[2]
			review_id = int(row[3])

			# review_writer ()
			self.review_writer[review_id]=reviewer
			# similar_reviewer(item_id, rating value) = reviewer set
			item_and_rating = str(item_id)+','+str(rating_value)
			if item_and_rating not in self.similar_reviewer:
				self.similar_reviewer[item_and_rating] = set()
			self.similar_reviewer[item_and_rating].add(reviewer)

	def construct_reviewer_follower(self):
		# data loading
		overall_vote_matrix = self.load_overall_vote_matrix()
		for row in overall_vote_matrix:
			voter = int(row[0])
			review_id = int(row[1])
			helpful_vote = row[2]

			if helpful_vote>=3:
				reviewer = self.review_writer[review_id]
				if reviewer not in self.reviewer_follower:
					# no set, duplicates are allowed
					self.reviewer_follower[reviewer]=[]
				self.reviewer_follower[reviewer].append(voter)

	def enumerate_similar_reviewer_pair(self):
		# 1. same review i.e. same movie rating

		################################################################################
		# naive version
		similar_reviewer_pairs = []
		for k, similar_reviewer_set in self.similar_reviewer.iteritems():
			same_reviewer_pair = itertools.combinations(similar_reviewer_set, 2)
			similar_reviewer_pairs.extend(map(lambda x: [x[0], x[1]], same_reviewer_pair))
		# but consider 1/log(degree+c)!!!
		return similar_reviewer_pairs

	def enumerate_reviewer_follower_pair(self):
		# 2. same thinking i.e. reviewer-follower
		reviewer_follower_pairs=[]
		################################################################################
		# naive version
		for reviewer, follower_list in self.reviewer_follower.iteritems():
			reviewer_follower_pair = []
			for follower in follower_list:
				reviewer_follower_pair.append([reviewer, follower])

			reviewer_follower_pairs.extend(reviewer_follower_pair)
		# but consider 1/log(degree+c)!!!
		return reviewer_follower_pairs

	def enumerate_rater_rater_pair(self):
		# 3. same review rating i.e. same helpful about a movie review
		pass



	def learn_embedding(self, train_data=None):
		#  Learn embedding by optimizing the Skipgram objective using negative sampling.
		train_data = [map(str,x) for x in train_data]
		print("learning sentence # :", len(train_data))
		if not self.user_embedding_model:
			# first train data
			self.user_embedding_model = Word2Vec(train_data, size=self.embedding_dim
				, sg=1, negative=20, window=2, min_count=0, workers=8, iter=self.word2vec_iter)
		else:
			# new sentences
			self.user_embedding_model.train(train_data)
		return

	def save_embedding(self):
		self.user_embedding_model.save(self.user_embedding_path)
		self.user_embedding_model.save_word2vec_format(self.user_embedding_csv_path)

	def load_embedding(self):
		return Word2Vec.load(self.user_embedding_path)

	def whole_process(self):
		self.construct_review_writer_and_similar_reviewer()
		self.construct_reviewer_follower()
		self.construct_item_weight()

		self.learn_embedding(self.enumerate_reviewer_follower_pair())
		self.learn_embedding(self.enumerate_similar_reviewer_pair())
		self.save_embedding()


	def sampling_item_via_weight(self, num_sample):
		sampled_item_id_list = np.random.choice(self.num_item, num_sample, self.item_weight_list)
        




if __name__ == "__main__":
	from parameter_controller import *
	exp_title = 'emb_32_rank_50_bandwagon_5%_1%_5%'
	print('Experiment Title', exp_title)
	params = parse_exp_title(exp_title)
	u2v = user2vec(params=params)
	u2v.item_degree()

	# u2v = user2vec()
	# u2v.whole_process()
	# pass
