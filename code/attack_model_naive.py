# 1. inject fake review/vote
# 	Input
# 		review 	<-- './intermediate/review.npy'
# 		vote 	<-- './intermediate/vote.npy'
# 	Output
# 		fake review 	--> './intermediate/fake_review.npy'
# 		fake vote 		--> './intermediate/fake_vote.npy'

import numpy as np
import csv

class attack_model_naive():
	def __init__(self, num_fake_users=10, num_fake_items=10, num_fake_reviews=100, num_fake_votes=500, fake_rating_value=5, fake_helpful_value=5):
		# input
		self.review_numpy_path = './intermediate/review.npy'
		self.vote_numpy_path = './intermediate/vote.npy'
		# output
		self.fake_review_numpy_path = './intermediate/fake_review.npy'
		self.fake_vote_numpy_path = './intermediate/fake_vote.npy'
		# for readability
		self.fake_review_csv_path = './intermediate/check_fake_review.csv'
		self.fake_vote_csv_path = './intermediate/check_fake_vote.csv'

		#######################################
		# origin review stats
		origin_review_matrix = np.load(self.review_numpy_path)
		origin_vote_matrix = np.load(self.vote_numpy_path)

		self.num_origin_users = len(np.unique(np.concatenate((origin_review_matrix[:,0], origin_vote_matrix[:,0]), axis=0)))
		self.num_origin_items = len(np.unique(origin_review_matrix[:,1]))
		self.num_origin_reviews = len(origin_review_matrix)
		self.num_origin_votes = len(origin_vote_matrix)
		
		#######################################
		# fake review stats
		self.fake_user_start = self.num_origin_users+1
		self.fake_item_start = self.num_origin_items+1
		self.fake_review_id_start = self.num_origin_reviews+1

		self.num_fake_users = int(self.num_origin_users*num_fake_users) if num_fake_users<1 else num_fake_users
		self.fake_user_id_list = list(range(self.fake_user_start, self.fake_user_start+self.num_fake_users))

		self.num_fake_items = int(self.num_origin_items*num_fake_items) if num_fake_items<1 else num_fake_items
		self.fake_item_id_list = list(range(self.fake_item_start, self.fake_item_start+self.num_fake_items))
		
		# #(fake reviews) cannot be larger than #(fake_user)*#(fake_item)
		self.num_fake_reviews = int(self.num_fake_users*self.num_fake_items*num_fake_reviews) if num_fake_reviews<1 else num_fake_reviews
		self.num_fake_reviews = min(self.num_fake_reviews, self.num_fake_users*self.num_fake_items)
		self.fake_review_id_list = list(range(self.fake_review_id_start,self.fake_review_id_start+self.num_fake_reviews))

		# #(fake votes) cannot be larger than #(fake users)*#(fake reviews)
		self.num_fake_votes = int(self.num_fake_reviews*self.num_fake_users*num_fake_votes) if num_fake_votes<1 else num_fake_votes
		self.num_fake_votes = min(self.num_fake_votes, self.num_fake_reviews*self.num_fake_users)
		
		# fake_rating value is usually extream value
		self.fake_rating_value = fake_rating_value
		self.fake_helpful_value = fake_helpful_value

		# importatnt data
		self.fake_review_matrix = []
		self.fake_vote_matrix = []

		print('[Origin] Users x Items : ', self.num_origin_users, self.num_origin_items, 'Reviews, Votes : ', self.num_origin_reviews, self.num_origin_votes)
		print('[Fake] Users x Items : ', self.num_fake_users, self.num_fake_items, 'Reviews, Votes : ',self.num_fake_reviews, self.num_fake_votes)

	def generate_fake_reviews(self):
		# generate naive fake review
		fake_review_coo = np.random.choice(self.num_fake_users*self.num_fake_items, self.num_fake_reviews, replace=False)

		fake_review_id = self.fake_review_id_start
		for coo in fake_review_coo:
			fake_u = self.fake_user_id_list[int(coo/self.num_fake_items)]
			fake_i = self.fake_item_id_list[coo%self.num_fake_items]

			self.fake_review_matrix.append([fake_u, fake_i, self.fake_rating_value, fake_review_id])
			fake_review_id+=1


	def generate_fake_votes(self):
		fake_vote_coo = np.random.choice(self.num_fake_users*self.num_fake_reviews, self.num_fake_votes, replace=False)
		for coo in fake_vote_coo:
			fake_u = self.fake_user_id_list[int(coo/self.num_fake_reviews)]
			fake_r = self.fake_review_id_list[coo%self.num_fake_reviews]
			self.fake_vote_matrix.append([fake_u,fake_r, self.fake_helpful_value])

	def save_fake_review_matrix(self):
		np.save(self.fake_review_numpy_path, np.array(self.fake_review_matrix))
		np.savetxt(self.fake_review_csv_path, np.array(self.fake_review_matrix))

	def save_fake_vote_matrix(self):
		np.save(self.fake_vote_numpy_path, np.array(self.fake_vote_matrix))
		np.savetxt(self.fake_vote_csv_path, np.array(self.fake_vote_matrix))


	# def inject_fake_helpfulness_vote(self, ratio_users=0.01, ratio_votes=0.01, vote_value=5):
	# 	# self.fake_review_id_list

	# 	num_fake_users = int(self.num_origin_users*ratio_users) if ratio_users<1 else ratio_users
	# 	num_fake_votes = int(self.num_origin_votes*ratio_votes) if ratio_votes<1 else ratio_votes
	# 	num_fake_reviews = len(self.fake_review_id_list)

	# 	fake_vote_f = open(self.fake_vote_path, 'w')
	# 	writer = csv.writer(fake_vote_f,  delimiter=",",lineterminator='\n')

	# 	fake_vote_coo = np.random.choice(num_fake_users*num_fake_reviews, min(num_fake_votes,num_fake_users*num_fake_reviews), replace=False)
	# 	for coo in fake_vote_coo:
	# 		fake_u = coo/num_fake_reviews+self.fake_user_start
	# 		fake_r = self.fake_review_id_list[coo%num_fake_reviews]
			
	# 		writer.writerow([fake_u,fake_r,vote_value])
	# 		self.fake_vote_matrix.append([fake_u,fake_r,vote_value])

	# 	# save fake_vote_matrix numpy
	# 	np.save(self.fake_vote_numpy_path, self.fake_vote_matrix)

if __name__=="__main__":
	am = attack_model_naive(num_fake_users=10, num_fake_items=10, num_fake_reviews=100, num_fake_votes=500, fake_rating_value=5)
	am.generate_fake_reviews()
	am.save_fake_review_matrix()
	print("finished fake review injection")
	am.generate_fake_votes()
	am.save_fake_vote_matrix()
	print("finished fake vote injection")