# # 1. inject fake review/vote
# # 	Input
# # 		review 	<-- './intermediate/review.npy'
# # 		vote 	<-- './intermediate/vote.npy'
# # 	Output
# # 		fake review 	--> './intermediate/fake_review.npy'
# # 		fake vote 		--> './intermediate/fake_vote.npy'

# import numpy as np
# import csv

# class attack_model_naive():
# 	def __init__(self, num_fake_user=10, num_fake_item=10, num_fake_review=100, num_fake_vote=500, fake_rating_value=5, fake_helpful_value=5):
# 		# input
# 		self.review_origin_numpy_path = './intermediate/review.npy'
# 		self.vote_origin_numpy_path = './intermediate/vote.npy'
# 		# output
# 		self.review_fake_numpy_path = './intermediate/fake_review.npy'
# 		self.vote_fake_numpy_path = './intermediate/fake_vote.npy'
# 		# for readability
# 		self.fake_review_csv_path = './intermediate/check_fake_review.csv'
# 		self.fake_vote_csv_path = './intermediate/check_fake_vote.csv'

# 		#######################################
# 		# origin review stats
# 		origin_review_matrix = np.load(self.review_origin_numpy_path)
# 		origin_vote_matrix = np.load(self.vote_origin_numpy_path)

# 		self.num_origin_user = len(np.unique(np.concatenate((origin_review_matrix[:,0], origin_vote_matrix[:,0]), axis=0)))
# 		self.num_origin_item = len(np.unique(origin_review_matrix[:,1]))
# 		self.num_origin_review = len(origin_review_matrix)
# 		self.num_origin_vote = len(origin_vote_matrix)
		
# 		#######################################
# 		# fake review stats
# 		self.fake_user_start = self.num_origin_user+1
# 		self.fake_item_start = self.num_origin_item+1
# 		self.fake_review_id_start = self.num_origin_review+1

# 		self.num_fake_user = int(self.num_origin_user*num_fake_user) if num_fake_user<1 else num_fake_user
# 		self.fake_user_id_list = list(range(self.fake_user_start, self.fake_user_start+self.num_fake_user))

# 		self.num_fake_item = int(self.num_origin_item*num_fake_item) if num_fake_item<1 else num_fake_item
# 		self.fake_item_id_list = list(range(self.fake_item_start, self.fake_item_start+self.num_fake_item))
		
# 		# #(fake reviews) cannot be larger than #(fake_user)*#(fake_item)
# 		self.num_fake_review = int(self.num_fake_user*self.num_fake_item*num_fake_review) if num_fake_review<1 else num_fake_review
# 		self.num_fake_review = min(self.num_fake_review, self.num_fake_user*self.num_fake_item)
# 		self.fake_review_id_list = list(range(self.fake_review_id_start,self.fake_review_id_start+self.num_fake_review))

# 		# #(fake votes) cannot be larger than #(fake users)*#(fake reviews)
# 		self.num_fake_vote = int(self.num_fake_review*self.num_fake_user*num_fake_vote) if num_fake_vote<1 else num_fake_vote
# 		self.num_fake_vote = min(self.num_fake_vote, self.num_fake_review*self.num_fake_user)
		
# 		# fake_rating value is usually extream value
# 		self.fake_rating_value = fake_rating_value
# 		self.fake_helpful_value = fake_helpful_value

# 		# importatnt data
# 		self.fake_review_matrix = []
# 		self.fake_vote_matrix = []

# 		print('[Origin] Users x Items : ', self.num_origin_user, self.num_origin_item, 'Reviews, Votes : ', self.num_origin_review, self.num_origin_vote)
# 		print('[Fake] Users x Items : ', self.num_fake_user, self.num_fake_item, 'Reviews, Votes : ',self.num_fake_review, self.num_fake_vote)

# 	def generate_fake_reviews(self):
# 		# generate naive fake review
# 		fake_review_coo = np.random.choice(self.num_fake_user*self.num_fake_item, self.num_fake_review, replace=False)

# 		fake_review_id = self.fake_review_id_start
# 		for coo in fake_review_coo:
# 			fake_u = self.fake_user_id_list[int(coo/self.num_fake_item)]
# 			fake_i = self.fake_item_id_list[coo%self.num_fake_item]

# 			self.fake_review_matrix.append([fake_u, fake_i, self.fake_rating_value, fake_review_id])
# 			fake_review_id+=1


# 	def generate_fake_votes(self):
# 		fake_vote_coo = np.random.choice(self.num_fake_user*self.num_fake_review, self.num_fake_vote, replace=False)
# 		for coo in fake_vote_coo:
# 			fake_u = self.fake_user_id_list[int(coo/self.num_fake_review)]
# 			fake_r = self.fake_review_id_list[coo%self.num_fake_review]
# 			self.fake_vote_matrix.append([fake_u,fake_r, self.fake_helpful_value])

# 	def save_fake_review_matrix(self):
# 		np.save(self.review_fake_numpy_path, np.array(self.fake_review_matrix))
# 		np.savetxt(self.fake_review_csv_path, np.array(self.fake_review_matrix))

# 	def save_fake_vote_matrix(self):
# 		np.save(self.vote_fake_numpy_path, np.array(self.fake_vote_matrix))
# 		np.savetxt(self.fake_vote_csv_path, np.array(self.fake_vote_matrix))


# 	# def inject_fake_helpfulness_vote(self, ratio_users=0.01, ratio_votes=0.01, vote_value=5):
# 	# 	# self.fake_review_id_list

# 	# 	num_fake_user = int(self.num_origin_user*ratio_users) if ratio_users<1 else ratio_users
# 	# 	num_fake_vote = int(self.num_origin_vote*ratio_votes) if ratio_votes<1 else ratio_votes
# 	# 	num_fake_review = len(self.fake_review_id_list)

# 	# 	fake_vote_f = open(self.fake_vote_path, 'w')
# 	# 	writer = csv.writer(fake_vote_f,  delimiter=",",lineterminator='\n')

# 	# 	fake_vote_coo = np.random.choice(num_fake_user*num_fake_review, min(num_fake_vote,num_fake_user*num_fake_review), replace=False)
# 	# 	for coo in fake_vote_coo:
# 	# 		fake_u = coo/num_fake_review+self.fake_user_start
# 	# 		fake_r = self.fake_review_id_list[coo%num_fake_review]
			
# 	# 		writer.writerow([fake_u,fake_r,vote_value])
# 	# 		self.fake_vote_matrix.append([fake_u,fake_r,vote_value])

# 	# 	# save fake_vote_matrix numpy
# 	# 	np.save(self.vote_fake_numpy_path, self.fake_vote_matrix)

# if __name__=="__main__":
# 	am = attack_model_naive(num_fake_user=10, num_fake_item=10, num_fake_review=100, num_fake_vote=500, fake_rating_value=5)
# 	am.generate_fake_reviews()
# 	am.save_fake_review_matrix()
# 	print("finished fake review injection")
# 	am.generate_fake_votes()
# 	am.save_fake_vote_matrix()
# 	print("finished fake vote injection")