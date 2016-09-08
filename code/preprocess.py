# 1. preprocess
# 	Input
# 		raw review 	<-- '../dataset/CiaoDVD/raw_review.txt'
# 		raw vote 	<-- '../dataset/CiaoDVD/raw_vote.txt'
# 	Output
# 		review 	--> './intermediate/review.npy'
# 		vote 	--> './intermediate/vote.npy'
import numpy as np
from time import time

class preprocess():
	def __init__(self, params=None):
		self.user_threshold = params.user_threshold
		self.item_threshold = params.item_threshold

		self.raw_review_path  = params.raw_review_path 
		self.raw_vote_path  = params.raw_vote_path 
		
		self.review_numpy_path  = params.review_numpy_path 
		self.vote_numpy_path  = params.vote_numpy_path 
		# for readability
		self.review_csv_path  = params.review_csv_path 
		self.vote_csv_path  = params.vote_csv_path 
		
		# data after preprocess
		self.review_matrix = []
		self.vote_matrix = []

		#################################
		self.user_review_count = dict()
		self.item_review_count = dict()

		self.renumbering_user=dict()
		self.current_num_user=0

		self.current_voter_id = 1000000
		
		self.renumbering_item=dict()
		self.current_num_item=0

		self.renumbering_review_id = dict()
		self.current_num_review_id = 0

	def preprocess_review_matrix(self):
		# raw_review file -> tmp_review_matrix -> self.review_matrix (above threshold)
		# ('old (U x I)', 17615, 16121)
		# ('new rating #', 72665, '(U x I)', 17615, 16121, 'U,I threshold', 1, 1)
		# ('new rating #', 54828, '(U x I)', 7573, 7667, 'U,I threshold', 2, 2)
		# ('new rating #', 45010, '(U x I)', 4695, 5059, 'U,I threshold', 3, 3)
		# ('new rating #', 33911, '(U x I)', 2582, 3066, 'U,I threshold', 5, 5)
		# ('new rating #', 20348, '(U x I)', 1112, 1495, 'U,I threshold', 10, 10)
		
		# construct statistics
		tmp_review_matrix = []
		f=open(self.raw_review_path)
		while True:
			line = f.readline()
			if not line: break
			line = line.strip().split(',')
			# size: 72,665
			# reviewer, item, genreID, reviewID, movieRating, date
			reviewer = int(line[0])
			item = int(line[1])
			rating = float(line[4])
			review_id  = int(line[3])

			# recording user data
			if reviewer not in self.user_review_count:
				self.user_review_count[reviewer]=0
			self.user_review_count[reviewer]+=1

			# recording item data
			if item not in self.item_review_count:
				self.item_review_count[item]=0
			self.item_review_count[item]+=1

			# fill temp review matrix
			tmp_review_matrix.append([reviewer, item, rating, review_id])
		f.close()

		# filter(tmp_review_matrix) --> self.review_matrix
		for tmp_review_row in tmp_review_matrix:
			tmp_reviewer = tmp_review_row[0]
			tmp_item = tmp_review_row[1]
			tmp_rating = tmp_review_row[2]
			tmp_review_id = tmp_review_row[3]

			if self.user_review_count[tmp_reviewer]>=self.user_threshold and self.item_review_count[tmp_item]>=self.item_threshold:
				# pass this review_row

				# renumber!!
				if tmp_reviewer not in self.renumbering_user:
					self.renumbering_user[tmp_reviewer]=self.current_num_user
					self.current_num_user+=1
				if tmp_item not in self.renumbering_item:
					self.renumbering_item[tmp_item]=self.current_num_item
					self.current_num_item+=1
				if tmp_review_id not in self.renumbering_review_id:
					self.renumbering_review_id[tmp_review_id]=self.current_num_review_id
					self.current_num_review_id+=1

				new_reviewer = self.renumbering_user[tmp_reviewer]
				new_item = self.renumbering_item[tmp_item]
				new_review_id = self.renumbering_review_id[tmp_review_id]
				# append filtered review matrix
				self.review_matrix.append([new_reviewer, new_item, tmp_rating, new_review_id])

	def preprocess_vote_matrix(self):
		# filter only the reviews in the preprocessed review_matrix
		# raw_vote file --(filtering)--> vote_matrix

		# users who do not review going to be 1M+a id
		f=open(self.raw_vote_path)
		while True:
			line = f.readline()
			if not line: break
			line = line.strip().split(',')
			# review_ratings.txt (size: 1,625,480 --> 1.6M)
			# userID, reviewID, reviewRating
			
			voter = int(line[0])
			review_id = int(line[1])
			helpful = float(line[2])
			
			# voter can be not in self.renumbering_user!!!!
			# but only review_id, which is in the self.renumbering_review_id, can survive

			if review_id in self.renumbering_review_id:
				# pass this vote

				# renumber voter!
				if voter not in self.renumbering_user:
					# this voter did not review any item -> so renumbering_user[voter]=1000000+
					self.renumbering_user[voter]=self.current_voter_id
					self.current_voter_id+=1

				new_voter = self.renumbering_user[voter]
				new_review_id = self.renumbering_review_id[review_id]
				self.vote_matrix.append([new_voter, new_review_id, helpful])

	def save_review_matrix(self):
		np.save(self.review_numpy_path, np.array(self.review_matrix))
		np.savetxt(self.review_csv_path, np.array(self.review_matrix))

	def save_vote_matrix(self):
		np.save(self.vote_numpy_path, np.array(self.vote_matrix))
		np.savetxt(self.vote_csv_path, np.array(self.vote_matrix))

	def whole_process(self):
		self.preprocess_review_matrix()
		self.preprocess_vote_matrix()
		self.save_review_matrix()
		self.save_vote_matrix()

if __name__=="__main__":
	from parameter_controller import *
	# exp_title = 'emb_32_rank_50_None'
	# exp_title = 'emb_32_rank_50_bandwagon_1%_1%_1%'
	# exp_title = 'emb_32_rank_50_bandwagon_1%_1%_10%'
	# exp_title = 'emb_32_rank_50_bandwagon_3%_3%_3%'
	exp_title = 'emb_32_rank_50_bandwagon_10%_10%_10%'
	# exp_title = 'emb_32_rank_50_average_1%_1%_1%'
	params = parse_exp_title(exp_title)

	pp = preprocess(params=params)
	pp.whole_process()





