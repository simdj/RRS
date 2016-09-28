# 1. preprocess
# 	Input
# 		raw review 	<-- '../dataset/CiaoDVD/raw_review.txt'
# 		raw vote 	<-- '../dataset/CiaoDVD/raw_vote.txt'
# 	Output
# 		review 	--> './intermediate/review.npy'
# 		vote 	--> './intermediate/vote.npy'

import numpy as np
from collections import Counter

class preprocess():
	def __init__(self, params=None):
		self.user_threshold = params.user_threshold
		self.item_threshold = params.item_threshold

		self.raw_review_path  = params.raw_review_path 
		self.raw_vote_path  = params.raw_vote_path 
		
		self.review_origin_path  = params.review_origin_path 
		self.vote_origin_path  = params.vote_origin_path 
		
		# data after preprocess
		self.review_matrix = []
		self.vote_matrix = []

		#################################
		self.user_review_count = dict()
		self.item_review_count = dict()

		self.renumbering_user=dict()
		self.current_num_user=0

		self.index_for_only_voter = 1000000
		
		self.renumbering_item=dict()
		self.current_num_item=0

		self.renumbering_review_id = dict()
		self.current_num_review_id = 0


	def removing_not_enough_obj(self, review_data):
		ret_review_data = []
		user_counter = Counter(review_data[:,0])
		item_counter = Counter(review_data[:,1])
		for review_row in review_data:
			reviewer = int(review_row[0])
			item = int(review_row[1])
			if user_counter[reviewer]>=self.user_threshold and item_counter[item]>=self.item_threshold:
				ret_review_data.append(review_row)
		
		return np.array(ret_review_data)

	def get_enough_rating_review(self, raw_review_data):
		review_data = raw_review_data
		last_num_data = len(review_data)
		while True:
			review_data = self.removing_not_enough_obj(review_data)
			current_num_data = len(review_data)
			# print current_num_data
			if last_num_data==current_num_data:
				break
			else:
				last_num_data = current_num_data

		return review_data

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
			# fill temp review matrix
			tmp_review_matrix.append([reviewer, item, rating, review_id])
		f.close()

		# leave only review writtern by user who rated enough item which also is rated enough times
		review_matrix_filtered = self.get_enough_rating_review(np.array(tmp_review_matrix))

		# renumbering for convinence
		for tmp_review_row in review_matrix_filtered:
			tmp_reviewer = int(tmp_review_row[0])
			tmp_item = int(tmp_review_row[1])
			tmp_rating = tmp_review_row[2]
			tmp_review_id = int(tmp_review_row[3])

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
				# pass this vote only when the review is alive
				# renumber voter!
				if voter not in self.renumbering_user:
					# this voter did not review any item -> so renumbering_user[voter]=1000000+
					self.renumbering_user[voter]=self.index_for_only_voter
					self.index_for_only_voter+=1

				new_voter = self.renumbering_user[voter]
				new_review_id = self.renumbering_review_id[review_id]
				self.vote_matrix.append([new_voter, new_review_id, helpful])

	def save_review_matrix(self):
		np.save(self.review_origin_path, np.array(self.review_matrix))
		# np.savetxt(self.review_csv_path, np.array(self.review_matrix))

	def save_vote_matrix(self):
		np.save(self.vote_origin_path, np.array(self.vote_matrix))
		# np.savetxt(self.vote_csv_path, np.array(self.vote_matrix))

	def whole_process(self):
		self.preprocess_review_matrix()
		self.preprocess_vote_matrix()
		self.save_review_matrix()
		self.save_vote_matrix()

if __name__=="__main__":
	from parameter_controller import *
	exp_title = 'bandwagon_1%_1%_1%_emb_32'
	params = parse_exp_title(exp_title)

	pp = preprocess(params=params)
	pp.whole_process()





