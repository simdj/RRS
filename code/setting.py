class setting():
	def __init__(self):
		# first input
		self.raw_review_path = '../dataset/CiaoDVD/raw_review.txt'
		self.raw_vote_path = '../dataset/CiaoDVD/raw_vote.txt'
		# main input
		self.review_origin_numpy_path = './intermediate/review.npy'
		self.vote_origin_numpy_path = './intermediate/vote.npy'
		# fake input
		self.review_fake_numpy_path = './intermediate/fake_review.npy'
		self.vote_fake_numpy_path = './intermediate/fake_vote.npy'
		
		# for readability
		self.review_csv_path = './intermediate/check_review.csv'
		self.vote_csv_path = './intermediate/check_vote.csv'
		self.fake_review_csv_path = './intermediate/fake_review.csv'
		self.fake_vote_csv_path = './intermediate/fake_vote.csv'