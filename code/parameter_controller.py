
class parameter_controller():
	def __init__(self):

		############################## 0. General ##############################
		self.attack_model = 'bandwagon'
		self.bad_flag = True
		self.fake_flag = True
		self.camo_flag = True

		# raw review/vote path
		self.raw_review_path = '../dataset/CiaoDVD/raw_review.txt'
		self.raw_vote_path = '../dataset/CiaoDVD/raw_vote.txt'
		
		# review path
		self.review_numpy_path = './intermediate/review.npy'
		self.fake_review_numpy_path = './intermediate/fake_review_bandwagon.npy'
		self.camo_review_numpy_path = './intermediate/camo_review_bandwagon.npy'

		# vote path (camo vote not yet...)
		self.vote_numpy_path = './intermediate/vote.npy'
		self.fake_vote_numpy_path = './intermediate/fake_vote_bandwagon.npy'

		# embedding path
		self.user_embedding_path = './intermediate/user2vec.emb'

		# helpful path
		self.helpful_numpy_path='./intermediate/helpful.npy'
		self.fake_helpful_numpy_path='./intermediate/fake_helpful.npy'
		self.camo_helpful_numpy_path='./intermediate/camo_helpful.npy'
		
		
		############################## for readability ##############################
		self.review_csv_path = './intermediate/check_review.csv'
		self.fake_review_csv_path = './intermediate/check_fake_review_bandwagon.csv'
		self.camo_review_csv_path = './intermediate/check_camo_review_bandwagon.csv'
		
		self.vote_csv_path = './intermediate/check_vote.csv'
		self.fake_vote_csv_path = './intermediate/check_fake_vote_bandwagon.csv'

        self.check_embedding_path = './intermediate/check_user2vec.emb'

        self.helpful_csv_path = './intermediate/check_helpful.csv'
        self.fake_helpful_csv_path='./intermediate/check_fake_helpful.csv'
        self.camo_helpful_csv_path='./intermediate/check_camo_helpful.csv'


		############################## 1. preprocess ##############################
		self.user_threshold = user_threshold
		self.item_threshold = item_threshold


		############################## 2. attack ##############################
		self.num_fake_user = num_fake_user
		self.num_fake_item = num_fake_item
		self.num_camo_item = num_camo_item

		############################## 3. user2vec ##############################
        self.embedding_dim = embedding_dim
        self.word2vec_iter = word2vec_iter

        ############################## 4. helpfulness ##############################
        self.doubt_weight = 5

        ############################## 5. matrix factorization ##############################
        self.rank = rank
        self.lda = lda
        self.max_iter = max_iter
