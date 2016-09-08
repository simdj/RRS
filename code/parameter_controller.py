
class parameter_controller():
	def __init__(self
		, attack_model='None', bad_flag=False, camo_flag=False
		, num_fake_user=0, num_fake_item=0, num_camo_item=0
		, embedding_dim=32, word2vec_iter=10
		, rank=50, lda=1, max_iter=5001
		, user_threshold=10, item_threshold=10
		, doubt_weight=10):
		############################## 0. General ##############################
		self.attack_model = attack_model
		self.fake_flag = True if attack_model!= 'None' else False

		self.bad_flag = bad_flag
		self.camo_flag = camo_flag

		############################## 1. preprocess ##############################
		self.user_threshold = user_threshold
		self.item_threshold = item_threshold


		############################## 2. attack ##############################
		self.num_fake_user = num_fake_user
		self.num_fake_item = num_fake_item
		self.num_camo_item = num_camo_item
		# (default)
		self.num_fake_review=1
		self.num_fake_vote=1
		self.fake_rating_value=5
		self.fake_helpful_value=5

		############################## 3. user2vec ##############################
		self.embedding_dim = embedding_dim
		self.word2vec_iter = word2vec_iter

		############################## 4. helpfulness ##############################
		self.doubt_weight = doubt_weight

		############################## 5. matrix factorization ##############################
		self.rank = rank
		self.lda = lda
		self.max_iter = max_iter

		#######################################################################################
		########################################################################################\
		self.intermediate_dir_path = './intermediate/'
		# review path
		self.raw_review_path = '../dataset/CiaoDVD/raw_review.txt'
		self.review_numpy_path = self.intermediate_dir_path+'review.npy'
		self.fake_review_numpy_path = self.injection_path(self.review_numpy_path, self.attack_model, 'fake')
		self.camo_review_numpy_path = self.injection_path(self.review_numpy_path, self.attack_model, 'camo')
		# self.fake_review_numpy_path = './intermediate/fake_review_bandwagon.npy'
		# self.camo_review_numpy_path = './intermediate/camo_review_bandwagon.npy'

		# vote path (camo vote not yet...)
		self.raw_vote_path = '../dataset/CiaoDVD/raw_vote.txt'
		self.vote_numpy_path = self.intermediate_dir_path+'vote.npy'
		self.fake_vote_numpy_path = self.injection_path(self.vote_numpy_path, self.attack_model, 'fake')
		self.camo_vote_numpy_path = self.injection_path(self.vote_numpy_path, self.attack_model, 'camo')
		# self.fake_vote_numpy_path = './intermediate/fake_vote_bandwagon.npy'
		# self.camo_vote_numpy_path = './intermediate/camo_vote_bandwagon.npy'

		# embedding path
		self.user_embedding_path = self.intermediate_dir_path+'user2vec.emb'

		# helpful path
		self.helpful_numpy_path=self.intermediate_dir_path+'helpful.npy'
		self.fake_helpful_numpy_path = self.injection_path(self.helpful_numpy_path, self.attack_model, 'fake')
		self.camo_helpful_numpy_path = self.injection_path(self.helpful_numpy_path, self.attack_model, 'camo')
		# self.fake_helpful_numpy_path='./intermediate/fake_helpful.npy'
		# self.camo_helpful_numpy_path='./intermediate/camo_helpful.npy'
		
		
		############################## for readability ##############################
		self.review_csv_path = self.readable_path(self.review_numpy_path)
		self.fake_review_csv_path = self.readable_path(self.fake_review_numpy_path)
		self.camo_review_csv_path = self.readable_path(self.camo_review_numpy_path)
		
		self.vote_csv_path = self.readable_path(self.vote_numpy_path)
		self.fake_vote_csv_path = self.readable_path(self.fake_vote_numpy_path)
		
		self.user_embedding_csv_path = self.readable_path(self.user_embedding_path)
		
		self.helpful_csv_path = self.readable_path(self.helpful_numpy_path)
		self.fake_helpful_csv_path = self.readable_path(self.fake_helpful_numpy_path)
		self.camo_helpful_csv_path = self.readable_path(self.camo_helpful_numpy_path)
		# self.review_csv_path = './intermediate/ZZZZ_review.csv'
		# self.fake_review_csv_path = './intermediate/ZZZZ_fake_review_bandwagon.csv'
		# self.camo_review_csv_path = './intermediate/ZZZZ_camo_review_bandwagon.csv'
		#
		# self.vote_csv_path = './intermediate/ZZZZ_vote.csv'
		# self.fake_vote_csv_path = './intermediate/ZZZZ_fake_vote_bandwagon.csv'
		#
		# self.check_embedding_path = './intermediate/ZZZZ_user2vec.emb'
		#
		# self.helpful_csv_path = './intermediate/ZZZZ_helpful.csv'
		# self.fake_helpful_csv_path='./intermediate/ZZZZ_fake_helpful.csv'
		# self.camo_helpful_csv_path='./intermediate/ZZZZ_camo_helpful.csv'
		#######################################################################################
		#######################################################################################

	def injection_path(self, origin_path, attack_model, injection_type):
		if attack_model=='None':
			return None
		origin_file_name = origin_path.split(self.intermediate_dir_path).pop()
		file_extension = origin_file_name.split('.').pop()
		file_name=origin_file_name.split('.')[0]
		return self.intermediate_dir_path+injection_type+'_'+file_name+'_'+attack_model+'.'+file_extension
	
	def readable_path(self, origin_path):
		if not origin_path:
			return None
		origin_file_name = origin_path.split(self.intermediate_dir_path).pop()
		file_extension = origin_file_name.split('.').pop()
		file_name=origin_file_name.split('.')[0]
		if file_name=='':
			return None
		return self.intermediate_dir_path+'ZZZZ_'+file_name+'.csv'


def parse_exp_title(exp_title):
	# exp_title = 'emb_32_rank_50_None'
	# exp_title = 'emb_32_rank_50_bandwagon_1%_1%_1%'
	param_list = exp_title.split('_')

	embedding_dim = int(param_list[1])
	rank = int(param_list[3])
	
	attack_model = param_list[4]
	if attack_model=='None':
		pc = parameter_controller(embedding_dim=embedding_dim, rank=rank, attack_model='None', bad_flag=False, camo_flag=False, num_fake_user=0, num_fake_item=0, num_camo_item=0)
		return pc
	else:
		num_fake_user = float(param_list[5].split('%')[0])
		num_fake_item = float(param_list[6].split('%')[0])
		num_camo_item = float(param_list[7].split('%')[0])
		
		if param_list[5].find('%')>-1:
			num_fake_user/=100.0
		if param_list[6].find('%')>-1:
			num_fake_item/=100.0
		if param_list[7].find('%')>-1:
			num_camo_item/=100.0
		pc = parameter_controller(embedding_dim=embedding_dim, rank=rank, attack_model=attack_model, bad_flag=True, camo_flag=True, num_fake_user=num_fake_user, num_fake_item=num_fake_item, num_camo_item=num_camo_item)
		

		return pc


if __name__=="__main__":

	# exp_title = 'emb_32_rank_50_None'
	# exp_title = 'emb_32_rank_50_bandwagon_1%_1%_1%'
	# exp_title = 'emb_32_rank_50_bandwagon_1%_1%_10%'
	# exp_title = 'emb_32_rank_50_bandwagon_3%_3%_3%'
	exp_title = 'emb_32_rank_50_bandwagon_10%_10%_10%'
	# exp_title = 'emb_32_rank_50_average_1%_1%_1%'
	params = parse_exp_title(exp_title)
	# parameter_controller(embedding_dim=32, rank=50, attack_model='bandwagon', bad_flag=True, camo_flag=True, num_fake_user=0.01, num_fake_item=0.01, num_camo_item=0.01)
	# params.readable_path(params.review_numpy_path)
	a=params.__dict__
	for k in a.keys():
		print k,a[k]
