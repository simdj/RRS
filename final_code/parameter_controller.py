import os

# Input
# 	attack model
# 	attack size
# 	embedding size

# output -- directory name = '[attack][attack size][embedding size]'
# 	review_origin
# 	review_fake
# 	review_camo

# 	vote_origin
# 	vote_fake
# 	vote_camo
	
# 	------------------------------------------
# 	embedding_clean

# 	helpful_origin_clean_naive
# 	helpful_origin_clean_robust
	
# 	------------------------------------------
# 	embedding_attacked
	
# 	helpful_origin_attacked_naive
# 	helpful_fake_attacked_naive
# 	helpful_camo_attacked_naive
			
# 	helpful_origin_attacked_robust
# 	helpful_fake_attacked_robust
# 	helpful_camo_attacked_robust

class parameter_controller():
	def __init__(self
		, attack_model='bandwagon', badly_rated_item_flag=True, camo_flag=True
		, num_fake_user=0, filler_size=0, selected_size=0, camo_vote_size_multiple=1
		, embedding_dim=32, word2vec_iter=5
		, rank=50, lda=1, max_iter=5001
		, user_threshold=5, item_threshold=5
		, doubt_weight=10, exp_title=''):
		############################## 0. General ##############################
		self.attack_model = attack_model
		self.fake_flag = True 

		self.badly_rated_item_flag = badly_rated_item_flag
		self.bad_item_threshold = 18 # 1%
		self.camo_flag = camo_flag

		self.exp_title = exp_title


		############################## 1. preprocess ##############################
		self.user_threshold = user_threshold
		self.item_threshold = item_threshold

		############################## 2. attack ##############################
		self.num_fake_user = num_fake_user
		self.num_fake_item = 1
		self.filler_size = filler_size
		self.selected_size = selected_size
		self.camo_vote_size_multiple = camo_vote_size_multiple
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
		self.similarity_threshold = 0.8
		self.base_helpful_numerator = 1.5
		self.base_helpful_denominator = 0.6

		# ############################# 5. matrix factorization ##############################
		self.rank = rank
		self.lda = lda
		self.max_iter = max_iter

		#######################################################################################
		########################################################################################\
		# directory path
		self.intermediate_dir_path = exp_title+'/'
		if not os.path.exists(self.intermediate_dir_path):
			os.makedirs(self.intermediate_dir_path)
		
		# review path
		self.raw_review_path = '../dataset/CiaoDVD/raw_review.txt'
		self.review_origin_path = self.intermediate_dir_path+'review_origin.npy'
		self.review_fake_path = self.intermediate_dir_path+'review_fake.npy'
		self.review_camo_path = self.intermediate_dir_path+'review_camo.npy'
		
		# vote path (camo vote not yet...)
		self.raw_vote_path = '../dataset/CiaoDVD/raw_vote.txt'
		self.vote_origin_path = self.intermediate_dir_path+'vote_origin.npy'
		self.vote_fake_path = self.intermediate_dir_path+'vote_fake.npy'
		self.vote_camo_path = self.intermediate_dir_path+'vote_camo.npy'
		

		# # clean dataset 
		# embedding path
		self.embedding_clean_path = self.intermediate_dir_path+'embedding_clean.emb'
		# helpful path (naive)
		self.helpful_origin_clean_naive_path = self.intermediate_dir_path+'helpful_origin_clean_naive.npy'
		# helpful path (robust)
		self.helpful_origin_clean_robust_path = self.intermediate_dir_path+'helpful_origin_clean_robust.npy'

		# # under attack
		# embedding path
		self.embedding_attacked_path = self.intermediate_dir_path+'embedding_attacked.emb'
		# helpful path (naive)
		self.helpful_origin_attacked_naive = self.intermediate_dir_path+'helpful_origin_attacked_naive.npy'
		self.helpful_fake_attacked_naive = self.intermediate_dir_path+'helpful_fake_attacked_naive.npy'
		self.helpful_camo_attacked_naive = self.intermediate_dir_path+'helpful_camo_attacked_naive.npy'
		# helpful path (robust)
		self.helpful_origin_attacked_robust = self.intermediate_dir_path+'helpful_origin_attacked_robust.npy'
		self.helpful_fake_attacked_robust = self.intermediate_dir_path+'helpful_fake_attacked_robust.npy'
		self.helpful_camo_attacked_robust = self.intermediate_dir_path+'helpful_camo_attacked_robust.npy'
	

		# output
		self.base_U_clean_path = self.intermediate_dir_path+'base_U_clean.npy'
		self.base_V_clean_path = self.intermediate_dir_path+'base_V_clean.npy'

		self.base_U_attacked_path = self.intermediate_dir_path+'base_U_attacked.npy'
		self.base_V_attacked_path = self.intermediate_dir_path+'base_V_attacked.npy'

		self.naive_U_clean_path = self.intermediate_dir_path+'naive_U_clean.npy'
		self.naive_V_clean_path = self.intermediate_dir_path+'naive_V_clean.npy'

		self.naive_U_attacked_path = self.intermediate_dir_path+'naive_U_attacked.npy'
		self.naive_V_attacked_path = self.intermediate_dir_path+'naive_V_attacked.npy'

		self.robust_U_clean_path = self.intermediate_dir_path+'robust_U_clean.npy'
		self.robust_V_clean_path = self.intermediate_dir_path+'robust_V_clean.npy'

		self.robust_U_attacked_path = self.intermediate_dir_path+'robust_U_attacked.npy'
		self.robust_V_attacked_path = self.intermediate_dir_path+'robust_V_attacked.npy'



		# for metric
		self.target_item_list_path = self.intermediate_dir_path+'target_item_list_path.npy'
		self.fake_user_id_list_path = self.intermediate_dir_path+'fake_user_id_list_path.npy'
		# 
		self.train_data_path = self.intermediate_dir_path+'train_data.npy'
		self.test_data_path = self.intermediate_dir_path+'test_data.npy'

		# self.test_target_data_path = self.intermediate_dir_path+'test_target_data.npy'
		# self.test_overall_data_path = self.intermediate_dir_path+'test_overall_data.npy'

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

	param_list = exp_title.split('_')

	embedding_dim = int(param_list[5])
	attack_model = param_list[0]

	num_fake_user = float(param_list[1].split('%')[0])
	filler_size = float(param_list[2].split('%')[0])
	selected_size = float(param_list[3].split('%')[0])
	
	if param_list[1].find('%')>-1:
		num_fake_user/=100.0
	if param_list[2].find('%')>-1:
		filler_size/=100.0
	if param_list[3].find('%')>-1:
		selected_size/=100.0
	pc = parameter_controller(embedding_dim=embedding_dim, attack_model=attack_model, num_fake_user=num_fake_user, filler_size=filler_size, selected_size=selected_size, exp_title=exp_title)
	return pc


if __name__=="__main__":

	exp_title = 'bandwagon_1%_1%_1%_emb_32'
	params = parse_exp_title(exp_title)


	# # parameter_controller(embedding_dim=32, rank=50, attack_model='bandwagon', badly_rated_item_flag=True, camo_flag=True, num_fake_user=0.01, filler_size=0.01, selected_size=0.01)
	# # params.readable_path(params.review_origin_path)
	

	# a=params.__dict__
	# for k in a.keys():
	# 	print k,':::::',a[k]
