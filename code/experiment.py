import time

from preprocess import preprocess
from attack_model_bandwagon import attack_model_bandwagon
from user2vec import user2vec
from helpful import helpful_measure

class Timer(object):
	def __init__(self, name=None):	
		self.name = name
	def __enter__(self):
		self.tstart = time.time()
	def __exit__(self, type, value, traceback):
		print('[%s]' % self.name, "Elapsed: %.2g sec" % (time.time() - self.tstart))

def parse_exp_title(exp_title):
	param_obj = dict()
	
	param_list = exp_title.split('_')

	param_obj['which_attack'] = param_list[0]
	param_obj['embedding_dim'] = int(param_list[2])
	param_obj['rank'] = int(param_list[4])
	if which_attack!='None':
		param_obj['num_fake_users'] = float(param_list[6])/100.0
		param_obj['num_fake_items'] = float(param_list[7])/100.0
		param_obj['num_camo_items'] = float(param_list[8])/100.0

	return param_obj

#######################################################################################

exp_title = 'badnwagon_emb_32_rank_50_attack_1_1_1'
exp_param = parse_exp_title(exp_title)

#######################################################################################


with Timer("1. preprocess"):
	pp=preprocess(user_threshold=5, item_threshold=5)
	pp.whole_process()

with Timer("2. attack"):
	if exp_param['which_attack'] != 'None':
		am = attack_model_bandwagon(
			num_fake_users=exp_param['num_fake_users']
			, num_fake_items=exp_param['num_fake_items']
			, num_camo_items=exp_param['num_camo_items'])
		am.whole_process()

with Timer("3. User2Vec"):
	u2v = user2vec(embedding_dim=exp_param['embedding_dim'], word2vec_iter=10)
	u2v.whole_process()

with Timer("4. Compute helpfulness"):
	hm = helpful_measure(doubt_weight=5)
	hm.whole_process()


with Timer("5. Matrix factorization"):
	from matrix_factorization import matrix_factorization
	mf = matrix_factorization(rank=50, lda=1, max_iter=5001)
	mf.whole_process()
