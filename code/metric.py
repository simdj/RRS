import numpy as np
from parameter_controller import *
class metric():
	def __init__(self, params):
		self.review_origin_path = params.review_origin_path
		self.review_fake_path = params.review_fake_path
		self.review_camo_path = params.review_camo_path

		self.U_path = params.U_path
		self.V_path = params.V_path

	def get_honest_user_list(self):
		return np.unique(np.load(self.review_origin_path)[:,0])

	def get_fake_user_list(self):
		return np.unique(np.load(self.review_fake_path)[:,0])

	def get_target_item_list(self):
		return np.unique(np.load(self.review_fake_path)[:,1])

	def mean_prediction_of_honest_rating_on_target(self):
		U_matrix = np.load(self.U_path)
		V_matrix = np.load(self.V_path)  # V_matrix.shape = (rank,I)
		
		# (user,target_item)
		honest_user_list = self.get_honest_user_list()
		target_item_list = self.get_target_item_list()

		target_V_matrix = U_matrix[map(int,honest_user_list),:]
		target_V_matrix = V_matrix[:,map(int, target_item_list)]
		matmul_result = np.matmul(U_matrix, target_V_matrix)
		# average
		# each_overall_rating = np.mean(matmul_result, axis=0)
		total_overall_rating = np.mean(matmul_result)
		return total_overall_rating

	# def RMSE(self, observed, prediction):
	# 	# order matters!
	# 	return np.sqrt( np.sum(np.square(observed-prediction)) / len(observed) )
	# def MAE(self, observed, prediction):
	# 	# order matters!
	# 	return np.sum(np.abs(observed-prediction)) / len(observed)

	# def error_of_target_prediction(self, target_item_list, U_matrix, V_matrix, error_type='RMSE'):
	# 	# The smaller error is, the Better result is

	# 	# 1. from rating R
	# 	target_rating_row = []
	# 	for row in self.review_matrix:
	# 		if int(row[1]) in target_item_list:
	# 			target_rating_row.append(row)
	# 	target_rating_row = np.array(target_rating_row)
		
	# 	# 2. from prediction U,V / by sparse format
	# 	target_prediction_value = []
	# 	for row in target_rating_row:
	# 		tmp = np.dot(U_matrix[int(row[0]),:],V_matrix[int(row[1]),:])
	# 		target_prediction_value.append(tmp)
	# 	target_prediction_value = np.array(target_prediction_value)
		
	# 	if error_type=='RMSE':
	# 		return self.RMSE(target_rating_row[:,2],target_prediction_value)
	# 	else:
	# 		return self.MAE(target_rating_row[:,2],target_prediction_value)


	# def error_without_attacker(self, U_matrix, V_matrix, error_type='RMSE'):
	# 	observed = self.review_matrix[:,2]
	# 	prediction = []
	# 	for row in self.review_matrix:
	# 		tmp = np.dot(U_matrix[int(row[0]),:],V_matrix[int(row[1]),:])
	# 		prediction.append(tmp)
	# 	prediction = np.array(prediction)

	# 	if error_type=='RMSE':
	# 		return self.RMSE(observed, prediction)
	# 	else:
	# 		return self.MAE(observed, prediction)

	# # def robustness_result(self):
	# # 	before_rmse
	# # 	after_rmse
	# # 	difference
	# # 	return before_rmse, after_rmse, difference
		




if __name__=="__main__":
	exp_title = 'bandwagon_3%_3%_3%_emb_32'
	params = parse_exp_title(exp_title)
	a=metric(params)




		

