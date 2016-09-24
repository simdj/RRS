import numpy as np
from parameter_controller import *
class metric():
	def __init__(self, params):
		pass
		# self.review_origin_path = params.review_origin_path
		# self.review_fake_path = params.review_fake_path
		# self.review_camo_path = params.review_camo_path


		# # origin user
		# self.review_matrix = np.load(self.review_origin_path)
		# # fake user
		# if params.fake_flag:
		# 	self.fake_review_matrix = np.load(self.review_fake_path)
		# # camo user
		# if params.camo_flag:
		# 	self.camo_review_matrix = np.load(self.review_camo_path)
		


		# self.U_before
		# self.V_before

		# self.U_after
		# self.V_after




	def RMSE(self, observed, prediction):
		# order matters!
		return np.sqrt( np.sum(np.square(observed-prediction)) / len(observed) )
	def MAE(self, observed, prediction):
		# order matters!
		return np.sum(np.abs(observed-prediction)) / len(observed)

	def error_of_target_prediction(self, target_item_list, U_matrix, V_matrix, error_type='RMSE'):
		# The smaller error is, the Better result is

		# 1. from rating R
		target_rating_row = []
		for row in self.review_matrix:
			if int(row[1]) in target_item_list:
				target_rating_row.append(row)
		target_rating_row = np.array(target_rating_row)
		
		# 2. from prediction U,V / by sparse format
		target_prediction_value = []
		for row in target_rating_row:
			tmp = np.dot(U_matrix[int(row[0]),:],V_matrix[int(row[1]),:])
			target_prediction_value.append(tmp)
		target_prediction_value = np.array(target_prediction_value)
		
		if error_type=='RMSE':
			return self.RMSE(target_rating_row[:,2],target_prediction_value)
		else:
			return self.MAE(target_rating_row[:,2],target_prediction_value)


	def error_without_attacker(self, U_matrix, V_matrix, error_type='RMSE'):
		observed = self.review_matrix[:,2]
		prediction = []
		for row in self.review_matrix:
			tmp = np.dot(U_matrix[int(row[0]),:],V_matrix[int(row[1]),:])
			prediction.append(tmp)
		prediction = np.array(prediction)

		if error_type=='RMSE':
			return self.RMSE(observed, prediction)
		else:
			return self.MAE(observed, prediction)

	# def robustness_result(self):
	# 	before_rmse
	# 	after_rmse
	# 	difference
	# 	return before_rmse, after_rmse, difference
		

	def overall_rating_of_target(self, target_list, U_matrix, V_matrix):
		# V_matrix.shape = (rank,I)
		target_V_matrix = V_matrix[:,np.array(target_list)]
		# (user,target_item)
		matmul_result = np.matmul(U_matrix, target_V_matrix)
		# average
		each_overall_rating = np.mean(matmul_result, axis=0)
		total_overall_rating = np.mean(matmul_result)
		return each_overall_rating, total_overall_rating




if __name__=="__main__":
	exp_title = 'bandwagon_3%_3%_3%_emb_32'
	params = parse_exp_title(exp_title)
	a=metric(params)




		

