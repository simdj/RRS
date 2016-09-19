import numpy as np
from parameter_controller import *
class metric():
	def __init__(self, params):
		self.review_origin_numpy_path = params.review_origin_numpy_path
		self.review_fake_numpy_path = params.review_fake_numpy_path
		self.review_camo_numpy_path = params.review_camo_numpy_path


		# origin user
		self.review_matrix = np.load(self.review_origin_numpy_path)
		# fake user
		if params.fake_flag:
			self.fake_review_matrix = np.load(self.review_fake_numpy_path)
		# camo user
		if params.camo_flag:
			self.camo_review_matrix = np.load(self.review_camo_numpy_path)
		


		self.U_before
		self.V_before

		self.U_after
		self.V_after




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

	def robustness_result(self):
		before_rmse
		after_rmse
		difference
		return before_rmse, after_rmse, difference
		




if __name__=="__main__":
	exp_title = 'emb_64_rank_50_bandwagon_1%_1%_5%'
	params = parse_exp_title(exp_title)
	a=metric(params)




		

