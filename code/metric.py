from parameter_controller import *

class metric():
	def __init__(self, params):
		self.review_numpy_path = params.review_numpy_path
		self.fake_review_numpy_path = params.fake_review_numpy_path
		self.camo_review_numpy_path = params.camo_review_numpy_path


		
		self.review_matrix = np.load(self.review_numpy_path)
		if params.fake_flag:
			self.fake_review_matrix = np.load(self.fake_review_numpy_path)
		if params.camo_flag:
			self.camo_review_matrix = np.load(self.camo_review_numpy_path)
		

		# self.U = np.load(self.output_U)
		# self.V = np.load(self.output_V)

		self.RRS_U_matrix
		self.RRS_V_matrix



	def MAE(self, observed, prediction):
		# order matters!
		np.sum(np.abs(observed-prediction))

	def error_of_target_prediction(self, target_list, U_matrix, V_matrix):
		# The smaller error is, the Better result is

		# 1. from rating R
		target_rating_row = []
		for row in self.review_matrix:
			if int(row[1]) in target_list:
				target_rating_row.append(row)
		target_rating_row = np.array(target_rating_row)
		
		# 2. from prediction U,V / by sparse format
		target_prediction_value = []
		for row in target_rating_row:
			tmp = np.dot(U_matrix[int(row[0]),:],V_matrix[int(row[1]),:])
			target_prediction_value.append(tmp)
		target_prediction_value = np.array(target_prediction_value)


		return self.MAE(target_rating_row[:,2],target_prediction_value)






		

