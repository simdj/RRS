from scipy.sparse import *
from scipy import *
import numpy as np
from sklearn.decomposition import NMF

from parameter_controller import *
params = parse_exp_title('bandwagon_3%_3%_3%_emb_32')

train = np.load(params.train_data_path)
test_overall = np.load(params.test_overall_data_path)
test_target = np.load(params.test_target_data_path)


every = np.concatenate((train, test_overall, test_target))
num_every_row = len(np.unique(every[:,0]))
num_every_col = len(np.unique(every[:,1]))

row = train[:,0]
col = train[:,1]
data = train[:,2]

X = csr_matrix((data,(row,col)), shape=(num_every_row, num_every_col))


model = NMF(n_components=30, init='random', l1_ratio=1.0)
model.fit(X)