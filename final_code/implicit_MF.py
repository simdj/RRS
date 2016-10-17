""" Implicit Alternating Least Squares """
import numpy as np
import time
import os
import logging
from scipy.sparse import csr_matrix
import math


log = logging.getLogger("implicit")


class metric():
    def __init__(self, params):
        self.review_origin_path = params.review_origin_path
        self.review_origin = np.load(self.review_origin_path)
        try:
            self.review_fake_path = params.review_fake_path
            self.review_fake = np.load(self.review_fake_path)
        except:
            # print("no attack")
            pass
        # self.review_fake_path = params.default_review_fake_path


        self.target_item_list = np.load(params.target_item_list_path)
        self.fake_user_id_list = np.load(params.fake_user_id_list_path)
        # self.get_honest_user_list = np.array(list(range(np.min(self.fake_user_id_list))))

        self.U_path = params.U_path
        self.V_path = params.V_path

    def get_honest_user_list(self):
        return np.unique(self.review_origin[:, 0])

    def get_honest_high_degree_user_list(self, num_user=100):
        review_data = np.load(self.review_origin_path)
        from collections import Counter
        user_counter = Counter(review_data[:, 0])
        return map(lambda x: x[0], user_counter.most_common(num_user))

    def rmse_rating_on_target(self, honest=True):
        selected_review = self.review_origin
        if not honest:
            try:
                selected_review = self.review_fake
            except:
                return -1.0

        rating_on_target = []
        for target_item in self.target_item_list:
            tmp = selected_review[selected_review[:, 1] == target_item, :]
            rating_on_target.append(tmp)
        rating_on_target = np.concatenate(rating_on_target)

        U_matrix = np.load(self.U_path)
        V_matrix = np.load(self.V_path)  # V_matrix.shape = (rank,I)

        rmse = 0.0
        for rating_row in rating_on_target:
            user_id = int(rating_row[0])
            target_item = int(rating_row[1])

            rating_observed = rating_row[2]
            rating_prediction = np.matmul(U_matrix[user_id, :], V_matrix[:, target_item])

            rmse += np.square(rating_observed - rating_prediction)
        rmse = np.sqrt(rmse / float(len(rating_on_target)))
        return rmse

    def mean_prediction_rating_on_target(self, honest=True):
        U_matrix = np.load(self.U_path)
        V_matrix = np.load(self.V_path)  # V_matrix.shape = (rank,I)

        # (user,target_item)
        if honest:
            focus_user_list = self.get_honest_user_list()
        # focus_user_list = self.get_honest_high_degree_user_list(100)
        else:
            focus_user_list = self.fake_user_id_list
        target_item_list = self.target_item_list

        target_U_matrix = U_matrix[map(int, focus_user_list), :]
        target_V_matrix = V_matrix[:, map(int, target_item_list)]
        matmul_result = np.matmul(target_U_matrix, target_V_matrix)
        # average
        # each_overall_rating = np.mean(matmul_result, axis=0)
        total_overall_rating = np.mean(matmul_result)
        print(
        '[RAW Rating Distribution]', np.min(matmul_result), np.percentile(matmul_result, 25), np.median(matmul_result),
        np.percentile(matmul_result, 75), np.max(matmul_result), total_overall_rating)

        # limit
        matmul_result[matmul_result < 1] = 1
        matmul_result[matmul_result > 5] = 5

        total_overall_rating = np.mean(matmul_result)
        print('[Limit Rating Distribution]', np.min(matmul_result), np.percentile(matmul_result, 25),
              np.median(matmul_result), np.percentile(matmul_result, 75), np.max(matmul_result), total_overall_rating)

        return total_overall_rating


def alternating_least_squares(Cui, Pui, factors, regularization=0.01,
                              iterations=5001, use_native=True, num_threads=0,
                              dtype=np.float64, good_mean=0.28):
    """ factorizes the matrix Cui using an implicit alternating least squares
    algorithm
    Args:
        Cui (csr_matrix): Confidence Matrix
        factors (int): Number of factors to extract
        regularization (double): Regularization parameter to use
        iterations (int): Number of alternating least squares iterations to
        run
        num_threads (int): Number of threads to run least squares iterations.
        0 means to use all CPU cores.
    Returns:
        tuple: A tuple of (row, col) factors
    """
    # _check_open_blas()

    users, items = Cui.shape

    X = np.random.randn(users, factors).astype(dtype) * 0.0004+good_mean
    Y = np.random.randn(items, factors).astype(dtype) * 0.0004+good_mean

    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()
    Pui, Piu = Pui.tocsr(), Pui.T.tocsr()



    # solver = _implicit_MF.least_squares if use_native else least_squares
    solver = least_squares
    s = time.time()
    for iteration in range(iterations):

        solver(Cui, Pui, X, Y, regularization, num_threads)
        solver(Ciu, Piu, Y, X, regularization, num_threads)

        if iteration>0 and iteration%100==0:
            # log.debug("finished iteration %i in %s", iteration, time.time() - s)
            # print("finished iteration %i in %s", iteration, time.time() - s)
            print( "iteration:", iteration, "time:", time.time()-s, "RMSE: ",rmse(X,Y,Pui), "Important", mean_prediction_rating_on_target(X,Y, honest_user_list, target_item_list))
            s = time.time()


    return X, Y


def least_squares(Cui, Pui, X, Y, regularization, num_threads):
    """ For each user in Cui, calculate factors Xu for them
    using least squares on Y.
    Note: this is at least 10 times slower than the cython version included
    here.
    """
    users, factors = X.shape
    # YtY = Y.T.dot(Y)

    for u in range(users):
        A = regularization * np.eye(factors)
        b = np.zeros(factors)
        for nonzero_index_of_row in nonzeros_index(Cui,u):
            i = Cui.indices[nonzero_index_of_row]
            factor = Y[i]
            confidence = Cui.data[nonzero_index_of_row]
            rating = Pui.data[nonzero_index_of_row]

            A+=confidence*np.outer(factor,factor)
            b += confidence * rating * factor



        # # accumulate YtCuY + regularization*I in A
        # A = YtY + regularization * np.eye(factors)
        #
        # # accumulate YtCuPu in b
        # b = np.zeros(factors)
        #
        # for index_of_indices in nonzeros_index(Cui,u):
        #     index = Cui.indices[index_of_indices]
        #     confidence = Cui.data[index_of_indices]
        #     factor = Y[index]
        #     A += (confidence - 1) * np.outer(factor, factor)
        #     b += confidence * factor * Pui.data[index_of_indices]

        # for i, confidence in nonzeros(Cui, u):
        #
        #     factor = Y[i]
        #     A += (confidence - 1) * np.outer(factor, factor)
        #     b += confidence * factor*Pui[u]

        # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
        X[u] = np.linalg.solve(A, b)

def nonzeros_index(m,row):
    for index_of_indices in range(m.indptr[row], m.indptr[row + 1]):
        yield index_of_indices

def nonzeros(m, row):
    """ returns the non zeroes of a row in csr_matrix """
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]

def rmse(rows,cols, data):
    ret=0
    users, factors = rows.shape

    for u in range(users):
        for nonzero_idx, value in nonzeros(data,u):
            # print(nonzero_idx,value)
            tmp = rows[u].dot(cols[nonzero_idx])-(value)
            #
            # if ret==0:
            #     print(rows[u])
            #     print(cols[nonzero_idx])
            #     print (rows[u].dot(cols[nonzero_idx]), (value), rows[u].dot(cols[nonzero_idx])-(value))
            ret+=tmp*tmp
    ret = math.sqrt(ret/data.nnz)
    return ret

def mean_prediction_rating_on_target(U,V, focus_user_list, target_item_list):
    U_=U[map(int,focus_user_list),:]
    V_=V[map(int,target_item_list),:]
    matmul_result = np.matmul(U_,V_.T)
    total_overall_rating = np.mean(matmul_result)
    print('[RAW Rating Distribution]', np.min(matmul_result), np.percentile(matmul_result, 25), np.median(matmul_result),np.percentile(matmul_result, 75), np.max(matmul_result), total_overall_rating)

    # limit
    matmul_result[matmul_result < 1] = 1
    matmul_result[matmul_result > 5] = 5

    total_overall_rating = np.mean(matmul_result)
    print('[Limit Rating Distribution]', np.min(matmul_result), np.percentile(matmul_result, 25), np.median(matmul_result),np.percentile(matmul_result, 75), np.max(matmul_result), total_overall_rating)

    return total_overall_rating

###########################################################################
np.random.seed(10)



intermediate_dir = 'bandwagon_1%_1%_1%_emb_32/'
review_origin = np.load(intermediate_dir+'review_origin.npy')
review_fake = np.load(intermediate_dir+'review_fake.npy')
review_camo = np.load(intermediate_dir+'review_camo.npy')

helpful_origin_attacked_naive = np.load(intermediate_dir+'helpful_origin_attacked_naive.npy')
helpful_fake_attacked_naive = np.load(intermediate_dir+'helpful_fake_attacked_naive.npy')
helpful_camo_attacked_naive = np.load(intermediate_dir+'helpful_camo_attacked_naive.npy')

helpful_origin_attacked_robust = np.load(intermediate_dir+'helpful_origin_attacked_robust.npy')
helpful_fake_attacked_robust = np.load(intermediate_dir+'helpful_fake_attacked_robust.npy')
helpful_camo_attacked_robust = np.load(intermediate_dir+'helpful_camo_attacked_robust.npy')


honest_user_list = np.unique(review_origin[:,0])
target_item_list = np.unique(review_fake[:,1])



################################################




# review_data = np.concatenate([review_origin,review_fake,review_camo])
# helpful_data = np.concatenate([helpful_origin,helpful_fake,helpful_camo])

# num_users = len(np.unique(review_data[:,0]))
# num_items = len(np.unique(review_data[:,1]))
# global_mean = np.mean(review_data[:,2])
# print('Users',num_users, 'Items', num_items, 'global mean', global_mean)

# # Cui = csr_matrix((helpful[:,2]/helpful[:,2],(helpful[:,0],helpful[:,1])))
# Cui = csr_matrix((helpful_data[:,2],(helpful_data[:,0],helpful_data[:,1])))
# Pui = csr_matrix((review_data[:,2],(review_data[:,0],review_data[:,1])))

# rank =20
# regularization = 0.1
# U, V = alternating_least_squares(Cui,Pui,rank,regularization, good_mean= np.sqrt(global_mean/rank))

# # print (rmse(U,V,Pui))





algorithm_model_list = ['base','base','naive','robust']
attack_flag_list = [False,True,True,True]

for algorithm_model, attack_flag in zip(algorithm_model_list, attack_flag_list):
    if algorithm_model == 'base' and attack_flag==False:
        review_data = review_origin
        helpful_data = review_origin
        helpful_data[:,-1]=3
    elif algorithm_model =='base' and attack_flag==True:
        review_data = np.concatenate([review_origin,review_fake,review_camo])
        helpful_data = review_data
        helpful_data[:,-1]=3
    elif algorithm_model == 'naive' and attack_flag==True:
        review_data = np.concatenate([review_origin,review_fake,review_camo])
        helpful_data = np.concatenate([helpful_origin_attacked_naive,helpful_fake_attacked_naive,helpful_camo_attacked_naive])
    elif algorithm_model=='robust' and attack_flag==True:
        review_data = np.concatenate([review_origin,review_fake,review_camo])
        helpful_data = np.concatenate([helpful_origin_attacked_robust,helpful_fake_attacked_robust,helpful_camo_attacked_robust])
        
    print 'algorithm_model', algorithm_model, 'attack', attack_flag
    num_users = len(np.unique(review_data[:,0]))
    num_items = len(np.unique(review_data[:,1]))
    global_mean = np.mean(review_data[:,2])
    print('Users',num_users, 'Items', num_items, 'global mean', global_mean)

    # Cui = csr_matrix((helpful[:,2]/helpful[:,2],(helpful[:,0],helpful[:,1])))
    Cui = csr_matrix((helpful_data[:,2],(helpful_data[:,0],helpful_data[:,1])))
    Pui = csr_matrix((review_data[:,2],(review_data[:,0],review_data[:,1])))

    rank =20
    regularization = 0.1
    U, V = alternating_least_squares(Cui,Pui,rank,regularization, iterations=10001, good_mean= np.sqrt(global_mean/rank))
    print( "Final:", "RMSE: ",rmse(U,V,Pui), "Important", mean_prediction_rating_on_target(U,V, honest_user_list, target_item_list))
    print ''
    # print (rmse(U,V,Pui))



# iter : 21 -> 100 seconds
# iter : 101 -> 500 seconds
# iter : 1001 -> 5000 seconds
# iter : 10001 -> 50000 seconds
