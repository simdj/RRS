import numpy as np
import time
import os
import logging
from scipy.sparse import csr_matrix
import math

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
            # print( "iteration:", iteration, "time:", time.time()-s, "RMSE: ",rmse(X,Y,Pui), "Important", mean_prediction_rating_on_target(X,Y, honest_user_list, target_item_list))
            print( "iteration:", iteration, "time:", time.time()-s, "RMSE: ",rmse(X,Y,Pui))
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

regularization = 1e-9
tolerance = math.sqrt(regularization)
tolerance = 0.001

rating_observed = csr_matrix([[1, 1, 0, 1, 0, 0],
                     [0, 1, 1, 1, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 1, 0, 0, 0, 1],
                     [0, 0, 0, 0, 1, 0]], dtype=np.float64)
helpful_observed = csr_matrix([[10, 10, 0, 10, 0, 0],
                     [0, 1, 1, 1, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 1, 0, 0, 0, 1],
                     [0, 0, 0, 0, 1, 0]], dtype=np.float64)

# try out pure python version
rows, cols = alternating_least_squares(rating_observed, helpful_observed, 7,regularization, use_native=False)

reconstructed = rows.dot(cols.T)
# print(np.sum(reconstructed-rating_observed))
np.set_printoptions(precision=1)
print reconstructed