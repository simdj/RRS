# 4. compute helpfulness
#     Input
#         review_numpy <-- './intermediate/review.npy'
#         vote_numpy <-- './intermediate/review.npy'

#         fake review     --> './intermediate/fake_review_[average|bandwagon].npy'
#         camo review     --> './intermediate/camo_review_[average|bandwagon].npy'
#         fake vote       --> './intermediate/fake_vote_[average|bandwagon].npy'
        
#         user_embedding <-- './intermediate/user2vec.emb'
#     Output
#         origin helpful  --> './intermediate/helpful.npy'
#         fake_helpful    --> './intermediate/fake_helpful.npy'
#         camo_helpful    --> './intermediate/camo_helpful.npy'

#     Intermediate
#         review_writer : review_writer(review)=reviewer
#         vote_tensor : T(voter,reviewer,item)=helpful vote

# star rating! 0~5 -> 0~10
import numpy as np
import itertools
from gensim.models import Word2Vec
from time import time

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))


def user_similarity_with_threshold(v1,v2):
    return (cosine_similarity(v1,v2)-0.9)*10

class helpful_measure():
    def __init__(self, params):
        # input        
        self.review_numpy_path = params.review_numpy_path
        self.vote_numpy_path = params.vote_numpy_path
        self.fake_review_numpy_path = params.fake_review_numpy_path
        self.camo_review_numpy_path = params.camo_review_numpy_path
        self.fake_vote_numpy_path = params.fake_vote_numpy_path
        self.user_embedding_path = params.user_embedding_path
        # output
        self.helpful_numpy_path= params.helpful_numpy_path
        self.fake_helpful_numpy_path= params.fake_helpful_numpy_path
        self.camo_helpful_numpy_path= params.camo_helpful_numpy_path
        # readable
        self.helpful_csv_path = params.helpful_csv_path
        self.fake_helpful_csv_path= params.fake_helpful_csv_path
        self.camo_helpful_csv_path= params.camo_helpful_csv_path
        # experiment condition
        self.fake_flag = params.fake_flag
        self.camo_flag = params.camo_flag
        self.doubt_weight = params.doubt_weight

        ####################################################################################
        # intermediate        
        self.user_embedding = Word2Vec.load(self.user_embedding_path)

        self.review_dict = dict()  # R(reviewer,item)= rating value
        self.helpful_matrix = []  # H(reviewer, item) = helpfulness score

        self.review_writer = dict()  # dict(review_id) = review_writer_id
        self.helpful_vote_tensor = dict()  # H(reviewer, item) = (cos_distance(reviewer,voter), helpfulness vote)

        # star rating! 0~5 -> 0~10
        self.base_helpful_numerator = 15.0
        self.base_helpful_denominator = 6.0

    def fill_review_dict(self):
        overall_review_matrix = np.load(self.review_numpy_path)
        if self.fake_flag:
            fake_review_matrix = np.load(self.fake_review_numpy_path)
            overall_review_matrix = np.concatenate((overall_review_matrix, fake_review_matrix))
            if self.camo_flag:
                camo_review_matrix = np.load(self.camo_review_numpy_path)
                overall_review_matrix = np.concatenate((overall_review_matrix, camo_review_matrix))

        for row in overall_review_matrix:
            reviewer = int(row[0])
            item_id = int(row[1])
            rating_value = row[2]
            review_id = int(row[3])
            # fill review_dict and review_writer
            self.review_dict[str(reviewer) + ',' + str(item_id)] = rating_value
            self.review_writer[str(review_id)] = (reviewer, item_id)
            
    def fill_helpful_vote_tensor(self):
        overall_vote_matrix = np.load(self.vote_numpy_path)
        if self.fake_flag:
            fake_vote_matrix = np.load(self.fake_vote_numpy_path)
            overall_vote_matrix = np.concatenate((overall_vote_matrix, fake_vote_matrix))
        
        for row in overall_vote_matrix:
            voter = int(row[0])
            review_id = int(row[1])
            helpful_vote = row[2]

            reviewer, item_id = self.review_writer[str(review_id)]
            # maybe user_id does not exist in embedding vocab....
            # if then, do not consider similarity
            vote_info = [0, helpful_vote]
            if (str(voter) in self.user_embedding) and (str(reviewer) in self.user_embedding):
                voter_vec = self.user_embedding[str(voter)]
                reviewer_vec = self.user_embedding[str(reviewer)]
                # (distance(reviewer,review rater) and vote_rating)
                vote_info = [user_similarity_with_threshold(voter_vec, reviewer_vec), helpful_vote]

            # fill helpful_vote_tensor
            reviewer_and_item = str(reviewer)+','+str(item_id)
            if reviewer_and_item not in self.helpful_vote_tensor:
                self.helpful_vote_tensor[reviewer_and_item] = []
            self.helpful_vote_tensor[reviewer_and_item].append(vote_info)

    def compute_reveiw_helpful(self, which_review_numpy_path, which_helpful_numpy_path, which_helpful_csv_path):
        helpful_matrix = []
        
        review_matrix = np.load(which_review_numpy_path)
        for row in review_matrix:
            reviewer = int(row[0])
            item_id = int(row[1])
            user_item_key = str(reviewer)+','+str(item_id)

            numerator = self.base_helpful_numerator
            denominator = self.base_helpful_denominator
            # gather evidence
            if user_item_key in self.helpful_vote_tensor:
                vote_info_list = self.helpful_vote_tensor[user_item_key]
                for vote_info in vote_info_list:
                    # star rating! 0~5 -> 0~10
                    # cosine_similarity: {(-1)~(+1)}
                    if vote_info[1] >=3:
                        # similar user agrees -> diminishing return
                        numerator += vote_info[1] * np.exp(-vote_info[0]*self.doubt_weight)
                        denominator += np.exp(-vote_info[0]*self.doubt_weight)
                    else:
                        # dissimilar user disagrees -> diminishing return
                        numerator += vote_info[1] * np.exp( vote_info[0]*self.doubt_weight)
                        denominator += np.exp( vote_info[0]*self.doubt_weight)
            # update posterior
            helpful_matrix.append([reviewer, item_id, numerator / denominator])

        np.save(which_helpful_numpy_path, np.array(helpful_matrix))
        np.savetxt(which_helpful_csv_path, np.array(helpful_matrix))

        
    def whole_process(self):
        self.fill_review_dict()
        self.fill_helpful_vote_tensor()
        self.compute_reveiw_helpful(self.review_numpy_path, self.helpful_numpy_path, self.helpful_csv_path)
        if self.fake_flag:
            self.compute_reveiw_helpful(self.fake_review_numpy_path, self.fake_helpful_numpy_path, self.fake_helpful_csv_path)
        if self.camo_flag:
            self.compute_reveiw_helpful(self.camo_review_numpy_path, self.camo_helpful_numpy_path, self.camo_helpful_csv_path)
    
def helpful_test():
    # is helpfulness well assigned???
    a=np.load('./intermediate/helpful.npy')
    b=np.load('./intermediate/fake_helpful_bandwagon.npy')

    print ('origin rating #',len(a), 'fake rating #',len(b))
    print('target helpful mean', np.mean(b[:, 2]))
    print("*********************************")
    print ('1',np.percentile(a[:, 2], 1), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 1))))
    print ('10',np.percentile(a[:, 2], 10), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 10))))
    print ('25',np.percentile(a[:, 2], 25), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 25))))
    print ('50',np.percentile(a[:, 2], 50), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 50))))
    print ('75',np.percentile(a[:, 2], 75), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 75))))
    print ('90',np.percentile(a[:, 2], 90), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 90))))
    print ('99',np.percentile(a[:, 2], 99), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 99))))

if __name__ == "__main__":

    # hm = helpful_measure()
    # hm.whole_process()
    
    helpful_test()




