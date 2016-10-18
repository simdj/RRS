# 4. compute helpfulness
#     Input
#         review_numpy <-- './intermediate/review.npy'
#         vote_numpy <-- './intermediate/review.npy'

#         fake review     --> './intermediate/fake_review_[average|bandwagon].npy'
#         camo review     --> './intermediate/camo_review_[average|bandwagon].npy'
#         fake vote       --> './intermediate/fake_vote_[average|bandwagon].npy'
        
#         embedding_attacked <-- './intermediate/user2vec.emb'
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
    # -1~1 -> -1.9~0.1 -> -19~1
    return (cosine_similarity(v1,v2)-0.9)*10
# clean? attacked?
# naive? robust?
class helpful_measure():
    def __init__(self, params, fake_flag=True, camo_flag=True, robust_flag=True):
         # experiment condition
        self.fake_flag = fake_flag
        self.camo_flag = camo_flag
        self.robust_flag = robust_flag
        self.doubt_weight = params.doubt_weight


        # input        
        self.review_origin_path = params.review_origin_path
        self.review_fake_path = params.review_fake_path
        self.review_camo_path = params.review_camo_path
        
        self.vote_origin_path = params.vote_origin_path
        self.vote_fake_path = params.vote_fake_path
        
        # output
        self.helpful_origin_path = ''
        self.helpful_fake_path = ''
        self.helpful_camo_path = ''

        if not fake_flag:
            self.embedding_model = Word2Vec.load(params.embedding_clean_path)
            if not robust_flag: # clean and naive
                self.helpful_origin_path = params.helpful_origin_clean_naive_path
            else:   # clean and robust
                self.helpful_origin_path = params.helpful_origin_clean_robust_path
        else:
            self.embedding_model = Word2Vec.load(params.embedding_attacked_path)
            if not robust_flag: # attacked and naive
                self.helpful_origin_path = params.helpful_origin_attacked_naive
                self.helpful_fake_path = params.helpful_fake_attacked_naive
                self.helpful_camo_path = params.helpful_camo_attacked_naive
            else:   # attacked and robust
                self.helpful_origin_path = params.helpful_origin_attacked_robust
                self.helpful_fake_path = params.helpful_fake_attacked_robust
                self.helpful_camo_path = params.helpful_camo_attacked_robust


        ####################################################################################
        # intermediate        
        self.review_dict = dict()  # R(reviewer,item)= rating value
        self.helpful_matrix = []  # H(reviewer, item) = helpfulness score

        self.review_writer = dict()  # dict(review_id) = review_writer_id
        self.helpful_vote_tensor = dict()  # H(reviewer, item) = (cos_distance(reviewer,voter), helpfulness vote)

        # star rating! 0~5 -> 0~10
        self.base_helpful_numerator = 1.5
        self.base_helpful_denominator = 0.6

    def fill_review_dict(self):
        overall_review_matrix = np.load(self.review_origin_path)
        if self.fake_flag:
            fake_review_matrix = np.load(self.review_fake_path)
            overall_review_matrix = np.concatenate((overall_review_matrix, fake_review_matrix))
            if self.camo_flag:
                camo_review_matrix = np.load(self.review_camo_path)
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
        overall_vote_matrix = np.load(self.vote_origin_path)
        if self.fake_flag:
            fake_vote_matrix = np.load(self.vote_fake_path)
            overall_vote_matrix = np.concatenate((overall_vote_matrix, fake_vote_matrix))
        
        for row in overall_vote_matrix:
            voter = int(row[0])
            review_id = int(row[1])
            helpful_vote = row[2]

            reviewer, item_id = self.review_writer[str(review_id)]
            # maybe user_id does not exist in embedding vocab....
            # if then, do not consider similarity
            vote_info = [0, helpful_vote]
            if (str(voter) in self.embedding_model) and (str(reviewer) in self.embedding_model):
                voter_vec = self.embedding_model[str(voter)]
                reviewer_vec = self.embedding_model[str(reviewer)]
                # (distance(reviewer,review rater) and vote_rating)
                vote_info = [cosine_similarity(voter_vec, reviewer_vec), helpful_vote]

            # fill helpful_vote_tensor
            reviewer_and_item = str(reviewer)+','+str(item_id)
            if reviewer_and_item not in self.helpful_vote_tensor:
                self.helpful_vote_tensor[reviewer_and_item] = []
            self.helpful_vote_tensor[reviewer_and_item].append(vote_info)

    def compute_review_helpful(self, which_review_path, which_helpful_path):
        helpful_matrix = []
        
        review_matrix = np.load(which_review_path)
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
                    if self.robust_flag:
                        # dissimilar (-1) ~ (+1) similar
                        sim = vote_info[0]
                        rating = vote_info[1]
                        if rating >=3:
                            # similar user agrees -> diminishing return
                            # sim>=0.9 -> positive value
                            # sim< 0.9 -> zero
                            sim_with_threshold = max(0, (sim-0.9)*10)
                            numerator += rating * np.exp(-sim_with_threshold*self.doubt_weight)
                            denominator += np.exp(-sim_with_threshold*self.doubt_weight)
                        else:
                            # dissimilar user disagrees -> diminishing return
                            # sim<=-0.9 -> negative value
                            # sim> 0.9 -> zero
                            sim_with_threshold = min(0, (sim+0.9)*10)
                            numerator += rating * np.exp( sim_with_threshold*self.doubt_weight)
                            denominator += np.exp( sim_with_threshold*self.doubt_weight)
                    else:
                        numerator += vote_info[1]
                        denominator += 1
            # update posterior
            helpful_matrix.append([reviewer, item_id, numerator / denominator])

        np.save(which_helpful_path, np.array(helpful_matrix))
        # np.savetxt(which_helpful_csv_path, np.array(helpful_matrix))

    
    def whole_process(self):
        self.fill_review_dict()
        self.fill_helpful_vote_tensor()

        self.compute_review_helpful(self.review_origin_path, self.helpful_origin_path)
        if self.helpful_fake_path:
            self.compute_review_helpful(self.review_fake_path, self.helpful_fake_path)
        if self.helpful_camo_path:
            self.compute_review_helpful(self.review_camo_path, self.helpful_camo_path)
    
    def compare_helpful_on_target(self):
        review_fake=np.load(self.review_fake_path)
        target_item_list = np.unique(review_fake[:,1])

        h=np.load(self.helpful_origin_path)
        origin_helpful_on_target = [h[h[:,1]==target_item,2] for target_item in target_item_list]
        
        if self.helpful_fake_path:
            hf = np.load(self.helpful_fake_path)
            hf = hf[:,2]




    def helpful_test(self):
        try:
            print("{Helpful test}")
            a=np.load(self.helpful_origin_path)
            print('origin helpful mean', np.mean(a[:, 2]))
            
            if not self.helpful_fake_path:
                return
            # is helpfulness well assigned???
            b=np.load(self.helpful_fake_path)

            print('target helpful mean', np.mean(b[:, 2]))
            print ('origin rating #',len(a), 'fake rating #',len(b))
            # print ('1',np.percentile(a[:, 2], 1), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 1))))
            print ('10',np.percentile(a[:, 2], 10), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 10))))
            # print ('25',np.percentile(a[:, 2], 25), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 25))))
            print ('50',np.percentile(a[:, 2], 50), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 50))))
            # print ('75',np.percentile(a[:, 2], 75), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 75))))
            print ('90',np.percentile(a[:, 2], 90), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 90))))
            # print ('99',np.percentile(a[:, 2], 99), np.sqrt(np.square(np.mean(b[:, 2])/np.percentile(a[:, 2], 99))))
        except:
            pass

if __name__ == "__main__":
    from parameter_controller import *

    exp_title = 'bandwagon_1%_1%_1%_emb_32'
    print('Experiment Title', exp_title)
    params = parse_exp_title(exp_title)

    print("Clean and naive")
    hm_clean_naive = helpful_measure(params=params, fake_flag=False, camo_flag=False, robust_flag=False)
    hm_clean_naive.whole_process()
    hm_clean_naive.helpful_test()

    print("Clean and robust")
    hm_clean_robust = helpful_measure(params=params, fake_flag=False, camo_flag=False, robust_flag=True)
    hm_clean_robust.whole_process()
    hm_clean_robust.helpful_test()

    print("Attacked and naive")
    hm_attacked_naive = helpful_measure(params=params, fake_flag=True, camo_flag=True, robust_flag=False)
    hm_attacked_naive.whole_process()
    hm_attacked_naive.helpful_test()

    print("Attacked and robust")
    hm_attacked_robust = helpful_measure(params=params, fake_flag=True, camo_flag=True, robust_flag=True)
    hm_attacked_robust.whole_process()
    hm_attacked_robust.helpful_test()
    




