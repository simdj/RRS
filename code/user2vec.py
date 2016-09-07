# 3. embedding
#     Input
#         review_numpy <-- './intermediate/review_numpy.npy'
#         vote_numpy <-- './intermediate/vote_numpy.npy'
#         fake review     --> './intermediate/fake_review_[average|bandwagon].npy'
#         camo review     --> './intermediate/camo_review_[average|bandwagon].npy'
#         fake vote       --> './intermediate/fake_vote_[average|bandwagon].npy'
#     Output
#         user_embedding --> './intermediate/user2vec.emb'
#     Intermediate
#         review_writer : review_writer(review)=reviewer
#         similar_reviewer : similar_reviewer(item,rating)= set of reviewers
#         reviewer_follower : reviewer_follower(reviewer) = list of followers (allowing duplicated)
#         // similar_voter : similar_voter(review,vote_value) = set of voters


import numpy as np
import csv
import itertools
from time import time
from gensim.models import Word2Vec

def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))

class user2vec():
    def __init__(self, embedding_dim=32, word2vec_iter=10
        , fake_flag = True
        , camo_flag = True
        , fake_review_numpy_path = './intermediate/fake_review_bandwagon.npy'
        , camo_review_numpy_path = './intermediate/camo_review_bandwagon.npy'
        , fake_vote_numpy_path = './intermediate/fake_vote_bandwagon.npy'
        , user_embedding_path = './intermediate/user2vec.emb'
        ):
        # input
        self.review_numpy_path = './intermediate/review.npy'
        self.vote_numpy_path = './intermediate/vote.npy'
        self.fake_review_numpy_path = fake_review_numpy_path
        self.camo_review_numpy_path = camo_review_numpy_path
        self.fake_vote_numpy_path = fake_vote_numpy_path
        
        # output
        self.user_embedding_path = user_embedding_path
        self.check_embedding_path = './intermediate/check_user2vec.emb'

        # embedding
        self.fake_flag = fake_flag
        self.camo_flag = camo_flag
        self.user_embedding = None
        self.embedding_dim = embedding_dim
        self.word2vec_iter = word2vec_iter

        # intermediate
        self.review_writer = dict()
        self.similar_reviewer = dict()
        self.reviewer_follower = dict()


    def enumerate_similar_reviewer_pair(self):
        '''
        1. same review i.e. same movie rating
        '''
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

            item_and_rating = str(item_id)+','+str(rating_value)
            if item_and_rating not in self.similar_reviewer:
                self.similar_reviewer[item_and_rating] = set()
            self.similar_reviewer[item_and_rating].add(reviewer)

        ################################################################################
        # naive version
        similar_reviewer_pairs = []
        for k, similar_reviewer_set in self.similar_reviewer.iteritems():
            same_reviewer_pair = itertools.combinations(similar_reviewer_set, 2)
            similar_reviewer_pairs.extend(map(lambda x: [x[0], x[1]], same_reviewer_pair))
        # but consider 1/log(degree+c)!!!
        return similar_reviewer_pairs

    def enumerate_reviewer_follower_pair(self):
        '''
        2. same thinking i.e. reviewer-follower
        '''
        reviewer_follower_pairs = []

        # 2-1) record (review_id, reviewer_user_id)
        overall_review_matrix = np.load(self.review_numpy_path)

        if self.fake_flag:
            fake_review_matrix = np.load(self.fake_review_numpy_path)
            overall_review_matrix = np.concatenate((overall_review_matrix, fake_review_matrix))
            if self.camo_flag:
                camo_review_matrix = np.load(self.camo_review_numpy_path)
                overall_review_matrix = np.concatenate((overall_review_matrix, camo_review_matrix))

        for row in overall_review_matrix:
            reviewer = int(row[0])
            # item_id = int(row[1])
            # rating_value = row[2]
            review_id = int(row[3])
            self.review_writer[review_id]=reviewer

        # 2-2) enumerate (reviewer_user_id, rater_user_id who vote the review helpful)

        # 0. Off topic / # 1. Not Helpful / # 2. Somewhat Helpful (less helpful)
        # 3. Helpful / # 4. Very Helpful / # 5. Exceptional
        vote_matrix = np.load(self.vote_numpy_path)
        for row in vote_matrix:
            voter = int(row[0])
            review_id = int(row[1])
            helpful_vote = row[2]

            if helpful_vote>=3:
                reviewer = self.review_writer[review_id]
                if reviewer not in self.reviewer_follower:
                    # no set, duplicates are allowed
                    self.reviewer_follower[reviewer]=[]
                self.reviewer_follower[reviewer].append(voter)
        ################################################################################
        # naive version
        for reviewer, follower_list in self.reviewer_follower.iteritems():
            reviewer_follower_pair = []
            for follower in follower_list:
                reviewer_follower_pair.append([reviewer, follower])

            reviewer_follower_pairs.extend(reviewer_follower_pair)
        # but consider 1/log(degree+c)!!!
        return reviewer_follower_pairs

    def enumerate_rater_rater_pair(self):
        '''
        3. same review rating i.e. same helpful about a movie review
        '''
        overall_review_matrix = np.load(self.review_numpy_path)

        if self.fake_flag:
            fake_review_matrix = np.load(self.fake_review_numpy_path)
            overall_review_matrix = np.concatenate((overall_review_matrix, fake_review_matrix))
            if self.camo_flag:
                camo_review_matrix = np.load(self.camo_review_numpy_path)
                overall_review_matrix = np.concatenate((overall_review_matrix, camo_review_matrix))

        # for row in overall_review_matrix:

    # def enumerate_user_pairs(self):
    #     '''
    #     1.reviewer-reviewr (review_numpy_path) --> self.same_review
    #     2.reviewer-rater (review_numpy_path, vote_numpy_path) --> self.reviewer_follower
    #     3.rater-rater (vote_numpy_path) --> self.same_review_rating
    #     '''
    #     # pair1 = self.enumerate_similar_reviewer_pair()
    #     pair2 = self.enumerate_reviewer_follower_pair()
    #     # pair3 = self.enumerate_rater_rater_pair() #1,600,000KB? RAM explorsion
    #     # print(len(pair1))  # 786186 ->  576826????
    #     print(len(pair2))  # 1518033
    #     # print(len(pair3))
    #
    #     return

    def learn_embedding(self, train_data=None):
        #  Learn embedding by optimizing the Skipgram objective using negative sampling.
        train_data = [map(str,x) for x in train_data]
        print("learning sentence # :", len(train_data))
        if not self.user_embedding:
            # first train data
            self.user_embedding = Word2Vec(train_data, size=self.embedding_dim
                , sg=1, negative=20, window=2, min_count=0, workers=8, iter=self.word2vec_iter)
        else:
            # new sentences
            self.user_embedding.train(train_data)
        return

    def save_embedding(self):
        self.user_embedding.save(self.user_embedding_path)
        self.user_embedding.save_word2vec_format(self.check_embedding_path)

    def load_embedding(self):
        return Word2Vec.load(self.user_embedding_path)

    def whole_process(self):
        self.learn_embedding(self.enumerate_reviewer_follower_pair())
        self.learn_embedding(self.enumerate_similar_reviewer_pair())
        self.save_embedding()

if __name__ == "__main__":
    u2v = user2vec()
    u2v.whole_process()
    
# a = u2v.load_embedding()
# print(cosine_distance(a['3745'], a['14742']))
# print(cosine_distance(a['3745'], a['5138']))
# print(cosine_distance(a['3745'], a['44']))
# print(cosine_distance(a['3745'], a['11686']))
# print(cosine_distance(a['3745'], a['534']))
# print(cosine_distance(a['3745'], a['11772']))
#
# print(cosine_distance(a['3745'], a['2299']))
# print(cosine_distance(a['3745'], a['1099']))
# print(cosine_distance(a['3745'], a['989']))
# print(cosine_distance(a['3745'], a['781']))
# # # print(u2v.similar_reviewer['2227,4'])
# print(a.most_similar(positive=['11772','44'],topn=10))




