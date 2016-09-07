import numpy as np 

a=np.load('./intermediate/helpful.npy')
b=np.load('./intermediate/fake_helpful.npy')
# print len(np.unique(np.concatenate((a[:,0], b[:,0]), axis=0)))
print np.percentile(a[:,2],25)
print np.percentile(a[:,2],50)
print np.percentile(a[:,2],75)
print np.percentile(a[:,2],95)
print np.percentile(a[:,2],99)
print np.percentile(a[:,2],99.9)
print np.mean(b[:,2])
print("*********************************")
print np.mean(b[:,2])-np.percentile(a[:,2],25)
print np.mean(b[:,2])-np.percentile(a[:,2],50)
print np.mean(b[:,2])-np.percentile(a[:,2],75)
print np.mean(b[:,2])-np.percentile(a[:,2],90)



# from gensim.models import Word2Vec
# user_embedding = Word2Vec.load('./intermediate/user2vec.emb')



# num_fake_users=100, num_fake_items=10
# , num_fake_reviews=1000, num_fake_votes=10000
# , fake_rating_value=5, filler_size=0.01
# alpha = 3
# 2.67210920324
# 3.12883905748
# 3.49071026501
# 3.75534007324
# 3.98485075712
# 4.34774549549
# 3.16570441739
# *********************************
# 0.493595214146
# 0.0368653599049
# -0.32500584762
# -0.508286419213


# num_fake_users=100, num_fake_items=10
# , num_fake_reviews=1000, num_fake_votes=10000
# , fake_rating_value=5, filler_size=0.01
# alpha = 5
# 2.61164955457
# 2.92509408983
# 3.26473219944
# 3.58550679404
# 3.79409802946
# 4.16591093305
# 2.79801866835
# *********************************
# 0.186369113774
# -0.127075421481
# -0.466713531095
# -0.688484683751



# num_fake_users=100, num_fake_items=10
# , num_fake_reviews=1000, num_fake_votes=10000
# , fake_rating_value=5, filler_size=0.01
# alpha = 10
# 2.53163304633
# 2.62299844808
# 2.79520371963
# 3.1169737405
# 3.33464159512
# 3.66828384863
# 2.52773564912
# *********************************
# -0.00389739721178
# -0.0952627989611
# -0.267468070513
# -0.469665627687