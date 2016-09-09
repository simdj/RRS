import numpy as np 

# a=np.load('./intermediate/helpful.npy')
# b=np.load('./intermediate/fake_helpful.npy')
# print len(np.unique(np.concatenate((a[:,0], b[:,0]), axis=0)))
# print np.percentile(a[:,2],25)
# print np.percentile(a[:,2],50)
# print np.percentile(a[:,2],75)
# print np.percentile(a[:,2],95)
# print np.percentile(a[:,2],99)
# print np.percentile(a[:,2],99.9)
# print np.mean(b[:,2])
# print("*********************************")
# print np.mean(b[:,2])-np.percentile(a[:,2],25)
# print np.mean(b[:,2])-np.percentile(a[:,2],50)
# print np.mean(b[:,2])-np.percentile(a[:,2],75)
# print np.mean(b[:,2])-np.percentile(a[:,2],90)





def cosine(v1, v2):
	return  np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))



def cosine_distance(v1, v2):
	return 1 - np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))

np.random.seed(1)
from gensim.models import Word2Vec
# model = Word2Vec(size=2, sg=1, negative=3, window=1, min_count=1, workers=8, iter=100)
# vocab = [ [str(x) for x in list(range(100))] ]
# model.build_vocab(vocab)
# print model
# print model['1']
# # print model['2']
# # print model['3']
# model.save('test_model')
# model = None

# new_model = Word2Vec.load('test_model')
# new_model.train([['1','2','3'], ['3','2','1'], ['2','3','180808080']])

model = Word2Vec.load('./intermediate/user2vec.emb')
print model.similarity('2651','2586')
print model.similarity('2651','2')