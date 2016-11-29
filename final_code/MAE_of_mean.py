import numpy as np



r=np.load('bandwagon_1%_1%_0%_emb_32/review_origin.npy')

rr=r[:,2]
MAE_list = []
RMSE_list = []
for i in xrange(10):
	np.random.shuffle(rr)

	r_len=len(rr)

	test = rr[:int(r_len/10)]
	train = rr[int(r_len/10):]



	overall_mean_rating = np.mean(train)+0.9
	print overall_mean_rating, np.mean(np.abs(test-overall_mean_rating))
	MAE_list.append(np.mean(np.abs(test-overall_mean_rating)))
	RMSE_list.append(np.sqrt(np.mean(np.square(test-overall_mean_rating))))

print 'final MAE', sum(MAE_list)/(len(MAE_list)*1.0)
print 'final RMSE', sum(RMSE_list)/(len(RMSE_list)*1.0)


rr[rr>5]=5
print np.mean(rr)
rr[rr<1]=1
print np.mean(rr)
