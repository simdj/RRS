
def amplify_cooccurence(corpus):
	pair_counter = dict()
	for x,y in corpus:
		key = str(int(x))+','+str(int(y))
		if key not in pair_counter:
			pair_counter[key]=0
		pair_counter[key]+=1

	ret = []
	for key, co_occur in pair_counter.iteritems():
		ret.extend([key.split(',')]*(co_occur*co_occur))
	return ret

a=[[1,2],[3,4],[1,2]]
print amplify_cooccurence(a)