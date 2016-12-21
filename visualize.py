import gensim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import numpy as np

def draw(file_name):
	model = gensim.models.Word2Vec.load(file_name)
	print model
	words = model.vocab.keys()

	reviewer_list = filter(lambda x:int(x)<3000 ,words)
	print 'reviewer list length', len(reviewer_list)
	vectors = [model[reviewer] for reviewer in reviewer_list]
	tsne = TSNE(n_components=2, random_state=0)
	vectors2d = tsne.fit_transform(vectors)

	plt.figure(figsize=(6, 6))

	first = True
	alternate = True
	for point, reviewer in zip(vectors2d , reviewer_list):
		# plot points
		if int(reviewer)>=1822 and int(reviewer)<=2000:

			plt.scatter(point[0]+np.random.rand(), point[1]+np.random.rand(), c='r', marker="D")
		else:
			plt.scatter(point[0], point[1], c='g',marker="x")
		# plot word annotations
		# plt.annotate(
		# 	'.',
		# 	xy = (point[0], point[1]),
		# 	xytext = (-7, -6) if first else (7, -6),
		# 	textcoords = 'offset points',
		# 	ha = 'right' if first else 'left',
		# 	va = 'bottom',
		# 	size = "x-large"
		# )
		# first = not first if alternate else first
	plt.tight_layout()

	red_patch = mpatches.Patch(color='red', label='Fake User')
	green_patch = mpatches.Patch(color='green', label='Normal User')
	plt.legend(handles=[green_patch,red_patch], loc='upper left')

	plt.show()
# X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# model = TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
# result = model.fit_transform(X)
# print result

file_name = 'code/bandwagon_1%_1%_1%_emb_32/embedding_attacked.emb'
# file_name = 'embedding_attacked.emb'
draw(file_name)

