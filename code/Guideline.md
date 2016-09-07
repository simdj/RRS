============================================================
1. preprocess
2. inject fake reviews/vote
	2-1. Average attack
	2-2. Bandwagon attack
3. embedding
4. compute helpfulness
5. MF(helpfulness*ratings)
============================================================


******************************************************************
(Done)
1. preprocess
	Input
		raw review 	<-- '../dataset/CiaoDVD/raw_review.txt'
		raw vote 	<-- '../dataset/CiaoDVD/raw_vote.txt'
	Output
		review 	--> './intermediate/review.npy'
		vote 	--> './intermediate/vote.npy'

******************************************************************
2. inject fake review/vote
	Input
		review 	<-- './intermediate/review.npy'
		vote 	<-- './intermediate/vote.npy'
	Output
		fake review 	--> './intermediate/fake_review_[average|bandwagon].npy'
		camo review 	--> './intermediate/camo_review_[average|bandwagon].npy'
		fake vote 		--> './intermediate/fake_vote_[average|bandwagon].npy'

******************************************************************
(Done)
3. embedding
	Input
		review_numpy <-- './intermediate/review.npy'
		vote_numpy <-- './intermediate/vote.npy'
		
		fake review 	--> './intermediate/fake_review_[average|bandwagon].npy'
		camo review 	--> './intermediate/camo_review_[average|bandwagon].npy'
		fake vote 		--> './intermediate/fake_vote_[average|bandwagon].npy'
	Output
		user_embedding --> './intermediate/user2vec.emb'
	Intermediate
		review_writer : review_writer(review)=reviewer
		similar_reviewer : similar_reviewer(item,rating)= set of reviewers
		reviewer_follower : reviewer_follower(reviewer) = list of followers (allowing duplicated)
		// similar_voter : similar_voter(review,vote_value) = set of voters

******************************************************************
(Done)
4. compute helpfulness
	Input
		review_numpy <-- './intermediate/review.npy'
		vote_numpy <-- './intermediate/review.npy'

		fake review 	--> './intermediate/fake_review_[average|bandwagon].npy'
		camo review 	--> './intermediate/camo_review_[average|bandwagon].npy'
		fake vote 		--> './intermediate/fake_vote_[average|bandwagon].npy'
		
		user_embedding <-- './intermediate/user2vec.emb'
	Output
		origin helpful 	--> './intermediate/helpful.npy'
		fake_helpful 	--> './intermediate/fake_helpful.npy'
		camo_helpful 	--> './intermediate/camo_helpful.npy'

	Intermediate
		review_writer : review_writer(review)=reviewer
		vote_tensor : T(voter,reviewer,item)=helpful vote

******************************************************************
5. MF(helpfulness*ratings)
	Input
		origin review 	<-- './intermediate/review.npy'
		fake review 	<-- './intermediate/fake_review_[average|bandwagon].npy'
		camo review 	<-- './intermediate/camo_review_[average|bandwagon].npy'

		origin helpful 	<-- './intermediate/helpful.npy'
		fake helpful 	<-- './intermediate/fake_helpful.npy'
		camo helpful 	<-- './intermediate/camo_helpful.npy'
		

	Output
		user_latent --> './output/user_latent.npy'
		item_latent --> './output/item_latent.npy'