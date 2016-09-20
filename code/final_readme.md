compare before/after attack

==========================================================================

1. before MF
	Input
		attack model
		attack size
		embedding size
	
	output -- directory name = '[attack][attack size][embedding size]'
		review_origin
		review_fake
		review_camo

		vote_origin
		vote_fake
		vote_camo
		
		------------------------------------------
		embedding_clean

		helpful_origin_clean_naive
		helpful_origin_clean_robust
		
		------------------------------------------
		embedding_attacked
		
		helpful_origin_attacked_naive
		helpful_fake_attacked_naive
		helpful_camo_attacked_naive
				
		helpful_origin_attacked_robust
		helpful_fake_attacked_robust
		helpful_camo_attacked_robust
	
==========================================================================


2. MF
	Input
		rank
		algorithm type : base, naive, robust
		
		review
		helpful
		target item id list

	Output
		base_U_clean
		base_V_clean

		base_U_attacked
		base_V_attacked
		------------------------
		naive_U_clean
		naive_V_clean

		naive_U_attacked
		naive_V_attacked
		------------------------
		robust_U_clean
		robust_V_clean

		robust_U_attacked
		robust_V_attacked

