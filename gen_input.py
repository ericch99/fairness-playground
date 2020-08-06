def generate_input():

    A = {'mean': 1, 
    	 'var': 1, 
    	 'prob': 0.3}
    
    B = {'mean': 0, 
    	 'var': 1, 
    	 'prob': 0.7}
    
    sim = {'metric': 'avg-position', 
    	   'dist': 'logit-normal', 
    	   'k': 10, 
           'r_policy': 'top-k', 
           's_policy': 'stochastic', 
           'query_len': 20}

    penalty = {'pos': 0.5, 
    		   'neg': -0.25, 
    		   'non': 0}

    return A, B, sim, penalty