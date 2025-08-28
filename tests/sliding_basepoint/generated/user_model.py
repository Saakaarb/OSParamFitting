# Ordering of parameters in trainable_parameters as provided by the user:
# ['c2', 'Dk', 'Dc']
# Ordering of integrated variables as provided by the user:
# ['x1', 'x2', 'v1', 'v2', 'k', 'c1']

def user_defined_system(t, y, trainable_parameters, fixed_parameters):
        # Trainable parameters
        c2 = trainable_parameters['c2']
        Dk = trainable_parameters['Dk']
        Dc = trainable_parameters['Dc']
        # Fixed parameters
        m1 = fixed_parameters['m1']
        m2 = fixed_parameters['m2']
        vf = fixed_parameters['vf']
        # Integrable variables (in order as provided by user)
        x1 = y[0]  # x1 is at index 0
        x2 = y[1]  # x2 is at index 1
        v1 = y[2]  # v1 is at index 2
        v2 = y[3]  # v2 is at index 3
        k = y[4]  # k is at index 4
        c1 = y[5]  # c1 is at index 5
    
        def F1(Fs,c1,v1):
    		return Fs-c1*np.abs(v1)*np.sign(v1)
    	def F2(Fs,c2,v2):
    
    		if np.abs(Fs) < c2 and np.abs(v2) < vf:
    			return 0
    		else:
    			return Fs - c2*np.sign(v2)
        Fs= k*(x2-x1)
        
        dx1_dt = v1
        dx2_dt = v2
        dv1_dt = (1/m1)*F1(Fs,c1,v1)
        dv2_dt = (1/m2)*F2(-1*Fs,c2,v2)
        P = np.abs(m1*v1*dv1_dt)
        dk_dt = Dk*P
        dc1_dt = Dc*P
    
        # ... your code here ...
    
        return [dx1_dt, dx2_dt, dv1_dt, dv2_dt, dk_dt, dc1_dt]

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
        # Trainable parameters
        c2 = trainable_parameters['c2']
        Dk = trainable_parameters['Dk']
        Dc = trainable_parameters['Dc']
        # Fixed parameters
        m1 = fixed_parameters['m1']
        m2 = fixed_parameters['m2']
        vf = fixed_parameters['vf']
    
        # ... your code here ...
    	max_cols=np.max(np.abs(dataset[:,2:4]),axis=0)
    	loss = np.sqrt(np.mean(np.square(np.divide(solution[:,2:4]-dataset[:,2:4],max_cols))))
        
        return loss

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

    c2 = trainable_parameters[&#39;c2&#39;]
    Dk = trainable_parameters[&#39;Dk&#39;]
    Dc = trainable_parameters[&#39;Dc&#39;]
    m1 = fixed_parameters[&#39;m1&#39;]
    m2 = fixed_parameters[&#39;m2&#39;]
    vf = fixed_parameters[&#39;vf&#39;]
    x1 = solution[:, 0]  # x1 is at column 0
    x2 = solution[:, 1]  # x2 is at column 1
    v1 = solution[:, 2]  # v1 is at column 2
    v2 = solution[:, 3]  # v2 is at column 3
    k = solution[:, 4]  # k is at column 4
    c1 = solution[:, 5]  # c1 is at column 5


    # Change the size of writeout_array as per your requirement
    Nts = solution_time.shape[0]
    writeout_array = np.zeros([Nts, 9])

    writeout_array[:,0]=solution_time
	writeout_array[:,1:5]=dataset
	writeout_array[:,5:9]=solution[:,:4]
# ... your code here ...

    return writeout_array