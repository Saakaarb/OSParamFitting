# Ordering of parameters in trainable_parameters as provided by the user:
# ['mu']
# Ordering of integrated variables as provided by the user:
# ['x1', 'x2']

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset):
        # Trainable parameters
        mu = trainable_parameters['mu']
        # Fixed parameters
        # Integrable variables (in order as provided by user)
        x1 = y[0]  # x1 is at index 0
        x2 = y[1]  # x2 is at index 1
    
        dx1_dt = mu*(x2-((1.0/3.0)*x1**3-x1))
    	dx2_dt = -1.0/mu*x1
    
        # ... your code here ...
    
        return [dx1_dt, dx2_dt]

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
        # Trainable parameters
        mu = trainable_parameters['mu']
        # Fixed parameters
    
        max_col=np.max(np.abs(solution),axis=0)
    	loss = np.sqrt(np.mean(np.square(np.divide(solution-dataset,max_col))))
    
        return loss

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

    mu = trainable_parameters['mu']
    x1 = solution[:, 0]  # x1 is at column 0
    x2 = solution[:, 1]  # x2 is at column 1


    # Change the size of writeout_array as per your requirement
    Nts = solution_time.shape[0]
    writeout_array = np.zeros([Nts, 2])

# ... your code here ...

    return writeout_array