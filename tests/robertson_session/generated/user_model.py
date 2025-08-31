# Ordering of parameters in trainable_parameters as provided by the user:
# ['k1', 'k2', 'k3']
# Ordering of integrated variables as provided by the user:
# ['y1', 'y2', 'y3']

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
        k1 = trainable_parameters['k1']
        k2 = trainable_parameters['k2']
        k3 = trainable_parameters['k3']
        y1 = y[0]
        y2 = y[1]
        y3 = y[2]
        dy1dt = -k1*y1 + k3*y3*y2
        dy2dt = k1*y1 - k2*y2**2 - k3*y2*y3
        dy3dt = k2*y2**2
        return np.array([dy1dt, dy2dt, dy3dt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
        scale_factor = np.max(dataset,axis=0)
        return np.sqrt(np.mean(np.square(np.divide(solution - dataset,scale_factor))))

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    # Change the size of writeout_array as per your requirement
    Nts = solution_time.shape[0]
    writeout_array = np.zeros([Nts,7])
    writeout_array[:,0]=solution_time
    writeout_array[:,1:4] = dataset
    writeout_array[:,4:7] = solution
    return writeout_array