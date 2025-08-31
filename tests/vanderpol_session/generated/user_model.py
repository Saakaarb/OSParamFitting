import numpy as np

# Key: 
# Nts: number of time steps in dataset
# Ny: number of state variables defined in the user_input.xml file
# N_col: number of columns in dataset (including time) 

def user_defined_system(t: float, y: np.ndarray, trainable_parameters: dict, fixed_parameters: dict, dataset: np.ndarray, t_eval: np.ndarray):

        # Arguments:
        # t: time (float)
        # y: state vector (np.ndarray of size [Ny])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file
        # dataset: dataset (np.ndarray of size [Nts,N_col-1]) (first dimension in dataset is time)
        # t_eval: evaluation times (np.ndarray of size [Nts])

        

        # this part is to be populated by the model
        #--------------------------------
        mu = trainable_parameters['mu']

        x1 = y[0]
        x2 = y[1]

        #--------------------------------

        # this part is to be populated by the user
        #--------------------------------
        dx1dt = mu*(x2-((1.0/3.0)*x1**3-x1))
        dx2dt = -1.0/mu*x1
        #--------------------------------
        derivatives = np.array([dx1dt, dx2dt]) # 
        return derivatives # of shape [Ny]. Each derivative term must be user defined

def _compute_loss_problem(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # Arguments:
        # solution_time: time (np.ndarray of size [Nts])
        # solution: state vector (np.ndarray of size [Nts,Ny])
        # dataset: dataset (np.ndarray of size [Nts,N_col])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file

        loss=0.0

        # this part is to be populated by the model
        #--------------------------------
        mu = trainable_parameters['mu']
        #--------------------------------


        # this part is to be populated by the user
        #--------------------------------
        max_col=np.max(np.abs(dataset),axis=0)
        loss = np.sqrt(np.mean(np.square(np.divide(solution-dataset,max_col))))
        #--------------------------------

        return loss # scalar
        

def write_problem_result(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # Arguments:
        # solution_time: time (np.ndarray of size [Nts])
        # solution: state vector (np.ndarray of size [Nts,Ny])
        # dataset: dataset (np.ndarray of size [Nts,N_col])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file
        # Change the size of writeout_array as per your requirement

        # this part is to be populated by the model
        #--------------------------------
        mu = trainable_parameters['mu']
        #--------------------------------

        writeout_array = np.zeros([solution_time.shape[0],5])

        # this part is to be populated by the user
        #--------------------------------
        writeout_array[:,0]=solution_time
        writeout_array[:,1:3]=dataset
        writeout_array[:,3:5]=solution

        #--------------------------------

        return writeout_array # of custom shape
