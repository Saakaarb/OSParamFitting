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

        derivatives = np.zeros(y.shape[0])

        # this part is to be populated by the model
        #--------------------------------
        k1 = trainable_parameters['k1']
        k2 = trainable_parameters['k2']
        k3 = trainable_parameters['k3']
        unused_constant = fixed_parameters['unused_constant']

        y1 = y[0]
        y2 = y[1]
        y3 = y[2]

        #--------------------------------

        # this part is to be populated by the user
        #--------------------------------
        dy1dt = -k1*y1 + k3*y3*y2
        dy2dt = k1*y1 - k2*y2**2 - k3*y2*y3
        dy3dt = k2*y2**2

        derivatives = np.array([dy1dt, dy2dt, dy3dt])
        #--------------------------------

        return derivatives

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
        k1 = trainable_parameters['k1']
        k2 = trainable_parameters['k2']
        k3 = trainable_parameters['k3']
        unused_constant = fixed_parameters['unused_constant']
        #--------------------------------


        # this part is to be populated by the user
        #--------------------------------
        scale_factor = np.max(dataset,axis=0)

        loss = np.sqrt(np.mean(np.square(np.divide(solution - dataset,scale_factor))))
        #--------------------------------

        return loss
        

def writeout_description(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # Arguments:
        # solution_time: time (np.ndarray of size [Nts])
        # solution: state vector (np.ndarray of size [Nts,Ny])
        # dataset: dataset (np.ndarray of size [Nts,N_col])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file
        # Change the size of writeout_array as per your requirement

        # this part is to be populated by the model
        #--------------------------------
        k1 = trainable_parameters['k1']
        k2 = trainable_parameters['k2']
        k3 = trainable_parameters['k3']
        unused_constant = fixed_parameters['unused_constant']
        #--------------------------------


        # this part is to be populated by the user
        #--------------------------------
        Nts = solution_time.shape[0]
        writeout_array = np.zeros([Nts,7])
        writeout_array[:,0]=solution_time
        writeout_array[:,1:4] = dataset
        writeout_array[:,4:7] = solution

        #--------------------------------

        return writeout_array
