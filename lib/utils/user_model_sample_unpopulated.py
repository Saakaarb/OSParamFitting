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


        #--------------------------------

        return writeout_array
