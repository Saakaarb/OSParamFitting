import numpy as np
from lib.algorithms.NODE.helper_functions import scale_value, unscale_value
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import os
from copy import deepcopy
import time
from pathlib import Path
from lib.utils.xmlread import XMLReader
#os.environ["EQX_ON_ERROR"]="nan"

class FitParamsNODE:

    def __init__(self, input_reader: XMLReader, problem_object, init_guess=None):
        """Initialize the Neural Ordinary Differential Equation (NODE) fitting parameters.

        This method sets up the NODE optimization parameters by:
        1. Storing input reader and problem object references
        2. Processing search bounds and scaling information
        3. Setting up NODE algorithm parameters
        4. Initializing search space boundaries
        5. Handling log-scale transformations if specified

        Parameters
        ----------
        input_reader : object
            Reader object containing optimization parameters:
            - n_search_axes : Number of dimensions in search space
            - axis_logscale : List of booleans indicating log scale for each axis
            - min_axis_values : List of minimum values for each axis
            - max_axis_values : List of maximum values for each axis
            - n_iters_grad : Number of iterations for gradient-based search
            - stepsize_rtol : Relative tolerance for step size
            - stepsize_atol : Absolute tolerance for step size
            - initial_timestep : Initial time step for ODE solver
            - max_steps : Maximum number of steps for ODE solver
            - init_value_lr : Initial learning rate
            - end_value_lr : Final learning rate
            - transition_steps_lr : Number of steps for learning rate transition
            - decay_rate_lr : Learning rate decay rate
        problem_object : object
            Object containing problem-specific information:
            - compute_all_losses : Method to compute loss values for all particles
        init_guess : list, optional
            Initial guess for the NODE parameters, by default None

        Attributes
        ----------
        problem_obj : object
            Reference to the problem object
        input_reader : object
            Reference to the input reader object
        min_search_list : list
            List of minimum values for scaled search space
        max_search_list : list
            List of maximum values for scaled search space
        min_search_axis : list
            List of minimum values for unscaled search space
        max_search_axis : list
            List of maximum values for unscaled search space
        axis_logscale : list
            List of booleans indicating log scale for each axis
        bounds_tuple : tuple
            Tuple of (min_search_list, max_search_list) for NODE bounds
        n_search_axes : int
            Number of dimensions in search space
        n_iters_grad : int
            Number of iterations for gradient-based search
        stepsize_rtol : float
            Relative tolerance for step size
        stepsize_atol : float
            Absolute tolerance for step size
        initial_timestep : float
            Initial time step for ODE solver
        max_steps : int
            Maximum number of steps for ODE solver
        init_value_lr : float
            Initial learning rate
        end_value_lr : float
            Final learning rate
        transition_steps_lr : int
            Number of steps for learning rate transition
        decay_rate_lr : float
            Learning rate decay rate

        Notes
        -----
        - Search space is scaled to [-1, 1] range for each dimension
        - Log-scale axes are transformed using log10 before scaling
        - Learning rate follows a decay schedule from init_value_lr to end_value_lr
        - ODE solver parameters (stepsize_rtol, stepsize_atol) control solution accuracy
        - Maximum steps limit prevents infinite integration

        See Also
        --------
        scale_value : Method to scale values to [-1, 1] range
        unscale_value : Method to unscale values from [-1, 1] range
        """

        self.input_reader = input_reader

        self.optimizer_name = 'adam'

        self.n_iters_grad = self.input_reader.n_iters_grad 
        self.constants = {}
        self.problem_obj = problem_object
        self.init_guess = init_guess
        # stores scaled value
        self.sc_min_search_list = []
        self.sc_max_search_list = []

        # stores numerical value (non scaled)
        self.min_search_axis = []
        self.max_search_axis = []

        self.axis_logscale = self.input_reader.axis_logscale

        for i_axis in range(input_reader.n_search_axes):

            if input_reader.axis_logscale[i_axis]:

                curr_axis_min = np.log10(input_reader.min_axis_values[i_axis])
                curr_axis_max = np.log10(input_reader.max_axis_values[i_axis])

            else:

                curr_axis_min = input_reader.min_axis_values[i_axis]
                curr_axis_max = input_reader.max_axis_values[i_axis]

            self.min_search_axis.append(curr_axis_min)
            self.max_search_axis.append(curr_axis_max)

            sc_min = self.scale_value(curr_axis_min, curr_axis_min, curr_axis_max)
            sc_max = self.scale_value(curr_axis_max, curr_axis_min, curr_axis_max)

            self.sc_min_search_list.append(sc_min)
            self.sc_max_search_list.append(sc_max)

        self.sc_min_search_list = np.array(self.sc_min_search_list)
        self.sc_max_search_list = np.array(self.sc_max_search_list)
        self.n_search_axes = input_reader.n_search_axes

        self.trainable_params = np.zeros(self.n_search_axes)

        # set initial guess

        if init_guess is None:
            raise NotImplementedError("Currently initial guess must be provided")
        else:

            for i_axis in range(self.n_search_axes):

                if self.input_reader.axis_logscale[i_axis]:

                    self.trainable_params[i_axis] = scale_value(
                        np.log10(init_guess[i_axis]),
                        self.min_search_axis[i_axis],
                        self.max_search_axis[i_axis],
                    )

                else:

                    self.trainable_params[i_axis] = scale_value(
                        init_guess[i_axis],
                        self.min_search_axis[i_axis],
                        self.max_search_axis[i_axis],
                    )

        print("Trainable_params:", self.trainable_params)

        if np.any(self.trainable_params > self.sc_max_search_list) or np.any(
            self.trainable_params < self.sc_min_search_list
        ):

            raise ValueError("starting position violates search limits")

        print(
            "Note: framework currently handles multiple trajectories of the same size and upto the same simulation time. Variable length and final time trajectories will be added in the future"
        )

        # self.constants['']

    def scale_value(self, val, min_val, max_val):

        return scale_value(val, min_val, max_val)

    def unscale_value(self, val, min_val, max_val):

        return unscale_value(val, min_val, max_val)

    # must hook into user/LLM defined loss function
    # TODO define order of arguments and static jit args
    # first two arguments are going to be constants and trainable_param
    # TODO construct infra to define N static arguments after

    def compute_loss(self, trainable_params):

        return self.problem_obj._compute_loss(trainable_params)

    def constrain_search_vars(self):

        self.trainable_params = jnp.clip(
            self.trainable_params, self.sc_min_search_list, self.sc_max_search_list
        )

    def train_NODE(self):
        """Train the Neural Ordinary Differential Equation (NODE) model.

        This method performs the gradient-based optimization of the NODE model by:
        1. Initializing the optimization process
        2. Computing initial loss and gradient
        3. Performing gradient descent iterations
        4. Updating learning rate according to schedule
        5. Logging progress to file and console

        Parameters
        ----------
        None

        Attributes
        ----------
        best_pos : numpy.ndarray
            Best position found during optimization
        best_cost : float
            Best cost value found during optimization
        current_pos : numpy.ndarray
            Current position in parameter space
        current_cost : float
            Current cost value
        current_grad : numpy.ndarray
            Current gradient of the cost function
        learning_rate : float
            Current learning rate for gradient descent

        Notes
        -----
        - Uses gradient descent with learning rate decay
        - Learning rate follows schedule: lr = init_value_lr * (decay_rate_lr ** (iter / transition_steps_lr))
        - Progress is printed every iteration
        - Log file format: "iteration, cost, learning_rate, iteration_time"
        - Stores best position and cost in class attributes

        Returns
        -------
        self.best_result: jax.numpy.ndarray
        self.best_loss: float

        See Also
        --------
        compute_losses : Method to compute loss values
        compute_gradients : Method to compute gradients
        """

        iter_write_freq = 1
        self.loss_history = []
        self.best_loss=np.inf
        self.best_result=None
        #TODO add gradient clipping
        print("TODO: Add gradient clipping")
        print("Note: L.R parameters and choice is currently hardcoded")

        

        #self.optimizer = optax.adam(self.learning_rate)
        if self.optimizer_name == 'lbfgs':
            self.optimizer = optax.lbfgs()
        elif self.optimizer_name == 'adam':
            self.learning_rate = optax.exponential_decay(
            init_value=self.input_reader.init_value_lr,
            transition_steps=self.input_reader.transition_steps_lr,
            decay_rate=self.input_reader.decay_rate_lr,
            end_value=self.input_reader.end_value_lr,
            )
            self.optimizer = optax.adam(self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported")

        self.opt_state = self.optimizer.init(self.trainable_params)

        grad_fn=jax.value_and_grad(self.compute_loss, argnums=0)

        # Create NODE log file
        log_path = Path(self.input_reader.output_dir) / "NODE_fitting.log"
        with open(log_path, 'w') as log_file:
            log_file.write(f"Total number of NODE iterations: {self.n_iters_grad}\n")
            log_file.write("-" * 50 + "\n\n")

        t1 = time.time()

        for i_iter in range(self.n_iters_grad):
            
            # Check for stop flag
            stop_flag = Path(self.input_reader.output_dir) / "stop_fitting.flag"
            if stop_flag.exists():
                print("Stop flag detected, stopping NODE optimization")
                break

            #try:
            value, grad_loss = grad_fn(
                    self.trainable_params
                )
            #except eqx.EquinoxRuntimeError as e:
            if value==self.input_reader.error_loss:
                print("Stopping G.D iterations, exiting with previous best solution, failed at iter:",i_iter)
                return self.best_result,self.best_loss
                
            self.loss_history.append(value)

            if value < self.best_loss:

                self.best_loss=value
                self.best_result=self.trainable_params
            if i_iter % iter_write_freq == 0 or i_iter == self.n_iters_grad - 1:

                print("Iteration:,loss value, best loss:", i_iter, value,self.best_loss)
                if i_iter != 0:
                    t2 = time.time()
                    iteration_time = t2 - t1
                    print("Iteration time:", iteration_time)
                    # Write to log file
                    with open(log_path, 'a') as f:
                        f.write(f"{i_iter}, {self.best_loss:.4E}, {iteration_time/iter_write_freq:.4f}\n")
                        f.flush()
                    t1=t2

            if self.optimizer_name == 'adam':
                updates, self.opt_state =self.optimizer.update(grad_loss, self.opt_state)
            elif self.optimizer_name == 'lbfgs':
                #updates, self.opt_state = self.optimizer.update(grad_loss, self.opt_state)
                updates, self.opt_state = self.optimizer.update(grad_loss, self.opt_state,self.trainable_params,value=value,grad=grad_loss,value_fn=self.compute_loss) #self.optimizer.update(grad_loss, self.opt_state)
            results = optax.apply_updates(self.trainable_params, updates)
            self.trainable_params = results
            self.constrain_search_vars()

        
        return self.best_result,self.best_loss
