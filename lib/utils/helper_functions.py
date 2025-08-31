import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from lib.utils.classes import ProblemObjectBase
from functools import partial
from lib.algorithms.PSO.classes import FitParamsPSO
from lib.algorithms.NODE.classes import FitParamsNODE
import xml.etree.ElementTree as ET
from lib.utils.xmlread import XMLReader
from pathlib import Path
import importlib.util
import sys

jax.config.update("jax_enable_x64", True)

class CreatedClass(ProblemObjectBase):
    def __init__(self, dataset: np.ndarray, t_eval: np.ndarray, y0: jnp.ndarray, input_reader: XMLReader, compute_loss_problem, write_problem_result):
        """
        Initialize the CreatedClass instance with problem configuration.

        Args:
            dataset (numpy.ndarray): Experimental data array with shape (time_steps, variables)
            t_eval (numpy.ndarray): Time points for solution evaluation
            y0 (jax.numpy.ndarray): Initial conditions for the ODE system
            input_reader (XMLReader): Configuration reader containing all problem parameters
            compute_loss_problem (callable): Function to compute loss for given parameters
            write_problem_result (callable): Function to write problem results

        The constructor sets up the complete problem environment including:
        - Parameter management (trainable vs fixed)
        - Integration settings (tolerance, step size, max steps)
        - Time domain configuration
        - Loss computation and result writing functions
        """
        super().__init__()
        self.y0 = y0
        self.input_reader = input_reader
        self.t_eval = t_eval
        self.dataset = np.array(dataset)
        self.num_columns_to_fit = self.dataset.shape[1]
        self.params_to_fit_names = self.input_reader.trainable_parameter_names
        self.fixed_param_names = self.input_reader.fixed_parameter_names
        self.fixed_param_values = self.input_reader.fixed_parameter_values
        self.fixed_val_dict = {}
        for i in range(len(self.input_reader.fixed_parameter_names)):
            self.fixed_val_dict[self.fixed_param_names[i]] = self.fixed_param_values[i]
        self.constants = {
            "dataset": jnp.array(self.dataset),
            "t_eval": t_eval,
            "init_cond": y0,
        }
        print("NOTE: currently, only simulations till the same final time are supported")
        self.constants["num_steps"] = self.dataset.shape[0]
        if self.input_reader.init_time is None:
            self.constants["init_time"] = self.t_eval[0]
        else:
            self.constants["init_time"] = self.input_reader.init_time
        self.constants["final_time"] = self.t_eval[-1]
        self.constants['stepsize_rtol'] = np.array(self.input_reader.stepsize_rtol)
        self.constants['stepsize_atol'] = np.array(self.input_reader.stepsize_atol)
        self.constants['init_timestep'] = self.input_reader.init_timestep
        self.constants['max_steps'] = self.input_reader.max_steps
        self.constants['fixed_parameters'] = self.fixed_val_dict
        self.constants['error_loss'] = self.input_reader.error_loss
        self._compute_loss_problem = compute_loss_problem
        self._write_problem_result = write_problem_result

    def _compute_all_losses(self, population: np.ndarray)-> np.ndarray:
        """
        Compute losses for an entire population of parameter sets.

        Args:
            population (numpy.ndarray): Array of parameter sets, shape (n_individuals, n_parameters)

        Returns:
            numpy.ndarray: Array of loss values for each parameter set, with NaN/Inf values
                        replaced by 1e10 to prevent optimization issues

        This method iterates through each parameter set in the population and computes
        the corresponding loss using the JIT-compiled _compute_loss method.
        """
        #losses = []

        # get number of visible devices
        n_devices_total = len(jax.devices("cpu"))

        all_devices = jax.devices("cpu")

        user_selected_n_devices=self.input_reader.processors

        if user_selected_n_devices > n_devices_total:
            print(f"User selected {user_selected_n_devices} devices, but only {n_devices_total} are available. The code is constrained to see a maximum of 8 devices. Using all available devices.")
            print("The hardcoded upper limit of 8 can be changed in fit_parameters.py")

        used_devices = all_devices[:min(n_devices_total, user_selected_n_devices)]

        n_devices = len(used_devices)

        n_particles = population.shape[0]
        # define some array splitting logic
        local_n_particles = np.ceil(n_particles / n_devices).astype(int)
        pad = local_n_particles * n_devices - n_particles
        # pad to make equal-sized shards per device
        if pad:
            population = jnp.pad(population, ((0, pad), (0, 0)))

        # reshape to [n_dev, local_n, D] for pmap
        population_sharded = population.reshape(n_devices, local_n_particles, *population.shape[1:])

        # define a shard loss function
        @jax.jit
        def shard_loss(shard_population: jax.Array)-> jax.Array:
            losses = jnp.zeros(local_n_particles)
            for i in range(shard_population.shape[0]):
                loss = self._compute_loss(shard_population[i])
                
                # JAX-compatible way to handle NaN/Inf values
                loss = jnp.where(jnp.logical_or(jnp.isnan(loss), jnp.isinf(loss)), 1e10, loss)
                losses = losses.at[i].set(loss)
            return losses

        # create a pmap with the number of devices
        # pmap over devices; broadcast data (in_axes=None)
        p_shard_loss = jax.pmap(shard_loss, in_axes=(0),devices=used_devices)
        shard_losses = p_shard_loss(population_sharded)   

        # flatten back and drop padding
        losses = shard_losses.reshape(-1)[:n_particles]


        # TODO replace with jax pmap
        #losses = jax.pmap(self._compute_loss, in_axes=(0,))(population)
        #losses = jax.pmap(self._compute_loss, in_axes=(0,))(population)
        # for i in range(population.shape[0]):
        #     loss = self._compute_loss(population[i])
            
        #     if np.isnan(loss) or np.isinf(loss):
        #         loss=1e10
        #     losses.append(loss)
        
        return np.array(losses)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_loss(self, design_pt: np.ndarray)-> float:
        """
        Compute loss for a single parameter set using JIT compilation.

        Args:
            design_pt (jax.numpy.ndarray): Single parameter set to evaluate

        Returns:
            float: Computed loss value for the given parameters

        This method is JIT-compiled for performance and calls the user-defined
        loss computation function with the problem constants and parameters.
        """
        return self._compute_loss_problem(self.constants, jnp.array(design_pt))

    def set_min_limit(self, min_lim: list[float])-> None:
        """
        Set minimum bounds for parameter search space.

        Args:
            min_lim (numpy.ndarray): Array of minimum values for each parameter

        The limits are stored in the constants dictionary and used by the
        optimization algorithms to constrain the search space.
        """
        self.constants["min_limits"] = jnp.array(min_lim)

    def set_max_limit(self, max_lim: list[float])-> None:
        """
        Set maximum bounds for parameter search space.

        Args:
            max_lim (numpy.ndarray): Array of maximum values for each parameter

        The limits are stored in the constants dictionary and used by the
        optimization algorithms to constrain the search space.
        """
        self.constants["max_limits"] = jnp.array(max_lim)

    def set_is_logscale(self, is_logscale: list[bool])-> None:
        """
        Set log-scale flag for parameter axes.

        Args:
            is_logscale (numpy.ndarray): Boolean array indicating which parameters
                                        should use log-scale transformation

        This affects how the optimization algorithms handle parameter scaling
        and search space exploration.
        """
        self.constants["is_logscale"] = jnp.array(is_logscale)

    def write_problem_result(self, design_point: np.ndarray, input_reader: XMLReader, label:str="default")-> None:
        """
        Write problem solution results to CSV files.

        Args:
            design_point (numpy.ndarray): Parameter set that produced the solution
            input_reader (XMLReader): Configuration reader containing output directory info
            label (str, optional): Label for the output file. Defaults to "default"

        The method calls the user-defined result writing function and saves the
        output to a CSV file in the format "{label}_solution.csv".
        """
        writeout_array = self._write_problem_result(self.constants, jnp.array(design_point))
        np.savetxt(input_reader.output_dir/Path(f"{label}_solution.csv"), writeout_array, delimiter=",")
      


def get_input_reader(path_to_input: Path)-> XMLReader:
    """
    Parse XML input file and create an XMLReader instance.

    Args:
        path_to_input (Path): Path to the XML configuration file

    Returns:
        XMLReader: Configured reader instance containing all problem parameters

    This function parses the XML file using ElementTree and initializes
    an XMLReader object with the parsed configuration data.
    """
    tree = ET.parse(path_to_input)
    root = tree.getroot()

    input_reader=XMLReader()
    input_reader.read_XML(root)

    return input_reader


def fit_generic_system(path_to_input: Path, path_to_output_dir: Path, generated_dir: Path,session_path: Path)-> np.ndarray:
    """Fit a generic system using a two-phase optimization approach.

    This function performs parameter fitting using a combination of:
    1. Population-based optimization (PSO) for global search
    2. Gradient-based optimization (NODE) for local refinement

    The process includes:
    1. Reading input parameters from XML
    2. Running PSO to find initial parameter estimates
    3. Using PSO results as initial guess for NODE
    4. Running NODE to refine the parameters
    5. Writing results to output directory

    Parameters
    ----------
    path_to_input : str or Path
        Path to the input XML file containing optimization parameters
    path_to_output_dir : str or Path
        Directory where output files will be written
    generated_dir : str or Path
        Directory containing generated files (user_model.py, etc.)

    Notes
    -----
    - PSO is used first to explore the parameter space globally
    - NODE uses the best PSO result as its initial guess
    - Results are written to the output directory:
        - final_design_point.csv : Best parameters found
        - result_solution.csv : Solution trajectory
        - fitting_error.txt : Error messages if any
    - Progress is logged to the output directory

    Returns
    -------
    numpy.ndarray: Best parameter set found during optimization

    See Also
    --------
    FitParamsPSO : Class for PSO optimization parameters
    FitParamsNODE : Class for NODE optimization parameters
    """
    try:
        # Dynamically import generated_script.py from the session's generated_dir
        generated_script_path = Path(generated_dir) / "generated_script.py"
        spec = importlib.util.spec_from_file_location("generated_script", generated_script_path)
        generated_script = importlib.util.module_from_spec(spec)
        sys.modules["generated_script"] = generated_script
        spec.loader.exec_module(generated_script)

        input_reader=get_input_reader(path_to_input)

        # variable and parameter name uniqueness 
        input_reader.check_name_uniqueness()

        # assign output dir
        input_reader.output_dir=path_to_output_dir

        y0=jnp.array(input_reader.integrated_variable_init_values)

        # load dataset
        dataset_path = session_path / Path(input_reader.user_input_dirname) / Path(input_reader.filename_data)
        with open(dataset_path, 'r', encoding='utf-8-sig') as f:
            all_data=np.genfromtxt(f, dtype=float, delimiter=',')
        # split into time and data
        t_eval=all_data[:,0]
        dataset=all_data[:,1:]
        # Run a test fitting
        problem_obj = CreatedClass(dataset=dataset, t_eval=t_eval, y0=y0, input_reader=input_reader,
                                  compute_loss_problem=generated_script._compute_loss_problem,
                                  write_problem_result=generated_script._write_problem_result)

        final_ans = fit_equation_system(input_reader, y0, t_eval, dataset, problem_obj)

        return final_ans
        
    except Exception as e:
        error_message = f"Fitting process failed with error: {str(e)}"
        print(error_message)
        
        # Write error to file
        error_file = Path(path_to_output_dir) / "fitting_error.txt"
        with open(error_file, 'w') as f:
            f.write(error_message)
        
        # Re-raise the exception to ensure the process exits with error code
        raise e


# fixed
def fit_equation_system(input_reader: XMLReader, y0: jnp.ndarray, t_eval: np.ndarray, dataset: np.ndarray, problem_obj: CreatedClass)-> np.ndarray:
    """
    Fit a system of equations using a two-phase optimization approach.

    This function performs parameter fitting for a system of equations using:
    1. Population-based optimization (PSO) for global search
    2. Gradient-based optimization (NODE) for local refinement

    The process includes:
    1. Running PSO to find initial parameter estimates
    2. Using PSO results as initial guess for NODE
    3. Running NODE to refine the parameters
    4. Writing results to output directory

    Args:
        input_reader (XMLReader): Reader object containing optimization parameters from XML
        y0 (jax.numpy.ndarray): Initial conditions for the system of equations
        t_eval (numpy.ndarray): Time points at which to evaluate the solution
        dataset (numpy.ndarray): Experimental data to fit against
        problem_obj (CreatedClass): Problem object containing loss computation and result writing methods

    Returns:
        numpy.ndarray: Best parameter set found during optimization

    Notes:
        - PSO is used first to explore the parameter space globally
        - NODE uses the best PSO result as its initial guess
        - Results are written to the output directory specified in input_reader
        - Progress is logged to the output directory
        - Final parameters are saved to final_design_point.csv
    """
    ## create fit object code
    # make sure this can be any algorithm
    print(
        "TODO replace with a function that takes the algorithm name as a string input"
    )

    fit_obj_PSO = FitParamsPSO(input_reader, problem_obj)

    # save axis limit info in problem_obj. The limits have already
    # been scaled depending on axis_logscale in fit_obj_PSO
    problem_obj.set_min_limit(fit_obj_PSO.min_search_axis)
    problem_obj.set_max_limit(fit_obj_PSO.max_search_axis)
    problem_obj.set_is_logscale(input_reader.axis_logscale)

    print("Writing pso log file")
    log_path = Path(input_reader.output_dir) / "pso_fitting.log"
    with open(log_path, 'w') as log_file:
        log_file.write(f"Total number of PSO iterations: {input_reader.n_iters_pop}\n")
        log_file.write("-" * 50 + "\n\n")

    best_position,best_cost = optimize_function(fit_obj_PSO, input_reader, log_path)

    unscaled_best_position = fit_obj_PSO.unscale_design_point(best_position)
    print("Best Position from PSO:", unscaled_best_position)
    print("Best cost from PSO:",best_cost)

    #problem_obj.plot_problem_result(best_position, label="PSO")

    # fine tuning using NODE

    fit_obj_NODE = FitParamsNODE(
        input_reader, problem_obj, init_guess=unscaled_best_position
    )

    try:
        tuned_best_position, tuned_best_loss = fit_obj_NODE.train_NODE()
    except Exception as e:
        print(f"Error in NODE training: {e}, stopping")
        tuned_best_position = best_position
        tuned_best_loss = 1e10 #problem_obj._compute_loss(unscaled_best_position)

    print("Tuned position from NODE(scaled):", tuned_best_position)
    print("Tuned best loss:",tuned_best_loss)

    unscaled_best_position_tuned = fit_obj_PSO.unscale_design_point(
        np.array(tuned_best_position)
    )
    print("Final best position:", unscaled_best_position_tuned)

    if input_reader.write_results:
        problem_obj.write_problem_result(tuned_best_position, input_reader,label="result")

    # save design point to file
    unscaled_best_position_tuned_np=np.array(unscaled_best_position_tuned)
    #TODO edit this to save the number obtained with the name of each parameter
    np.savetxt(input_reader.output_dir/Path(f"final_design_point.csv"),unscaled_best_position_tuned_np,delimiter=",")

    return unscaled_best_position_tuned

def optimize_function(fit_obj: FitParamsPSO, input_reader: XMLReader, file_obj: Path)-> tuple[np.ndarray, float]:
    """
    Execute PSO optimization iterations with logging and error handling.

    This function runs the PSO optimization algorithm for the specified number
    of iterations, logging progress and handling any errors that occur during
    the optimization process.

    Args:
        fit_obj (FitParamsPSO): PSO optimization object configured with problem parameters
        input_reader (XMLReader): Configuration reader containing iteration count and output directory
        file_obj (Path): Path to the log file for writing optimization progress

    Returns:
        tuple: (best_position, best_cost) - Best parameter set found and its corresponding cost

    Raises:
        Exception: If optimization fails, with error details written to fitting_error.txt

    Notes:
        - The function checks for a stop_fitting.flag file to allow early termination
        - Progress is logged to the specified log file
        - Errors are captured and written to fitting_error.txt in the output directory
        - If optimization fails, the current best position is returned if available
    """  
    print("Starting Search Iterations")
    # Iterations
    try:
        for iter_no in range(input_reader.n_iters_pop):
            
            # Check for stop flag
            stop_flag = Path(input_reader.output_dir) / "stop_fitting.flag"
            if stop_flag.exists():
                print("Stop flag detected, stopping PSO optimization")
                break

            fit_obj.search_iteration(iter_no, file_obj)
    except Exception as e:
        error_message = f"PSO optimization failed at iteration {iter_no}: {str(e)}"
        print(error_message)
        
        # Write error to file
        error_file = Path(input_reader.output_dir) / "fitting_error.txt"
        with open(error_file, 'w') as f:
            f.write(error_message)
        
        # Return current best position if available, otherwise raise
        if hasattr(fit_obj, 'best_pos') and fit_obj.best_pos is not None:
            return fit_obj.best_pos, fit_obj.swarm_obj.best_cost
        else:
            raise e

    return fit_obj.best_pos,fit_obj.swarm_obj.best_cost


# class CreatedClass(ProblemObjectBase):
#     def __init__(self, dataset, t_eval, y0, input_reader, compute_loss_problem, write_problem_result):
#         super().__init__()
#         self.y0 = y0
#         self.input_reader = input_reader
#         self.t_eval = t_eval
#         self.dataset = np.array(dataset)
#         self.num_columns_to_fit = self.dataset.shape[1]
#         self.params_to_fit_names = self.input_reader.trainable_parameter_names
#         self.fixed_param_names = self.input_reader.fixed_parameter_names
#         self.fixed_param_values = self.input_reader.fixed_parameter_values
#         self.fixed_val_dict = {}
#         for i in range(len(self.input_reader.fixed_parameter_names)):
#             self.fixed_val_dict[self.fixed_param_names[i]] = self.fixed_param_values[i]
#         self.constants = {
#             "dataset": jnp.array(self.dataset),
#             "t_eval": t_eval,
#             "init_cond": y0,
#         }
#         print("NOTE: currently, only simulations till the same final time are supported")
#         self.constants["num_steps"] = self.dataset.shape[0]
#         if self.input_reader.init_time is None:
#             self.constants["init_time"] = self.t_eval[0]
#         else:
#             self.constants["init_time"] = self.input_reader.init_time
#         self.constants["final_time"] = self.t_eval[-1]
#         self.constants['stepsize_rtol'] = np.array(self.input_reader.stepsize_rtol)
#         self.constants['stepsize_atol'] = np.array(self.input_reader.stepsize_atol)
#         self.constants['init_timestep'] = self.input_reader.init_timestep
#         self.constants['max_steps'] = self.input_reader.max_steps
#         self.constants['fixed_parameters'] = self.fixed_val_dict
#         self.constants['error_loss'] = self.input_reader.error_loss
#         self._compute_loss_problem = compute_loss_problem
#         self._write_problem_result = write_problem_result

#     def _compute_all_losses(self, population):
#         losses = []
#         for i in range(population.shape[0]):
#             loss = self._compute_loss(population[i])
            
#             if np.isnan(loss) or np.isinf(loss):
#                 loss=1e10
#             losses.append(loss)
#         return np.array(losses)

#     @partial(jax.jit, static_argnums=(0,))
#     def _compute_loss(self, design_pt):
#         return self._compute_loss_problem(self.constants, jnp.array(design_pt))

#     def set_min_limit(self, min_lim):
#         self.constants["min_limits"] = jnp.array(min_lim)

#     def set_max_limit(self, max_lim):
#         self.constants["max_limits"] = jnp.array(max_lim)

#     def set_is_logscale(self, is_logscale):
#         self.constants["is_logscale"] = jnp.array(is_logscale)

#     def write_problem_result(self, design_point, input_reader, label="default"):
#         writeout_array = self._write_problem_result(self.constants, jnp.array(design_point))
#         np.savetxt(input_reader.output_dir/Path(f"{label}_solution.csv"), writeout_array, delimiter=",")
        
        