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

def get_input_reader(path_to_input):
    tree = ET.parse(path_to_input)
    root = tree.getroot()

    input_reader=XMLReader()
    input_reader.read_XML(root)

    return input_reader


def fit_generic_system(path_to_input, path_to_output_dir, generated_dir):
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
    None

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

        tree = ET.parse(path_to_input)
        root = tree.getroot()

        input_reader=XMLReader()
        input_reader.read_XML(root)
        # variable and parameter name uniqueness 
        input_reader.check_name_uniqueness()

        # assign output dir
        input_reader.output_dir=path_to_output_dir

        y0=jnp.array(input_reader.integrated_variable_init_values)

        # load dataset
        dataset_path = Path("sessions") / Path(input_reader.input_dirname) / Path(input_reader.filename_data)
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
def fit_equation_system(input_reader, y0, t_eval, dataset, problem_obj):
    """Fit a system of equations using a two-phase optimization approach.

    This function performs parameter fitting for a system of equations using:
    1. Population-based optimization (PSO) for global search
    2. Gradient-based optimization (NODE) for local refinement

    The process includes:
    1. Running PSO to find initial parameter estimates
    2. Using PSO results as initial guess for NODE
    3. Running NODE to refine the parameters
    4. Writing results to output directory

    Parameters
    ----------
    input_reader : XMLReader
        Reader object containing optimization parameters from XML
    y0 : jax.numpy.ndarray
        Initial conditions for the system of equations
    t_eval : numpy.ndarray
        Time points at which to evaluate the solution
    dataset : numpy.ndarray
        Experimental data to fit against
    problem_obj : CreatedClass
        Problem object containing loss computation and result writing methods

    Notes
    -----
    - PSO is used first to explore the parameter space globally
    - NODE uses the best PSO result as its initial guess
    - Results are written to the output directory specified in input_reader
    - Progress is logged to the output directory

    Returns
    -------
    best_position: numpy.ndarray
        Best position found during optimization
    

    See Also
    --------
    FitParamsPSO : Class for PSO optimization parameters
    FitParamsNODE : Class for NODE optimization parameters
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

def optimize_function(fit_obj, input_reader, file_obj):
    
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


class CreatedClass(ProblemObjectBase):
    def __init__(self, dataset, t_eval, y0, input_reader, compute_loss_problem, write_problem_result):
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

    def _compute_all_losses(self, population):
        losses = []
        for i in range(population.shape[0]):
            loss = self._compute_loss(population[i])
            
            if np.isnan(loss) or np.isinf(loss):
                loss=1e10
            losses.append(loss)
        return np.array(losses)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_loss(self, design_pt):
        return self._compute_loss_problem(self.constants, jnp.array(design_pt))

    def set_min_limit(self, min_lim):
        self.constants["min_limits"] = jnp.array(min_lim)

    def set_max_limit(self, max_lim):
        self.constants["max_limits"] = jnp.array(max_lim)

    def set_is_logscale(self, is_logscale):
        self.constants["is_logscale"] = jnp.array(is_logscale)

    def write_problem_result(self, design_point, input_reader, label="default"):
        writeout_array = self._write_problem_result(self.constants, jnp.array(design_point))
        np.savetxt(input_reader.output_dir/Path(f"{label}_solution.csv"), writeout_array, delimiter=",")
        
        