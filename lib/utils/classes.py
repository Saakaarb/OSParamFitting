import numpy as np
import xml.etree.ElementTree as ET
from lib.utils.xmlread import XMLReader
from pathlib import Path
from functools import partial
import jax
import jax.numpy as jnp
# Base class for any problem defined


class ProblemObjectBase:
    """
    Abstract base class for defining optimization problems in the OSParamFitting framework.
    
    This class provides a common interface for all problem types, including ODE systems,
    parameter estimation problems, and other optimization tasks. It defines the basic
    structure and methods that concrete problem implementations must provide.
    
    The base class handles common attributes like parameter names, initial conditions,
    time evaluation points, and dataset information. Concrete implementations should
    override the abstract methods to provide specific problem-solving logic.
    
    Attributes:
        params_to_fit_names (dict): Dictionary mapping parameter names to their indices
        y0 (numpy.ndarray, optional): Initial conditions for unsteady problems
        t_eval (numpy.ndarray, optional): Time points for solution evaluation
        dataset (numpy.ndarray, optional): Experimental or reference data to fit against
        num_columns_to_fit (int, optional): Number of data columns to fit
        fixed_params_names (list, optional): Names of parameters that are held constant
        fixed_param_values (list, optional): Values of the fixed parameters
    """
    
    def __init__(self):
        """
        Initialize the ProblemObjectBase with default attribute values.
        
        Sets up the basic structure for problem objects with empty or None values
        that should be populated by concrete implementations or external configuration.
        """
        self.params_to_fit_names = {}

        # unsteady problems
        self.y0 = None

        self.t_eval = None
        self.dataset = None

        self.num_columns_to_fit=None
        self.params_to_fit_names=None
        self.fixed_params_names=None

        self.fixed_param_values=None

    def integrate_system(self, *args):
        """
        Integrate the system for given parameters and return the solution.
        
        This is a public interface method that delegates to the concrete
        implementation's _integrate_system method.
        
        Args:
            *args: Variable arguments passed to the concrete integration method
            
        Returns:
            The result of the system integration (typically solution trajectories)
            
        Note:
            Concrete implementations must override _integrate_system to provide
            actual integration logic.
        """
        return self._integrate_system(*args)

    #@partial(jax.jit,static_argnums=(0,))
    def compute_loss(self, *args):
        """
        Compute the loss/objective function value for given parameters.
        
        This is a public interface method that delegates to the concrete
        implementation's _compute_loss method. The loss function quantifies
        how well the model fits the experimental data.
        
        Args:
            *args: Variable arguments (typically parameter values) passed to the loss computation
            
        Returns:
            float: The computed loss value (lower values indicate better fits)
            
        """
        return self._compute_loss(*args)

    def compute_all_losses(self, population_points: np.ndarray):
        """
        Compute loss values for an entire population of parameter sets.
        
        This method evaluates the loss function for multiple parameter combinations,
        which is useful for population-based optimization algorithms like PSO.
        
        Args:
            population_points (numpy.ndarray): Array of parameter sets with shape
                                            (n_individuals, n_parameters)
            
        Returns:
            numpy.ndarray: Array of loss values for each parameter set
            
        Note:
            Concrete implementations must override _compute_all_losses to provide
            efficient batch loss computation logic.
        """
        return self._compute_all_losses(population_points)

    def plot_result(self, design_point: np.ndarray, label: str="default"):
        """
        Plot the result for a given parameter set.
        
        This method provides visualization capabilities for the problem results,
        allowing users to inspect the quality of fits and parameter estimates.
        
        Args:
            design_point (numpy.ndarray): Parameter set to visualize
            label (str, optional): Label for the plot. Defaults to "default"
            
        Returns:
            The result of the plotting operation (typically matplotlib figure or None)
            
        Note:
            Concrete implementations must override _plot_result to provide
            actual plotting logic. This method is useful for debugging and
            result analysis.
        """
        return self._plot_result(design_point, label)



