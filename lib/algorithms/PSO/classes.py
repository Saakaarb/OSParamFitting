import numpy as np
import pyswarms
from lib.utils.doe_space_sampling import get_spacefilled_DoE
import time
from lib.utils.xmlread import XMLReader
from lib.utils.classes import ProblemObjectBase
from pathlib import Path



class FitParamsPSO:

    def __init__(self, input_reader: XMLReader, problem_object: ProblemObjectBase):
        """Initialize the Particle Swarm Optimization (PSO) fitting parameters.

        This method sets up the PSO algorithm parameters by:
        1. Storing input reader and problem object references
        2. Processing search bounds and scaling information
        3. Setting up PSO algorithm parameters
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
            - n_particles : Number of particles in the swarm
            - n_iters_pop : Number of iterations for population-based search
        problem_object : ProblemObjectBase
            Object containing problem-specific information:
            - compute_all_losses : Method to compute loss values for all particles

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
            Tuple of (min_search_list, max_search_list) for PSO bounds
        n_particles : int
            Number of particles in the swarm
        n_search_axes : int
            Number of dimensions in search space
        swarm_start_options : dict
            Initial PSO parameters:
            - c1 : Cognitive parameter (2.0)
            - c2 : Social parameter (0.5)
            - w : Inertia weight (1.0)
        swarm_end_options : dict
            Final PSO parameters:
            - c1 : Cognitive parameter (0.5)
            - c2 : Social parameter (2.0)
            - w : Inertia weight (0.5)
        vel_clamp : tuple
            Velocity clamping bounds (-0.05, 0.05)

        Notes
        -----
        - Search space is scaled to [-1, 1] range for each dimension
        - Log-scale axes are transformed using log10 before scaling
        - PSO parameters (c1, c2, w) will linearly vary from start to end values
        - Velocity clamping prevents particles from moving too far in one step

        See Also
        --------
        initialize_swarm : Method to initialize the particle swarm
        scale_value : Method to scale values to [-1, 1] range
        unscale_value : Method to unscale values from [-1, 1] range
        """

        self.problem_obj = problem_object
        self.input_reader = input_reader
        # Set bounds and scaling info

        self.min_search_list = []
        self.max_search_list = []

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

            self.min_search_list.append(sc_min)
            self.max_search_list.append(sc_max)

        self.bounds_tuple = (self.min_search_list, self.max_search_list)

        self.n_particles = input_reader.n_particles
        self.n_search_axes = input_reader.n_search_axes
        self.swarm_start_options = {"c1": 2.0, "c2": 0.5, "w": 1.0}
        self.swarm_end_options = {"c1": 0.5, "c2": 2.0, "w": 0.5}
        self.vel_clamp = (-0.05, 0.05)

        self.initialize_swarm()

    def initialize_swarm(self)-> None:
        """Initialize the particle swarm optimization (PSO) algorithm.

        This method sets up the PSO algorithm by:
        1. Creating a star topology for particle communication
        2. Initializing the swarm with specified number of particles and dimensions
        3. Setting up boundary and velocity handlers
        4. Creating an options handler for dynamic parameter adjustment
        5. Performing space-filling design of experiments (DoE) for initial particle positions

        The initialization includes:
        - Star topology for particle communication
        - Initial swarm parameters (c1, c2, w) for particle movement
        - Velocity clamping to prevent excessive movement
        - Boundary handling strategy set to 'nearest'
        - Velocity handling strategy set to 'invert'
        - Options handler for linear variation of parameters

        Parameters
        ----------
        None
            Uses class attributes set during initialization

        Attributes
        ----------
        my_topology : pyswarms.backend.topology.Star
            Star topology object for particle communication
        swarm_obj : pyswarms.backend.swarms.Swarm
            Swarm object containing particle positions, velocities, and costs
        bh : pyswarms.backend.handlers.BoundaryHandler
            Handler for keeping particles within search bounds
        vh : pyswarms.backend.handlers.VelocityHandler
            Handler for managing particle velocities
        oh : pyswarms.backend.handlers.OptionsHandler
            Handler for dynamically adjusting PSO parameters
        unscaled_position : numpy.ndarray
            Array to store unscaled particle positions

        Notes
        -----
        - Uses space-filling DoE to initialize particle positions
        - Initial swarm parameters are set to c1=2.0, c2=0.5, w=1.0
        - Velocity is clamped between -0.05 and 0.05
        - Parameters will linearly vary to end values (c1=0.5, c2=2.0, w=0.5)

        See Also
        --------
        get_spacefilled_DoE : Function used for initial particle positioning
        """

        self.my_topology = pyswarms.backend.topology.Star()
        self.swarm_obj = pyswarms.backend.generators.create_swarm(
            self.n_particles,
            self.n_search_axes,
            options=self.swarm_start_options,
            bounds=self.bounds_tuple,
            clamp=self.vel_clamp,
        )

        
        doe_axis_lims = []
        for i_axis in range(self.input_reader.n_search_axes):

            doe_axis_lims.append(
                [self.min_search_list[i_axis], self.max_search_list[i_axis]]
            )

        q = 100

        print("Creating initial sampling")
        best_sampling, phi_p_best = get_spacefilled_DoE(
            self.n_particles, np.array(doe_axis_lims), q
        )

        
        self.swarm_obj.position = np.array(best_sampling)
        self.bh = pyswarms.backend.handlers.BoundaryHandler(strategy="nearest")
        self.vh = pyswarms.backend.handlers.VelocityHandler(strategy="invert")
        self.oh = pyswarms.backend.handlers.OptionsHandler(
            strategy={
                "w": "lin_variation",
                "c1": "lin_variation",
                "c2": "lin_variation",
            }
        )

        # declare array for unscaled position
        self.unscaled_position = np.zeros_like(self.swarm_obj.position)

    def scale_value(self, val: float, min_val: float, max_val: float)-> float:

        return 2 * (val - min_val) / (max_val - min_val) - 1

    def unscale_value(self, val: float, min_val: float, max_val: float):

        return (1 + val) * (max_val - min_val) / 2 + min_val

    def unscale_design_point(self, position: np.ndarray)-> np.ndarray:

        if not isinstance(position, np.ndarray):
            raise ValueError("Query point to unscale_design_point must be a np array")

        unscaled_position = np.zeros_like(position)

        for i_axis in range(self.input_reader.n_search_axes):

            if self.axis_logscale[i_axis]:

                unscaled_position[i_axis] = 10 ** (
                    self.unscale_value(
                        position[i_axis],
                        self.min_search_axis[i_axis],
                        self.max_search_axis[i_axis],
                    )
                )
            else:
                unscaled_position[i_axis] = self.unscale_value(
                    position[i_axis],
                    self.min_search_axis[i_axis],
                    self.max_search_axis[i_axis],
                )

        return unscaled_position

    def compute_losses(self)-> np.ndarray:

        #all_losses=np.zeros(self.swarm_obj.position.shape[0])

        all_losses = self.problem_obj.compute_all_losses(self.swarm_obj.position)
        return all_losses

    def search_iteration(self, iter_no: int, file_obj: Path)-> None:
        """Perform a single iteration of the particle swarm optimization search.

        This method executes one iteration of the PSO algorithm, which includes:
        1. Computing losses for all particles
        2. Updating personal best positions and costs
        3. Updating global best position and cost
        4. Computing new velocities and positions
        5. Updating PSO parameters
        6. Logging progress to file and console

        Parameters
        ----------
        iter_no : int
            Current iteration number (0-based)
        file_obj : PosixPath or None
            Path to the log file where iteration results will be written.
            If None, results are only printed to console.

        Attributes
        ----------
        swarm_obj : pyswarms.backend.swarms.Swarm
            Swarm object containing:
            - position : Current positions of all particles
            - velocity : Current velocities of all particles
            - current_cost : Current cost values for all particles
            - pbest_pos : Personal best positions for all particles
            - pbest_cost : Personal best costs for all particles
            - best_pos : Global best position found so far
            - best_cost : Global best cost found so far

        Notes
        -----
        - First iteration (iter_no = 0) initializes personal bests
        - Progress is printed every iteration (print_freq = 1)
        - Log file format: "iteration, best_cost, iteration_time"
        - On final iteration, stores best position in self.best_pos

        Returns
        -------
        numpy.ndarray: Best parameter set found during optimization

        See Also
        --------
        compute_losses : Function used to calculate particle costs
        """

        print_freq = 1

        t1 = time.time()

        self.swarm_obj.current_cost = self.compute_losses()
        
        if iter_no == 0:

            self.swarm_obj.pbest_cost = self.swarm_obj.current_cost
            self.swarm_obj.pbest_pos = self.swarm_obj.position

        self.swarm_obj.pbest_pos, self.swarm_obj.pbest_cost = (
            pyswarms.backend.compute_pbest(self.swarm_obj)
        )

        # update best if new best is found

        if np.min(self.swarm_obj.pbest_cost) < self.swarm_obj.best_cost:

            self.swarm_obj.best_pos, self.swarm_obj.best_cost = (
                self.my_topology.compute_gbest(self.swarm_obj)
            )

        self.swarm_obj.velocity = self.my_topology.compute_velocity(self.swarm_obj)

        self.swarm_obj.position = self.my_topology.compute_position(
            self.swarm_obj, self.bounds_tuple, self.bh
        )

        new_options = self.oh(
            self.swarm_start_options,
            iternow=iter_no,
            itermax=self.input_reader.n_iters_pop,
            end_opts=self.swarm_end_options,
        )

        self.swarm_obj.options = new_options

        t2 = time.time()
        iteration_time = t2 - t1

        if iter_no % print_freq == 0:
            # Print human readable output to terminal
            print("Iteration: {}".format(iter_no + 1))
            print("Best Cost: {0:.4E}".format(self.swarm_obj.best_cost))
            print("Best position:", self.swarm_obj.best_pos)
            print("Time for iteration:", iteration_time)
            print("-------")

            # Write formatted output to log file
            if file_obj is not None:
                with open(file_obj, 'a') as f:
                    f.write(f"{iter_no + 1}, {self.swarm_obj.best_cost:.4E}, {iteration_time:.4f}\n")
                    f.flush()

        if iter_no == self.input_reader.n_iters_pop - 1:
            print("Final best position found(scaled):", self.swarm_obj.best_pos)
            print("Done with PSO search")
            self.best_pos = self.swarm_obj.best_pos
