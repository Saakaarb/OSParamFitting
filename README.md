# OSParamFitting

<div style="display: flex; justify-content: space-between; align-items: center; margin: 20px 0;">
  <div style="flex: 1; margin: 0 10px;">
    <h4>Parameter Convergence</h4>
    <video width="100%" controls>
      <source src="img/param_convergence_vid.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
  
  <div style="flex: 1; margin: 0 10px;">
    <h4>Fitting Process</h4>
    <video width="100%" controls>
      <source src="img/fit_vid.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
  
  <div style="flex: 1; margin: 0 10px;">
    <h4>Loss Plot</h4>
    <video width="100%" controls>
      <source src="img/loss_plot_vid.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>

This repository implements an Agentic AI framework for parameter estimation/identification of complex ODE systems. The workflow is designed to be greatly simplified such that no expertise in numerical optimization, machine learning or even programming is required to efficiently fit user-defined ODE models to time-series data, experimental or simulated. After filling out a simple XML file and a couple of simple python functions that describe the problem, the framework automatically sets up and solves the estimation problem, returning the identified parameters to the user and abstracting away complex processes like:

1. Setting up a constrained optimization problem
2. Writing efficient, differentiable, JIT compiled code
3. Checking the optimization setup for correctness

# Value Proposition

ODE parameter identification using data is inherently complex, typically requiring optimization techniques including gradient-based methods, population/genetic algorithms, and various heuristic approaches. While gradient-based methods offer efficiency, they can be unstable without proper differentiable programming implementations, which remain inaccessible to many users due to the specialized knowledge required. The framework addresses these challenges by requiring users to simply populate a single XML file and define basic Python functions using NumPy. The agentic system then automatically generates a code skeleton, validates the XML and code for correctness, converts NumPy code into JAX-jittable format, and implements a two-layer optimization strategy for maximum efficiency. This approach first employs a population method to explore the parameter space hypercube, followed by gradient-based refinement using differentiable programming, ensuring robust convergence while abstracting away the underlying complexity.

# Installation

1. Create a virtual environment and install the required libraries:

```
pip/pip3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Directory Structure


# Getting Started

Before running the framework, a directory named "sessions" must be created 

Shown below is an example fitting the robertson equations to solution data. The robertson system is a classical ODE system used to demonstrate ODE parameter identification algorithms due to it's stiff nature.

## Problem Setup

To setup the problem fitting and start using the agentic workflow, the user must complete a few short steps:

Step 1: in the base dir, make a directory called sessions: ```mkdir sessions```
Step 2: Within the ```sessions``` directory, make a subdir robertson_session: ```cd sessions; mkdir robertson_session```
Step 3: Within the ```robertson_session``` directory, make a subdir inputs: ```cd robertson_session; mkdir inputs```
Step 4: Create the input XML. The user can copy the example input XML to the current session: ```cp ../../examples/robertson_example/inputs/user_input.xml /inputs```
Step 5: Put the data to fit to in the ```/inputs``` folder. The user can copy the robertson data from examples: ```cp ../../examples/robertson_example/inputs/robertson_data.csv /inputs ```

## Running the agentic framework

Running the agentic parameter estimation is divided into 3 stages:

1. Creating the user model skeleton: First, using the provided XML, the agent sets up the code skeleton for the user to populate with the problem definition in generated/user_model.py. To launch this stage, run ```python create_user_model.py session_dirname```, where session_dirname is the name of the subdir in session. Continuing the robertson example, the session_dirname is robertson_session. Once generated/user_model.py has been generated, the user can populate the functions in user_model.py

2. Correctness check: Once the user has populated the functions, the input XML and user_model are checked by the agent for correctness and feasibility. The agent returns a report on critical errors and warnings. The user can address the critical errors and warnings before proceeding to the next stage

3. Parameter estimation: Once the user is satisfied with their input setup (after reading the critical errors and warnings), the user can run the parameter estimation. This can take a while depending on problem complexity. Once completed, this stage returns the estimated parameters.

# API Documentation

# Best Practices


# Citation
