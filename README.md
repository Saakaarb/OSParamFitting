# OSParamFitting

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Documentation-blue.svg)](https://github.com/Saakaarb/OSParamFitting/actions)




https://github.com/user-attachments/assets/c46ff6ab-75d6-4947-aac2-a1f332b31b2d



https://github.com/user-attachments/assets/b0e91488-e3f4-4adf-8806-08bd6044ceae



https://github.com/user-attachments/assets/f3d6be48-d1ad-479d-878a-b2ca8c9e13be



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

This guide walks you through setting up and running your first ODE parameter fitting problem using OSParamFitting. We'll use the Robertson system as an example - a classical stiff ODE system that demonstrates the framework's capabilities.

## Prerequisites

Before starting, ensure you have:
- OSParamFitting installed (see [Installation](#installation) above)
- a .env file in the base directory with the OpenAI API key-value pair: `OPENAI_ENV_KEY`= key
- Basic understanding of ODE systems
- Your experimental or simulated time-series data

## Quick Start: Robertson System Example

### 1. Create Session Structure

Set up the required directory structure for your fitting session:

```bash
# Create the main sessions directory
mkdir sessions

# Create a session for the Robertson system
cd sessions
mkdir robertson_session
cd robertson_session

# Create the inputs directory
mkdir inputs
```

Your directory structure should look like:
```
sessions/
└── robertson_session/
    └── inputs/
```

### 2. Prepare Input Files

Copy the example files to get started quickly:

```bash
# Copy the example XML configuration
cp ../../examples/robertson_example/inputs/user_input.xml inputs/

# Copy the example data file
cp ../../examples/robertson_example/inputs/robertson_data.csv inputs/
```

### 3. Run the Three-Stage Workflow

OSParamFitting uses an intelligent, three-stage approach to parameter estimation:

#### Stage 1: Generate Code Skeleton
The AI agent analyzes your XML configuration and creates a Python template:

```bash
python create_user_model.py robertson_session
```

This generates `generated/user_model.py` with functions you need to implement:
- `user_defined_system`: Define how the trainable parameters, fixed constants and integrable variables make up the ODE system
- `_compute_loss_problem`: Define how to compute the loss between integrated model solution and data
- `_write_problem_result`: Define how to write results to files

#### Stage 2: Validate Your Setup
Check for errors and warnings in your configuration:

```bash
python check_user_input.py robertson_session
```

The agent will:
- Analyze your XML and Python code
- Identify critical errors and warnings
- Provide actionable feedback
- Ensure your setup is ready for optimization

#### Stage 3: Parameter Estimation
Run the actual parameter fitting process:

```bash
python fit_parameters.py robertson_session
```

This stage implements a sophisticated two-layer optimization strategy:
1. **Population-based search** (PSO) explores the parameter space globally
2. **Gradient-based refinement** (NODE) fine-tunes the best results
3. **Automatic convergence** ensures robust parameter estimates

## Expected Results

After successful completion, you'll find in your output directory:
- `final_design_point.csv`: Best parameter values found
- `result_solution.csv`: Solution trajectory with fitted parameters  
- `pso_fitting.log`, `NODE_fitting.log`: Detailed optimization log
- `fitting_error.txt`: Any errors encountered (if applicable)

## Customizing for Your Problem

To adapt this workflow for your own ODE system:

1. **Modify the XML configuration**:
   - Update parameter names and bounds
   - Adjust optimization settings
   - Specify your data file name

2. **Implement the required functions**:
   - `_compute_loss_problem`: Return scalar loss value
   - `_write_problem_result`: Return solution array

3. **Prepare your data**:
   - CSV format with time in first column
   - Data columns matching your ODE variables
   - Consistent time points

## Troubleshooting

**Common Issues:**
- **Directory errors**: Ensure you're in the base directory and sessions structure is correct
- **API key errors**: Verify `OPENAI_ENV_KEY` is present in the .env
- **Convergence issues**: Adjust parameter bounds or optimization settings in XML

**Getting Help:**
- Check error messages in `fitting_error.txt`
- Review the PSO log file for optimization details
- Verify your XML configuration syntax

## Next Steps

- Explore the [API Documentation](#api-documentation) for detailed reference
- Check the examples directory for more complex use cases
- Experiment with different optimization settings
- Apply this workflow to your own research problems

# API Documentation

# Best Practices


# Citation
