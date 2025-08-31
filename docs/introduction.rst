Introduction
============

OSParamFitting is an Agentic AI framework for parameter estimation/identification of complex ODE systems. The workflow is designed to be greatly simplified such that no expertise in numerical optimization, machine learning or even programming is required to efficiently fit user-defined ODE models to time-series data, experimental or simulated.


Framework Overview
-----------------

After filling out a simple XML file and a couple of simple Python functions that describe the problem, the framework automatically sets up and solves the estimation problem, returning the identified parameters to the user and abstracting away complex processes like:

1. Setting up a constrained optimization problem
2. Writing efficient, differentiable, JIT compiled code
3. Checking the optimization setup for correctness

Key Features
-----------

* **Simplified Workflow**: Fill out a simple XML file and define basic Python functions
* **Automatic Setup**: Framework automatically sets up and solves estimation problems
* **Two-Phase Optimization**: Combines population-based (PSO) and gradient-based (NODE) methods
* **JAX Integration**: Leverages JAX for efficient, differentiable programming
* **No Expertise Required**: Abstracts away the underlying complexity

Value Proposition
----------------

ODE parameter identification using data is inherently complex, typically requiring optimization techniques including gradient-based methods, population/genetic algorithms, and various heuristic approaches. While gradient-based methods offer efficiency, they can be unstable without proper differentiable programming implementations, which remain inaccessible to many users due to the specialized knowledge required.

The framework addresses these challenges by requiring users to simply populate a single XML file and define basic Python functions using NumPy. The agentic system then automatically generates a code skeleton, validates the XML and code for correctness, converts NumPy code into JAX-jittable format, and implements a two-layer optimization strategy for maximum efficiency.

This approach first employs a population method to explore the parameter space hypercube, followed by gradient-based refinement using differentiable programming, ensuring robust convergence while abstracting away the underlying complexity.

Installation
-----------

1. Create a virtual environment and install the required libraries:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -r requirements.txt

2. Set your OpenAI API key in a .env file in the base directory:

   .. code-block:: bash

      OPENAI_ENV_KEY=your-api-key-here

