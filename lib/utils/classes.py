import numpy as np
import xml.etree.ElementTree as ET
from lib.utils.xmlread import XMLReader
from pathlib import Path
from functools import partial
import jax
import jax.numpy as jnp
# Base class for any problem defined


class ProblemObjectBase:
    def __init__(self):
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
        return self._integrate_system(*args)

    #@partial(jax.jit,static_argnums=(0,))
    def compute_loss(self, *args):
        return self._compute_loss(*args)

    def compute_all_losses(self, population_points):
        return self._compute_all_losses(population_points)

    def plot_result(self, design_point, label="default"):
        return self._plot_result(design_point, label)



