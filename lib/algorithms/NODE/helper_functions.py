import numpy as np
import jax
import jax.numpy as jnp


def scale_value(val, min_val, max_val):

    return 2.0 * ((val - min_val) / (max_val - min_val)) - 1.0


def unscale_value(val, min_val, max_val):

    return (1.0 + val) * (max_val - min_val) / 2.0 + min_val
