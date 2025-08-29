from lib.utils.classes import ProblemObjectBase
import jax
import jax.numpy as jnp
import diffrax
from diffrax import RESULTS
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from functools import partial

jax.config.update("jax_enable_x64", True)


# fixed
@jax.jit
def unscale_value(val, min_val, max_val, is_logscale):

    lin_unscaled = ((val + 1.0) / 2.0) * (max_val - min_val) + min_val
    unscaled = jnp.where(is_logscale, 10.0**lin_unscaled, lin_unscaled)

    return unscaled


# fixed
@jax.jit
def scale_value(unscaled_val, min_val, max_val, is_logscale):
    # If logscaled, take log10 first
    lin_val = jnp.where(is_logscale, jnp.log10(unscaled_val), unscaled_val)
    # Linearly map [min_val, max_val] → [-1, 1]
    scaled = 2.0 * (lin_val - min_val) / (max_val - min_val) - 1.0
    return scaled


# Robertson stiff system
@jax.jit
def user_defined_system(t, y, other_args):

    # fixed
    # ----------------------
    trainable_variables = other_args["trainable_variables"]
    constants = other_args["constants"]
    dataset = constants["dataset"]
    t_eval=constants["t_eval"]
    fixed_parameters=constants["fixed_parameters"] # dict {"m":1, "Cp":800}
    #TODO the order of parameters here should be the same as the ordering in the input
    min_val = constants["min_limits"]
    max_val = constants["max_limits"]
    is_logscale = constants["is_logscale"]


    # ----------------------

    k1, k2, k3 = unscale_value(trainable_variables, min_val, max_val, is_logscale)

    y1, y2, y3 = y

    # this part is user entered
    #---------------------------------------------------
    


    #---------------------------------------------------
    
    return jnp.array([dy1dt, dy2dt, dy3dt])


# TODO add ability to declare arbitrary number of static variables
# fixed
@jax.jit
def _integrate_system(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)
    solver = diffrax.Kvaerno5()
    t_eval = constants["t_eval"]
    init_cond = constants["init_cond"]
    init_time=constants["init_time"]
    dataset = constants["dataset"]
    saveat = diffrax.SaveAt(ts=t_eval)

    # TODO add ability to add more arbitrary inputs here
    other_args = {"constants": constants, "trainable_variables": trainable_variables}
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=init_time,
        t1=t_eval[-1],
        max_steps=,
        dt0=constants['init_timestep'],
        y0=init_cond,
        args=other_args,
        saveat=saveat,
        throw=False,
        stepsize_controller=diffrax.PIDController(rtol=constants['stepsize_rtol'], atol=constants['stepsize_atol']),
    )
    return sol.ts, sol.ys, sol.result


@jax.jit
def _compute_loss_problem(constants, trainable_variables):

    #fixed
    #---------------------------------------------------
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    failed = jnp.logical_or(result == RESULTS.max_steps_reached, result==RESULTS.singular)
    #---------------------------------------------------
    # if required, use sim to get further results

    
    # this part is user entered
    #---------------------------------------------------
    
    #---------------------------------------------------

    #fixed
    #--------------------------------------
    
    loss= jnp.where(failed, 
    constants["error_loss"],
    loss_value
    )

    return loss

# the purpose of this function is to write out a CSV containing info
# that is to be plotted
def _write_problem_result(constants,trainable_variables):

    #fixed
    #---------------------------------------------------
    dataset = constants["dataset"]

    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    #---------------------------------------------------

    # the rest is user entered
    #---------------------------------------------------

    #---------------------------------------------------
   


    return writeout_array


