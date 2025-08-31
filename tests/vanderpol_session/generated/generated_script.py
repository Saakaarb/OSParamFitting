import jax
import jax.numpy as jnp
import diffrax
from diffrax import RESULTS

jax.config.update("jax_enable_x64", True)

@jax.jit
def unscale_value(val, min_val, max_val, is_logscale):
    lin_unscaled = ((val + 1.0) / 2.0) * (max_val - min_val) + min_val
    unscaled = jnp.where(is_logscale, 10.0**lin_unscaled, lin_unscaled)
    return unscaled

@jax.jit
def scale_value(unscaled_val, min_val, max_val, is_logscale):
    lin_val = jnp.where(is_logscale, jnp.log10(unscaled_val), unscaled_val)
    scaled = 2.0 * (lin_val - min_val) / (max_val - min_val) - 1.0
    return scaled

@jax.jit
def user_defined_system(t, y, other_args):
    trainable_variables = other_args["trainable_variables"]
    constants = other_args["constants"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    fixed_parameters = constants["fixed_parameters"]
    min_val = constants["min_limits"]
    max_val = constants["max_limits"]
    is_logscale = constants["is_logscale"]

    # Trainable parameter ordering (trainable_variables): mu
    mu, = unscale_value(trainable_variables, min_val, max_val, is_logscale)

    # Ny=2 (from XML): y = [x1, x2]
    x1 = y[0]
    x2 = y[1]

    dx1dt = mu * (x2 - ((1.0 / 3.0) * x1 ** 3 - x1))
    dx2dt = -1.0 / mu * x1

    derivatives = jnp.array([dx1dt, dx2dt])
    return derivatives

@jax.jit
def _integrate_system(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)
    solver = diffrax.Kvaerno5()
    t_eval = constants["t_eval"]
    init_cond = constants["init_cond"]
    init_time = constants["init_time"]
    saveat = diffrax.SaveAt(ts=t_eval)
    other_args = {"constants": constants, "trainable_variables": trainable_variables}

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=init_time,
        t1=t_eval[-1],
        max_steps=10000,
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
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    failed = jnp.logical_or(result == RESULTS.max_steps_reached, result == RESULTS.singular)

    # Trainable parameter ordering: mu
    # Loss calculation
    # dataset shape [Nts, N_col], solution shape [Nts, Ny]
    # dataset omits time column --> [Nts, Ny]
    # max_col of dataset (axis=0)
    max_col = jnp.max(jnp.abs(dataset), axis=0)
    loss_value = jnp.sqrt(jnp.mean(jnp.square(jnp.divide(solution - dataset, max_col))))

    loss = jnp.where(failed, constants["error_loss"], loss_value)
    return loss

def _write_problem_result(constants, trainable_variables):
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)

    # Writeout array: shape [Nts, 5]
    # Column 0: solution_time
    # Columns 1,2: dataset
    # Columns 3,4: solution
    writeout_array = jnp.zeros((solution_time.shape[0], 5), dtype=solution.dtype)
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1:3].set(dataset)
    writeout_array = writeout_array.at[:, 3:5].set(solution)
    return writeout_array
