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
    # Only one parameter, order: ['mu']
    mu = unscale_value(trainable_variables[0], min_val[0], max_val[0], is_logscale[0])

    # Integrated variables order: ['x1', 'x2']
    x1 = y[0]
    x2 = y[1]
    dx1_dt = mu * (x2 - ((1.0/3.0) * x1**3 - x1))
    dx2_dt = -1.0 / mu * x1
    return jnp.array([dx1_dt, dx2_dt])

@jax.jit
def _integrate_system(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)
    solver = diffrax.Kvaerno5()
    t_eval = constants["t_eval"]
    init_cond = constants["init_cond"]
    init_time = constants["init_time"]
    dataset = constants["dataset"]
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
        stepsize_controller=diffrax.PIDController(
            rtol=constants['stepsize_rtol'], atol=constants['stepsize_atol']
        ),
    )
    return sol.ts, sol.ys, sol.result

@jax.jit
def _compute_loss_problem(constants, trainable_variables):
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    failed = jnp.logical_or(result == RESULTS.max_steps_reached, result == RESULTS.singular)
    # Compute loss: normalized RMSE w.r.t. each column's max absolute value
    max_col = jnp.max(jnp.abs(solution), axis=0)
    loss_value = jnp.sqrt(jnp.mean(jnp.square((solution - dataset) / max_col)))
    loss = jnp.where(failed, constants["error_loss"], loss_value)
    return loss

def _write_problem_result(constants, trainable_variables):
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    # Writeout: columns x1 and x2 for each time (shape [N,2] as in user sample)
    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros((Nts, 2))
    writeout_array = solution[:, :2]
    return writeout_array
