import jax
import jax.numpy as jnp
import diffrax
from diffrax import RESULTS

jax.config.update("jax_enable_x64", True)

@jax.jit
def unscale_value(val, min_val, max_val, is_logscale):
    lin_unscaled = ((val + 1.0) / 2.0) * (max_val - min_val) + min_val
    unscaled = jnp.where(is_logscale, 10.0 ** lin_unscaled, lin_unscaled)
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

    # ['c2', 'Dk', 'Dc']
    c2, Dk, Dc = unscale_value(trainable_variables, min_val, max_val, is_logscale)
    # fixed ['m1','m2','vf']
    m1 = fixed_parameters["m1"]
    m2 = fixed_parameters["m2"]
    vf = fixed_parameters["vf"]

    # y ordering: ['x1', 'x2', 'v1', 'v2', 'k', 'c1']
    x1 = y[0]
    x2 = y[1]
    v1 = y[2]
    v2 = y[3]
    k = y[4]
    c1 = y[5]

    def F1(Fs, c1, v1):
        return Fs - c1 * jnp.abs(v1) * jnp.sign(v1)

    def F2(Fs, c2, v2):
        cond = jnp.logical_and(jnp.abs(Fs) < c2, jnp.abs(v2) < vf)
        return jnp.where(cond, 0.0, Fs - c2 * jnp.sign(v2))

    Fs = k * (x2 - x1)
    dx1_dt = v1
    dx2_dt = v2
    dv1_dt = (1.0 / m1) * F1(Fs, c1, v1)
    dv2_dt = (1.0 / m2) * F2(-1.0 * Fs, c2, v2)
    P = jnp.abs(m1 * v1 * dv1_dt)
    dk_dt = Dk * P
    dc1_dt = Dc * P

    return jnp.array([dx1_dt, dx2_dt, dv1_dt, dv2_dt, dk_dt, dc1_dt])

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
            rtol=constants['stepsize_rtol'],
            atol=constants['stepsize_atol']
        ),
    )
    return sol.ts, sol.ys, sol.result

@jax.jit
def _compute_loss_problem(constants, trainable_variables):
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    failed = jnp.logical_or(result == RESULTS.max_steps_reached, result == RESULTS.singular)

    # Columns 2:4 => v1, v2
    max_cols = jnp.max(jnp.abs(dataset[:, 2:4]), axis=0)
    loss_value = jnp.sqrt(jnp.mean(jnp.square((solution[:, 2:4] - dataset[:, 2:4]) / max_cols)))

    loss = jnp.where(failed, constants["error_loss"], loss_value)
    return loss

def _write_problem_result(constants, trainable_variables):
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros([Nts, 9])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1:5].set(dataset)
    writeout_array = writeout_array.at[:, 5:9].set(solution[:, 0:4])
    return writeout_array
