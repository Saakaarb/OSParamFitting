import numpy as np
import jax.numpy as jnp
import jax
import random
import itertools
import time

# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# get space-filled doe sampling based on value ranges


# Driver function that returns spacefilled DoE
# tests many different DoE distirbutions until the most spacefilled one is found


# Arguments
# n_samples: number of samples to test while getting space filled DoE space
# value ranges: list of value ranges for each axis, in the format [[min_value,max_value],[val1,val2,vel3...]....] if the axis is continuously or discretely sampled
# moris mitchell criteria parameters


# @njit
def get_spacefilled_DoE(n_samples: int, value_ranges: np.ndarray, q: int):
    """
    Generate a space-filling Design of Experiments (DoE) sampling using the Morris-Mitchell criterion.

    This function generates candidate DoE samplings and selects the one that maximizes space-filling
    (minimizes the Morris-Mitchell phi_p criterion) over a specified number of trials.
    The function always includes the center and all corners of the parameter space, and fills the rest
    with random samples. It then evaluates the space-filling quality using pairwise distances.

    Args:
        n_samples (int):
            Number of samples (points) to generate in the DoE.
        value_ranges (list of list):
            List of value ranges for each axis/parameter. Each element should be a list or tuple of two values [min, max].
            Example: [[0, 1], [10, 20]] for a 2D space.
        q (int):
            Exponent parameter for the Morris-Mitchell criterion (typically q=1 or 2).

    Returns:
        tuple:
            - best_sampling (ndarray): The DoE sampling (array of points) with the best space-filling property found.
            - phi_p_best (float): The value of the Morris-Mitchell criterion for the best sampling.

    Notes:
        - The function currently uses only one trial (n_trials=1), but can be extended for more robust optimization.
        - The Morris-Mitchell criterion is used to quantify space-filling: lower phi_p means better space-filling.
        - Useful for generating initial samples for surrogate modeling, sensitivity analysis, or global optimization.
    """

    # initialize
    phi_p_best = np.inf
    
    n_trials = 20
    print("Note: using n_trials=",n_trials)

    for trial_no in range(n_trials):
        t1=time.time()
        #sampling = create_doe_sampling_with_corners(n_samples, value_ranges)
        sampling = create_doe_sampling_random_points(n_samples, value_ranges)

        pairwise_dists = pairwise_dist(sampling, value_ranges)

        phi_p = pow(
            sum([1.0 / pairwise_dists[i] for i in range(len(pairwise_dists))]), 1.0 / q
        )

        if phi_p < phi_p_best:
            phi_p_best = phi_p
            
            best_sampling = sampling
        t2=time.time()
        
    return best_sampling, phi_p_best


# returns a DoE pt distribution to test for space-filledness

# Arguments:
# 1) n_samples: number of samples in DoE pt distribution
# 2) value_ranges: list of value ranges for each axis, in the format [[min_value,max_value],[val1,val2,vel3...]....] if the axis is continuously or discretely sampled


def create_doe_sampling_with_corners(n_samples: int, value_ranges):
    """
    Generate a Design of Experiments (DoE) sampling of points in a multi-dimensional parameter space.

    This function creates a set of sample points that span the provided value ranges for each axis/parameter.
    It ensures that all corners and the center of the space are included, and fills the remaining points with random samples.

    Args:
        n_samples (int):
            Total number of samples (points) to generate. Must be at least 2**n_axes + 1, where n_axes is the number of parameters/axes.
        value_ranges (list of list):
            List of value ranges for each axis/parameter. Each element should be a list or tuple of two values [min, max].
            Example: [[0, 1], [10, 20]] for a 2D space.

    Returns:
        list of list:
            A list of sample points, where each point is a list of values (one per axis).
            The first point is the center, followed by all corners, then random samples.

    Raises:
        ValueError: If n_samples is less than the required minimum (2**n_axes + 1).

    Notes:
        - The function always includes the center and all corners of the parameter space.
        - Remaining points are filled with random samples uniformly distributed within the value ranges.
        - The output can be used for space-filling DoE or sensitivity analysis.
    """

    n_axes = len(value_ranges)
    # sampling=[]
    # add corners and centre of DoE space

    # number of points added mandatorily
    num_comp_pts = 2 ** len(value_ranges) + 1
    # sampling=[]
    if n_samples < num_comp_pts:

        raise ValueError("Number of DoE points must be at least ", num_comp_pts)

    else:

        sampling = []
        # sampling=np.zeros([n_samples,n_axes])

        mean_entry = []
        # mean_entry=np.zeros([n_axes])
        for i_ax in range(n_axes):

            mean_entry.append((value_ranges[i_ax][0] + value_ranges[i_ax][1]) / 2.0)
            # mean_entry[i_ax]=(value_ranges[i_ax][0]+value_ranges[i_ax][1])/2.0

        sampling.append(mean_entry)
        # sampling[0,:]=mean_entry

        sampling_rest = list(itertools.product(*value_ranges))
        # sampling_rest=list(itertools.product(*value_ranges))

        sampling_rest = [list(x) for x in sampling_rest]
        # sampling_rest=[np.array(x) for x in sampling_rest]

        sampling.extend(sampling_rest)

    for i_sample in range(n_samples - num_comp_pts):
        sample = []

        for i_axis in range(n_axes):
            val = random.uniform(0, 1) * (
                max(value_ranges[i_axis]) - min(value_ranges[i_axis])
            ) + min(value_ranges[i_axis])

            sample.append(val)
        sampling.append(sample)

    return sampling

def create_doe_sampling_random_points(n_samples: int, value_ranges: np.ndarray):
    """
    Generate a Design of Experiments (DoE) sampling of points in a multi-dimensional parameter space.

    This function creates a set of sample points that span the provided value ranges for each axis/parameter.
    It ensures that all corners and the center of the space are included, and fills the remaining points with random samples.

    Args:
        n_samples (int):
            Total number of samples (points) to generate. Must be at least 2**n_axes + 1, where n_axes is the number of parameters/axes.
        value_ranges (list of list):
            List of value ranges for each axis/parameter. Each element should be a list or tuple of two values [min, max].
            Example: [[0, 1], [10, 20]] for a 2D space.

    Returns:
        list of list:
            A list of sample points, where each point is a list of values (one per axis).
            The first point is the center, followed by all corners, then random samples.

    Raises:
        ValueError: If n_samples is less than the required minimum (2**n_axes + 1).

    Notes:
        - The function always includes the center and all corners of the parameter space.
        - Remaining points are filled with random samples uniformly distributed within the value ranges.
        - The output can be used for space-filling DoE or sensitivity analysis.
    """

    n_axes = value_ranges.shape[0]
    # sampling=[]
    # add corners and centre of DoE space

    sampling = np.zeros((n_samples, n_axes))
    sample = np.zeros(n_axes)

    for i_sample in range(n_samples):
        

        for i_axis in range(n_axes):
            val = random.uniform(0, 1) * (
                max(value_ranges[i_axis]) - min(value_ranges[i_axis])
            ) + min(value_ranges[i_axis])

            sample[i_axis]=val
        sampling[i_sample,:] = sample

    return sampling

@jax.jit
def l2_dist(s1, s2):

    return jnp.linalg.norm(s1 - s2)


# Compute pairwie dists of points in DoE sampling


# Arguments:
# sampling: list of lists: contains n_samples number of DoE samples, each of which is a list
# value_ranges: as above
@jax.jit
def pairwise_dist(sampling, value_ranges):
    """
    Compute all pairwise Euclidean distances between points in a DoE sampling, scaled to [0, 1] for each axis.

    This function normalizes each axis of the sampling to the [0, 1] range using the provided value_ranges, then computes
    the Euclidean (L2) distance between every unique pair of points in the sampling.

    Args:
        sampling (array-like):
            2D array or list of shape (n_samples, n_axes), where each row is a sample point in the parameter space.
        value_ranges (array-like):
            2D array or list of shape (n_axes, 2), where each row is [min, max] for that axis.

    Returns:
        array float:
            Array of all unique pairwise Euclidean distances between the scaled sample points.

    Notes:
        - Scaling ensures that distances are comparable across axes with different units or ranges.
        - The output list contains distances for all unique pairs (i < j) in the sampling.
        - Useful for evaluating the space-filling quality of a DoE sampling (e.g., Morris-Mitchell criterion).
    """

    # scale down value ranges to [0,1]
    max_vals = value_ranges[:, 1]  # [max(value_range) for value_range in value_ranges]
    min_vals = value_ranges[:, 0]  # [min(value_range) for value_range in value_ranges]

    range_vals = max_vals - min_vals

    # Vectorized scaling
    scaled_samples = (sampling - min_vals) / range_vals

    n_samples = scaled_samples.shape[0]
    i_idx, j_idx = jnp.triu_indices(n_samples, k=1)

    # Gather the corresponding pairs
    xi = scaled_samples[i_idx]
    xj = scaled_samples[j_idx]

    # Vectorized L2 distances
    pairwise_dists = jax.vmap(l2_dist)(xi, xj)
    return pairwise_dists
