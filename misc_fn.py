# misc_fn.py
import numpy as np
from domain_utils import *
import config
import jax.numpy as jnp

def build_analytical_functions_from_strings(aly_strings):
    """
    aly_strings: list of strings like ['x+y', 'y**2+3*y+z', ...]
    Returns:
        aly_list: list of callables f(x,y,z) that work with numpy arrays.
    """
    aly_list = []
    for expr in aly_strings:
        # Define a function f(x,y,z) that evaluates 'expr'
        def make_f(e):
            def f(x, y, z):
                # x, y, z can be arrays; eval runs elementwise ops
                return jnp.array(eval(e, {"__builtins__": {}}, {"x": x, "y": y, "z": z, "jnp": jnp}))
            return f
        aly_list.append(make_f(expr))
    return aly_list

def build_flux_grad_fn(grad_strings):
    """
    grad_strings: list of strings like ['1', 'z/3', ...]
    Returns:
        grad_list: list of callables f(x,y,z) that work with arrays.
    """
    grad_list = []
    for expr in grad_strings:
        def make_f(e):
            def f(x, y, z):
                return jnp.array(eval(e, {"__builtins__": {}}, {"x": x, "y": y, "z": z, "jnp": jnp}))
            return f
        grad_list.append(make_f(expr))
    return grad_list


def compute_Tmax(aly_list, domain_inputs, domain_id):
    """
    aly_list: [f0, ..., f_{N-1}], each f(x,y,z) using normalized coords.
    domain_inputs: (N_total, 3)     Put physical co-ordinates
    domain_id:  (N_total,)
    """
    num_domains = len(aly_list)
    X_all = domain_inputs[:, 0]
    Y_all = domain_inputs[:, 1]
    Z_all = domain_inputs[:, 2]

    max_val = -jnp.inf

    for d in range(num_domains):
        mask = (domain_id == d)
        aly_all = aly_list[d](X_all, Y_all, Z_all)
        aly_masked = np.where(mask, np.abs(aly_all), -np.inf)
        max_val = np.maximum(max_val, np.max(aly_masked))

    return max_val


def compute_L_max(dimensions=None):
    """
    Compute longest physical dimension from config.dimensions or given tuple.

    dimensions: [(x_min,x_max),(y_min,y_max),(z_min,z_max)]
                If None, uses config.dimensions.
    """
    if dimensions is None:
        dimensions = config.dimensions

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = dimensions
    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min
    return max(Lx, Ly, Lz)



ALY_LIST = build_analytical_functions_from_strings(config.aly_soln)
dp = DomainPoints(seed=config.domain_random_seed)
X_dom, dom_id = dp.sample_domain_points()

# Scaling values (T_max from analytical solution, L_max from dimensions)
max_T_val_global = compute_Tmax(
    ALY_LIST,           # analytical functions in PHYSICAL coords
    X_dom,         # physical points (NumPy ok inside compute_Tmax)
    dom_id
)
L_max_global = compute_L_max(config.dimensions)
