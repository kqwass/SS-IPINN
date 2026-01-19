import jax
import jax.numpy as jnp
from jax import vmap, jacfwd, jacrev

import config
from physics_loss import PDELoss, BoundaryLoss, InterfaceLoss
from neuron import DomainNetworks
import domain_utils
import misc_fn
from neuron import *
import numpy as np


def objective(params):
    """
    Modular objective function using physics loss classes.
    
    Args:
        params: list of neural network parameters for each domain
        domain_points: dict with keys like 'X_dom_norm', 'dom_id'
        boundary_points: dict with normalized and physical boundary coordinates
        interface_points: dict with normalized and physical interface coordinates
    
    Returns:
        total_loss: scalar loss
    """
    
    # Initialize networks with params
    nets = DomainNetworks(params)
    # Initialize loss calculators
    pde_loss_calc = PDELoss(nets)
    boundary_loss_calc = BoundaryLoss(nets)
    interface_loss_calc = InterfaceLoss(nets)
    
    # Compute PDE loss (domain interior)
    eq_loss, eq_losses_per_domain = pde_loss_calc.compute_domain_loss()
    
    # Compute boundary loss (Dirichlet conditions)
    total_boundary_loss, boundary_losses = boundary_loss_calc.compute_boundary_loss()
    
    # Compute interface loss (Dirichlet + flux jump conditions)
    total_interface_loss, \
    (dir_interface_loss, flux_interface_loss), \
    interface_losses = interface_loss_calc.compute_interface_loss()
    
    
    # Loss weights
    pde_weight = 20.0
    interface_dir_weight = 1.0
    interface_flux_weight = 1.0
    boundary_weight = 1.0
    
    # Total loss
    total_loss = (
        pde_weight * eq_loss
        + interface_dir_weight * dir_interface_loss
        + interface_flux_weight * flux_interface_loss
        + boundary_weight * total_boundary_loss
    )
    
    # Logging
    #print("=" * 80)
    #for i, loss in enumerate(eq_losses_per_domain):
        #print("PDE Loss - Domain ",i+1,jnp.array(loss))
    """ print(f"Total PDE Loss: {float(eq_loss):.6e}")
        print(f"Boundary Loss: {float(total_boundary_loss):.6e}")
        print(f"Dirichlet Interface Loss: {float(dir_interface_loss):.6e}")
        print(f"Flux Interface Loss: {float(flux_interface_loss):.6e}")
        print("=" * 80)"""
    
    return total_loss, {
        "pde": eq_loss,
        "boundary": total_boundary_loss,
        "int_dir": dir_interface_loss,
        "int_flux": flux_interface_loss
    }

"""
param_making = WeightBias(config.layer_sizes, config.param_scale)
init_params = param_making.generate_weight_bias()

foo = objective(init_params)
print("Objective function output on test params:", foo)"""