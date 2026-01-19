import jax
from jax import jit, value_and_grad
import jax.numpy as jnp
# Using the example_libraries as requested
from jax.example_libraries.optimizers import adam 
import config
import RMSE
from objective import objective


# 1. Initialize the optimizer functions
# These return functions to init the state, update it, and extract params
opt_init, opt_update, get_params = adam(config.step_size, b1=0.9, b2=0.999, eps=1e-08)

@jit
def train_step(step_count, opt_state):
    """
    Performs a single training step.
    Returns the updated state, total loss, and a dict of sub-losses.
    """
    # Extract current parameters from the optimizer state
    params = get_params(opt_state)
    
    # value_and_grad with has_aux=True allows the objective function 
    # to return (loss, aux_data). grads is computed only for 'loss'.
    (loss_value, aux_losses), grads = value_and_grad(objective, has_aux=True)(params)
    
    # Update the optimizer state using the gradients
    opt_state = opt_update(step_count, grads, opt_state)
    
    return opt_state, loss_value, aux_losses


def train(params, train_iters=config.train_iters):
    """Main training loop"""
    
    print(f"Starting training for {train_iters} iterations...")
    print("-" * 50)
    
    # 2. Initialize the optimizer state with the starting parameters
    opt_state = opt_init(params)
    
    for i in range(train_iters):
        # Perform the compiled update step
        opt_state, total_loss, aux = train_step(i, opt_state)
        
        # Log results every 1000 iterations
        if i % 1000 == 0:
            print(f"Iteration {i:5d} | Total Loss: {float(total_loss):.6e}")
            
            # Print individual components from the 'aux' dictionary
            # These keys must match what you return in objective.py
            print(f"    - PDE Loss:      {float(aux['pde']):.6e}")
            print(f"    - Boundary Loss: {float(aux['boundary']):.6e}")
            print(f"    - Int. Dirichlet: {float(aux['int_dir']):.6e}")
            print(f"    - Int. Flux:      {float(aux['int_flux']):.6e}")
            print("-" * 50)

    # 3. Training finished: Extract final parameters from the state
    final_params = get_params(opt_state)
    
    print("\nTraining Complete.")
    print("Final RMSE Calculation:")
    foo = RMSE.FinalError(final_params)
    foo.print_data()

    return final_params, opt_state
