import jax
from jax import jit, value_and_grad
import optax
import config

from objective import objective
import RMSE


optimizer = optax.adam(learning_rate=config.step_size, b1=0.9, b2=0.999, eps=1e-8)

@jit
def train_step(params, opt_state):
    """Compute gradient and update parameters"""
    value, grads = value_and_grad(objective)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
    return params, opt_state, value


def train(params,
          train_iters=config.train_iters, step_size=config.step_size):
    """Main training loop"""
    
    # Initialize optimizer
    
    opt_state = optimizer.init(params)
    
    for i in range(train_iters):
        params, opt_state, value = train_step(
            params, opt_state
        )
        
        if i % 1000 == 0:
            print(f"Iteration {i:5d} objective {float(value):.6e}")
    print("RMSE Calculation:")
    foo = RMSE.FinalError(params)
    foo.print_data()

    return params, opt_state


