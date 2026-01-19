import numpy as np
import numpy.random as npr
import jax
import jax.numpy as jnp
import config

class WeightBias:
    """
    Generates initial random weights and biases for the NN
    """
    def __init__(self, layer_sizes, scale):
        self.layer_sizes = layer_sizes
        self.scale = scale

    def generate_weight_bias(self, seed=10):
        rng = npr.RandomState(seed)

        params = [
            (self.scale * rng.randn(m, n),   # weights
             self.scale * rng.randn(n))      # biases
            for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]
        #print(params)
        return params
    

def get_activation(name):
    if name == "tanh":
        return jnp.tanh
    if name == "swish":
        return jax.nn.swish
    if name == "relu":
        return jax.nn.relu
    if name == "elu":
        return jax.nn.elu
    if name == "sigmoid":
        return jax.nn.sigmoid
    raise ValueError(f"Unknown activation '{name}'")


class DomainNetworks:
    """
    One JAX MLP per domain, using provided parameters.
    """

    def __init__(self, params_list):
        self.layer_sizes = config.layer_sizes
        self.activation_names = config.activation
        self.scale = config.param_scale
        self.num_domains = len(self.activation_names)
        
        # Use provided params
        self.params_list = params_list
        
        # activations
        self.acts_list = [get_activation(name) for name in self.activation_names]

    def mlp_forward(self, params, act_fn, X):
        """
        X: (N, in_dim) jnp.array.
        params: list of (W, b).
        """
        h = X
        # hidden layers
        for (W, b) in params[:-1]:
            h = act_fn(jnp.dot(h, W) + b)
        # output layer
        W_out, b_out = params[-1]
        y = jnp.dot(h, W_out) + b_out
        return y

    def forward(self, domain_id, X):
        params = self.params_list
        act_fn = self.acts_list[domain_id]
        return self.mlp_forward(params, act_fn, X)

    
