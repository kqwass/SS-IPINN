import jax
import jax.numpy as jnp
import numpy as np

import config
from neuron import DomainNetworks, WeightBias
from domain_utils import DomainPoints
from optimizer import *
from physics_loss import PDELoss


init_params = WeightBias(config.layer_sizes, config.param_scale).generate_weight_bias()

def main():
    train(init_params)


if __name__ == "__main__":
    main()
    