# physics_loss.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jacfwd, jacrev

import config
import misc_fn
from domain_utils import DomainPoints
from neuron import *
import RMSE

# ======================================================================
# Build analytical solution functions ONCE at module import
# ======================================================================

# aly_list[d](x, y, z) expects PHYSICAL coordinates and returns physical T
ALY_LIST = misc_fn.build_analytical_functions_from_strings(config.aly_soln)
GRAD_LIST = misc_fn.build_flux_grad_fn(config.aly_soln_grad)



# ======================================================================
# PDE loss (domain interior)
# ======================================================================

class PDELoss:
    """
    Handles PDE residual loss for an arbitrary number of domains.

    Expects:
    - nets: DomainNetworks instance
    """

    def __init__(self, nets, seed=config.domain_random_seed):
        self.nets = nets

        # Sample domain points and normalize once
        dp = DomainPoints(seed=seed)
        self.X_dom, self.dom_id = dp.sample_domain_points()
        self.X_dom_norm = dp.normalize_points(self.X_dom)
        """
        # Convert to JAX arrays
        self.X_dom = jnp.array(X_dom)
        self.X_dom_norm = jnp.array(X_dom_norm)
        self.dom_id = jnp.array(dom_id)
        """
        # Physics parameters
        self.K_list = np.array(config.kappa_values)
        self.forcing_list = np.array(config.forcing_functions)
        self.num_domains = len(self.K_list)
        self.max_T_val = misc_fn.max_T_val_global
        self.L_max = misc_fn.L_max_global


    def _pde_residual_single(self, f, X, Kd, f_rhs_const):
        """
        f:            f(x,y,z) -> u_pred (normalized output)
        X:            (N,3) normalized coords (JAX array)
        Kd:           kappa for this domain
        f_rhs_const:  scalar forcing for this domain (e.g. 1.0)
        """
        T_max = self.max_T_val
        L_max = self.L_max

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        g = lambda x, y, z: f(x, y, z)

        f_xx = jax.vmap(jacfwd(jacrev(g, argnums=0), argnums=0))
        f_yy = jax.vmap(jacfwd(jacrev(g, argnums=1), argnums=1))
        f_zz = jax.vmap(jacfwd(jacrev(g, argnums=2), argnums=2))

        u_xx = f_xx(x, y, z).reshape(-1, 1)
        u_yy = f_yy(x, y, z).reshape(-1, 1)
        u_zz = f_zz(x, y, z).reshape(-1, 1)

        lap_u = u_xx + u_yy + u_zz
        rhs = f_rhs_const * jnp.ones_like(lap_u)

        # Your scaled PDE: ((Kd * T_max) / L_max) * ∇²u - forcing = 0
        eq = ((Kd * T_max) / (L_max**2)) * lap_u - rhs
        return eq

    def compute_domain_loss(self):
        """
        Compute PDE residual loss for all domains.

        Returns:
            total_eq_loss: scalar
            eq_losses: list of per-domain scalar losses
        """
        dom_id = self.dom_id
        X_dom_norm = self.X_dom_norm
        eq_losses = []

        for d in range(self.num_domains):
            mask = (dom_id == d)
            Kd = self.K_list[d]
            f_rhs_const = self.forcing_list[d]

            def f_d(x, y, z):
                X = jnp.stack([x, y, z], axis=-1)
                return self.nets.forward(d, X)  # normalized u

            eq = self._pde_residual_single(f_d, X_dom_norm, Kd, f_rhs_const)
            #masked_eq = jnp.where(mask[:, None], eq, 0.0)
            sum_sq = jnp.sum(eq**2)
            count = jnp.sum(mask.astype(jnp.float32))
            eq_loss_d = jnp.where(count > 0, sum_sq / count, 0.0)
            eq_losses.append(eq_loss_d)

        total_eq_loss = jnp.sum(jnp.stack(eq_losses))
        return total_eq_loss, eq_losses


# ======================================================================
# Boundary loss (Dirichlet, with normalized NN and physical analytical)
# ======================================================================

class BoundaryLoss:
    """
    Dirichlet boundary loss with:
    - NN input: normalized coords (Xn,Yn,Zn)
    - NN output: normalized temperature (scaled by T_max)
    - Analytical solution: evaluated on PHYSICAL coords (x,y,z)
    """

    def __init__(self, nets):
        """
        nets:   DomainNetworks instance

        """
        self.nets = nets
        self.T_max = misc_fn.max_T_val_global
        self.L_max = misc_fn.L_max_global
        self.aly_list = ALY_LIST          # same global analytical functions
        self.num_domains = len(self.aly_list)
        boundary_points, boundary_id = DomainPoints().sample_boundary_points()
        self.boundary_points = np.array(boundary_points)
        self.boundary_id = np.array(boundary_id)

    def boundary_residual_single_domain(self, X, f, domain_id):

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        X_norm = DomainPoints().normalize_points(X)
        x_n = X_norm[:, 0]
        y_n = X_norm[:, 1]
        z_n = X_norm[:, 2]

        pred_boundary = f(x_n, y_n, z_n).reshape(-1, 1)  # normalized
        aly_boundary = self.aly_list[domain_id](x, y, z).reshape(-1, 1) / self.T_max  # normalized

        res = pred_boundary - aly_boundary
        loss = jnp.mean(res**2)
        return loss



    def compute_boundary_loss(self):
        """
        Compute Dirichlet boundary loss for all domains.

        Returns:
            total_boundary_loss: scalar
            boundary_losses: list of per-domain scalar losses
        """
        X_bound = self.boundary_points
        bound_id = self.boundary_id

        boundary_losses = []

        for d in range(self.num_domains):
            mask = (bound_id == d)

            def f_d(x, y, z):
                X = jnp.stack([x, y, z], axis=-1)
                return self.nets.forward(d, X)  # normalized u

            X_norm = DomainPoints().normalize_points(X_bound)
            x_n = X_norm[:, 0]
            y_n = X_norm[:, 1]
            z_n = X_norm[:, 2]
            x = X_bound[:, 0]
            y = X_bound[:, 1]
            z = X_bound[:, 2]
            pred_boundary = f_d(x_n, y_n, z_n).reshape(-1, 1)  # normalized
            aly_boundary = self.aly_list[d](x, y, z).reshape(-1, 1) / self.T_max  # normalized
            res = pred_boundary - aly_boundary
            loss_all = res**2
            mean_sq = jnp.mean(loss_all)
            boundary_losses.append(mean_sq)

        total_boundary_loss = jnp.sum(jnp.stack(boundary_losses))
        return total_boundary_loss, boundary_losses
    

class InterfaceLoss:
    """
    Interface Dirichlet + flux (Neumann) jump loss for arbitrary number of interfaces.

    - NN input: normalized coords (Xn,Yn,Zn) on the interface.
    - NN output: normalized u; scaled by T_max to physical T.
    - Analytical solutions: evaluated on PHYSICAL coords (x,y,z).
    - Flux gradients: from aly_soln_grad strings evaluated on normalized coords.
    """

    def __init__(self, nets):
        """
        nets:   DomainNetworks instance
        T_max:  maximum physical temperature (from PDELoss.max_T_val)
        L_max:  longest physical dimension (from PDELoss.L_max)
        """
        self.nets = nets
        self.T_max = misc_fn.max_T_val_global
        self.L_max = misc_fn.L_max_global
        self.aly_list = ALY_LIST           # global analytical functions (physical coords)
        self.K_list = np.array(config.kappa_values)  # kappa per domain
        self.num_domains = len(self.aly_list)
        self.aly_grad_list = GRAD_LIST    # gradient functions physical coords)


    def _interface_loss_single(
        self,
        d,                  # interface index (between domain d and d+1)
        X_intf, intf_id,  # full interface points and ids
        f1, # function 1 to get NN output (left)
        f2  # function 2 to get NN output (right)
    ):
        """
        Compute both Dirichlet jump and flux jump for one interface.
        
        Returns:
            (dir_res, flux_res): residual arrays
            (dir_loss, flux_loss): scalar losses
        """
        i_left = d
        i_right = d + 1
        mask = (intf_id == d)
        aly_left = self.aly_list[i_left]
        aly_right = self.aly_list[i_right]
        K_left = self.K_list[i_left]
        K_right = self.K_list[i_right]
        x_phys = X_intf[:, 0]
        y_phys = X_intf[:, 1]
        z_phys = X_intf[:, 2]
        X_phys = X_intf
        X_norm = DomainPoints().normalize_points(X_phys)
        Xn = X_norm[:, 0]
        Yn = X_norm[:, 1]
        Zn = X_norm[:, 2]

        # ========== DIRICHLET JUMP ==========
        # NN predictions (normalized input -> normalized output)
        def f_left_norm(X):
            return self.nets.forward(i_left, X)

        def f_right_norm(X):
            return self.nets.forward(i_right, X)

        # Ensure JAX arrays

        # NN predictions at normalized coords
        X_in = jnp.stack([Xn, Yn, Zn], axis=-1)  # (N,3)
        u_left_norm = f_left_norm(X_in).reshape(-1, 1)   # (N,1) normalized
        u_right_norm = f_right_norm(X_in).reshape(-1, 1) # (N,1) normalized

        # Normalized jumps (both sides)
        pred_jump = u_right_norm - u_left_norm

        # Analytical jump (physical coords -> normalized)
        T_left_aly = aly_left(x_phys, y_phys, z_phys).reshape(-1, 1) / self.T_max
        T_right_aly = aly_right(x_phys, y_phys, z_phys).reshape(-1, 1) / self.T_max
        aly_jump = T_right_aly - T_left_aly

        # Dirichlet residual and loss
        dir_res = pred_jump - aly_jump
        dir_loss = jnp.mean(dir_res**2)
        # ========== FLUX JUMP ==========
        g = lambda x, y, z: f1(x, y, z)  # left NN
        h = lambda x, y, z: f2(x, y, z)  # right NN
        f1_z = jax.vmap(jacfwd(g, argnums=2))
        f2_z = jax.vmap(jacfwd(h, argnums=2))



        uz_left  = f1_z(Xn, Yn, Zn).reshape(-1, 1)
        uz_right = f2_z(Xn, Yn, Zn).reshape(-1, 1)

        K_left = self.K_list[i_left]
        K_right = self.K_list[i_right]
        flux_pred = (K_left * uz_left - K_right * uz_right)

        # Analytical gradients from config.aly_soln_grad per DOMAIN
        grad_left_fn = self.aly_grad_list[i_left]
        grad_right_fn = self.aly_grad_list[i_right]
        
        grad_left_aly  = grad_left_fn(x_phys, y_phys, z_phys).reshape(-1, 1)
        grad_right_aly = grad_right_fn(x_phys, y_phys, z_phys).reshape(-1, 1)

        flux_aly = (K_left * grad_left_aly - K_right * grad_right_aly) / ((self.T_max / self.L_max))

        flux_res = flux_pred - flux_aly
        flux_loss = jnp.mean(flux_res**2)

        return (dir_res, flux_res), (dir_loss, flux_loss)

    def compute_interface_loss(self):
        """
        Compute interface Dirichlet and flux jump losses for all interfaces.

        Returns:
            total_interface_loss: scalar
            (dir_interface_loss, flux_interface_loss): tuple of scalar losses
            interface_losses: list of per-interface (dir_loss, flux_loss) tuples
            
        """

        interface_losses = []
        dir_interface_losses = []
        flux_interface_losses = []
        # Sample interface points once
        dp = DomainPoints()
        X_intf, intf_id = dp.sample_interface_points()
        X_intf = jnp.array(X_intf)
        intf_id = jnp.array(intf_id)

        for d in range(self.num_domains - 1):

            def f_left(x, y, z):
                X = jnp.stack([x, y, z], axis=-1)
                return self.nets.forward(d, X)  # normalized u

            def f_right(x, y, z):
                X = jnp.stack([x, y, z], axis=-1)
                return self.nets.forward(d+1, X)  # normalized u

            (dir_res_d, flux_res_d), (dir_loss_d, flux_loss_d) = self._interface_loss_single(
                d,
                X_intf,
                intf_id,
                f_left,
                f_right
            )

            dir_interface_losses.append(dir_loss_d)
            flux_interface_losses.append(flux_loss_d)
            interface_losses.append((dir_loss_d, flux_loss_d))

        dir_interface_loss = jnp.sum(jnp.stack(dir_interface_losses))
        flux_interface_loss = jnp.sum(jnp.stack(flux_interface_losses))
        total_interface_loss = dir_interface_loss + flux_interface_loss
        return total_interface_loss, (dir_interface_loss, flux_interface_loss), interface_losses
    

"""
param_making = WeightBias(config.layer_sizes, config.param_scale)
init_params = param_making.generate_weight_bias()
nets = DomainNetworks(init_params)
domain_loss = PDELoss(nets)
eq_loss, eq_losses = domain_loss.compute_domain_loss()
print("Total PDE loss on test array:", eq_loss)
boundary_loss_calc = BoundaryLoss(nets)
total_boundary_loss, boundary_losses = boundary_loss_calc.compute_boundary_loss()
print("Total Boundary loss on test array:", type(total_boundary_loss))
interface_loss_calc = InterfaceLoss(nets)
total_interface_loss, (dir_interface_loss, flux_interface_loss), interface_losses = interface_loss_calc.compute_interface_loss()
print("Total Interface loss on test array:", total_interface_loss)
print("Dirichlet Interface loss on test array:", dir_interface_loss)
print("Flux Interface loss on test array:", flux_interface_loss)
"""