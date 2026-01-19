"""
Docstring for RMSE
Generate RMSE and RRMSE of preidcted vs actual values for all the domains and boundaries.
"""

import numpy as np
import neuron
import domain_utils
import config
import misc_fn


max_T_val = misc_fn.max_T_val_global


class FinalError:
    def __init__(self, trained_params):
        self.final_params = trained_params
        
    def compute_RMSE(self,points, domain_id):
        """
        Compute RMSE and RRMSE between predicted and actual values.
        
        Args:
            predicted: jnp.array of predicted values
            actual: jnp.array of actual values
        """
        # Initialize network for the specific domain
        net = neuron.DomainNetworks(self.final_params)
        points_normalised = domain_utils.DomainPoints().normalize_points(points)
        # Predict values using the trained network
        predicted = net.forward(domain_id, points_normalised) * max_T_val
        predicted = np.array(predicted).flatten()
        
        # Compute actual values using analytical solution
        aly_soln_str = config.aly_soln[domain_id]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        actual = eval(aly_soln_str)
        actual = np.array(actual).flatten()
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((predicted - actual)**2))
        
        # Compute RRMSE
        rrmse = 1 #rmse / (np.max(actual) - np.min(actual))
        
        print("Predicted range:", predicted.min(), predicted.max())
        print("Actual range:", actual.min(), actual.max())


        return rmse, rrmse
    
    def compute_all_boundary_RMSE(self):
        """
        Compute RMSE and RRMSE for all boundaries.
        """
        boundary_errors = {}
        
        # Get boundary points
        boundary_points, boundary_id = domain_utils.DomainPoints(seed=config.domain_random_seed).sample_boundary_points()
        for i in range(len(config.activation)):
            points = boundary_points[boundary_id == i]
            rmse, rrmse = self.compute_RMSE(points, i)
            boundary_errors[f'boundary_{i}'] = {'RMSE': rmse, 'RRMSE': rrmse}
        
        return boundary_errors
    

    def compute_all_domain_RMSE(self):
        """
        Compute RMSE and RRMSE for all domains.
        """
        domain_errors = {}
        
        domain_points, domain_id = domain_utils.DomainPoints(seed=config.domain_random_seed).sample_domain_points()
        for i in range(len(config.activation)):
            points = domain_points[domain_id == i]
            rmse, rrmse = self.compute_RMSE(points, i)
            domain_errors[f'domain_{i}'] = {'RMSE': rmse, 'RRMSE': rrmse}

        return domain_errors
    
    def print_data(self):
        boundary_errors = self.compute_all_boundary_RMSE()
        for boundary, errors in boundary_errors.items():
            print(f"Boundary: {boundary}, RMSE: {errors['RMSE']}, RRMSE: {errors['RRMSE']}")
        
        domain_errors = self.compute_all_domain_RMSE()
        for domain_id, errors in domain_errors.items():
            print(f"Domain: {domain_id}, RMSE: {errors['RMSE']}, RRMSE: {errors['RRMSE']}")