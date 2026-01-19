import config
import numpy as np

class DomainPoints:
    def __init__(self, seed=config.domain_random_seed):
        # Read from config once
        self.rng = np.random.RandomState(seed)
        self.dimensions = config.dimensions
        self.interface_planes = config.interface_planes
        self.domain_points = config.domain_points
        self.interface_number_points = config.interface_number_points
        self.boundary_point_density = config.boundary_point_density

        # Precompute domain bounds in z
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.dimensions
        planes = [z_min] + self.interface_planes + [z_max]
        self.domain_bounds = [(planes[i], planes[i+1]) for i in range(len(planes)-1)]
        Lx = x_max - x_min
        Ly = y_max - y_min
        Lz = z_max - z_min
        self.L_max = max(Lx, Ly, Lz)

    def sample_domain_points(self):
        """Return X_domain_all, domain_id."""
        (x_min, x_max), (y_min, y_max), _ = self.dimensions
        X_list = []
        ids_list = []

        for d, (z_min, z_max) in enumerate(self.domain_bounds):
            n = self.domain_points[d]
            x = self.rng.uniform(x_min, x_max, size=n)
            y = self.rng.uniform(y_min, y_max, size=n)
            z = self.rng.uniform(z_min, z_max, size=n)
            X_d = np.column_stack([x, y, z])
            X_list.append(X_d)
            ids_list.append(np.full(n, d, dtype=int))

        X_domain = np.concatenate(X_list, axis=0)
        domain_id = np.concatenate(ids_list, axis=0)
        return X_domain, domain_id

    def sample_interface_points(self):
        """Return X_interface_all, interface_id."""
        (x_min, x_max), (y_min, y_max), _ = self.dimensions
        X_list = []
        ids_list = []

        for k, z_plane in enumerate(self.interface_planes):
            n = self.interface_number_points[k]
            x = self.rng.uniform(x_min, x_max, size=n)
            y = self.rng.uniform(y_min, y_max, size=n)
            z = np.full(n, z_plane)
            X_k = np.column_stack([x, y, z])
            X_list.append(X_k)
            ids_list.append(np.full(n, k, dtype=int))

        X_interface = np.concatenate(X_list, axis=0)
        interface_id = np.concatenate(ids_list, axis=0)
        return X_interface, interface_id

    def sample_boundary_points(self):
            """Returns X_boundary (N, 3) and domain_id (N,)."""
            (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.dimensions
            n = self.boundary_point_density
            X_list = []
            ids_list = []

            # 1. SIDE WALLS (X and Y faces)
            # We slice these vertically so points are assigned to the correct domain ID
            for d, (z_start, z_end) in enumerate(self.domain_bounds):
                # x = x_min and x = x_max
                for x_val in [x_min, x_max]:
                    y = self.rng.uniform(y_min, y_max, size=n)
                    z = self.rng.uniform(z_start, z_end, size=n)
                    X_list.append(np.column_stack([np.full(n, x_val), y, z]))
                    ids_list.append(np.full(n, d, dtype=int))

                # y = y_min and y = y_max
                for y_val in [y_min, y_max]:
                    x = self.rng.uniform(x_min, x_max, size=n)
                    z = self.rng.uniform(z_start, z_end, size=n)
                    X_list.append(np.column_stack([x, np.full(n, y_val), z]))
                    ids_list.append(np.full(n, d, dtype=int))

            # 2. BOTTOM FACE (z = z_min)
            # Always falls in the first domain (Index 0)
            x_bot = self.rng.uniform(x_min, x_max, size=n)
            y_bot = self.rng.uniform(y_min, y_max, size=n)
            X_list.append(np.column_stack([x_bot, y_bot, np.full(n, z_min)]))
            ids_list.append(np.full(n, 0, dtype=int))

            # 3. TOP FACE (z = z_max)
            # Always falls in the last domain
            last_domain_id = len(self.domain_bounds) - 1
            x_top = self.rng.uniform(x_min, x_max, size=n)
            y_top = self.rng.uniform(y_min, y_max, size=n)
            X_list.append(np.column_stack([x_top, y_top, np.full(n, z_max)]))
            ids_list.append(np.full(n, last_domain_id, dtype=int))

            X_boundary = np.concatenate(X_list, axis=0)
            boundary_id = np.concatenate(ids_list, axis=0)
            
            return X_boundary, boundary_id
    
    def normalize_points(self, X):
        """Normalize (N, 3) points by the longest domain dimension.

        X: array of shape (N, 3) with columns [x, y, z].
        Returns:
            X_norm: (N, 3) with coordinates divided by L_max.
        """
        return X / self.L_max
