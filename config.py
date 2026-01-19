
domain_random_seed = 20

dimensions = [(0,2),(0,2),(0,2)]                # x,y,z


"""Interface planes are defined from the bottom, 10 means 10 units from bottom which means
that the top most boundary is at the position z=zmax

Note: Number of interface planes = Number of domains - 1  """
interface_planes = [0.5,1.5]

#Number of activation functions = Number of domains
activation = ['elu','swish','tanh']    #Only activation functions required

#Domain and interface points
domain_points = [1000,2000,1000]  #Domain 0 is the bottom most domain
interface_number_points = [1000,1000]

"""The boundary_point_density is the number of points in the top or bottom boundary. The other boundaries will
have similar density as the top and bottom"""

boundary_point_density = 1000 

#Hyperparameters

layer_sizes = [3,20,20,1]
param_scale = 0.1
step_size = 1e-3
train_iters = 10000
step = step_size

#Analytical Solutions

aly_soln = ['(x**2) + (y**2) + (z**2)',
            'x*y + y*z + x*z',
            'x+y+z'
            ]

aly_soln_grad = ['2*z',
                 'x+y',
                 '1']


kappa_values = [1.0,1.0,1.0]        #\kappa values

forcing_functions = [6,0,0]     #forcing_functions



