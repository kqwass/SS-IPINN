""
domain_random_seed = 20

dimensions = [(0,2000),(0,1000),(0,200)]                # x,y,z


"""Interface planes are defined from the bottom, 10 means 10 units from bottom which means
that the top most boundary is at the position z=zmax

Note: Number of interface planes = Number of domains - 1  """
interface_planes = [20,50,80,150]

#Number of activation functions = Number of domains
activation = ['swish','tanh','swish','tanh','swish']    #Only activation functions required

#Domain and interface points
domain_points = [100,100,300,400,500]  #Domain 1 is the bottom most domain
interface_number_points = [30,20,40,30]

"""The boundary_point_density is the number of points in the top or bottom boundary. The other boundaries will
have similar density as the top and bottom"""

boundary_point_density = 100   

#Hyperparameters

layer_sizes = [3,5,5,5,1]
param_scale = 0.1
step_size = 1e-3
train_iters = 10000
step = step_size

#Analytical Solutions

aly_soln = ['((1/8)*(x**2) + (y) + (z))',
            '(1/6)*((x**2)+(y**2)+((z)**2))',
            '((1)*(x*y + y*z + (1/4)*(z**2)))',
            '((1)*(x + y**2 + z))',
            '((1/32)*((y**2) + (z**2)))']

aly_soln_grad = ['1',
                 'z/3',
                 'y + z/2',
                 '1',
                 'z/16']


kappa_values = [4.0,1.0,2.0,0.5,8.0]        #\kappa values

forcing_functions = [1,1,1,1,1]     #forcing_functions

""