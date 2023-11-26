import random
import numpy as np
from matplotlib import pyplot as plotting_my_graph

#Given to create the 1 million Random_data_values
#We need to check for the 100 queries
#We need to do for the Three type of metric distances
norm_types = ['L1', 'L2', 'Linf']
near_distances, far_distances = {norm: [] for norm in norm_types}, {norm: [] for norm in norm_types}
#We need to do for the following Dimensions
dimensions = [1, 2, 4, 8, 16, 32, 64]
#We need to do for the folloing precision limit
np.set_printoptions(precision=5)
#We are looping for the each dimension
for i in dimensions:
    #get the random data point 
    Random_data_values = np.random.rand(1000000, i)
    #create the temporary arrays to store the 100 randon query resullts
    temp_near,temp_far={norm: [] for norm in norm_types}, {norm: [] for norm in norm_types}
    #The below is the loop for the 100 queries
    for _ in range(100):
        idx = random.randint(0, 999999)
        differences = Random_data_values - Random_data_values[idx]

        norms = {
            'L1': np.linalg.norm(differences, ord=1, axis=1),
            'L2': np.linalg.norm(differences, ord=2, axis=1),
            'Linf': np.linalg.norm(differences, ord=np.inf, axis=1),
        }

        for norm_type in norm_types:
            sorted_norms = np.sort(norms[norm_type])
            if sorted_norms[0] == 0:
                near_distance=sorted_norms[1]
            else :
                near_distance=sorted_norms[0]
            temp_near[norm_type].append(near_distance)
            temp_far[norm_type].append(norms[norm_type][999999])
    for norm_type in norm_types:
        near_distances[norm_type].append(sum(temp_near[norm_type]) / len(temp_near[norm_type]))
        far_distances[norm_type].append(sum(temp_far[norm_type]) / len(temp_far[norm_type]))
 
#The below is the code for the printing the graph
for norm_type in norm_types:
    near_key, far_key = f'{norm_type}_y1', f'{norm_type}_y2'

    plotting_my_graph.clf()
    plotting_my_graph.plot(dimensions, near_distances[norm_type], label='Near Distance', color='pink')
    plotting_my_graph.plot(dimensions, far_distances[norm_type], label='Far Distance', color='blue')
    plotting_my_graph.xlabel('Dimensions (d)')
    plotting_my_graph.ylabel(f'{norm_type.capitalize()} Norm Distance')
    plotting_my_graph.title(f'Nearest and Farthest distances vs. Dimensions ({norm_type.capitalize()} Norm)')
    plotting_my_graph.legend()
    plotting_my_graph.savefig(f'{norm_type}_output.png')
