import numpy as np
import os
#import matplotlib.pyplot as plt
#from hypervolume import HyperVolume

root_dir = "outputs_housing"
file_generation = '{}/generation_number.npy'.format(root_dir)
generation = np.load(file_generation)
# fitness_score_list = []
# active_nodes_list = []
population = []
for gen in range(0, generation+1):
    file_pop = '{}/gen{}_pop.npy'.format(root_dir, gen)
    population += np.load(file_pop).tolist()
scores = []

for individual in population:
    scores.append(individual.fitness.values[0])
sample_best = population[np.random.choice(a=np.where(np.min(scores)==scores)[0], size=1)[0]]
print(sample_best)