from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt
import numpy as np
import random

def WorldGen(size_x, size_y, seed_id):
    color_map = np.zeros((size_x,size_y, 3))
    height_map = np.zeros((size_x,size_y, 3))
    noise1 = PerlinNoise(octaves=3, seed=seed_id)
    noise2 = PerlinNoise(octaves=6, seed=seed_id) 
    for i in range(size_x):
        for j in range(size_y):
            noise_val = noise1([i/50, j/50])
            noise_val += noise2([i/50, j/50]) + 0.2
            height_map[i,j] = noise_val
            if noise_val < 0:
                color_map[i,j] = (0, 0, 1) #blue
            elif noise_val < 0.15:
                color_map[i,j] = (1, 1, 0) #yellow 
            elif noise_val < 0.5:
                color_map[i,j] = (0, 1, 0)#green
            elif noise_val < 0.7:
                color_map[i,j] = (0.5, 0.3, 0.1) #brown
            else:
                color_map[i,j] = (1, 1, 1) #white
    return color_map, height_map

class Ant:

    def __init__(self, height_matrix):
        self.pos_x = random.randint(0,len(height_matrix))
        self.pos_y = random.randint(0,len(height_matrix[0]))


        
[color_matrix, height_matrix] = WorldGen(300,300,50)

print(Ant(height_matrix))

plt.imshow(color_matrix)
plt.axis('off')
plt.show()
