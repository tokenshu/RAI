from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

# ---------------- WORLD ----------------

def WorldGen(size_x, size_y, seed_id):

    color_map = np.zeros((size_x, size_y, 3))
    height_map = np.zeros((size_x, size_y))

    noise1 = PerlinNoise(octaves=3, seed=seed_id)
    noise2 = PerlinNoise(octaves=6, seed=seed_id)

    for i in range(size_x):
        for j in range(size_y):

            noise_val = noise1([i / 50, j / 50])
            noise_val += noise2([i / 50, j / 50]) + 0.2

            height_map[i, j] = noise_val

            if noise_val < 0:
                color_map[i, j] = (0, 0, 1)  # water
            elif noise_val < 0.15:
                color_map[i, j] = (1, 1, 0)  # sand
            elif noise_val < 0.5:
                color_map[i, j] = (0, 1, 0)  # grass
            elif noise_val < 0.7:
                color_map[i, j] = (0.5, 0.3, 0.1)  # mountain
            else:
                color_map[i, j] = (1, 1, 1)  # snow

    return color_map, height_map


# ---------------- FOOD ----------------

class Food:
    def __init__(self, x, y):
        self.pos_x = x
        self.pos_y = y


# ---------------- ANT ----------------

class Ant:

    def __init__(self, height_matrix):

        self.height_matrix = height_matrix

        while True:
            x = random.randint(0, len(height_matrix) - 1)
            y = random.randint(0, len(height_matrix[0]) - 1)

            if height_matrix[x, y] > 0:
                self.pos_x = x
                self.pos_y = y
                break

    def move(self, foods, dead_food):

        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        dx, dy = random.choice(directions)

        new_x = self.pos_x + dx
        new_y = self.pos_y + dy

        # pohyb
        if (
            0 <= new_x < len(self.height_matrix)
            and 0 <= new_y < len(self.height_matrix[0])
            and self.height_matrix[new_x, new_y] > 0
        ):
            self.pos_x = new_x
            self.pos_y = new_y

        # ---------------- EAT FOOD ----------------
        for i, food in enumerate(foods):

            if food.pos_x == self.pos_x and food.pos_y == self.pos_y:

                foods.pop(i)
                dead_food.append(50)   # cooldown
                break


# ---------------- MAIN ----------------

color_matrix, height_matrix = WorldGen(50, 50, 50)

foods = []
dead_food = []

# initial food
for _ in range(100):

    while True:
        x = random.randint(0, len(height_matrix)-1)
        y = random.randint(0, len(height_matrix[0])-1)

        if height_matrix[x, y] > 0:
            foods.append(Food(x, y))
            break

ant = Ant(height_matrix)

# ---------------- PLOT ----------------

fig, ax = plt.subplots()
ax.imshow(color_matrix)

food_scatter = ax.scatter(
    [f.pos_y for f in foods],
    [f.pos_x for f in foods],
    c='red',
    s=10
)

ant_plot = ax.scatter(
    ant.pos_y,
    ant.pos_x,
    c='orange',
    s=30,
    marker='s'
)

ax.axis('off')


# ---------------- UPDATE ----------------

def update(frame):

    ant.move(foods, dead_food)

    # ---------------- RESPAWN SYSTEM ----------------
    new_dead = []

    for t in dead_food:
        t -= 1

        if t <= 0:

            # spawn nové jedlo
            while True:
                x = random.randint(0, len(height_matrix)-1)
                y = random.randint(0, len(height_matrix[0])-1)

                if height_matrix[x, y] > 0:
                    foods.append(Food(x, y))
                    break
        else:
            new_dead.append(t)

    dead_food[:] = new_dead

    # ---------------- RENDER ----------------
    ant_plot.set_offsets([[ant.pos_y, ant.pos_x]])

    food_scatter.set_offsets([
        [f.pos_y, f.pos_x] for f in foods
    ])

    return ant_plot, food_scatter


ani = animation.FuncAnimation(
    fig,
    update,
    interval=200,
    blit=False,
    cache_frame_data=False
)

plt.show()
