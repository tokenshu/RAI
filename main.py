from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from collections import deque

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

def GenerateNest(color_map, height_map, existing_nests=[], nest_color=(0.5,0.5,0.5)):

    size_x = len(height_map)
    size_y = len(height_map[0])

    valid_positions = []

    for i in range(1, size_x - 1):
        for j in range(1, size_y - 1):

            valid = True

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:

                    h = height_map[i + dx, j + dy]

                    if not (0.15 <= h < 0.5):
                        valid = False

            if valid:
                valid_positions.append((i, j))

    # prvé hniezdo náhodne
    if len(existing_nests) == 0:
        nest = random.choice(valid_positions)

    else:
        best_pos = None
        best_distance = -1

        for pos in valid_positions:

            min_dist = min(
                abs(pos[0] - nest[0]) + abs(pos[1] - nest[1])
                for nest in existing_nests
            )

            if min_dist > best_distance:
                best_distance = min_dist
                best_pos = pos

        nest = best_pos

    # vykreslenie hniezda
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            color_map[nest[0] + dx, nest[1] + dy] = nest_color

    return nest

# ---------------- ANT ----------------

class Food:
    def __init__(self, x, y):
        self.pos_x = x
        self.pos_y = y

class Ant:
    def __init__(self, height_matrix, nest_pos, colony_type):
        self.nest_pos = nest_pos
        self.colony_type = colony_type
        self.carrying_food = False
        self.height_matrix = height_matrix
        
        self.pos_x, self.pos_y = nest_pos

        # Vizuálne spomalenie: koľko frameov musí mravec čakať na aktuálnom políčku
        self.wait_ticks = 0

        self.visited = set()
        self.visited.add((self.pos_x, self.pos_y))
        
        self.discovered = set()
        self._discover_neighbors(self.pos_x, self.pos_y)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                self.visited.add((nest_pos[0] + dx, nest_pos[1] + dy))
        
        self.current_path = []

    def _discover_neighbors(self, x, y):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < len(self.height_matrix)
                and 0 <= ny < len(self.height_matrix[0])
                and self.height_matrix[nx, ny] > 0
                and (nx, ny) not in self.visited
            ):
                self.discovered.add((nx, ny))

    def terrain_weight(self, x, y):
        h = self.height_matrix[x, y]
        if h < 0.15: return 3   # sand
        elif h < 0.5: return 1  # grass
        elif h < 0.7: return 6  # mountain
        else: return 10         # snow

    def move(self, foods):
        # Ak mravec čaká kvôli náročnému terénu, znížime počítadlo a nepohneme sa
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return None

        # --- Logika pohybu s jedlom ---
        if self.carrying_food:
            if not self.current_path:
                self.current_path = self._find_path_home()

            if self.current_path:
                self.pos_x, self.pos_y = self.current_path.pop(0)
                # Nastavíme vizuálne spomalenie podľa terénu, na ktorý mravec práve stúpil
                self.wait_ticks = self.terrain_weight(self.pos_x, self.pos_y) - 1

            if (self.pos_x, self.pos_y) == self.nest_pos:
                self.carrying_food = False
                self.current_path = []
                return "DELIVERED"
            return None

        # --- Logika hľadania jedla (prieskum) ---
        if not self.current_path:
            if not self.discovered:
                return

            path_to_target = self._find_path_to_nearest_unvisited()
            
            if path_to_target:
                self.current_path = path_to_target
            else:
                self.discovered.clear()
                return

        if self.current_path:
            self.pos_x, self.pos_y = self.current_path.pop(0)
            # Nastavíme vizuálne spomalenie podľa terénu
            self.wait_ticks = self.terrain_weight(self.pos_x, self.pos_y) - 1
            
            self.visited.add((self.pos_x, self.pos_y))
            self.discovered.discard((self.pos_x, self.pos_y))
            self._discover_neighbors(self.pos_x, self.pos_y)

        # Kontrola, či našiel jedlo
        for i, food in enumerate(foods):
            if food.pos_x == self.pos_x and food.pos_y == self.pos_y:
                foods.pop(i)
                self.carrying_food = True
                self.current_path = []
                break

    # ČISTÝ BFS ALGORITMUS PRE PRIESKUM
    def _find_path_to_nearest_unvisited(self):
        start = (self.pos_x, self.pos_y)
        queue = deque([start])
        
        parent = {}
        visited_in_bfs = {start}
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            curr = queue.popleft()
            
            if curr in self.discovered:
                path = []
                while curr != start:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                return path
                
            for dx, dy in directions:
                nx, ny = curr[0] + dx, curr[1] + dy
                neighbor = (nx, ny)
                
                if neighbor not in visited_in_bfs:
                    if neighbor in self.visited or neighbor in self.discovered:
                        visited_in_bfs.add(neighbor)
                        parent[neighbor] = curr
                        queue.append(neighbor)
                        
        return []

    # ČISTÝ BFS ALGORITMUS PRE CESTU DOMOV
    def _find_path_home(self):
        start = (self.pos_x, self.pos_y)
        goal = self.nest_pos

        queue = deque([start])
        parent = {}
        visited_in_bfs = {start}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            curr = queue.popleft()

            if curr == goal:
                path = []
                while curr != start:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                return path

            for dx, dy in directions:
                nx, ny = curr[0] + dx, curr[1] + dy
                neighbor = (nx, ny)

                if neighbor in self.visited and neighbor not in visited_in_bfs:
                    visited_in_bfs.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)

        return []

# ---------------- MAIN ----------------

color_matrix, height_matrix = WorldGen(45, 45, 50)

nests = []

bfs_nest = GenerateNest(color_matrix, height_matrix, nests, nest_color=(0.5, 0.5, 0.5))
nests.append(bfs_nest)

dfs_nest = GenerateNest(color_matrix, height_matrix, nests, nest_color=(0.5, 0.5, 0))
nests.append(dfs_nest)

astar_nest = GenerateNest(color_matrix, height_matrix, nests, nest_color=(0.5, 0, 0.5))
nests.append(astar_nest)

nest_tiles = set()

for nest in [bfs_nest, dfs_nest, astar_nest]:

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:

            nest_tiles.add((nest[0] + dx, nest[1] + dy))

foods = []
dead_food = []

food_collected = {
    "BFS": 0,
    "DFS": 0,
    "ASTAR": 0
}

spawn_threshold = {
    "BFS": 10,
    "DFS": 10,
    "ASTAR": 10
}

for _ in range(100):
    while True:
        x = random.randint(0, len(height_matrix)-1)
        y = random.randint(0, len(height_matrix[0])-1)
        if height_matrix[x, y] > 0 and (x, y) not in nest_tiles:
            foods.append(Food(x, y))
            break

ants = []

ants.append(Ant(height_matrix, bfs_nest, "BFS"))
ants.append(Ant(height_matrix, dfs_nest, "DFS"))
ants.append(Ant(height_matrix, astar_nest, "ASTAR"))

# ---------------- PLOT ----------------

fig, ax = plt.subplots()
ax.imshow(color_matrix)
bfs_text = ax.text(
    bfs_nest[1], bfs_nest[0],
    "0", color='black',
    ha='center', va='center',
    fontsize=10, fontweight='bold'
)

dfs_text = ax.text(
    dfs_nest[1], dfs_nest[0],
    "0", color='black',
    ha='center', va='center',
    fontsize=10, fontweight='bold'
)

astar_text = ax.text(
    astar_nest[1], astar_nest[0],
    "0", color='black',
    ha='center', va='center',
    fontsize=10, fontweight='bold'
)

food_scatter = ax.scatter(
    [f.pos_y for f in foods], [f.pos_x for f in foods],
    c='red', s=10
)

ant_plot = ax.scatter([], [], s=30, marker='s')
ax.axis('off')

# ---------------- UPDATE ----------------

def update(frame):
    global food_collected
    global spawn_threshold

    for ant in ants:
        result = ant.move(foods)
        if result == "DELIVERED":

            colony = ant.colony_type
            food_collected[colony] += 1

            if food_collected[colony] >= spawn_threshold[colony]:

                if colony == "BFS":
                    nest = bfs_nest
                elif colony == "DFS":
                    nest = dfs_nest
                else:
                    nest = astar_nest

                ants.append(Ant(height_matrix, nest, colony))

                spawn_threshold[colony] += 10

    # Render
    positions = []
    colors = []

    for ant in ants:
        positions.append([ant.pos_y, ant.pos_x])
        if ant.colony_type == "BFS":
            base_color = 'orange'

        elif ant.colony_type == "DFS":
            base_color = 'blue'

        else:
            base_color = 'green'

        if ant.carrying_food:
            colors.append('magenta')
        else:
            colors.append(base_color)

    ant_plot.set_offsets(positions)
    ant_plot.set_color(colors)

    food_scatter.set_offsets([[f.pos_y, f.pos_x] for f in foods])
    bfs_text.set_text(str(food_collected["BFS"]))
    dfs_text.set_text(str(food_collected["DFS"]))
    astar_text.set_text(str(food_collected["ASTAR"]))

    return ant_plot, food_scatter

ani = animation.FuncAnimation(
    fig, update, interval=10, blit=False, cache_frame_data=False
)

plt.show()
