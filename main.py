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

def GenerateNest(color_map, height_map):

    size_x = len(height_map)
    size_y = len(height_map[0])

    for i in range(1, size_x - 1):
        for j in range(1, size_y - 1):

            valid = True

            # kontrola 3x3 okolia
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:

                    h = height_map[i + dx, j + dy]

                    # musí byť tráva
                    if not (0.15 <= h < 0.5):
                        valid = False

            if valid:

                # vykreslenie mraveniska
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        color_map[i + dx, j + dy] = (0.5, 0.5, 0.5)

                return (i, j)

    return None

# ---------------- ANT ----------------

class Food:
    def __init__(self, x, y):
        self.pos_x = x
        self.pos_y = y

class Ant:
    def __init__(self, height_matrix, nest_pos):
        self.nest_pos = nest_pos

        self.carrying_food = False
        self.height_matrix = height_matrix
        
        # Nájdenie štartu v mravenisku
        self.pos_x, self.pos_y = nest_pos

        # Globálna pamäť mravca, kde už všade fyzicky stál
        self.visited = set()
        self.visited.add((self.pos_x, self.pos_y))
        
        # Množina políčok, ktoré mravec "videl" (susedia navštívených), ale ešte na nich nestál
        self.discovered = set()
        self._discover_neighbors(self.pos_x, self.pos_y)
        
        # Aktuálny plynulý plán cesty (krok za krokom)
        self.current_path = []

    def _discover_neighbors(self, x, y):
        """Pomocná funkcia: mravec sa poobzerá okolo seba a objaví nové políčka."""
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

    def move(self, foods):
                # ---------------- RETURNING HOME ----------------

        if self.carrying_food:

            # ak nemá plán domov, vypočíta ho
            if not self.current_path:
                self.current_path = self._find_path_home()

            # krok po kroku ide domov
            if self.current_path:
                self.pos_x, self.pos_y = self.current_path.pop(0)

            # prišiel do hniezda
            if (self.pos_x, self.pos_y) == self.nest_pos:

                self.carrying_food = False

                self.current_path = []

                return "DELIVERED"
            return None
        # 1. Ak mravec nemá plán, kam urobiť ďalší krok, nájde si najbližší nepreskúmaný cieľ
        if not self.current_path:
            if not self.discovered:
                return  # Celý ostrov je kompletne preskúmaný

            # Použijeme klasické BFS na nájdenie najkratšej cesty k najbližšiemu 'discovered' políčku
            # Tento výpočet prebehne okamžite v pamäti a mravcovi dá plynulú trasu krok za krokom
            path_to_target = self._find_path_to_nearest_unvisited()
            
            if path_to_target:
                self.current_path = path_to_target
            else:
                # Ak sa k cieľu nedá dostať (napr. izolovaný ostrov), odstránime ho a skúsime znova
                self.discovered.clear()
                return

        # 2. FYZICKÝ POHYB: mravec zoberie presne jeden krok z plánu
        if self.current_path:
            self.pos_x, self.pos_y = self.current_path.pop(0)
            self.visited.add((self.pos_x, self.pos_y))
            self.discovered.discard((self.pos_x, self.pos_y))
            self._discover_neighbors(self.pos_x, self.pos_y)

        # 3. Kontrola a zber jedla na aktuálnej pozícii
        for i, food in enumerate(foods):
            if food.pos_x == self.pos_x and food.pos_y == self.pos_y:
                foods.pop(i)

                self.carrying_food = True

                self.current_path = []

                break

    def _find_path_to_nearest_unvisited(self):
        """
        Klasické BFS, ktoré hľadá najkratšiu cestu z aktuálnej pozície mravca
        k najbližšiemu políčku z množiny self.discovered, pričom smie chodiť
        iba po už navštívených políčkach (self.visited) + cieľovom políčku.
        """
        start = (self.pos_x, self.pos_y)
        queue = deque([start])
        parent = {}
        local_visited = {start}
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            curr = queue.popleft()
            
            # Ak sme našli políčko, ktoré sme chceli preskúmať, zrekonštruujeme cestu
            if curr in self.discovered:
                path = []
                while curr != start:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                return path
                
            for dx, dy in directions:
                nx, ny = curr[0] + dx, curr[1] + dy
                
                # Mravec smie pri plánovaní cesty stupiť len na miesta, ktoré už pozná (visited)
                # ALEBO na cieľové nepreskúmané políčko (discovered)
                if (nx, ny) not in local_visited:
                    if (nx, ny) in self.visited or (nx, ny) in self.discovered:
                        local_visited.add((nx, ny))
                        parent[(nx, ny)] = curr
                        queue.append((nx, ny))

    def _find_path_home(self):

        start = (self.pos_x, self.pos_y)

        goal = self.nest_pos

        queue = deque([start])

        parent = {}
        local_visited = {start}

        directions = [(-1,0),(1,0),(0,-1),(0,1)]

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

                nx = curr[0] + dx
                ny = curr[1] + dy

                if (nx, ny) not in local_visited:

                    if (nx, ny) in self.visited:

                        local_visited.add((nx, ny))

                        parent[(nx, ny)] = curr

                        queue.append((nx, ny))

        return []
    
# ---------------- MAIN ----------------

color_matrix, height_matrix = WorldGen(35, 35, 50)

nest_pos = GenerateNest(color_matrix, height_matrix)

nest_tiles = set()

for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        nest_tiles.add((nest_pos[0] + dx, nest_pos[1] + dy))

foods = []
dead_food = []

food_collected = 0
spawn_threshold = 10
# initial food
for _ in range(100):

    while True:
        x = random.randint(0, len(height_matrix)-1)
        y = random.randint(0, len(height_matrix[0])-1)

        if height_matrix[x, y] > 0 and (x, y) not in nest_tiles:
            foods.append(Food(x, y))
            break

ants = [Ant(height_matrix, nest_pos)]

# ---------------- PLOT ----------------

fig, ax = plt.subplots()
ax.imshow(color_matrix)
counter_text = ax.text(
    nest_pos[1],
    nest_pos[0],
    "0",
    color='black',
    ha='center',
    va='center',
    fontsize=10,
    fontweight='bold'
)

food_scatter = ax.scatter(
    [f.pos_y for f in foods],
    [f.pos_x for f in foods],
    c='red',
    s=10
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

            food_collected += 1

            # spawn nového mravca každých 10 jedál
            if food_collected >= spawn_threshold:

                ants.append(Ant(height_matrix, nest_pos))

                spawn_threshold += 10

    # ---------------- RESPAWN SYSTEM ----------------
    new_dead = []

    for t in dead_food:
        t -= 1

        if t <= 0:

            # spawn nové jedlo
            while True:
                x = random.randint(0, len(height_matrix)-1)
                y = random.randint(0, len(height_matrix[0])-1)

                if height_matrix[x, y] > 0 and (x, y) not in nest_tiles:
                    foods.append(Food(x, y))
                    break
        else:
            new_dead.append(t)

    dead_food[:] = new_dead

    # ---------------- RENDER ----------------
    positions = []
    colors = []

    for ant in ants:

        positions.append([ant.pos_y, ant.pos_x])

        if ant.carrying_food:
            colors.append('magenta')
        else:
            colors.append('orange')

    ant_plot.set_offsets(positions)
    ant_plot.set_color(colors)

    food_scatter.set_offsets([
        [f.pos_y, f.pos_x] for f in foods
    ])

    counter_text.set_text(str(food_collected))

    return ant_plot, food_scatter


ani = animation.FuncAnimation(
    fig,
    update,
    interval=20,
    blit=False,
    cache_frame_data=False
)

plt.show()
