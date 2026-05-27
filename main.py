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

def HandleCombat(ants, foods):
    """
    Opravený bojový systém. Mravce s HP <= 0 už nemôžu útočiť 
    a do konzoly sa vypíše smrť každého jedného mravca.
    """
    position_map = {}

    # 1. Krok: Zoskupenie žijúcich mravcov podľa ich aktuálnych súradníc
    for ant in ants:
        if ant.hp > 0:
            pos = (ant.pos_x, ant.pos_y)
            if pos not in position_map:
                position_map[pos] = []
            position_map[pos].append(ant)

    # 2. Krok: Vyhodnotenie bojov
    for pos, ants_at_pos in position_map.items():
        if len(ants_at_pos) > 1:
            
            first_colony = ants_at_pos[0].colony_type
            has_enemy = any(ant.colony_type != first_colony for ant in ants_at_pos)

            if has_enemy:
                for ant in ants_at_pos:
                    # FIX 1: Ak mravec zomrel počas tohto frame (dostal ranu od niekoho predtým), nemôže útočiť!
                    if ant.hp <= 0:
                        continue
                        
                    if ant.combat_lock == 0:
                        # FIX 2: Útočiť sa dá len na nepriateľov, ktorí ešte ŽIJÚ (majú HP > 0)
                        enemies = [e for e in ants_at_pos if e.colony_type != ant.colony_type and e.hp > 0]
                        
                        if enemies:
                            target = random.choice(enemies)
                            
                            print(f"BOJ! {ant.colony_type}#{ant.ant_id}(HP:{ant.hp}) utoci na {target.colony_type}#{target.ant_id}(HP:{target.hp})")
                            
                            target.hp -= ant.attack
                            ant.combat_lock = 15

    # 3. Krok: Správa padlých mravcov (Zaloguje smrť KAŽDÉHO mravca)
    for ant in ants:
        if ant.hp <= 0:
            if ant.carrying_food:
                print(f"PADOL! Mravec {ant.colony_type}#{ant.ant_id} padol v boji a pustil jedlo na [{ant.pos_x}, {ant.pos_y}]")
                foods.append(Food(ant.pos_x, ant.pos_y))
                ant.carrying_food = False  # Ošetrenie proti duplicitnému dropu
            else:
                # FIX 3: Výpis aj pre mravca, ktorý jedlo neniesol
                print(f"PADOL! Mravec {ant.colony_type}#{ant.ant_id} zomrel v boji na [{ant.pos_x}, {ant.pos_y}]")
            
            # Nastavíme HP na hlboké mínus, aby sme ho v ďalšom frame (ak by náhodou prežil filter) nezalogovali znova
            ant.hp = -999
    # 3. Krok: Sprava padlych mravcov (Drop jedla)
    for ant in ants:
        if ant.hp <= 0 and ant.carrying_food:
            # Bezpecny text bez emoji a diakritiky
            print(f"PADOL! Mravec {ant.colony_type}#{ant.ant_id} padol v boji a pustil jedlo na [{ant.pos_x}, {ant.pos_y}]")
            foods.append(Food(ant.pos_x, ant.pos_y))
            ant.carrying_food = False  # Osetrenie, aby nepushol jedlo viackrat
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

        self.hp = 6
        self.attack = 2

        self.combat_lock = 0
        self.ant_id = random.randint(1, 999)

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

color_matrix, height_matrix = WorldGen(50, 50, random.randint(0,10000))

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

    HandleCombat(ants, foods)           
    for ant in ants:
        if ant.combat_lock > 0:
            ant.combat_lock -= 1
    # 4. CRITICAL: Odstránenie mŕtvych mravcov z poľa PRED RENDEROM!
    ants[:] = [ant for ant in ants if ant.hp > 0]
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
