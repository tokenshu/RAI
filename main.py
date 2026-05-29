from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from collections import deque
import heapq  # POTŘEBNÉ PRO PRIORITNÍ FRONTU V A*

MAP_SIZE = 55
FOOD_QUANTITY = 150

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
                            if target.hp <= 0:
                                stats[ant.colony_type]["kills"] += 1

                            ant.combat_lock = 15

    # 3. Krok: Správa padlých mravcov (Zaloguje smrť KAŽDÉHO mravca)
    for ant in ants:
        if ant.hp <= 0:
            stats[ant.colony_type]["deaths"] += 1
            if ant.carrying_food:
                print(f"PADOL! Mravec {ant.colony_type}#{ant.ant_id} padol v boji a pustil jedlo na [{ant.pos_x}, {ant.pos_y}]")
                foods.append(Food(ant.pos_x, ant.pos_y))
                ant.carrying_food = False  # Ošetrenie proti duplicitnému dropu
            else:
                # FIX 3: Výpis aj pre mravca, ktorý jedlo neniesol
                print(f"PADOL! Mravec {ant.colony_type}#{ant.ant_id} zomrel v boji na [{ant.pos_x}, {ant.pos_y}]")
            
            # Nastavíme HP na hlboké mínus, aby sme ho v ďalšom frame (ak by náhodou prežil filter) nezalogovali znova
            ant.hp = -999

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
        self.wait_ticks = 0

        self.visited = set()
        self.visited.add((self.pos_x, self.pos_y))
        self.discovered = set()
        self._discover_neighbors(self.pos_x, self.pos_y)

        self.food_pickup_time = None
        self.total_path_cost = 0
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                self.visited.add((nest_pos[0] + dx, nest_pos[1] + dy))
        
        self.current_path = []

        # PRIRADENIE STRATEGIE PODLA TYPU KOLONIE
        if self.colony_type == "BFS":
            self.strategy = BfsStrategy()
        elif self.colony_type == "DFS":
            self.strategy = BfsStrategy() # zmeniť na DFS po jej pridaní
        elif self.colony_type == "ASTAR":
            self.strategy = AStarStrategy() # ZMENENÉ NA ASTARSTRATEGY

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
                stats[self.colony_type]["tiles_discovered"].add((nx, ny))

    def terrain_weight(self, x, y):
        h = self.height_matrix[x, y]
        if h < 0.15: return 3   # sand
        elif h < 0.5: return 1  # grass
        elif h < 0.7: return 6  # mountain
        else: return 10         # snow

    def move(self, foods, frame):
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return None

        # --- Logika pohybu s jedlom ---
        if self.carrying_food:
            if not self.current_path:
                # Volanie strategie pre navrat domov
                self.current_path = self.strategy.find_path_home(self)

            if self.current_path:
                self.pos_x, self.pos_y = self.current_path.pop(0)

                terrain_cost = self.terrain_weight(self.pos_x, self.pos_y)

                self.wait_ticks = terrain_cost - 1
                self.total_path_cost += terrain_cost

            if (self.pos_x, self.pos_y) == self.nest_pos:
                self.carrying_food = False
                self.current_path = []
                return "DELIVERED"
            return None

        # --- Logika hladania jedla (prieskum) ---
        if not self.current_path:
            if not self.discovered:
                return

            # Volanie strategie pre najblizsie nepreskumane policko
            path_to_target = self.strategy.find_path_to_unvisited(self)
            
            if path_to_target:
                self.current_path = path_to_target
            else:
                self.discovered.clear()
                return

        if self.current_path:
            self.pos_x, self.pos_y = self.current_path.pop(0)
            self.wait_ticks = self.terrain_weight(self.pos_x, self.pos_y) - 1
            terrain_cost = self.terrain_weight(self.pos_x, self.pos_y)

            self.wait_ticks = terrain_cost - 1
            self.total_path_cost += terrain_cost
            
            self.visited.add((self.pos_x, self.pos_y))
            stats[self.colony_type]["moves"] += 1
            self.discovered.discard((self.pos_x, self.pos_y))
            self._discover_neighbors(self.pos_x, self.pos_y)

        # Kontrola jedla
        for i, food in enumerate(foods):
            if food.pos_x == self.pos_x and food.pos_y == self.pos_y:
                foods.pop(i)
                self.carrying_food = True
                self.food_pickup_time = frame
                self.current_path = []
                break
    
# ---------------- PATHFINDING STRATEGIES ----------------

class PathfindingStrategy:
    """Base trieda pre vsetky vyhladavacie algoritmy (Interface)"""
    def find_path_to_unvisited(self, ant):
        raise NotImplementedError

    def find_path_home(self, ant):
        raise NotImplementedError


class BfsStrategy(PathfindingStrategy):
    """Tvoj povodny, plne funkcny BFS algoritmus"""
    def find_path_to_unvisited(self, ant):
        start = (ant.pos_x, ant.pos_y)
        queue = deque([start])
        parent = {}
        visited_in_bfs = {start}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            curr = queue.popleft()
            if curr in ant.discovered:
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
                    if neighbor in ant.visited or neighbor in ant.discovered:
                        visited_in_bfs.add(neighbor)
                        parent[neighbor] = curr
                        queue.append(neighbor)
        return []

    def find_path_home(self, ant):
        start = (ant.pos_x, ant.pos_y)
        goal = ant.nest_pos
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
                if neighbor in ant.visited and neighbor not in visited_in_bfs:
                    visited_in_bfs.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)
        return []


class DfsStrategy(PathfindingStrategy):
    """MIESTO PRE KAMARATA 1 (DFS)"""
    def find_path_to_unvisited(self, ant):
        # TODO: Sem napisat cisty algorytmus pre DFS prieskum
        # Mozes vyuzivat: ant.pos_x, ant.pos_y, ant.discovered, ant.visited, ant.height_matrix
        # Na konci musis vratit zoznam tuplov: [(x1, y1), (x2, y2)...]
        return []

    def find_path_home(self, ant):
        # TODO: Sem napisat cisty DFS algoritmus pre cestu domov (ant.nest_pos).
        return []


class AStarStrategy(PathfindingStrategy):
    """MIESTO PRE KAMARATA 2 (A*) - ZABUDOVANÉ"""
    def _heuristic(self, p1, p2):
        # Manhattanova vzdialenost (vzdialenost v mriezke bez diagonal)
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def find_path_to_unvisited(self, ant):
        start = (ant.pos_x, ant.pos_y)
        
        # Prvky vo fronte: (f_score, g_score, aktualna_pozicia)
        # Na zaciatku prieskumu nie je jeden fixny ciel, heuristika je 0 (Dijkstra)
        queue = [(0, 0, start)]
        heapq.heapify(queue)
        
        parent = {}
        g_score = {start: 0}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            _, current_g, curr = heapq.heappop(queue)
            
            # Nasli sme najblizsi a najlacnejsi nepreskumany bod
            if curr in ant.discovered:
                path = []
                while curr != start:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                return path
                
            for dx, dy in directions:
                nx, ny = curr[0] + dx, curr[1] + dy
                neighbor = (nx, ny)
                
                # Moze prejst len cez to, co uz pozna alebo prave vidi
                if neighbor in ant.visited or neighbor in ant.discovered:
                    weight = ant.terrain_weight(nx, ny)
                    tentative_g = current_g + weight
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = curr
                        f_score = tentative_g  # h=0, kedze ciele sa dynamicky menia
                        heapq.heappush(queue, (f_score, tentative_g, neighbor))
        return []

    def find_path_home(self, ant):
        start = (ant.pos_x, ant.pos_y)
        goal = ant.nest_pos

        # Prvky vo fronte: (f_score, g_score, aktualna_pozicia)
        queue = [(self._heuristic(start, goal), 0, start)]
        heapq.heapify(queue)
        
        parent = {}
        g_score = {start: 0}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            _, current_g, curr = heapq.heappop(queue)

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

                # Pri ceste domov chodi striktne po tom, co uz bolo zmapovane
                if neighbor in ant.visited:
                    weight = ant.terrain_weight(nx, ny)
                    tentative_g = current_g + weight

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = curr
                        f_score = tentative_g + self._heuristic(neighbor, goal)
                        heapq.heappush(queue, (f_score, tentative_g, neighbor))
        return []

# ---------------- MAIN ----------------

color_matrix, height_matrix = WorldGen(MAP_SIZE, MAP_SIZE, random.randint(0,10000))

walkable_tiles = np.sum(height_matrix > 0)

nests = []

bfs_nest = GenerateNest(color_matrix, height_matrix, nests, nest_color=(0.7, 0.42, 0))
nests.append(bfs_nest)

dfs_nest = GenerateNest(color_matrix, height_matrix, nests, nest_color=(0.0, 0.0, 0.7))
nests.append(dfs_nest)

astar_nest = GenerateNest(color_matrix, height_matrix, nests, nest_color=(0, 0.4, 0))
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

stats = {
    "BFS": {
        "food_delivered": 0,
        "tiles_discovered": set(),
        "return_times": [],
        "deaths": 0,
        "kills": 0,
        "path_costs": [],
        "moves": 0
    },

    "DFS": {
        "food_delivered": 0,
        "tiles_discovered": set(),
        "return_times": [],
        "deaths": 0,
        "kills": 0,
        "path_costs": [],
        "moves": 0
    },

    "ASTAR": {
        "food_delivered": 0,
        "tiles_discovered": set(),
        "return_times": [],
        "deaths": 0,
        "kills": 0,
        "path_costs": [],
        "moves": 0
    }
}

spawn_threshold = {
    "BFS": 10,
    "DFS": 10,
    "ASTAR": 10
}

for _ in range(FOOD_QUANTITY):
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

fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(right=0.72)

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
bfs_stats_text = fig.text(
    0.75, 0.75,
    "",
    fontsize=9,
    va='top',
    bbox=dict(facecolor='orange', alpha=0.25)
)
dfs_stats_text = fig.text(
    0.75, 0.55,
    "",
    fontsize=9,
    va='top',
    bbox=dict(facecolor='blue', alpha=0.2)
)
astar_stats_text = fig.text(
    0.75, 0.35,
    "",
    fontsize=9,
    va='top',
    bbox=dict(facecolor='green', alpha=0.2)
)

ax.axis('off')

# ---------------- UPDATE ----------------

def update(frame):
    global food_collected
    global spawn_threshold

    for ant in ants:
        result = ant.move(foods, frame)
        if result == "DELIVERED":

            colony = ant.colony_type

            stats[colony]["food_delivered"] += 1

            if ant.food_pickup_time is not None:
                return_time = frame - ant.food_pickup_time
                stats[colony]["return_times"].append(return_time)

            stats[colony]["path_costs"].append(ant.total_path_cost)

            ant.total_path_cost = 0
            ant.food_pickup_time = None

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

    if foods:
        food_scatter.set_offsets([[f.pos_y, f.pos_x] for f in foods])
    else:
        # Pokud jídlo došlo, předáme prázdné 2D pole, což Matplotlib schválí a tečky zmizí
        food_scatter.set_offsets(np.empty((0, 2)))

    bfs_text.set_text(str(food_collected["BFS"]))
    dfs_text.set_text(str(food_collected["DFS"]))
    astar_text.set_text(str(food_collected["ASTAR"]))

    def colony_report(colony):

        s = stats[colony]

        avg_return = (
            sum(s["return_times"]) / len(s["return_times"])
            if s["return_times"] else 0
        )

        avg_cost = (
            sum(s["path_costs"]) / len(s["path_costs"])
            if s["path_costs"] else 0
        )

        coverage = (
            len(s["tiles_discovered"]) / walkable_tiles * 100
        )

        efficiency = 1 / avg_cost if avg_cost > 0 else 0

        return (
            f"{colony}\n"
            f"Food: {s['food_delivered']}\n"
            f"Tiles: {len(s['tiles_discovered'])}\n"
            f"Avg return: {avg_return:.1f}\n"
            f"Deaths: {s['deaths']}\n"
            f"Kills: {s['kills']}\n"
            f"Avg cost: {avg_cost:.1f}\n"
            f"Efficiency: {efficiency:.2f}\n"
            f"Coverage: {coverage:.1f}%"
        )

    bfs_stats_text.set_text(colony_report("BFS"))
    dfs_stats_text.set_text(colony_report("DFS"))
    astar_stats_text.set_text(colony_report("ASTAR"))

    return ant_plot, food_scatter

ani = animation.FuncAnimation(
    fig, update, interval=10, blit=False, cache_frame_data=False
)

plt.show()
