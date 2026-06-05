from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from collections import deque
import heapq  # potřebné pro prioritní frontu v A*

# ---------------- SEED ----------------
## Pro zcela náhodný běh programu zakomentovat tyto 3 řádky:
SEED = 1  # Může se zvolit jakékoliv celé číslo
random.seed(SEED)
np.random.seed(SEED)

# ---------------- PARAMETERS ----------------

MAP_SIZE = 70

FOOD_QUANTITY = 170
FOOD_RESPAWN = False
FOOD_RESPAWN_CHANCE = 0.1
MAX_FOOD = 200

ANIM_SPEED = 1

ANT_HP = 6
ANT_ATTACK = 2
COMBAT_COOLDOWN = 15

VISION_RAD = 10

STARTING_ANTS = 3        # počet mravenců v každé kolonii na začátku
SPAWN_AMOUNT = 4         # kolik mravenců se přidá po každém dosažení prahu
SPAWN_THRESHOLD_STEP = 10 # za každých X přinesených jídel se zvýší spawnovací práh o tuto hodnotu

# ceny terénů
SAND_COST = 3
GRASS_COST = 1
MOUNTAIN_COST = 6
SNOW_COST = 10

SHARE_MEMORY = True
colony_memory = {
    "BFS": {
        "visited": set(),
        "discovered": set(),
        "targeted": set()
    },
    "DFS": {
        "visited": set(),
        "discovered": set(),
        "targeted": set()
    },
    "ASTAR": {
        "visited": set(),
        "discovered": set(),
        "targeted": set()
    }
}

next_ant_id = 1

# ---------------- WORLD ----------------

def WorldGen(size_x, size_y, seed_id):
    # vygeneruje barevnou a výškovou mapu pomocí Perlinova šumu
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

    # první hnízdo umístíme náhodně, další už optimalizovaně s ohledem na stávající hnízda (aby nebyla moc blízko sebe)
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

    # vykreslení hnízda (3x3 blok) do color_map
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            color_map[nest[0] + dx, nest[1] + dy] = nest_color

    return nest

def HandleCombat(ants, foods):
    position_map = {}

    # 1. Krok: Seskupení mravenců podle pozice
    for ant in ants:
        if ant.hp > 0:
            pos = (ant.pos_x, ant.pos_y)
            if pos not in position_map:
                position_map[pos] = []
            position_map[pos].append(ant)

    # 2. Krok: Vyhodnocení boje na pozicích, kde je více než jeden mravenec
    for pos, ants_at_pos in position_map.items():
        if len(ants_at_pos) > 1:
            # kontrola, jestli jsou mravenci z různých kolonií
            first_colony = ants_at_pos[0].colony_type
            has_enemy = any(ant.colony_type != first_colony for ant in ants_at_pos)

            if has_enemy:
                for ant in ants_at_pos:
                    # pokud už je mravenec mrtvý, přeskočíme ho
                    if ant.hp <= 0:
                        continue
                        
                    # mravenec může útočit, jen když mu vypršel cooldown
                    if ant.combat_lock == 0:
                        # seznam mravenců jiných kolonií na stejné pozici, kteří jsou stále naživu
                        enemies = [e for e in ants_at_pos if e.colony_type != ant.colony_type and e.hp > 0]
                        
                        # útok
                        if enemies:
                            target = random.choice(enemies)
                            
                            print(f"BOJ! {ant.colony_type}#{ant.ant_id}(HP:{ant.hp}) utoci na {target.colony_type}#{target.ant_id}(HP:{target.hp})")
                            
                            target.hp -= ant.attack
                            if target.hp <= 0:
                                stats[ant.colony_type]["kills"] += 1

                            ant.combat_lock = COMBAT_COOLDOWN

    # 3. Krok: Správa padlých mravenců (odstranění z cílových rezervací)
    for ant in ants:
        if ant.hp <= 0:
            # aktualizace statistik
            stats[ant.colony_type]["deaths"] += 1

            if ant.carrying_food:
                print(f"ZEMREL! Mravenec {ant.colony_type}#{ant.ant_id} zemrel v boji a pustil jidlo na [{ant.pos_x}, {ant.pos_y}]")
                # položení jídla
                foods.append(Food(ant.pos_x, ant.pos_y))
                ant.carrying_food = False  # Ošetrenie proti duplicitnému dropu
            else:
                print(f"ZEMREL! Mravenec {ant.colony_type}#{ant.ant_id} zemrel v boji na [{ant.pos_x}, {ant.pos_y}]")
            
            # pojistka
            ant.hp = -999

def spawn_food():
    while True:
        x = random.randint(0, len(height_matrix)-1)
        y = random.randint(0, len(height_matrix[0])-1)

        occupied = any(
            food.pos_x == x and food.pos_y == y
            for food in foods
        )

        occupied_by_ant = any(
            ant.pos_x == x and ant.pos_y == y
            for ant in ants
        )

        if (
            height_matrix[x, y] > 0
            and (x, y) not in nest_tiles
            and not occupied
            and not occupied_by_ant
        ):
            foods.append(Food(x, y))
            break

def draw_hp_texts(ax, ants):
    global hp_texts

    # odstranit staré texty HP
    for t in hp_texts:
        t.remove()
    hp_texts.clear()

    for ant in ants:
        txt = ax.text(
            ant.pos_y,
            ant.pos_x,
            str(ant.hp),
            color="white",
            fontsize=6,
            ha="center",
            va="center",
            fontweight="bold"
        )
        hp_texts.append(txt)

# ---------------- ANT ----------------

class Food:
    def __init__(self, x, y):
        self.pos_x = x
        self.pos_y = y

class Ant:
    # inicializace
    def __init__(self, height_matrix, nest_pos, colony_type):
        self.nest_pos = nest_pos
        self.colony_type = colony_type
        self.carrying_food = False
        self.height_matrix = height_matrix
        self.pos_x, self.pos_y = nest_pos

        self.hp = ANT_HP
        self.attack = ANT_ATTACK
        self.vision_radius = VISION_RAD
        self.combat_lock = 0
        global next_ant_id

        self.ant_id = next_ant_id
        next_ant_id += 1
        self.wait_ticks = 0

        # sdílení paměti mezi mravenci stejné kolonie (navštívené, objevené a zacílené pozice)
        if SHARE_MEMORY:
            self.visited = colony_memory[self.colony_type]["visited"]
            self.discovered = colony_memory[self.colony_type]["discovered"]
            self.targeted = colony_memory[self.colony_type]["targeted"]  # PRIDANÉ
        else:
            self.visited = set()
            self.discovered = set()
            self.targeted = set()

        self.targeted_tile = None  # které políčko si tento mravenec zacítil pro průzkum (používá se pro rezervaci)
        self.visited.add((self.pos_x, self.pos_y))
        self._discover_neighbors(self.pos_x, self.pos_y)

        self.food_pickup_time = None
        self.total_path_cost = 0
        
        # označení plochy hnízda jako navštívené
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                tile = (nest_pos[0] + dx, nest_pos[1] + dy)
                self.visited.add(tile)
        
        self.current_path = []

        # přiřazení strategie podle typu kolonie
        if self.colony_type == "BFS":
            self.strategy = BfsStrategy()
        elif self.colony_type == "DFS":
            self.strategy = DfsStrategy()
        elif self.colony_type == "ASTAR":
            self.strategy = AStarStrategy()

    # prozkoumávání okolí
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

    # váhy terénu
    def terrain_weight(self, x, y):
        h = self.height_matrix[x, y]
        if h < 0.15: return SAND_COST   # sand
        elif h < 0.5: return GRASS_COST  # grass
        elif h < 0.7: return MOUNTAIN_COST  # mountain
        else: return SNOW_COST        # snow

    # pohyb mravence
    def move(self, foods, frame):
        # pokud je mrtvý, nehýbe se
        if self.hp <= 0:
            return None

        # pokud je spomalený, čeká
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return None
        
        # 1. HOME MODE (nese jídlo)
        if self.carrying_food:
            # Má naplánovanou cestu domů?
            if not self.current_path:
                # plánování cesty domů
                self.current_path = self.strategy.find_path_home(self)

            # krok domů
            if self.current_path:
                self.pos_x, self.pos_y = self.current_path.pop(0)

                terrain_cost = self.terrain_weight(self.pos_x, self.pos_y)

                self.wait_ticks = terrain_cost - 1
                self.total_path_cost += terrain_cost
                stats[self.colony_type]["moves"] += 1

            # pokud je v hnízdě, odevzdá jídlo a resetuje stav
            if (self.pos_x, self.pos_y) == self.nest_pos:
                self.carrying_food = False
                self.current_path = []
                return "DELIVERED"
            return None

        # 2. FOOD MODE
        visible_food = self.get_visible_food(foods)

        # pokud vidí jídlo, nic nenese a nemá naplánovanou cestu
        if visible_food and not self.carrying_food and not self.current_path:
            # najít nejbližší jídlo (Manhattanova vzdálenost)
            target = min(
                visible_food,
                key=lambda f: abs(f.pos_x - self.pos_x) + abs(f.pos_y - self.pos_y)
            )
            
            # plánování cesty k jídlu
            path_to_food = self.strategy.find_path_to_food(self, target)
            
            if path_to_food:
                # zrušení případné rezervace políčka pro průzkum, nastavení trasy k jídlu
                if self.targeted_tile:
                    self.targeted.discard(self.targeted_tile)
                    self.targeted_tile = None
                self.current_path = path_to_food

        # 3. EXPLORATION MODE
        # pokud nemá naplánovanou cestu
        if not self.current_path:
            # zrušení případné rezervace políčka pro průzkum
            if self.targeted_tile:
                self.targeted.discard(self.targeted_tile)
                self.targeted_tile = None

            # pokud okolo sebe nemá žádná objevená políčka, zůstane stát
            if not self.discovered:
                return

            # plánování cesty k nejbližšímu neobjevenému políčku
            path_to_target = self.strategy.find_path_to_unvisited(self)
            
            if path_to_target:
                self.current_path = path_to_target
                # rezervace cílového políčka pro průzkum, aby ho nezacítil jiný mravenec
                self.targeted_tile = path_to_target[-1]
                self.targeted.add(self.targeted_tile)
            else:
                self.current_path = []
                return

        # PŘESUN PRO 2. A 3. MODE
        if self.current_path:
            self.pos_x, self.pos_y = self.current_path.pop(0)
            terrain_cost = self.terrain_weight(self.pos_x, self.pos_y)

            # zpomalení
            self.wait_ticks = terrain_cost - 1
            self.total_path_cost += terrain_cost
            
            self.visited.add((self.pos_x, self.pos_y))
            stats[self.colony_type]["moves"] += 1
            self.discovered.discard((self.pos_x, self.pos_y))
            
            # dosažení cíle
            if (self.pos_x, self.pos_y) == self.targeted_tile:
                self.targeted.discard(self.targeted_tile)
                self.targeted_tile = None

            # objevení okolí na nové pozici
            self._discover_neighbors(self.pos_x, self.pos_y)

        # kontrola, jestli na pozici není jídlo
        for i, food in enumerate(foods):
            if food.pos_x == self.pos_x and food.pos_y == self.pos_y:
                foods.pop(i)
                self.carrying_food = True
                self.food_pickup_time = frame
                self.current_path = []
                if self.targeted_tile:
                    self.targeted.discard(self.targeted_tile)
                    self.targeted_tile = None
                break
        stats[self.colony_type]["total_energy_spent"] += terrain_cost
            
    def get_visible_food(self, foods):
        visible = []

        for food in foods:
            dist = abs(food.pos_x - self.pos_x) + abs(food.pos_y - self.pos_y)
            if dist <= self.vision_radius:
                visible.append(food)
        return visible
 
# ---------------- PATHFINDING STRATEGIES ----------------

class PathfindingStrategy:
    # abstraktní třída pro algoritmy strategie hledání cesty
    def find_path_to_unvisited(self, ant):
        raise NotImplementedError

    def find_path_home(self, ant):
        raise NotImplementedError
    def find_path_to_food(self, ant, food):
        raise NotImplementedError


class BfsStrategy(PathfindingStrategy):
    def find_path_to_unvisited(self, ant):
            start = (ant.pos_x, ant.pos_y)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            # 1. PRŮCHOD: Prioritně hledáme NEZAREZERVOVANÉ políčko
            # fronta FIFO
            queue = deque([start])
            parent = {}
            visited_in_bfs = {start}
            
            # dokud jsou ve frontě políčka k prozkoumání
            while queue:
                curr = queue.popleft()
                # pokud je v seznamu objevených a nikdo ho nemá rezervované
                if curr in ant.discovered and curr not in ant.targeted:
                    path = []
                    # dokud se zpětným stopováním nedostaneš ke startu, komponuj cestu
                    while curr != start:
                        path.append(curr)
                        curr = parent[curr]
                    path.reverse()
                    return path
                    
                # pokud ještě nemá cíl, prozkoumá sousedy
                for dx, dy in directions:
                    nx, ny = curr[0] + dx, curr[1] + dy
                    neighbor = (nx, ny)
                    # kontrola
                    if (
                    0 <= nx < len(ant.height_matrix) 
                    and 0 <= ny < len(ant.height_matrix[0])
                    and neighbor not in visited_in_bfs
                    and ant.height_matrix[nx, ny] > 0
                    ):
                        # označí za navštívené a přidá do fronty
                        visited_in_bfs.add(neighbor)
                        parent[neighbor] = curr
                        queue.append(neighbor)
                            
            # 2. PRŮCHOD (Fallback): Pokud jsou všechny políčka rezervované, vezme první dostupné
            queue = deque([start])
            parent = {}
            visited_in_bfs = {start}
            
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
            # pokud jsme nalezli hnízdo, sestavíme cestu
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
                # při cestě domů chodí jen po navštívených polích
                if neighbor in ant.visited and neighbor not in visited_in_bfs:
                    visited_in_bfs.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)
        return []
    
    def find_path_to_food(self, ant, food):
        start = (ant.pos_x, ant.pos_y)
        goal = (food.pos_x, food.pos_y)

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

                # k jídlu může jít i přes navštívená i přes objevená políčka
                if (
                    neighbor not in visited_in_bfs
                    and (neighbor in ant.visited or neighbor in ant.discovered)
                ):
                    visited_in_bfs.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)
        return []

class DfsStrategy(PathfindingStrategy):
    def find_path_to_unvisited(self, ant):
        start = (ant.pos_x, ant.pos_y)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 1. PRŮCHOD: Prioritně hledáme NEZAREZERVOVANÉ políčko
        # místo fronty je použitý zásobník stack
        stack = [start]
        parent = {}
        visited_in_dfs = {start}
        
        while stack:
            curr = stack.pop() # ZMĚNA: pop() místo popleft()  
            # pokud je v seznamu objevených a nikdo ho nemá rezervované
            if curr in ant.discovered and curr not in ant.targeted:
                path = []
                # dokud se zpětným stopováním nedostaneš ke startu, komponuj cestu
                while curr != start:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                return path
                
            # pokud ještě nemá cíl, prozkoumá sousedy
            for dx, dy in directions:
                nx, ny = curr[0] + dx, curr[1] + dy
                neighbor = (nx, ny)
                # kontrola
                if (
                    0 <= nx < len(ant.height_matrix) 
                    and 0 <= ny < len(ant.height_matrix[0])
                    and neighbor not in visited_in_dfs
                    and ant.height_matrix[nx, ny] > 0
                ):
                    # označí za navštívené a přidá do fronty
                    visited_in_dfs.add(neighbor)
                    parent[neighbor] = curr
                    stack.append(neighbor)
                        
        # 2. PRŮCHOD (Fallback): Pokud jsou všechny políčka rezervované, vezme první dostupné
        stack = [start]
        parent = {}
        visited_in_dfs = {start}
        
        while stack:
            curr = stack.pop()
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
                if neighbor not in visited_in_dfs:
                    if neighbor in ant.visited or neighbor in ant.discovered:
                        visited_in_dfs.add(neighbor)
                        parent[neighbor] = curr
                        stack.append(neighbor)
        return []

    def find_path_home(self, ant):
        start = (ant.pos_x, ant.pos_y)
        goal = ant.nest_pos
        
        stack = [start]
        parent = {}
        visited_in_dfs = {start}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while stack:
            curr = stack.pop()
            # pokud jsme nalezli hnízdo, sestavíme cestu
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
                # při cestě domů chodí jen po navštívených polích
                if neighbor in ant.visited and neighbor not in visited_in_dfs:
                    visited_in_dfs.add(neighbor)
                    parent[neighbor] = curr
                    stack.append(neighbor)  
        return []
    
    def find_path_to_food(self, ant, food):
        start = (ant.pos_x, ant.pos_y)
        goal = (food.pos_x, food.pos_y)

        stack = [start]
        parent = {}
        visited_in_dfs = {start}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while stack:
            curr = stack.pop()
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

                # k jídlu může jít i přes navštívená i přes objevená políčka
                if (
                    neighbor not in visited_in_dfs
                    and (neighbor in ant.visited or neighbor in ant.discovered)
                ):
                    visited_in_dfs.add(neighbor)
                    parent[neighbor] = curr
                    stack.append(neighbor)
        return []

class AStarStrategy(PathfindingStrategy):
    def _heuristic(self, p1, p2):
        # Manhattanova vzdálenost jako heuristika pro A*
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def find_path_to_unvisited(self, ant):
        start = (ant.pos_x, ant.pos_y)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 1. PRŮCHOD: Prioritně hledáme NEZAREZERVOVANÉ políčko
        # prioritní fronta
        queue = [(0, 0, start)]
        heapq.heapify(queue)
        parent = {}
        g_score = {start: 0}
        
        # dokud jsou ve frontě políčka k prozkoumání
        while queue:
            # vem políčko s nejnižší cenou
            _, current_g, curr = heapq.heappop(queue)
            # pokud je v seznamu objevených a nikdo ho nemá rezervované
            if curr in ant.discovered and curr not in ant.targeted:
                path = []
                # dokud se zpětným stopováním nedostaneš ke startu, komponuj cestu
                while curr != start:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                return path
                
            # pokud ještě nemá cíl, prozkoumá sousedy
            for dx, dy in directions:
                nx, ny = curr[0] + dx, curr[1] + dy
                neighbor = (nx, ny)
                # kontrola
                if (
                    0 <= nx < len(ant.height_matrix)
                    and 0 <= ny < len(ant.height_matrix[0])
                    and ant.height_matrix[nx, ny] > 0
                ):
                    # výpočet ceny
                    weight = ant.terrain_weight(nx, ny)
                    tentative_g = current_g + weight
                    
                    # pokud je levnější než současná cesta, nebo jsme ho ještě nenavštívili, zapíše do fronty
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = curr
                        heapq.heappush(queue, (tentative_g, tentative_g, neighbor))
                        
        # 2. PRŮCHOD (Fallback): Pokud jsou všechny políčka rezervované, vezme první dostupné
        queue = [(0, 0, start)]
        heapq.heapify(queue)
        parent = {}
        g_score = {start: 0}
        
        while queue:
            _, current_g, curr = heapq.heappop(queue)
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
                if neighbor in ant.visited or neighbor in ant.discovered:
                    weight = ant.terrain_weight(nx, ny)
                    tentative_g = current_g + weight
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = curr
                        heapq.heappush(queue, (tentative_g, tentative_g, neighbor))
        return []

    def find_path_home(self, ant):
        start = (ant.pos_x, ant.pos_y)
        goal = ant.nest_pos

        # prvky ve frontě (f_score, g_score, pozice)
        queue = [(self._heuristic(start, goal), 0, start)]
        heapq.heapify(queue)
        
        parent = {}
        g_score = {start: 0}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            _, current_g, curr = heapq.heappop(queue)

            # pokud jsme nalezli hnízdo, sestavíme cestu
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

                # při cestě domů chodí jen po navštívených polích
                if neighbor in ant.visited:
                    # výpočet ceny
                    weight = ant.terrain_weight(nx, ny)
                    tentative_g = current_g + weight

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = curr
                        f_score = tentative_g + self._heuristic(neighbor, goal)
                        heapq.heappush(queue, (f_score, tentative_g, neighbor))
        return []
    
    def find_path_to_food(self, ant, food):
        start = (ant.pos_x, ant.pos_y)
        goal = (food.pos_x, food.pos_y)

        queue = [(0, 0, start)]
        heapq.heapify(queue)

        parent = {}
        g_score = {start: 0}

        directions = [(-1,0),(1,0),(0,-1),(0,1)]

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
                nx, ny = curr[0]+dx, curr[1]+dy
                neighbor = (nx, ny)

                # k jídlu může jít i přes navštívená i přes objevená políčka
                if neighbor in ant.visited or neighbor in ant.discovered:
                    # výpočet ceny
                    weight = ant.terrain_weight(nx, ny)
                    tentative_g = current_g + weight

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = curr
                        f = tentative_g + self._heuristic(neighbor, goal)
                        heapq.heappush(queue, (f, tentative_g, neighbor))

        return []

# ---------------- MAIN ----------------

color_matrix, height_matrix = WorldGen(MAP_SIZE, MAP_SIZE, random.randint(0,10000))

walkable_tiles = np.sum(height_matrix > 0)

# generování hnízd
# definujeme si konfiguraci pro jednotlivá hnízda (typ a barvu)
nest_configs = [
    {"type": "BFS", "color": (0.7, 0.42, 0)},
    {"type": "DFS", "color": (0.0, 0.0, 0.7)},
    {"type": "ASTAR", "color": (0, 0.4, 0)}
]

# zamícháme pořadí konfigurací – pokaždé bude pořadí generování jiné
random.shuffle(nest_configs)

# použijeme slovník, abychom si vygenerovaná hnízda uložili podle jejich typu
nests_dict = {}
nests = []

# vygenerujeme hnízda v tomto náhodném pořadí
for config in nest_configs:
    colony_type = config["type"]
    color = config["color"]
    
    generated_pos = GenerateNest(color_matrix, height_matrix, nests, nest_color=color)
    nests.append(generated_pos)

    nests_dict[colony_type] = generated_pos

# extrahujeme si hnízda pro konkrétní proměnné
bfs_nest = nests_dict["BFS"]
dfs_nest = nests_dict["DFS"]
astar_nest = nests_dict["ASTAR"]

# zápis políček hnízd do setu
nest_tiles = set()

for nest in [bfs_nest, dfs_nest, astar_nest]:
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nest_tiles.add((nest[0] + dx, nest[1] + dy))

# inicializace statistik
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
        "moves": 0,
        "total_energy_spent": 0
    },

    "DFS": {
        "food_delivered": 0,
        "tiles_discovered": set(),
        "return_times": [],
        "deaths": 0,
        "kills": 0,
        "path_costs": [],
        "moves": 0,
        "total_energy_spent": 0
    },

    "ASTAR": {
        "food_delivered": 0,
        "tiles_discovered": set(),
        "return_times": [],
        "deaths": 0,
        "kills": 0,
        "path_costs": [],
        "moves": 0,
        "total_energy_spent": 0
    }
}

spawn_threshold = {
    "BFS": SPAWN_THRESHOLD_STEP,
    "DFS": SPAWN_THRESHOLD_STEP,
    "ASTAR": SPAWN_THRESHOLD_STEP
}

ants = []
hp_texts = []

# generování jídla
for _ in range(FOOD_QUANTITY):
    spawn_food()

# inicializace mravenců
for _ in range(STARTING_ANTS):
    ants.append(Ant(height_matrix, bfs_nest, "BFS"))
    ants.append(Ant(height_matrix, dfs_nest, "DFS"))
    ants.append(Ant(height_matrix, astar_nest, "ASTAR"))

# ---------------- PLOT ----------------

fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(right=0.72)

# mapa
ax.imshow(color_matrix)

# texty s počtem nasbíraného jídla u hnízd
bfs_text = ax.text(
    bfs_nest[1], bfs_nest[0],
    "0", color='white',
    ha='center', va='center',
    fontsize=10, fontweight='bold'
)
dfs_text = ax.text(
    dfs_nest[1], dfs_nest[0],
    "0", color='white',
    ha='center', va='center',
    fontsize=10, fontweight='bold'
)
astar_text = ax.text(
    astar_nest[1], astar_nest[0],
    "0", color='white',
    ha='center', va='center',
    fontsize=10, fontweight='bold'
)

# jídlo
food_scatter = ax.scatter(
    [f.pos_y for f in foods], [f.pos_x for f in foods],
    c='red', s=10
)

# mravenci
ant_plot = ax.scatter([], [], s=30, marker='s')

# statistiky
bfs_stats_text = fig.text(
    0.75, 0.85,
    "",
    fontsize=9,
    va='top',
    bbox=dict(facecolor='orange', alpha=0.25)
)
dfs_stats_text = fig.text(
    0.75, 0.60,
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

    # respawnování jídla
    if (
        FOOD_RESPAWN
        and len(foods) < MAX_FOOD
        and random.random() < FOOD_RESPAWN_CHANCE
    ):
        spawn_food()

    # pohyb mravenců
    for ant in ants:
        result = ant.move(foods, frame)

        # pokud mravec donesl jídlo do hnízda, aktualizujeme statistiky a případně spawneme nové mravce
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

                for _ in range(SPAWN_AMOUNT):
                    ants.append(Ant(height_matrix, nest, colony))

                spawn_threshold[colony] += SPAWN_THRESHOLD_STEP

    # boj mravenců
    HandleCombat(ants, foods)
    for ant in ants:
        if ant.combat_lock > 0:
            ant.combat_lock -= 1

    # vymazání rezervací mrtvých mravenců
    for ant in ants:
        if ant.hp <= 0 and ant.targeted_tile:
            ant.targeted.discard(ant.targeted_tile)
            ant.targeted_tile = None
    
    # odstranění mrtvých mravenců ze simulace
    ants[:] = [ant for ant in ants if ant.hp > 0]

    # RENDER

    positions = []
    colors = []

    # získávání souřadnic a barev mravenců
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

    # aktualizace pozic a barev mravenců
    ant_plot.set_offsets(positions)
    ant_plot.set_color(colors)

    # aktualizace pozic jídla
    if foods:
        food_scatter.set_offsets([[f.pos_y, f.pos_x] for f in foods])
    else:
        # pokud jídlo došlo, předáme prázdné 2D pole
        food_scatter.set_offsets(np.empty((0, 2)))

    # přepsání textů s počtem nasbíraného jídla
    bfs_text.set_text(str(food_collected["BFS"]))
    dfs_text.set_text(str(food_collected["DFS"]))
    astar_text.set_text(str(food_collected["ASTAR"]))
    draw_hp_texts(ax, ants)

    # textové statistiky
    def colony_report(colony):

        s = stats[colony]

        alive = sum(1 for ant in ants if ant.colony_type == colony)

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

        # Kolik jídla přinesl 1 mravenec na 100 jednotek energie
        efficiency = (s['food_delivered'] / s['total_energy_spent'] * 100) if s['total_energy_spent'] > 0 else 0

        return (
            f"{colony}\n"
            f"Alive ants: {alive}\n"
            f"Food: {s['food_delivered']}\n"
            f"Tiles: {len(s['tiles_discovered'])}\n"
            f"Avg return: {avg_return:.1f}\n"
            f"Deaths: {s['deaths']}\n"
            f"Kills: {s['kills']}\n"
            f"Total energy spent: {s['total_energy_spent']}\n"
            f"Avg cost: {avg_cost:.1f}\n"
            f"Efficiency: {efficiency:.2f}%\n"
            f"Coverage: {coverage:.1f}%"
        )

    bfs_stats_text.set_text(colony_report("BFS"))
    dfs_stats_text.set_text(colony_report("DFS"))
    astar_stats_text.set_text(colony_report("ASTAR"))

    return ant_plot, food_scatter

# spuštění animace
ani = animation.FuncAnimation(
    fig, update, interval=ANIM_SPEED, blit=False, cache_frame_data=False
)

plt.show()
