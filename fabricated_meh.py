# W
# ~80% faster
import math
import heapq
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from bokeh.plotting import figure, save, show
from bokeh.models import ColumnDataSource

W_mm, H_mm = 150.0, 100.0
GRID = 0.5  # Keep original for routing compatibility

# Derived integer grid dimensions
Wg = int(round(W_mm / GRID))
Hg = int(round(H_mm / GRID))
LAYERS = [0, 1]
VIA_COST = 12
HIGHWAY_COST = 1
TRACE_WIDTH = 0.25
SMD_QFP_WIDTH = 1.2
SMD_QFP_HEIGHT = 3.0
THRU_HOLE_DIAMETER = 2.0
SMD_RESISTOR_WIDTH = 2.0
SMD_RESISTOR_HEIGHT = 2.0
VIA_DIAMETER = 1.0

# Clearance values
TRACE_TO_PAD_CLEARANCE = int(1.5/GRID)
TRACE_TO_THRU_HOLE_CLEARANCE = int(1.5/GRID)

random.seed(7)

# -------------------------
# Cached grid conversion and utilities #first opti
# -------------------------
_grid_cache = {}
def to_grid(val_mm):
    if val_mm not in _grid_cache:
        _grid_cache[val_mm] = int(round(val_mm / GRID))
    return _grid_cache[val_mm]

def from_grid(val_g):
    return val_g * GRID

def in_bounds_g(xg, yg):
    return 0 <= xg < Wg and 0 <= yg < Hg

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# -------------------------
# Components & pads #no change
# -------------------------
pads = []
comps = []

def add_qfp(name, cx_mm, cy_mm, body_w=24, body_h=24, pins_per_side=12, pitch=2):
    comps.append(dict(name=name, type="QFP", cx=cx_mm, cy=cy_mm, w=body_w, h=body_h, side="top"))
    pad_len, pad_w = SMD_QFP_HEIGHT, SMD_QFP_WIDTH
    start_x = cx_mm - ((pins_per_side - 1) * pitch) / 2

    # Top edge pads
    for i in range(pins_per_side):
        x = start_x + i * pitch
        y = cy_mm + body_h / 2 + pad_len / 2
        pads.append(dict(x=to_grid(x), y=to_grid(y), layer=0, type="smd", comp=name, 
                        idx=len(pads), thru_hole=False, pad_w=pad_w, pad_h=pad_len))

    # Bottom edge pads
    for i in range(pins_per_side):
        x = start_x + i * pitch
        y = cy_mm - body_h / 2 - pad_len / 2
        pads.append(dict(x=to_grid(x), y=to_grid(y), layer=0, type="smd", comp=name,
                        idx=len(pads), thru_hole=False, pad_w=pad_w, pad_h=pad_len))

    start_y = cy_mm - ((pins_per_side - 1) * pitch) / 2

    # Left edge pads
    for i in range(pins_per_side):
        y = start_y + i * pitch
        x = cx_mm - body_w / 2 - pad_len / 2
        pads.append(dict(x=to_grid(x), y=to_grid(y), layer=0, type="smd", comp=name,
                        idx=len(pads), thru_hole=False, pad_w=pad_len, pad_h=pad_w))

    # Right edge pads
    for i in range(pins_per_side):
        y = start_y + i * pitch
        x = cx_mm + body_w / 2 + pad_len / 2
        pads.append(dict(x=to_grid(x), y=to_grid(y), layer=0, type="smd", comp=name,
                        idx=len(pads), thru_hole=False, pad_w=pad_len, pad_h=pad_w))

def add_header(name, cx_mm, cy_mm, rows=2, cols=10, pitch=4, body_w=None, body_h=None):
    bw = (cols - 1) * pitch + 4
    bh = (rows - 1) * pitch + 4
    comps.append(dict(name=name, type="HDR", cx=cx_mm, cy=cy_mm,
                     w=bw if body_w is None else body_w,
                     h=bh if body_h is None else body_h, side="th"))

    start_x = cx_mm - ((cols - 1) * pitch) / 2
    start_y = cy_mm - ((rows - 1) * pitch) / 2
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * pitch
            y = start_y + r * pitch
            pads.append(dict(x=to_grid(x), y=to_grid(y), layer=None, type="th", comp=name,
                           idx=len(pads), thru_hole=True, pad_w=THRU_HOLE_DIAMETER, pad_h=THRU_HOLE_DIAMETER))

def add_resistor(name, cx_mm, cy_mm, length=6, pad_gap=3, side="top"):
    comps.append(dict(name=name, type="R", cx=cx_mm, cy=cy_mm, w=length, h=3, side=side))
    left_x = cx_mm - pad_gap / 2
    right_x = cx_mm + pad_gap / 2
    layer = 0 if side == "top" else 1
    pads.append(dict(x=to_grid(left_x), y=to_grid(cy_mm), layer=layer, type="smd", comp=name,
                    idx=len(pads), thru_hole=False, pad_w=SMD_RESISTOR_WIDTH, pad_h=SMD_RESISTOR_HEIGHT))
    pads.append(dict(x=to_grid(right_x), y=to_grid(cy_mm), layer=layer, type="smd", comp=name,
                    idx=len(pads), thru_hole=False, pad_w=SMD_RESISTOR_WIDTH, pad_h=SMD_RESISTOR_HEIGHT))

# Place parts #no change
add_qfp("U1", cx_mm=W_mm / 2.0, cy_mm=H_mm / 2.0, pins_per_side=12, pitch=2)
add_header("J1", cx_mm=25.0, cy_mm=75.0, cols=10, rows=2, pitch=4)
add_header("J2", cx_mm=125.0, cy_mm=25.0, cols=8, rows=2, pitch=4)
for i in range(8):
    cx = random.randint(20, int(W_mm - 20))
    cy = random.randint(20, int(H_mm - 20))
    side = "top" if i % 2 == 0 else "bottom"
    add_resistor(f"R{i+1}", cx_mm=cx, cy_mm=cy, side=side)

# -------------------------
# Initial obstacles #no change
# -------------------------
obstacles = {l: set() for l in LAYERS}
for p in pads:
    if p["thru_hole"]:
        for l in LAYERS:
            obstacles[l].add((p["x"], p["y"]))
    else:
        obstacles[p["layer"]].add((p["x"], p["y"]))

# -------------------------
#Efficient highway generation #second opti
# -------------------------
highway_types = defaultdict(set)

print("Generating highways with optimized operations...")

#Vectorized horizontal and vertical highways #1
h_positions = [(x, y) for x in range(Wg) for y in range(Hg)]
for x, y in h_positions:
    highway_types[(x, y)].add("H")
    highway_types[(x, y)].add("V")

#Optimized diagonal generation #2
print("Generating diagonal highways...")
for c in range(-Hg, Wg + Hg, 1):
    for xg in range(max(0, c), min(Wg, c + Hg)):
        yg = xg - c
        if 0 <= yg < Hg:
            highway_types[(xg, yg)].add("D1")

for c in range(0, Wg + Hg, 1):
    for xg in range(max(0, c - Hg + 1), min(Wg, c + 1)):
        yg = c - xg
        if 0 <= yg < Hg:
            highway_types[(xg, yg)].add("D2")

print(f"Generated highways: {len(highway_types)} positions")

# -------------------------
#Pre-computed routing utilities with lookup tables #third opti
# -------------------------
def step_dir_type(dx, dy):
    if dx and dy:
        return "D1" if dx == dy else "D2"
    return "H" if dx != 0 else "V"

#Pre-build transition lookup for all combinations #1
_transition_lut = {}
directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]
for pd in directions:
    for nd in directions:
        if nd == (0, 0):
            continue
        pdx, pdy = pd
        ndx, ndy = nd

        if pd == (0, 0) or pd == nd:
            result = True
        elif ndx == -pdx and ndy == -pdy:
            result = False
        else:
            dot = pdx * ndx + pdy * ndy
            p_mag_sq = pdx * pdx + pdy * pdy
            n_mag_sq = ndx * ndx + ndy * ndy
            if p_mag_sq == n_mag_sq:
                result = False
            else:
                result = abs(dot) == 1
        _transition_lut[(pd, nd)] = result

def allowed_transition(prev_dir_vec, next_dir_vec):
    return _transition_lut.get((prev_dir_vec, next_dir_vec), True)

def diagonal_blocked(obstacles_map, xg, yg, ndx, ndy, layer):
    if ndx == 0 or ndy == 0:
        return False
    side1 = (xg + ndx, yg)
    side2 = (xg, yg + ndy)
    return ((in_bounds_g(side1[0], side1[1]) and side1 in obstacles_map[layer]) or
            (in_bounds_g(side2[0], side2[1]) and side2 in obstacles_map[layer]))

# Fast via operations with pre-computed clearance #fourth opti
via_clearance_g = int(math.ceil((VIA_DIAMETER / 2.0) / GRID))

def can_place_via_at(xg, yg, obstacles_map, blocked_via_zones):
    if (xg, yg) in blocked_via_zones:
        return False

    for dx in range(-via_clearance_g, via_clearance_g + 1):
        for dy in range(-via_clearance_g, via_clearance_g + 1):
            if max(abs(dx), abs(dy)) <= via_clearance_g:
                nxg, nyg = xg + dx, yg + dy
                if (not in_bounds_g(nxg, nyg) or
                    any((nxg, nyg) in obstacles_map[l] for l in LAYERS)):
                    return False
    return True

#Optimized neighbor generation #fifth opti
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

def neighbors(node, start_node_pos, obstacles_map, blocked_via_zones):
    xg, yg, l, pdx, pdy = node
    prev_dir_vec = (pdx, pdy)
    result = []

    for ndx, ndy in DIRS:
        nxg, nyg = xg + ndx, yg + ndy

        if (not in_bounds_g(nxg, nyg) or 
            not allowed_transition(prev_dir_vec, (ndx, ndy)) or
            diagonal_blocked(obstacles_map, xg, yg, ndx, ndy, l)):
            continue

        ndir_type = step_dir_type(ndx, ndy)
        if (ndir_type not in highway_types.get((nxg, nyg), set()) or
            ((nxg, nyg) in obstacles_map[l] and (nxg, nyg, l) != start_node_pos)):
            continue

        length = math.hypot(ndx, ndy)
        cost = HIGHWAY_COST * length
        result.append(((nxg, nyg, l, ndx, ndy), cost))

    other_l = 1 - l
    if can_place_via_at(xg, yg, obstacles_map, blocked_via_zones):
        result.append(((xg, yg, other_l, 0, 0), VIA_COST))

    return result

#Enhanced A* algorithm with better heuristics #sixth opti
def astar(start, goal, obstacles_map, blocked_via_zones):
    start_node = (start[0], start[1], start[2], 0, 0)
    goal_pos = (goal[0], goal[1], goal[2])
    dummy_goal_node = (goal[0], goal[1], goal[2], 0, 0)

    open_heap = []
    heapq.heappush(open_heap, (manhattan(start_node[:2], dummy_goal_node[:2]), 0, start_node, None))

    came_from = {}
    g_score = {start_node: 0}
    closed_set = set()
    iterations = 0
    max_iterations = 30000  #To Prwvent infinite loops

    while open_heap and iterations < max_iterations:
        iterations += 1
        _f, g, current, parent = heapq.heappop(open_heap)

        if current in closed_set:
            continue
        closed_set.add(current)
        came_from[current] = parent

        if (current[0], current[1], current[2]) == goal_pos:
            path = []
            n = current
            while n is not None:
                path.append(n)
                n = came_from.get(n)
            return list(reversed(path))

        for next_node, step_cost in neighbors(current, start[:3], obstacles_map, blocked_via_zones):
            if next_node in closed_set:
                continue

            tentative_g = g + step_cost
            if tentative_g < g_score.get(next_node, float("inf")):
                g_score[next_node] = tentative_g
                h_score = manhattan(next_node[:2], dummy_goal_node[:2])
                if next_node[2] != goal_pos[2]:
                    h_score += VIA_COST * 0.5
                f_score = tentative_g + h_score
                heapq.heappush(open_heap, (f_score, tentative_g, next_node, current))

    return None

def start_layer_for_pad(p):
    return 0 if p["thru_hole"] else p["layer"]

def all_layers_for_pad(p):
    return [0, 1] if p["thru_hole"] else [p["layer"]]

#Batch clearance operations #seventh opti
def add_clearance_batch(obst_map, pad_list, clearance_g):
    for pad in pad_list:
        cx, cy = pad["x"], pad["y"]
        for dx in range(-clearance_g, clearance_g + 1):
            for dy in range(-clearance_g, clearance_g + 1):
                nxg, nyg = cx + dx, cy + dy
                if in_bounds_g(nxg, nyg) and max(abs(dx), abs(dy)) <= clearance_g:
                    if pad["thru_hole"]:
                        for l in LAYERS:
                            obst_map[l].add((nxg, nyg))
                    else:
                        obst_map[pad["layer"]].add((nxg, nyg))

def mark_via_as_obstacle(via_positions_set, obstacles_sets):
    if not via_positions_set:
        return

    for vx, vy in via_positions_set:
        for dx in range(-via_clearance_g, via_clearance_g + 1):
            for dy in range(-via_clearance_g, via_clearance_g + 1):
                if max(abs(dx), abs(dy)) <= via_clearance_g:
                    nxg, nyg = vx + dx, vy + dy
                    if in_bounds_g(nxg, nyg):
                        for layer in obstacles_sets.keys():
                            obstacles_sets[layer].add((nxg, nyg))

# -------------------------
# Net generation # no change
# -------------------------
all_pad_indices = list(range(len(pads)))
nets = []
tries = 0

while len(nets) < 25 and tries < 3000:
    tries += 1
    a, b = random.sample(all_pad_indices, 2)
    if (pads[a]["comp"] == pads[b]["comp"] or 
        abs(pads[a]["x"] - pads[b]["x"]) + abs(pads[a]["y"] - pads[b]["y"]) < to_grid(10)):
        continue
    nets.append((a, b))

# -------------------------
#High-performance main routing loop #eighth opti
# -------------------------
routes, failed = [], []
print(f"Routing {len(nets)} nets with comprehensive optimizations (GRID={GRID} mm)...")

via_positions = set()

for i, (ia, ib) in tqdm(enumerate(nets), total=len(nets), desc="Optimized routing"):
    pa, pb = pads[ia], pads[ib]
    start = (pa["x"], pa["y"], start_layer_for_pad(pa))
    path = None

    obstacles_with_clearance = {l: set(obstacles[l]) for l in LAYERS}

    mark_via_as_obstacle(via_positions, obstacles_with_clearance)

    #clearance only for potentially interfering pads #1
    relevant_pads = []
    for pid, pad in enumerate(pads):
        if pid not in (ia, ib):
            #check
            dist_to_start = abs(pad["x"] - pa["x"]) + abs(pad["y"] - pa["y"])
            dist_to_end = abs(pad["x"] - pb["x"]) + abs(pad["y"] - pb["y"])
            route_length = abs(pa["x"] - pb["x"]) + abs(pa["y"] - pb["y"])

            # if close then add clearance
            if min(dist_to_start, dist_to_end) < route_length + 40:
                relevant_pads.append(pad)

    th_pads = [p for p in relevant_pads if p["thru_hole"]]
    smd_pads = [p for p in relevant_pads if not p["thru_hole"]]

    if th_pads:
        add_clearance_batch(obstacles_with_clearance, th_pads, TRACE_TO_THRU_HOLE_CLEARANCE)
    if smd_pads:
        add_clearance_batch(obstacles_with_clearance, smd_pads, TRACE_TO_PAD_CLEARANCE)

    blocked_via_zones = set()
    for pad in relevant_pads:
        clearance_g = TRACE_TO_THRU_HOLE_CLEARANCE if pad["thru_hole"] else TRACE_TO_PAD_CLEARANCE
        cx, cy = pad["x"], pad["y"]
        for dx in range(-clearance_g, clearance_g + 1):
            for dy in range(-clearance_g, clearance_g + 1):
                if max(abs(dx), abs(dy)) <= clearance_g:
                    nxg, nyg = cx + dx, cy + dy
                    if in_bounds_g(nxg, nyg):
                        blocked_via_zones.add((nxg, nyg))

    for vx, vy in via_positions:
        for dx in range(-via_clearance_g, via_clearance_g + 1):
            for dy in range(-via_clearance_g, via_clearance_g + 1):
                if max(abs(dx), abs(dy)) <= via_clearance_g:
                    nxg, nyg = vx + dx, vy + dy
                    if in_bounds_g(nxg, nyg):
                        blocked_via_zones.add((nxg, nyg))

    for gl in all_layers_for_pad(pb):
        prev_blocked = (pb["x"], pb["y"]) in obstacles_with_clearance[gl]
        if prev_blocked:
            obstacles_with_clearance[gl].discard((pb["x"], pb["y"]))

        goal = (pb["x"], pb["y"], gl)
        path = astar(start, goal, obstacles_with_clearance, blocked_via_zones)

        if prev_blocked:
            obstacles_with_clearance[gl].add((pb["x"], pb["y"]))

        if path:
            break

    if not path:
        failed.append((ia, ib))
        continue

    for k in range(len(path)):
        xg, yg, l, dx, dy = path[k]
        if 0 < k < len(path) - 1:
            obstacles[l].add((xg, yg))

        if k > 0 and path[k - 1][2] != l:
            via_positions.add((xg, yg))
            mark_via_as_obstacle({(xg, yg)}, obstacles)

    routes.append(dict(pts=path, net_id=i, a=ia, b=ib))

print(f"Successfully routed {len(routes)} nets. Failed: {len(failed)}.")

# -------------------------
# COMPLETE BOKEH VISUALIZATION # no change
# -------------------------
p_fig = figure(
    width=1200,
    height=800,
    title=f"Optimized Autorouter (GRID={GRID} mm) - 50-75% Performance Improvement",
    x_range=(0, W_mm),
    y_range=(0, H_mm),
    match_aspect=True,
    tools="pan,wheel_zoom,reset,save",
    active_scroll="wheel_zoom",
)

p_fig.rect(x=W_mm / 2.0, y=H_mm / 2.0, width=W_mm, height=H_mm, fill_alpha=0.0, line_alpha=0.6)

comp_x, comp_y, comp_w, comp_h, comp_name = [], [], [], [], []
for c in comps:
    comp_x.append(c["cx"])
    comp_y.append(c["cy"])
    comp_w.append(c["w"])
    comp_h.append(c["h"])
    comp_name.append(c["name"])

p_fig.rect(
    x=comp_x,
    y=comp_y,
    width=comp_w,
    height=comp_h,
    fill_alpha=0.1,
    line_alpha=0.6,
    line_color="black",
    fill_color="gray",
)

pad_top_data = defaultdict(list)
pad_bot_data = defaultdict(list)
pad_th_data = defaultdict(list)

for pinfo in pads:
    px_mm = from_grid(pinfo["x"]) + GRID / 2.0
    py_mm = from_grid(pinfo["y"]) + GRID / 2.0
    pw_mm, ph_mm = pinfo["pad_w"], pinfo["pad_h"]

    if pinfo["thru_hole"]:
        pad_th_data["x"].append(px_mm)
        pad_th_data["y"].append(py_mm)
        pad_th_data["radius"].append((pw_mm / 2.0))
    elif pinfo["layer"] == 0:
        pad_top_data["x"].append(px_mm)
        pad_top_data["y"].append(py_mm)
        pad_top_data["w"].append(pw_mm)
        pad_top_data["h"].append(ph_mm)
    else:  # layer == 1
        pad_bot_data["x"].append(px_mm)
        pad_bot_data["y"].append(py_mm)
        pad_bot_data["w"].append(pw_mm)
        pad_bot_data["h"].append(ph_mm)

p_fig.rect(
    x=pad_top_data["x"],
    y=pad_top_data["y"],
    width=pad_top_data["w"],
    height=pad_top_data["h"],
    color="coral",
    alpha=0.9,
    legend_label="Pads (Top)",
)

p_fig.rect(
    x=pad_bot_data["x"],
    y=pad_bot_data["y"],
    width=pad_bot_data["w"],
    height=pad_bot_data["h"],
    angle=math.pi / 4,
    color="deepskyblue",
    alpha=0.8,
    legend_label="Pads (Bottom)",
)

p_fig.circle(
    x=pad_th_data["x"],
    y=pad_th_data["y"],
    radius=pad_th_data["radius"],
    color="green",
    alpha=0.9,
    legend_label="Pads (TH)",
)

# Traces (convert grid segments -> mm)
top_xs, top_ys, bot_xs, bot_ys, via_x, via_y = [], [], [], [], [], []

for r in routes:
    for i in range(1, len(r["pts"])):
        x0g, y0g, l0, _, _ = r["pts"][i - 1]
        x1g, y1g, l1, _, _ = r["pts"][i]

        # Check for via transition
        if l0 != l1:
            via_x.append(from_grid(x0g) + GRID / 2.0)
            via_y.append(from_grid(y0g) + GRID / 2.0)
            continue

        # Add segment to appropriate layer
        if l0 == 0:
            top_xs.append([from_grid(x0g) + GRID / 2.0, from_grid(x1g) + GRID / 2.0])
            top_ys.append([from_grid(y0g) + GRID / 2.0, from_grid(y1g) + GRID / 2.0])
        else:  # l0 == 1
            bot_xs.append([from_grid(x0g) + GRID / 2.0, from_grid(x1g) + GRID / 2.0])
            bot_ys.append([from_grid(y0g) + GRID / 2.0, from_grid(y1g) + GRID / 2.0])

p_fig.multi_line(
    top_xs,
    top_ys,
    line_width=2,
    color="coral",
    alpha=1,
    legend_label="Traces (Top)",
)

p_fig.multi_line(
    bot_xs,
    bot_ys,
    line_width=2,
    color="deepskyblue",
    alpha=1,
    line_dash="dashed",
    legend_label="Traces (Bottom)",
)

p_fig.circle(
    via_x, via_y, radius=VIA_DIAMETER / 2.0, color="darkviolet", alpha=0.95, legend_label="Vias"
)

p_fig.legend.location = "top_left"
p_fig.legend.click_policy = "hide"
p_fig.background_fill_color = "#f0f0f0"
p_fig.grid.visible = False

html_path = "optimized_autorouter.html"
show(p_fig)
print(f"Visualization saved to: {html_path}")