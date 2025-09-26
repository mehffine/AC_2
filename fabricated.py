import random
import math
import heapq
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from bokeh.plotting import figure, save, show
from bokeh.models import ColumnDataSource

# -------------------------
# Board & grid parameters (change GRID to any fractional value)
# -------------------------
W_mm, H_mm = 150.0, 100.0    # board size in mm (unchanged)
GRID = 0.5            # grid size in mm (fractional)
# Derived integer grid dimensions
Wg = int(round(W_mm / GRID))
Hg = int(round(H_mm / GRID))

LAYERS = [0, 1]  # 0 top, 1 bottom
VIA_COST = 12
HIGHWAY_COST = 1

TRACE_WIDTH = 0.25
SMD_QFP_WIDTH = 1.2
SMD_QFP_HEIGHT = 3.0
THRU_HOLE_DIAMETER = 2.0
SMD_RESISTOR_WIDTH = 2.0
SMD_RESISTOR_HEIGHT = 2.0
VIA_DIAMETER = 1.0

# Clearance values are provided in grid units (as in your original script)
TRACE_TO_PAD_CLEARANCE = int(1.5/GRID)  # Clearance in grid units (we will convert mm -> grid when used)
TRACE_TO_THRU_HOLE_CLEARANCE = int(1.5/GRID) 

random.seed(7)


# -------------------------
# Grid conversion helpers
# -------------------------
def to_grid(val_mm):
    """Convert mm value to integer grid coordinate."""
    return int(round(val_mm / GRID))


def from_grid(val_g):
    """Convert integer grid coordinate back to mm (float)."""
    return val_g * GRID


def in_bounds_g(xg, yg):
    return 0 <= xg < Wg and 0 <= yg < Hg


def manhattan(a, b):
    (x1, y1, l1, _dx1, _dy1) = a
    (x2, y2, l2, _dx2, _dy2) = b
    return abs(x1 - x2) + abs(y1 - y2)


# -------------------------
# Components & pads
# -------------------------
pads = []
comps = []


def add_qfp(name, cx_mm, cy_mm, body_w=24, body_h=24, pins_per_side=12, pitch=2):
    comps.append(
        dict(name=name, type="QFP", cx=cx_mm, cy=cy_mm, w=body_w, h=body_h, side="top")
    )
    pad_len = SMD_QFP_HEIGHT
    pad_w = SMD_QFP_WIDTH
    start_x = cx_mm - ((pins_per_side - 1) * pitch) / 2
    # Top edge pads
    for i in range(pins_per_side):
        x = start_x + i * pitch
        y = cy_mm + body_h / 2 + pad_len / 2
        pads.append(
            dict(
                x=to_grid(x),
                y=to_grid(y),
                layer=0,
                type="smd",
                comp=name,
                idx=len(pads),
                thru_hole=False,
                pad_w=pad_w,
                pad_h=pad_len,
            )
        )
    # Bottom edge pads
    for i in range(pins_per_side):
        x = start_x + i * pitch
        y = cy_mm - body_h / 2 - pad_len / 2
        pads.append(
            dict(
                x=to_grid(x),
                y=to_grid(y),
                layer=0,
                type="smd",
                comp=name,
                idx=len(pads),
                thru_hole=False,
                pad_w=pad_w,
                pad_h=pad_len,
            )
        )
    start_y = cy_mm - ((pins_per_side - 1) * pitch) / 2
    # Left edge pads
    for i in range(pins_per_side):
        y = start_y + i * pitch
        x = cx_mm - body_w / 2 - pad_len / 2
        pads.append(
            dict(
                x=to_grid(x),
                y=to_grid(y),
                layer=0,
                type="smd",
                comp=name,
                idx=len(pads),
                thru_hole=False,
                pad_w=pad_len,
                pad_h=pad_w,
            )
        )
    # Right edge pads
    for i in range(pins_per_side):
        y = start_y + i * pitch
        x = cx_mm + body_w / 2 + pad_len / 2
        pads.append(
            dict(
                x=to_grid(x),
                y=to_grid(y),
                layer=0,
                type="smd",
                comp=name,
                idx=len(pads),
                thru_hole=False,
                pad_w=pad_len,
                pad_h=pad_w,
            )
        )


def add_header(name, cx_mm, cy_mm, rows=2, cols=10, pitch=4, body_w=None, body_h=None):
    bw = (cols - 1) * pitch + 4
    bh = (rows - 1) * pitch + 4
    comps.append(
        dict(
            name=name,
            type="HDR",
            cx=cx_mm,
            cy=cy_mm,
            w=bw if body_w is None else body_w,
            h=bh if body_h is None else body_h,
            side="th",
        )
    )
    start_x = cx_mm - ((cols - 1) * pitch) / 2
    start_y = cy_mm - ((rows - 1) * pitch) / 2
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * pitch
            y = start_y + r * pitch
            pads.append(
                dict(
                    x=to_grid(x),
                    y=to_grid(y),
                    layer=None,
                    type="th",
                    comp=name,
                    idx=len(pads),
                    thru_hole=True,
                    pad_w=THRU_HOLE_DIAMETER,
                    pad_h=THRU_HOLE_DIAMETER,
                )
            )


def add_resistor(name, cx_mm, cy_mm, length=6, pad_gap=3, side="top"):
    comps.append(dict(name=name, type="R", cx=cx_mm, cy=cy_mm, w=length, h=3, side=side))
    left_x = cx_mm - pad_gap / 2
    right_x = cx_mm + pad_gap / 2
    layer = 0 if side == "top" else 1
    pads.append(
        dict(
            x=to_grid(left_x),
            y=to_grid(cy_mm),
            layer=layer,
            type="smd",
            comp=name,
            idx=len(pads),
            thru_hole=False,
            pad_w=SMD_RESISTOR_WIDTH,
            pad_h=SMD_RESISTOR_HEIGHT,
        )
    )
    pads.append(
        dict(
            x=to_grid(right_x),
            y=to_grid(cy_mm),
            layer=layer,
            type="smd",
            comp=name,
            idx=len(pads),
            thru_hole=False,
            pad_w=SMD_RESISTOR_WIDTH,
            pad_h=SMD_RESISTOR_HEIGHT,
        )
    )


# Place parts (keep positions in mm as before)
add_qfp("U1", cx_mm=W_mm / 2.0, cy_mm=H_mm / 2.0, pins_per_side=12, pitch=2)
add_header("J1", cx_mm=25.0, cy_mm=75.0, cols=10, rows=2, pitch=4)
add_header("J2", cx_mm=125.0, cy_mm=25.0, cols=8, rows=2, pitch=4)
for i in range(8):
    cx = random.randint(20, int(W_mm - 20))
    cy = random.randint(20, int(H_mm - 20))
    side = "top" if i % 2 == 0 else "bottom"
    add_resistor(f"R{i+1}", cx_mm=cx, cy_mm=cy, side=side)

# -------------------------
# Occupancy (use sets instead of large 2D arrays)
# obstacles[layer] = set of (xg, yg) grid coords that are blocked
# -------------------------
obstacles = {l: set() for l in LAYERS}

# Mark pad centers as occupied (thru holes block all layers at their center)
for p in pads:
    if p["thru_hole"]:
        for l in LAYERS:
            obstacles[l].add((p["x"], p["y"]))
    else:
        obstacles[p["layer"]].add((p["x"], p["y"]))

# -------------------------
# Highways (with types): stored sparsely as dict{(xg,yg): set(types)}
# We'll produce full coverage for HV and diagonals on grid coords
# -------------------------
highway_types = defaultdict(set)
# Spacing in grid units (SP = number of grid steps between highways)
SP_HV = 1
SP_D = 1

# Horizontal (H) and Vertical (V)
for xg in range(0, Wg, SP_HV):
    for yg in range(Hg):
        highway_types[(xg, yg)].add("V")
for yg in range(0, Hg, SP_HV):
    for xg in range(Wg):
        highway_types[(xg, yg)].add("H")

# Diagonals D1 (x - y = const) and D2 (x + y = const)
# We iterate over possible constants and mark points in bounds
for c in range(-Hg, Wg + Hg, SP_D):
    for xg in range(Wg):
        yg = xg - c
        if 0 <= yg < Hg:
            highway_types[(xg, yg)].add("D1")
for c in range(0, Wg + Hg, SP_D):
    for xg in range(Wg):
        yg = c - xg
        if 0 <= yg < Hg:
            highway_types[(xg, yg)].add("D2")

# -------------------------
# Utility helpers for routing
# -------------------------
def step_dir_type(dx, dy):
    if dx and dy:
        return "D1" if dx == dy else "D2"
    return "H" if dx != 0 else "V"


def allowed_transition(prev_dir_vec, next_dir_vec):
    pdx, pdy = prev_dir_vec
    ndx, ndy = next_dir_vec
    if (pdx, pdy) == (0, 0):
        return True
    if (pdx, pdy) == (ndx, ndy):
        return True
    if ndx == -pdx and ndy == -pdy:
        return False

    dot = pdx * ndx + pdy * ndy
    p_mag_sq = pdx * pdx + pdy * pdy
    n_mag_sq = ndx * ndx + ndy * ndy

    if p_mag_sq == n_mag_sq:
        return False
    return abs(dot) == 1


def diagonal_blocked(obstacles_map, xg, yg, ndx, ndy, layer):
    # Only applies to diagonals
    if ndx == 0 or ndy == 0:
        return False
    side1 = (xg + ndx, yg)
    side2 = (xg, yg + ndy)
    s1_ok = in_bounds_g(side1[0], side1[1])
    s2_ok = in_bounds_g(side2[0], side2[1])
    return (s1_ok and (side1 in obstacles_map[layer])) or (
        s2_ok and (side2 in obstacles_map[layer])
    )


DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

def can_place_via_at(xg, yg, obstacles_map, blocked_via_zones):
    """
    Return True if a via centered at (xg, yg) does not overlap any obstacles/pad clearances
    or existing via clearances on ANY layer. We test the full via footprint (radius in grid units).
    """
    via_clearance_g = int(math.ceil((VIA_DIAMETER / 2.0) / GRID))
    # Check each cell within the via footprint
    for dx in range(-via_clearance_g, via_clearance_g + 1):
        for dy in range(-via_clearance_g, via_clearance_g + 1):
            if max(abs(dx), abs(dy)) <= via_clearance_g:
                nxg, nyg = xg + dx, yg + dy
                if not in_bounds_g(nxg, nyg):
                    return False
                # If the cell is within any layer's obstacle set, via would overlap pad/clearance
                for l in LAYERS:
                    if (nxg, nyg) in obstacles_map[l]:
                        return False
                # Also ensure it does not intersect blocked via zones (existing via clearances)
                if (nxg, nyg) in blocked_via_zones:
                    return False
    return True


def neighbors(node, start_node_pos, obstacles_map, blocked_via_zones):
    xg, yg, l, pdx, pdy = node
    prev_dir_vec = (pdx, pdy)

    for ndx, ndy in DIRS:
        nxg, nyg = xg + ndx, yg + ndy
        if not in_bounds_g(nxg, nyg):
            continue
        if not allowed_transition(prev_dir_vec, (ndx, ndy)):
            continue
        if diagonal_blocked(obstacles_map, xg, yg, ndx, ndy, l):
            continue

        ndir_type = step_dir_type(ndx, ndy)
        # require that this grid cell has the highway type
        if ndir_type not in highway_types.get((nxg, nyg), set()):
            continue
        # allow stepping into the start pad even if marked occupied (so we can exit/enter pad)
        if (nxg, nyg) in obstacles_map[l] and (nxg, nyg, l) != start_node_pos:
            continue

        length = math.hypot(ndx, ndy)
        cost = HIGHWAY_COST * length
        yield (nxg, nyg, l, ndx, ndy), cost

    other_l = 1 - l
    # Via allowed only if the via footprint does not overlap obstacles/clearance on any layer
    # AND the coordinate footprint is not in blocked_via_zones (existing vias)
    if can_place_via_at(xg, yg, obstacles_map, blocked_via_zones):
        # Also ensure the via's center cell on the other layer is not considered occupied
        # (neighbors() caller expects that start pad cell may be allowed, etc.)
        yield (xg, yg, other_l, 0, 0), VIA_COST


def astar(start, goal, obstacles_map, blocked_via_zones):
    start_node = (start[0], start[1], start[2], 0, 0)
    goal_pos = (goal[0], goal[1], goal[2])
    dummy_goal_node = (goal[0], goal[1], goal[2], 0, 0)

    open_heap = []
    heapq.heappush(open_heap, (manhattan(start_node, dummy_goal_node), 0, start_node, None))

    came_from = {}
    g_score = {start_node: 0}
    closed_set = set()

    while open_heap:
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
                f_score = tentative_g + manhattan(next_node, dummy_goal_node)
                heapq.heappush(open_heap, (f_score, tentative_g, next_node, current))

    return None


def start_layer_for_pad(p):
    return 0 if p["thru_hole"] else p["layer"]


def all_layers_for_pad(p):
    return [0, 1] if p["thru_hole"] else [p["layer"]]


# -------------------------
# Clearance helper (works in grid units)
# -------------------------
def add_clearance(obst_map, pad, clearance_g):
    """
    Mark cells within Chebyshev distance <= clearance_g around pad as blocked
    obst_map is a dict {layer: set((xg, yg), ...)}
    For thru-hole pads, mark both layers; for SMD, mark pad layer only.
    """
    cx, cy = pad["x"], pad["y"]
    for dx in range(-clearance_g, clearance_g + 1):
        for dy in range(-clearance_g, clearance_g + 1):
            nxg, nyg = cx + dx, cy + dy
            if not in_bounds_g(nxg, nyg):
                continue
            if max(abs(dx), abs(dy)) <= clearance_g:
                if pad["thru_hole"]:
                    for l in LAYERS:
                        obst_map[l].add((nxg, nyg))
                else:
                    obst_map[pad["layer"]].add((nxg, nyg))


def mark_via_as_obstacle(via_positions_set, obstacles_sets):
    """
    Marks via positions (with diameter clearance) as obstacles on ALL layers.
    via_positions_set: set of (xg, yg)
    obstacles_sets: dict { layer_index: set((xg, yg), ...) }
    """
    if not via_positions_set:
        return
    via_clearance_g = int(math.ceil((VIA_DIAMETER / 2.0) / GRID))
    for vx, vy in via_positions_set:
        for dx in range(-via_clearance_g, via_clearance_g + 1):
            for dy in range(-via_clearance_g, via_clearance_g + 1):
                if max(abs(dx), abs(dy)) <= via_clearance_g:
                    nxg, nyg = vx + dx, vy + dy
                    if not in_bounds_g(nxg, nyg):
                        continue
                    for layer in obstacles_sets.keys():
                        obstacles_sets[layer].add((nxg, nyg))


# -------------------------
# Random nets (same logic, but using grid coordinates)
# -------------------------
all_pad_indices = list(range(len(pads)))
nets = []
tries = 0
while len(nets) < 25 and tries < 3000:
    tries += 1
    a, b = random.sample(all_pad_indices, 2)
    if pads[a]["comp"] == pads[b]["comp"]:
        continue
    if abs(pads[a]["x"] - pads[b]["x"]) + abs(pads[a]["y"] - pads[b]["y"]) < to_grid(10):
        continue
    nets.append((a, b))

# -------------------------
# Route
# -------------------------
routes, failed = [], []
print(f"Attempting to route {len(nets)} nets on a universal highway grid with pad clearance (GRID={GRID} mm)...")

via_positions = set()

for i, (ia, ib) in tqdm(enumerate(nets), total=len(nets), desc="Routing nets"):
    pa, pb = pads[ia], pads[ib]
    start = (pa["x"], pa["y"], start_layer_for_pad(pa))
    path = None

    # Copy obstacles & mark all existing vias as blocked (shallow copy of sets)
    obstacles_with_clearance = {l: set(obstacles[l]) for l in LAYERS}
    # mark existing vias as obstacles with diameter-based clearance
    mark_via_as_obstacle(via_positions, obstacles_with_clearance)

    # Add clearance around all pads except start/end
    for pid, pad in enumerate(pads):
        if pid in (ia, ib):
            continue
        clearance_g = TRACE_TO_THRU_HOLE_CLEARANCE if pad["thru_hole"] else TRACE_TO_PAD_CLEARANCE
        add_clearance(obstacles_with_clearance, pad, clearance_g)

    # Precompute blocked positions where vias are forbidden:
    blocked_via_zones = set()
    # Include pad clearance zones (these should forbid vias)
    for pad in pads:
        clearance_g = TRACE_TO_THRU_HOLE_CLEARANCE if pad["thru_hole"] else TRACE_TO_PAD_CLEARANCE
        cx, cy = pad["x"], pad["y"]
        for dx in range(-clearance_g, clearance_g + 1):
            for dy in range(-clearance_g, clearance_g + 1):
                if max(abs(dx), abs(dy)) <= clearance_g:
                    nxg, nyg = cx + dx, cy + dy
                    if in_bounds_g(nxg, nyg):
                        blocked_via_zones.add((nxg, nyg))
    # Include existing vias with via-diameter clearance
    via_clearance_g = int(math.ceil((VIA_DIAMETER / 2.0) / GRID))
    for vx, vy in via_positions:
        for dx in range(-via_clearance_g, via_clearance_g + 1):
            for dy in range(-via_clearance_g, via_clearance_g + 1):
                if max(abs(dx), abs(dy)) <= via_clearance_g:
                    nxg, nyg = vx + dx, vy + dy
                    if in_bounds_g(nxg, nyg):
                        blocked_via_zones.add((nxg, nyg))

    # Try all allowed layers for target pad
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

    # Mark traces & vias in the global obstacles map
    for k in range(len(path)):
        xg, yg, l, dx, dy = path[k]
        if k > 0 and k < len(path) - 1:
            obstacles[l].add((xg, yg))
        # Detect via and add to global list
        if k > 0 and path[k - 1][2] != l:
            # previous node had different layer -> this node is via location
            via_positions.add((xg, yg))
            # Block via on all layers (using diameter-based blocking)
            mark_via_as_obstacle({(xg, yg)}, obstacles)

    routes.append(dict(pts=path, net_id=i, a=ia, b=ib))

print(f"Successfully routed {len(routes)} nets. Failed: {len(failed)}.")

# -------------------------
# Bokeh figure (convert back to mm coords for plotting)
# -------------------------
p_fig = figure(
    width=1200,
    height=800,
    title=f"Autorouter (GRID={GRID} mm) - Universal Highways, 135° Bends Only, Pad Clearance",
    x_range=(0, W_mm),
    y_range=(0, H_mm),
    match_aspect=True,
    tools="pan,wheel_zoom,reset,save",
    active_scroll="wheel_zoom",
)

p_fig.rect(x=W_mm / 2.0, y=H_mm / 2.0, width=W_mm, height=H_mm, fill_alpha=0.0, line_alpha=0.6)

# Components (use mm coords)
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

# Pads (convert grid coords to mm)
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

# Use rect for SMD pads to specify width and height in data units
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

        # Add segment to appropriate layer (each segment as [x0,x1], [y0,y1] in mm)
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

html_path = "autorouter.html"
# save(p_fig, filename=html_path)
show(p_fig, filename=html_path)
print(f"Saved output to: {html_path}")
