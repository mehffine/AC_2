# fabricated_faster_v2.py
# Faster A* demo with a tuple/set baseline vs an integer/array optimized router.
# Usage:
#   python fabricated_faster_v2.py --bench --nets 400 --seed 1
#   python fabricated_faster_v2.py --bench --nets 1200 --seed 2      # larger = clearer speedup
#   python fabricated_faster_v2.py --route --nets 200

import math, random, heapq, time, argparse
from typing import List, Tuple

# ---------------- Grid & layers ----------------
W_MM, H_MM = 150.0, 100.0
GRID = 0.5
Wg = int(round(W_MM / GRID))
Hg = int(round(H_MM / GRID))

TOP, BOT = 0, 1

def to_g(mm: float) -> int: return int(round(mm / GRID))
def inb(x: int, y: int) -> bool: return 0 <= x < Wg and 0 <= y < Hg

# ---------------- Schematic synth ----------------
pads = []  # dicts: x,y,layer or thru_hole

def add_qfp(name: str, cx, cy, body_w=24, body_h=24, pins=12, pitch=2):
    pad_len=3.0
    sx = cx - ((pins-1)*pitch)/2
    for i in range(pins):
        pads.append(dict(x=to_g(sx+i*pitch), y=to_g(cy+body_h/2+pad_len/2), layer=TOP, thru_hole=False, comp=name))
        pads.append(dict(x=to_g(sx+i*pitch), y=to_g(cy-body_h/2-pad_len/2), layer=TOP, thru_hole=False, comp=name))
    sy = cy - ((pins-1)*pitch)/2
    for i in range(pins):
        pads.append(dict(x=to_g(cx-body_w/2-pad_len/2), y=to_g(sy+i*pitch), layer=TOP, thru_hole=False, comp=name))
        pads.append(dict(x=to_g(cx+body_w/2+pad_len/2), y=to_g(sy+i*pitch), layer=TOP, thru_hole=False, comp=name))

def add_header(name: str, cx, cy, rows=2, cols=10, pitch=4):
    sx = cx - ((cols-1)*pitch)/2
    sy = cy - ((rows-1)*pitch)/2
    for r in range(rows):
        for c in range(cols):
            pads.append(dict(x=to_g(sx+c*pitch), y=to_g(sy+r*pitch), layer=None, thru_hole=True, comp=name))

def add_res(name: str, cx, cy, side="top"):
    layer = TOP if side=="top" else BOT
    pads.append(dict(x=to_g(cx-1.5), y=to_g(cy), layer=layer, thru_hole=False, comp=name))
    pads.append(dict(x=to_g(cx+1.5), y=to_g(cy), layer=layer, thru_hole=False, comp=name))

def synth(seed=0):
    random.seed(seed); pads.clear()
    add_qfp("U1", W_MM/2, H_MM/2, pins=12, pitch=2)
    add_header("J1", 25.0, 75.0, rows=2, cols=10, pitch=4)
    add_header("J2", 125.0, 25.0, rows=2, cols=8, pitch=4)
    for i in range(10):
        add_res(f"R{i+1}", random.randint(20, int(W_MM-20)), random.randint(20, int(H_MM-20)), side=("top" if i%2==0 else "bottom"))

# ---------------- Obstacles ----------------
def build_obstacles_bool():
    obst = [[[False]*Wg for _ in range(Hg)] for __ in range(2)]
    for p in pads:
        if p["thru_hole"]:
            obst[TOP][p["y"]][p["x"]] = True
            obst[BOT][p["y"]][p["x"]] = True
        else:
            obst[p["layer"]][p["y"]][p["x"]] = True
    return obst

def build_obstacles_set():
    obst = {TOP:set(), BOT:set()}
    for p in pads:
        if p["thru_hole"]:
            obst[TOP].add((p["x"],p["y"])); obst[BOT].add((p["x"],p["y"]))
        else:
            obst[p["layer"]].add((p["x"],p["y"]))
    return obst

# ---------------- Nets ----------------
def gen_nets(n=400, min_mm=15.0, seed=0):
    random.seed(seed)
    idx = list(range(len(pads)))
    nets=[]; thr=to_g(min_mm); tries=0
    while len(nets)<n and tries<6000:
        tries+=1
        a,b = random.sample(idx,2)
        if abs(pads[a]["x"]-pads[b]["x"]) + abs(pads[a]["y"]-pads[b]["y"]) < thr:
            continue
        nets.append((a,b))
    return nets

# ---------------- A* shared ----------------
DIRS4 = [(1,0),(-1,0),(0,1),(0,-1)]
DIRS8 = DIRS4 + [(1,1),(-1,-1),(1,-1),(-1,1)]

# Baseline (float costs)
STEP_F = {(1,0):1.0,(-1,0):1.0,(0,1):1.0,(0,-1):1.0,(1,1):math.sqrt(2.0),(-1,-1):math.sqrt(2.0),(1,-1):math.sqrt(2.0),(-1,1):math.sqrt(2.0)}
VIA_F  = 1.4

# Optimized (integer costs: straight=10, diag=14, via=20)
STEP_I = {(1,0):10,(-1,0):10,(0,1):10,(0,-1):10,(1,1):14,(-1,-1):14,(1,-1):14,(-1,1):14}
VIA_I  = 20

# ---------------- Baseline (set obstacles, tuple states, float) ----------------
def neighbors_base(obst_set, x,y,l, sx,sy,sl, gx,gy,gl):
    o = obst_set[l]
    for dx,dy in DIRS8:
        nx,ny = x+dx, y+dy
        if not inb(nx,ny): continue
        if dx and dy and (((x+dx,y) in o) or ((x,y+dy) in o)):
            continue
        if ((nx,ny) in o) and not ((nx,ny,l)==(sx,sy,sl) or (nx,ny,l)==(gx,gy,gl)):
            continue
        yield (nx,ny,l,dx,dy), STEP_F[(dx,dy)]
    nl = 1-l
    if (x,y) not in obst_set[nl]:
        yield (x,y,nl,0,0), VIA_F

def astar_base(obst_set, start, goal):
    sx,sy,sl=start; gx,gy,gl=goal
    openh=[(abs(sx-gx)+abs(sy-gy), 0.0, (sx,sy,sl,0,0), None)]
    came={}; gsc={(sx,sy,sl,0,0):0.0}; closed=set()
    while openh:
        _f,gs,cur,parent = heapq.heappop(openh)
        if cur in closed: continue
        closed.add(cur); came[cur]=parent
        x,y,l,dx,dy=cur
        if (x,y,l)==(gx,gy,gl): return True, cur
        for nxt,c in neighbors_base(obst_set,x,y,l,sx,sy,sl,gx,gy,gl):
            if nxt in closed: continue
            tg=gs+c
            if tg < gsc.get(nxt,1e18):
                gsc[nxt]=tg
                f=tg+abs(nxt[0]-gx)+abs(nxt[1]-gy)
                heapq.heappush(openh,(f,tg,nxt,cur))
    return False, None

# ---------------- Optimized (bool obstacles, int arrays, int costs) ----------------
# position-only encoding: id = (l*Hg + y)*Wg + x
def pid(x:int,y:int,l:int) -> int: return (l*Hg + y)*Wg + x

def astar_fast(obst_bool, start, goal):
    sx,sy,sl=start; gx,gy,gl=goal
    # arrays sized to grid*layers
    N = Wg*Hg*2
    INF = 10**12
    g = [INF]*N
    closed = [False]*N

    sid = pid(sx,sy,sl); gid = pid(gx,gy,gl)
    g[sid] = 0

    # f, g, x, y, l
    # integer octile heuristic: h = 14*min(dx,dy) + 10*(max-dmin)
    def h(x:int,y:int)->int:
        dx = abs(x-gx); dy = abs(y-gy)
        m = dx if dx<dy else dy
        M = dx if dx>dy else dy
        return 14*m + 10*(M-m) + (5 if sl!=gl else 0)  # tiny bias for target layer

    openh = [(h(sx,sy), 0, sx, sy, sl)]
    while openh:
        _f, gs, x, y, l = heapq.heappop(openh)
        cid = pid(x,y,l)
        if closed[cid]: 
            continue
        closed[cid] = True
        if cid == gid:
            return True, (x,y,l)  # we only need goal pos for reservation

        o = obst_bool[l]
        # 8-way neighbors
        # hoist locals
        yrow = o[y]
        for dx,dy in DIRS8:
            nx,ny = x+dx, y+dy
            if nx<0 or nx>=Wg or ny<0 or ny>=Hg: 
                continue
            # corner-cut prevention (boolean index lookups)
            if dx and dy:
                if yrow[x+dx] or obst_bool[l][y+dy][x]:
                    continue
            # allow stepping onto endpoints
            if o[ny][nx] and not ((nx,ny,l)==(sx,sy,sl) or (nx,ny,l)==(gx,gy,gl)):
                continue
            nid = pid(nx,ny,l)
            if closed[nid]:
                continue
            step = STEP_I[(dx,dy)]
            ng = gs + step
            if ng < g[nid]:
                g[nid] = ng
                heapq.heappush(openh, (ng + h(nx,ny), ng, nx, ny, l))

        # via
        nl = 1 - l
        if not obst_bool[nl][y][x]:
            nid = pid(x,y,nl)
            if not closed[nid]:
                ng = gs + VIA_I
                if ng < g[nid]:
                    g[nid] = ng
                    heapq.heappush(openh, (ng + h(x,y), ng, x, y, nl))

    return False, None

# ---------------- Route-all & Benchmark ----------------
def route_all_baseline(obst_set, nets):
    routed=0
    for a,b in nets:
        pa,pb=pads[a],pads[b]
        sa=(pa["x"],pa["y"], TOP if pa["thru_hole"] else pa["layer"])
        sb=(pb["x"],pb["y"], TOP if pb["thru_hole"] else pb["layer"])
        ok,last = astar_base(obst_set, sa, sb)
        if not ok: continue
        routed+=1
        x,y,l,_,_ = last
        obst_set[l].add((x,y))  # reserve goal only (sparse)
    return routed

def route_all_fast(obst_bool, nets):
    routed=0
    for a,b in nets:
        pa,pb=pads[a],pads[b]
        sa=(pa["x"],pa["y"], TOP if pa["thru_hole"] else pa["layer"])
        sb=(pb["x"],pb["y"], TOP if pb["thru_hole"] else pb["layer"])
        ok,goal = astar_fast(obst_bool, sa, sb)
        if not ok: continue
        routed+=1
        x,y,l = goal
        obst_bool[l][y][x] = True  # reserve goal only (sparse)
    return routed

def benchmark(nets_count=400, seed=1):
    synth(seed)
    nets = gen_nets(nets_count, min_mm=15.0, seed=seed)

    ob_set = build_obstacles_set()
    t0=time.perf_counter(); r0=route_all_baseline(ob_set, nets); t1=time.perf_counter()

    ob_bool = build_obstacles_bool()
    t2=time.perf_counter(); r1=route_all_fast(ob_bool, nets); t3=time.perf_counter()

    base=t1-t0; opt=t3-t2; speed=(base-opt)/base*100 if base>0 else 0.0
    print(f"Baseline: {base:.3f}s -> routes {r0}/{len(nets)}")
    print(f"Optimized: {opt:.3f}s -> routes {r1}/{len(nets)}")
    print(f"Speedup: {speed:.1f}%")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--route", action="store_true")
    ap.add_argument("--nets", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    args=ap.parse_args()

    if args.bench:
        benchmark(args.nets, args.seed); return

    synth(args.seed)
    nets = gen_nets(args.nets, 15.0, args.seed)
    ob_bool = build_obstacles_bool()
    r = route_all_fast(ob_bool, nets)
    print(f"Optimized router placed {r}/{len(nets)} nets")

if __name__=="__main__":
    main()
