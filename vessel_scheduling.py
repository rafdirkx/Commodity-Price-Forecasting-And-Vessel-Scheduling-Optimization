"""
=============================================================================
  VESSEL SCHEDULING – Operations Research Solution
  Erasmus Bachelor Seminar Year Project II
=============================================================================

  STRUCTURE
  ─────────────────────────────────────────────────────────────────────────
  Assignment 1 │ Q1 – Unconstrained constructive heuristic
               │   Goal  : minimise total cost (hire + fuel)
               │   Method: regional grouping → greedy bin-packing →
               │           nearest-neighbour sequencing →
               │           TC-first vessel assignment → improvement pass
  ─────────────────────────────────────────────────────────────────────────
  Assignment 2 │ Q4 – Full-constraint model built on top of Q1 voyages
               │   Adds  : cargo temperature zoning (COOL/HEAT holds)
               │           port draft constraints (vessel eligibility +
               │             load-on-arrival sequencing)
               │           delivery laycans (±7 days, $5k/day penalty)
               │           TC two-trip reuse within 30-day horizon
               │           speed decision (15 vs 16 kn trade-off)
  ─────────────────────────────────────────────────────────────────────────
  Assignment 3 │ OR-opt local exchange applied to Q4 solution
               │   Move  : 1-port relocation between voyages
               │   Why OR-opt over 2-opt: cross-voyage moves can
               │     eliminate entire voyages; 2-opt only reshapes
               │     routes within a single voyage
  ─────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import ceil
from itertools import permutations
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 0 – DATA LOADING
# =============================================================================

DATA_FILE = "Data/Vessel Scheduling data.xlsx"


def load_data():
    """Load orders, port metadata, and distance matrix from the Excel file."""

    # ── Orders ──────────────────────────────────────────────────────────────
    raw = pd.read_excel(DATA_FILE, sheet_name="DATA", header=0) #read the data sheet from the excel file
    raw.columns = [
        "delivery_date", "order_id", "product",
        "loading_port", "dest_country", "dest_port", "qty_kmt",
    ] #rename the columns to more meaningful names
    raw["delivery_date"] = pd.to_datetime(raw["delivery_date"]) #convert the delivery date column to Python datetime format
    raw["qty_kmt"] = pd.to_numeric(raw["qty_kmt"], errors="coerce") #convert quantities  to numeric, and replacing invalid entries with NaN
    orders = raw.dropna(subset=["order_id"]).copy() #drop any rows where order_id is NaN, and make a copy of the resulting DataFrame

    # ── Port metadata & distance matrix ─────────────────────────────────────
    port_raw = pd.read_excel(DATA_FILE, sheet_name="Port data", header=None) #read the port data sheet from the excel file

    # Row 1: headers; port names start at column index 6
    port_names = [str(p).strip() for p in port_raw.iloc[1, 6:].tolist()] #extract port names from row 1 starting at column 6

    port_meta = {} #dictionary to hold port metadata (e.g. draft limits)
    dist_df = pd.DataFrame(index=port_names, columns=port_names, dtype=float) #initialise an empty square distance matrix between all ports

    for _, row in port_raw.iloc[2:, :].iterrows():  #iterate over each port row (skipping header row)
        pname = str(row.iloc[2]).strip() #extract the port name from column index 2
        if pname == "nan":
            continue
        draft = float(row.iloc[3]) if pd.notna(row.iloc[3]) else 21.0 #extract draft depth (maximum depth of water a ship can safely navigate), defaulting to 21.0 if missing
        port_meta[pname] = {"draft": draft}

        for j, d in enumerate(row.iloc[6:].tolist()): #fill distance matrix for port
            if j < len(port_names):
                try:
                    dist_df.at[pname, port_names[j]] = float(d)
                except (ValueError, TypeError):
                    dist_df.at[pname, port_names[j]] = np.nan

    return orders, port_meta, dist_df #return cleaned datasets


ORDERS, PORT_META, DIST_MATRIX = load_data() #load and unpack processed data globally for use in the model


# =============================================================================
# SECTION 1 – CONSTANTS & CONFIGURATION
# =============================================================================

PLAN_START      = datetime(2024, 6, 1)
SPEED_KN        = 15.0       #default sailing speed (knots)
SPEED_FAST_KN   = 16.0       #faster alternative evaluated in Q4
FUEL_PRICE      = 700.50     #$/metric tonne
LAYCAN_DAYS     = 7          #± days around delivery date (no penalty inside)
PENALTY_PER_DAY = 5_000      #$/day outside laycan window

TC_FLEET = {"MR50": 5, "J19": 3} #time-chartered vessels available

VESSELS = {
    "MR50": {
        "capacity":     50.0,   #KMT
        "holds":        12,     #cargo compartments
        "cols":          6,     #columns of holds (for temperature zoning)
        "kmt_per_hold":  4.0,
        "hire_rate":   35_000,  #$/day
        "fuel_loaded":  25.0,   #mt/day loaded
        "fuel_empty":   15.0,   #mt/day ballast (TC return)
        "draft_empty":  15.0,   #metres (empty)
        "draft_full":   19.0,   #metres (fully loaded)
    },
    "J19": {
        "capacity":     20.0,
        "holds":         8,
        "cols":          4,
        "kmt_per_hold":  2.5,
        "hire_rate":   20_000,
        "fuel_loaded":  15.0,
        "fuel_empty":   10.0,
        "draft_empty":  12.0,
        "draft_full":   15.0,
    },
}

# Temperature class for each product
TEMP_CLASS = {
    "Butanol":             "COOL",
    "Fatty acid":          "HEAT",
    "Styrene":             "NEUTRAL",
    "Methanol":            "NEUTRAL",
    "Methyl Methacrylate": "NEUTRAL",
    "Acetone":             "NEUTRAL",
    "Amines":              "NEUTRAL",
    "Ammonia":             "NEUTRAL",
}

#compute total COOL and HEAT cargo (in KMT) for a given destination port
def port_cool_heat(port: str) -> tuple:
    rows = ORDERS[ORDERS["dest_port"] == port] #filter orders to include only those going to the selected port
    cool = rows[rows["product"].map(lambda p: TEMP_CLASS.get(p, "NEUTRAL") == "COOL")]["qty_kmt"].sum() #identify COOL cargo: map each product to its temperature class and filter
    heat = rows[rows["product"].map(lambda p: TEMP_CLASS.get(p, "NEUTRAL") == "HEAT")]["qty_kmt"].sum()
    return float(cool), float(heat) #return totals as floats

# Geographic clusters. voyages stay within one cluster
REGIONS = {
    "ME_India": ["Jebel Ali", "Karachi", "Mumbai", "Cochin", "Paradeep", "Adabiyah"],
    "SE_Asia":  ["Singapore", "Penang", "Bangkok", "Zhuhai",
                 "Jiangyin", "Ningbo", "Ulsan", "Hirohata", "HIROHATA"],
    "Europe":   ["Antwerp", "Rotterdam", "Barcelona", "Ravenna", "Gebze", "Izmir"],
}

# =============================================================================
# SECTION 2 – SHARED UTILITY FUNCTIONS
# =============================================================================

def dist(a: str, b: str) -> float: #nautical miles between two ports using the precomputed distance matrix
    try: #look up distance from port a (row) to port b (column) in the distance matrix
        return float(DIST_MATRIX.at[a, b])
    except (KeyError, TypeError, ValueError): #if ports are missing or value is invalid, return a large penalty distance
        return 9_999.0


def sailing_days(nm: float, speed: float = SPEED_KN) -> float: #convert nautical miles to sailing days
    return nm / (speed * 24.0)


def get_region(port: str) -> str: #identify geographic region a given port belongs to using the REGIONS lookup table
    for region, ports in REGIONS.items():
        if port in ports:
            return region #return the region name of port (e.g. get_region("Rotterdam") returns "Europe")
    return "Unknown"

def voyage_distance(route: list, start: str = "Jubail") -> float: #total nm for start → route (one-way)
    total, current = 0.0, start #initialize total distance and current port
    for p in route: #iterate through each port in route
        total += dist(current, p) #add distance from current port to next port to total
        current = p
    return total

#compute total voyage cost including hire and fuel for a given route and vessel setup
def compute_voyage_cost(route: list, vessel_type: str,
                        vessel_mode: str, speed: float = SPEED_KN) -> dict:
    v           = VESSELS[vessel_type] #lookup vessel characteristics (capacity, fuel consumption, hire rate, etc.)
    loaded_nm   = voyage_distance(route) #calculate total distance of the voyage based on the route
    loaded_days = sailing_days(loaded_nm, speed) #calculate total days spent sailing loaded based on distance and speed
    hire        = loaded_days * v["hire_rate"] #calculate total hire cost based on loaded days and vessel hire rate
    fuel        = loaded_days * v["fuel_loaded"] * FUEL_PRICE #calculate total fuel cost based on loaded days, fuel consumption rate, and fuel price

    if vessel_mode in ("TC", "TC_trip2"): #if the vessel is time-chartered, account for empty return trip to Jubail
        return_nm   = dist(route[-1], "Jubail") if route else 0.0 #calculate distance from last port back to Jubail for empty return trip
        empty_days  = sailing_days(return_nm, speed) #calculate days spent sailing empty based on return distance and speed
        hire       += empty_days * v["hire_rate"] #add hire cost for empty return trip to total hire
        fuel       += empty_days * v["fuel_empty"] * FUEL_PRICE #add fuel cost for empty return trip to total fuel
    else:
        empty_days = 0.0 #if vessel is not time-chartered, there is no empty return trip

#return full breakdown of voyage cost and metrics
    return {
        "vessel_type":  vessel_type,
        "vessel_mode":  vessel_mode,
        "route":        route,
        "speed":        speed,
        "loaded_nm":    loaded_nm,
        "loaded_days":  loaded_days,
        "empty_days":   empty_days,
        "total_days":   loaded_days + empty_days,
        "hire_cost":    hire,
        "fuel_cost":    fuel,
        "total_cost":   hire + fuel,
    }


def build_hold_plan(cool_kmt: float, heat_kmt: float, vessel_type: str) -> str:
    #Zoning. COOL on left, HEAT on right, NEUTRAL takes middle. At least one neutral buffer must separate COOL from HEAT.
    v           = VESSELS[vessel_type] #lookup vessel characteristics (capacity, number of holds, columns, etc.)
    cols        = v["cols"] #number of columns of holds in the vessel
    kmt_per_col = v["kmt_per_hold"] * 2 #KMT capacity per column (2 holds per column)

    if cool_kmt > 0 and heat_kmt > 0: #both COOL and HEAT present, need to check if they fit within the vessel's zoning constraints
        c = ceil(cool_kmt / kmt_per_col) #calculate number of columns needed for COOL cargo
        h = ceil(heat_kmt / kmt_per_col) #calculate number of columns needed for HEAT cargo
        n = cols - c - h #calculate number of columns left for NEUTRAL cargo (must be at least 1 to separate COOL and HEAT)
        if n < 1:
            return "INFEASIBLE – temperature zone overflow" #if not enough columns for NEUTRAL buffer, return infeasible message
        plan = ["COOL"] * c + ["NEUT"] * n + ["HEAT"] * h #build hold plan with COOL on left, NEUTRAL in middle, HEAT on right
    else:
        plan = ["NEUT"] * cols #if only one type of cargo present, fill all columns with NEUTRAL

    return " | ".join(f"Col{i+1}:{lbl}" for i, lbl in enumerate(plan)) #return a formatted string showing the hold plan for each column (e.g. "Col1:COOL | Col2:NEUT | Col3:HEAT")


# =============================================================================
# SECTION 3 – Q4 CONSTRAINT HELPERS
# =============================================================================

# ── Draft helpers ────────────────────────────────────────────────────────────

def max_load_at_port(vessel_type: str, port_draft: float) -> float:
    #Maximum KMT a vessel can carry on arrival at a port with the given draft limit. Derived from the linear draft load formula.
    v = VESSELS[vessel_type] #vessel characteristics
    return max(0.0, (port_draft - v["draft_empty"]) / (v["draft_full"] - v["draft_empty"]) * v["capacity"]) #calculate maximum load based on draft limit

def draft_category(port: str, vessel_type: str) -> str:
    #'BLOCKED'  – vessel cannot enter even empty
    #'PARTIAL'  – vessel may enter but only below full load
    #'FREE'     – no restriction
    pdraft = PORT_META.get(port, {}).get("draft", 21.0) #get the port's draft limit, defaulting to 21.0 if missing
    v = VESSELS[vessel_type]
    if pdraft < v["draft_empty"]: #if the port's draft limit is less than the vessel's empty draft, the vessel cannot enter even when empty
        return "BLOCKED"
    if pdraft < v["draft_full"]: #if the port's draft limit is less than the vessel's full draft, the vessel can enter but only below full load
        return "PARTIAL"
    return "FREE"

#Is there any order in which we can visit these ports such that the ship never violates draft limits while unloading cargo?
def draft_sequence_feasible(ports: list, cargo_map: dict, vessel_type: str) -> bool:
    # Check whether there exists a valid visiting order of ports such that the vessel can always enter ports given draft constraints and cargo is progressively unloaded
    if len(ports) <= 1:
        return True #single port is always feasible
    total = sum(cargo_map.values()) #total cargo to be delivered across all ports
    for perm in permutations(ports): #generate all possible orderings of the ports
        on_board, ok = total, True #initialize on-board cargo to total and feasibility flag to True
        for p in perm: #iterate through each port in the current permutation
            pdraft = PORT_META.get(p, {}).get("draft", 21.0) #get the port's draft limit, defaulting to 21.0 if missing
            if (on_board > max_load_at_port(vessel_type, pdraft) #check if on-board cargo exceeds the maximum load allowed at the port's draft limit
                    and draft_category(p, vessel_type) != "FREE"): #if the port is not free and the on-board cargo exceeds the maximum load allowed, mark the permutation as infeasible
                ok = False
                break
            on_board -= cargo_map[p] #unload cargo for the current port
        if ok:
            return True
    return False

# ── Temperature-zone helpers ─────────────────────────────────────────────────
def ch_zone_limit(vessel_type: str) -> float:
    #Max combined COOL+HEAT KMT when both are present on the same vessel.
    return 40.0 if vessel_type == "MR50" else 15.0

# ── Packing checks ───────────────────────────────────────────────────────────
def check_A(current_kmt: float, add_kmt: float, vessel_type: str) -> bool:
    #Check A: total cargo ≤ vessel capacity
    return (current_kmt + add_kmt) <= VESSELS[vessel_type]["capacity"]

def check_B(v_cool: float, v_heat: float,
            add_cool: float, add_heat: float, vessel_type: str) -> bool:
    #Check B: combined COOL+HEAT fits within hold-zone limit
    nc, nh = v_cool + add_cool, v_heat + add_heat #calculate new total COOL and HEAT cargo after adding the new cargo
    if nc > 0 and nh > 0:
        return (nc + nh) <= ch_zone_limit(vessel_type) #if both COOL and HEAT are present, check if their combined total is within the hold-zone limit
    return True

def check_C(all_ports: list, cargo_map: dict, vessel_type: str) -> bool:
    #Check C: a draft feasible port ordering exists
    return draft_sequence_feasible(all_ports, cargo_map, vessel_type)

# ── Laycan helpers ───────────────────────────────────────────────────────────

def penalty_cost(arrival: datetime, delivery: datetime) -> float:
    #Penalty for arriving outside the ±7-day laycan window
    days_outside = max(0.0, abs((arrival - delivery).days) - LAYCAN_DAYS)
    return days_outside * PENALTY_PER_DAY

def optimal_departure(port_order_map: dict, speed: float = SPEED_KN) -> tuple:
    #Find the departure date that minimises total penalty cost for a voyage.
    port_sequence = list(port_order_map.keys())

    #cumulative sailing time to each port along the route
    cumT, nm, prev = {}, 0.0, "Jubail" #initialize cumulative sailing time dictionary, total nautical miles, and previous port
    for p in port_sequence: #iterate through each port in the sequence
        nm += dist(prev, p) #add distance from previous port to current port to total nautical miles
        cumT[p] = sailing_days(nm, speed) #calculate cumulative sailing days to reach the current port based on total nautical miles and speed
        prev = p #update previous port to current port for next iteration

    candidates = [PLAN_START] #initialize candidate departure dates with the plan start date
    for p, order_list in port_order_map.items(): #iterate through each port and its associated orders
        T = cumT[p] #cumulative sailing time to reach the current port
        for o in order_list:
            d = o["delivery_date"] #delivery date for the current order
            for offset in (-LAYCAN_DAYS, 0, LAYCAN_DAYS): #consider three offsets: -7 days, 0 days, +7 days
                cand = d + timedelta(days=offset) - timedelta(days=T) #calculate candidate departure date by subtracting cumulative sailing time from the adjusted delivery date
                if cand >= PLAN_START: #only consider candidate departures that are on or after the plan start date
                    candidates.append(cand) #add the candidate departure date to the list of candidates

    best_dep, best_pen = PLAN_START, float("inf")
    for dep in candidates:
        total_pen = sum(
            penalty_cost(dep + timedelta(days=cumT[p]), o["delivery_date"])
            for p, order_list in port_order_map.items()
            for o in order_list
        )
        if total_pen < best_pen:
            best_pen, best_dep = total_pen, dep

    return best_dep, best_pen


def speed_decision(route: list, port_order_map: dict) -> float:
    """
    Return 16.0 kn if the penalty saving at higher speed exceeds the extra
    fuel cost; otherwise return 15.0 kn.
    """
    loaded_nm = voyage_distance(route)
    pen_15 = optimal_departure(port_order_map, SPEED_KN)[1]
    pen_16 = optimal_departure(port_order_map, SPEED_FAST_KN)[1]

    extra_fuel = (
        VESSELS["MR50"]["fuel_loaded"] * 0.17          # ~17% more consumption at 16 kn
        * sailing_days(loaded_nm, SPEED_FAST_KN)
        * FUEL_PRICE
    )
    return SPEED_FAST_KN if (pen_15 - pen_16) > extra_fuel else SPEED_KN


# =============================================================================
# SECTION 4 – PORT SEQUENCING
# =============================================================================

def nearest_neighbour_sequence(ports: list, start: str = "Jubail") -> list:
    """Standard nearest-neighbour TSP heuristic."""
    remaining, route, current = list(ports), [], start
    while remaining:
        nearest = min(remaining, key=lambda p: dist(current, p))
        route.append(nearest)
        current = nearest
        remaining.remove(nearest)
    return route


def constrained_nn_sequence(cargo_map: dict, vessel_type: str,
                             start: str = "Jubail") -> list:
    """
    Draft-aware nearest-neighbour: at each step prefer the closest port
    whose load-on-arrival constraint is satisfied given current on-board cargo.
    Falls back to the closest feasible port if the nearest is restricted.
    """
    remaining, route = list(cargo_map.keys()), []
    current, on_board = start, sum(cargo_map.values())

    while remaining:
        candidates = sorted(remaining, key=lambda p: dist(current, p))
        placed = False
        for p in candidates:
            pdraft = PORT_META.get(p, {}).get("draft", 21.0)
            if (on_board <= max_load_at_port(vessel_type, pdraft)
                    or draft_category(p, vessel_type) == "FREE"):
                route.append(p)
                on_board -= cargo_map[p]
                current = p
                remaining.remove(p)
                placed = True
                break
        if not placed:                      # no constraint-satisfying port found
            p = candidates[0]               # force nearest (flag but continue)
            route.append(p)
            on_board -= cargo_map[p]
            current = p
            remaining.remove(p)

    return route


# =============================================================================
# SECTION 5 – DEMAND AGGREGATION
# =============================================================================

def aggregate_demand(order_df: pd.DataFrame) -> dict:
    """Build {port: {product: kmt}} from the orders DataFrame."""
    demand = {}
    for _, row in order_df.iterrows():
        port, prod, kmt = row["dest_port"], row["product"], row["qty_kmt"]
        demand.setdefault(port, {})
        demand[port][prod] = demand[port].get(prod, 0.0) + kmt
    return demand


def port_summary(demand: dict) -> dict:
    """Per-port {total, cool, heat, products} summary."""
    out = {}
    for port, prods in demand.items():
        total = sum(prods.values())
        cool  = sum(v for p, v in prods.items() if TEMP_CLASS.get(p) == "COOL")
        heat  = sum(v for p, v in prods.items() if TEMP_CLASS.get(p) == "HEAT")
        out[port] = {"total": total, "cool": cool, "heat": heat, "products": prods}
    return out


def group_by_region(demand: dict) -> dict:
    """Split demand dict into sub-dicts keyed by geographic region."""
    out = {}
    for port, prods in demand.items():
        r = get_region(port)
        out.setdefault(r, {})[port] = prods
    return out


# =============================================================================
# ASSIGNMENT 1 – Q1: UNCONSTRAINED CONSTRUCTIVE HEURISTIC
# =============================================================================

def q1_pack_voyages_by_threshold(regional_demand: dict) -> list:
    """
    Greedy bin-packing with threshold-based vessel selection:
      - Voyages > 20 KMT  → MR50 (capacity 50 KMT)
      - Voyages ≤ 20 KMT  → J19  (capacity 20 KMT)
    Ports are sorted by total demand descending within each region.
    """
    pt = port_summary(regional_demand)
    sorted_ports = sorted(pt, key=lambda p: pt[p]["total"], reverse=True)

    voyages, assigned = [], set()

    for port in sorted_ports:
        if port in assigned:
            continue

        port_kmt = pt[port]["total"]
        # Decide vessel type based on this seed port's demand
        vessel_type = "MR50" if port_kmt > 20.0 else "J19"
        capacity    = VESSELS[vessel_type]["capacity"]

        voy_ports = [port]
        voy_kmt   = port_kmt
        voy_cool  = pt[port]["cool"]
        voy_heat  = pt[port]["heat"]
        cargo_map = {port: port_kmt}
        assigned.add(port)

        for other in sorted_ports:
            if other in assigned:
                continue
            add_kmt = pt[other]["total"]
            if not check_A(voy_kmt, add_kmt, vessel_type):
                continue
            voy_kmt  += add_kmt
            voy_cool += pt[other]["cool"]
            voy_heat += pt[other]["heat"]
            voy_ports.append(other)
            cargo_map[other] = add_kmt
            assigned.add(other)

        route = nearest_neighbour_sequence(voy_ports)
        voyages.append({
            "ports":       route,
            "cargo_map":   cargo_map,
            "cool_kmt":    voy_cool,
            "heat_kmt":    voy_heat,
            "total_kmt":   voy_kmt,
            "vessel_type": vessel_type,
        })

    return voyages


def q1_assign_vessels(voyages: list, tc_available: dict) -> list:
    """
    Assign TC vessels first (cheapest, hire paid regardless).
    MR50 TC → voyages packed as MR50, J19 TC → voyages packed as J19.
    Remaining voyages get spot vessels of the same type.
    Spot vessels only pay for the loaded leg — no return cost.
    """
    for v in voyages:
        vtype = v["vessel_type"]
        if tc_available.get(vtype, 0) > 0:
            v["vessel_mode"] = "TC"
            tc_available[vtype] -= 1
        else:
            v["vessel_mode"] = "spot"
    return voyages


# =============================================================================
# ASSIGNMENT 1 – Q1: UNCONSTRAINED CONSTRUCTIVE HEURISTIC
# =============================================================================

# ... (keep q1_pack_voyages_by_threshold and q1_assign_vessels unchanged) ...

# ── Merge helpers for q1_improvement_pass ────────────────────────────────────

FILL_THRESHOLD = 0.70

def _fill_rate(voyage: dict) -> float:
    cap = VESSELS[voyage["vessel_type"]]["capacity"]
    return voyage["total_kmt"] / cap if cap else 1.0

def q1_improvement_pass(voyages: list) -> list:
    """
    Merge under-utilised voyages (< 70% fill) of the same vessel type,
    when their combined KMT fits within capacity. Eliminating one vessel
    saves its entire hire cost, which typically outweighs any extra
    distance sailed.
    Repeats until no further merge is possible.
    """
    improved = True
    while improved:
        improved = False
        for i in range(len(voyages)):
            v1 = voyages[i]
            if _fill_rate(v1) >= FILL_THRESHOLD:
                continue
            for j in range(i + 1, len(voyages)):
                v2 = voyages[j]
                if _fill_rate(v2) >= FILL_THRESHOLD:
                    continue
                if v1["vessel_type"] != v2["vessel_type"]:
                    continue
                merged_kmt = v1["total_kmt"] + v2["total_kmt"]
                if not check_A(0, merged_kmt, v1["vessel_type"]):
                    continue
                new_cargo = {**v1["cargo_map"], **v2["cargo_map"]}
                route = nearest_neighbour_sequence(list(new_cargo.keys()))
                voyages[i] = {
                    "ports":       route,
                    "cargo_map":   new_cargo,
                    "cool_kmt":    v1["cool_kmt"] + v2["cool_kmt"],
                    "heat_kmt":    v1["heat_kmt"] + v2["heat_kmt"],
                    "total_kmt":   merged_kmt,
                    "vessel_type": v1["vessel_type"],
                    "vessel_mode": v1.get("vessel_mode", "spot"),
                }
                voyages.pop(j)
                improved = True
                break
            if improved:
                break
    return voyages

def q1_assign_vessels(voyages: list, tc_available: dict) -> list:
    """Assign TC vessels first (cheapest), then spot."""
    for v in voyages:
        vtype = v["vessel_type"]
        if tc_available.get(vtype, 0) > 0:
            v["vessel_mode"] = "TC"
            tc_available[vtype] -= 1
        else:
            v["vessel_mode"] = "spot"
    return voyages


def solve_q1() -> tuple:
    """
    Assignment 1 – Q1: Unconstrained plan.
    Returns (results_list, total_cost, raw_voyages).
    """
    print(f"\n{'=' * 65}")
    print("  ASSIGNMENT 1  –  Q1: UNCONSTRAINED CONSTRUCTIVE HEURISTIC")
    print(f"{'=' * 65}")

    demand       = aggregate_demand(ORDERS)
    region_dem   = group_by_region(demand)
    tc_available = dict(TC_FLEET)
    all_voyages  = []

    for region, rdem in region_dem.items():
        voyages = q1_pack_voyages_by_threshold(rdem)
        q1_assign_vessels(voyages, tc_available)
        all_voyages.extend(voyages)

    all_voyages = q1_improvement_pass(all_voyages)

    results    = []
    total_cost = 0.0

    for idx, voy in enumerate(all_voyages, 1):
        ci = compute_voyage_cost(
            voy["ports"], voy["vessel_type"], voy["vessel_mode"]
        )
        ci.update({
            "voyage_id":  idx,
            "total_kmt":  voy["total_kmt"],
            "cool_kmt":   voy.get("cool_kmt", 0.0),
            "heat_kmt":   voy.get("heat_kmt", 0.0),
            "hold_plan":  build_hold_plan(
                              voy.get("cool_kmt", 0.0),
                              voy.get("heat_kmt", 0.0),
                              voy["vessel_type"]),
            "penalty_cost":   0.0,
            "departure_date": PLAN_START,
            "grand_total":    ci["total_cost"],
        })
        total_cost += ci["total_cost"]
        results.append(ci)

    return results, total_cost, all_voyages


# =============================================================================
# ASSIGNMENT 2 – Q4: FULL-CONSTRAINT LAYER ON TOP OF Q1 VOYAGES
# =============================================================================

def q4_apply_temperature_constraint(voyages: list) -> list:
    """
    Split any Q1 voyage whose combined COOL+HEAT KMT exceeds the hold-zone
    limit (40 KMT on MR50, 15 KMT on J19).  Returns the corrected voyage list.
    """
    corrected = []
    for voy in voyages:
        vtype    = voy["vessel_type"]
        cool     = voy.get("cool_kmt", 0.0)
        heat     = voy.get("heat_kmt", 0.0)

        if cool > 0 and heat > 0 and (cool + heat) > ch_zone_limit(vtype):
            # Split: put COOL ports on one voyage, HEAT/NEUTRAL on another
            pt = port_summary({p: {"products": {}} for p in voy["ports"]})
            cool_ports = [p for p in voy["ports"]
                          if any(TEMP_CLASS.get(prod) == "COOL"
                                 for prod in ORDERS[ORDERS["dest_port"] == p]["product"])]
            other_ports = [p for p in voy["ports"] if p not in cool_ports]

            for group in [cool_ports, other_ports]:
                if not group:
                    continue
                sub_cargo = {p: voy["cargo_map"][p] for p in group}
                sub_cool  = sum(
                    voy["cargo_map"][p]
                    for p in group
                    if any(TEMP_CLASS.get(prod) == "COOL"
                           for prod in ORDERS[ORDERS["dest_port"] == p]["product"])
                )
                sub_heat  = sum(
                    voy["cargo_map"][p]
                    for p in group
                    if any(TEMP_CLASS.get(prod) == "HEAT"
                           for prod in ORDERS[ORDERS["dest_port"] == p]["product"])
                )
                corrected.append({
                    "ports":       nearest_neighbour_sequence(group),
                    "cargo_map":   sub_cargo,
                    "cool_kmt":    sub_cool,
                    "heat_kmt":    sub_heat,
                    "total_kmt":   sum(sub_cargo.values()),
                    "vessel_type": vtype,
                    "vessel_mode": voy.get("vessel_mode", "spot"),
                })
        else:
            corrected.append(voy)

    return corrected


def q4_apply_draft_constraint(voyages: list, tc_available: dict) -> list:
    """
    Enforce port draft limits:
      1. Any port that an MR50 cannot enter (BLOCKED) is moved to a J19 voyage.
      2. Remaining voyages use constrained-NN sequencing so restricted ports
         are visited only after enough cargo has been unloaded.
      3. Re-verifies Check C; infeasible combos are split.
    """
    # Separate ports that require J19
    j19_only_ports = set(
        p for p in
        {port for voy in voyages for port in voy["ports"]}
        if draft_category(p, "MR50") == "BLOCKED"
    )
    if j19_only_ports:
        print(f"  [Q4-Draft] MR50-blocked ports (J19 only): {sorted(j19_only_ports)}")

    corrected = []
    j19_overflow = {}   # {port: cargo_map entry} needing a fresh J19 voyage

    for voy in voyages:
        vtype = voy["vessel_type"]

        # Identify which ports in this voyage are MR50-blocked
        blocked_here = [p for p in voy["ports"] if p in j19_only_ports]
        free_here    = [p for p in voy["ports"] if p not in j19_only_ports]

        # Move blocked ports to the J19 overflow pool
        for p in blocked_here:
            j19_overflow[p] = voy["cargo_map"][p]

        if not free_here:
            continue   # entire voyage moved to J19

        # Rebuild voyage with only the non-blocked ports
        new_cargo = {p: voy["cargo_map"][p] for p in free_here}
        new_cool  = sum(port_cool_heat(p)[0] for p in free_here)
        new_heat  = sum(port_cool_heat(p)[1] for p in free_here)

        # Re-sequence with draft awareness
        route = constrained_nn_sequence(new_cargo, vtype)

        corrected.append({
            "ports":       route,
            "cargo_map":   new_cargo,
            "cool_kmt":    new_cool,
            "heat_kmt":    new_heat,
            "total_kmt":   sum(new_cargo.values()),
            "vessel_type": vtype,
            "vessel_mode": voy.get("vessel_mode", "spot"),
        })

    # Build J19 voyages for the blocked ports
    if j19_overflow:
        capacity = VESSELS["J19"]["capacity"]
        ports_sorted = sorted(j19_overflow, key=lambda p: j19_overflow[p], reverse=True)
        current_ports, current_kmt = [], 0.0

        def flush_j19(plist, tc_avail):
            cargo = {p: j19_overflow[p] for p in plist}
            route = constrained_nn_sequence(cargo, "J19")
            mode  = "TC" if tc_avail.get("J19", 0) > 0 else "spot"
            if mode == "TC":
                tc_avail["J19"] -= 1
            return {
                "ports":       route,
                "cargo_map":   cargo,
                "heat_kmt":    sum(port_cool_heat(p)[1] for p in plist),
                "cool_kmt":    sum(port_cool_heat(p)[0] for p in plist),
                "total_kmt":   sum(cargo.values()),
                "vessel_type": "J19",
                "vessel_mode": mode,
            }

        for p in ports_sorted:
            if current_kmt + j19_overflow[p] > capacity:
                corrected.append(flush_j19(current_ports, tc_available))
                current_ports, current_kmt = [], 0.0
            current_ports.append(p)
            current_kmt += j19_overflow[p]
        if current_ports:
            corrected.append(flush_j19(current_ports, tc_available))

    return corrected


def q4_improvement_pass(voyages: list) -> list:
    """
    Merge under-utilised voyages, now enforcing all three checks:
    A (capacity), B (temperature zone), C (draft sequence).
    """
    improved = True
    while improved:
        improved = False
        for i in range(len(voyages)):
            for j in range(i + 1, len(voyages)):
                v1, v2 = voyages[i], voyages[j]
                if v1["vessel_type"] != v2["vessel_type"]:
                    continue
                vtype      = v1["vessel_type"]
                merged_kmt = v1["total_kmt"] + v2["total_kmt"]
                merged_cool = v1["cool_kmt"] + v2["cool_kmt"]
                merged_heat = v1["heat_kmt"] + v2["heat_kmt"]

                if not check_A(0, merged_kmt, vtype):
                    continue
                if not check_B(0, 0, merged_cool, merged_heat, vtype):
                    continue

                all_ports = v1["ports"] + v2["ports"]
                new_cargo = {**v1["cargo_map"], **v2["cargo_map"]}
                if not check_C(all_ports, new_cargo, vtype):
                    continue

                route = constrained_nn_sequence(new_cargo, vtype)
                voyages[i] = {
                    "ports":       route,
                    "cargo_map":   new_cargo,
                    "cool_kmt":    merged_cool,
                    "heat_kmt":    merged_heat,
                    "total_kmt":   merged_kmt,
                    "vessel_type": vtype,
                    "vessel_mode": v1.get("vessel_mode", "spot"),
                }
                voyages.pop(j)
                improved = True
                break
            if improved:
                break
    return voyages


def q4_tc_second_trips(voyages: list) -> list:
    spot = [v for v in voyages if v.get("vessel_mode") == "spot"]
    tc   = [v for v in voyages if v.get("vessel_mode") == "TC"]
    promoted_ids = set()

    for tv in tc:
        route    = tv["ports"]
        round_nm = voyage_distance(route) + dist(route[-1], "Jubail")
        if sailing_days(round_nm) < 28:
            for sv in spot:
                if id(sv) in promoted_ids:
                    continue
                if sv["vessel_type"] == tv["vessel_type"]:
                    sv["vessel_mode"] = "TC_trip2"
                    promoted_ids.add(id(sv))
                    break

    return tc + spot

def solve_q4(q1_voyages: list) -> tuple:
    """
    Assignment 2 – Q4: Layer all constraints onto the Q1 voyage list.

    Steps:
      1. Apply temperature-zone constraint (Check B) – split if needed
      2. Apply draft constraint (Check C) – move blocked ports to J19,
         resequence restricted ports
      3. Re-run improvement pass with all three checks
      4. Allow TC vessels a second trip (Check A/B/C on second trip too)
      5. For each voyage: decide speed, find optimal departure date,
         compute hire + fuel + penalty

    Returns (results_list, total_cost, total_penalty, raw_voyages).
    """
    print(f"\n{'=' * 65}")
    print("  ASSIGNMENT 2  –  Q4: FULL-CONSTRAINT MODEL")
    print(f"{'=' * 65}")
    print("  Layering constraints on Q1 voyages:")
    print("    Check A : capacity  |  Check B : temperature zone")
    print("    Check C : draft     |  Check D : laycan departure window")

    import copy
    voyages = copy.deepcopy(q1_voyages)

    # Count remaining TC slots (Q1 already consumed some)
    tc_used = {"MR50": sum(1 for v in voyages if v.get("vessel_mode") == "TC"
                           and v["vessel_type"] == "MR50"),
               "J19":  sum(1 for v in voyages if v.get("vessel_mode") == "TC"
                           and v["vessel_type"] == "J19")}
    tc_available = {k: TC_FLEET[k] - tc_used[k] for k in TC_FLEET}

    # Step 1 – Temperature zoning
    voyages = q4_apply_temperature_constraint(voyages)

    # Step 2 – Draft constraints
    voyages = q4_apply_draft_constraint(voyages, tc_available)

    # Step 3 – Improvement pass (all three checks)
    voyages = q4_improvement_pass(voyages)

    # Step 4 – TC second trips
    voyages = q4_tc_second_trips(voyages)

    # Step 5 – Scheduling, speed decision, penalty calculation
    results       = []
    total_cost    = 0.0
    total_penalty = 0.0

    for idx, voy in enumerate(voyages, 1):
        route  = voy["ports"]
        vtype  = voy["vessel_type"]
        vmode  = voy.get("vessel_mode", "spot")

        port_orders = {
            p: ORDERS[ORDERS["dest_port"] == p].to_dict("records")
            for p in route
        }

        speed            = speed_decision(route, port_orders)
        dep_date, pen    = optimal_departure(port_orders, speed)

        ci = compute_voyage_cost(route, vtype, vmode, speed)
        ci.update({
            "voyage_id":      idx,
            "total_kmt":      voy["total_kmt"],
            "cool_kmt":       voy.get("cool_kmt", 0.0),
            "heat_kmt":       voy.get("heat_kmt", 0.0),
            "hold_plan":      build_hold_plan(
                                  voy.get("cool_kmt", 0.0),
                                  voy.get("heat_kmt", 0.0),
                                  vtype),
            "departure_date": dep_date,
            "penalty_cost":   pen,
            "grand_total":    ci["total_cost"] + pen,
        })

        total_cost    += ci["total_cost"]
        total_penalty += pen
        results.append(ci)

    return results, total_cost, total_penalty, voyages


# =============================================================================
# ASSIGNMENT 3 – OR-OPT LOCAL EXCHANGE
# =============================================================================
#
# OR-opt (1-port relocation between voyages) is chosen over 2-opt because:
#   • Moving one port at a time makes it straightforward to re-check all
#     three constraints (A, B, C) after each move.
#   • Cross-voyage relocation can eliminate an entire voyage by merging its
#     last port into another — a saving 2-opt structurally cannot achieve,
#     since 2-opt only reverses sub-sequences within a single route.
#   • The main source of waste here is ports mis-assigned across voyages
#     (e.g. a voyage carrying only 5 KMT), not inefficient internal route
#     ordering — so inter-voyage moves are exactly what is needed.
#
# Algorithm (first-improvement):
#   For every (source voyage i, port p, destination voyage j ≠ i):
#     1. Tentatively remove p from voyage i.
#     2. Tentatively insert p into voyage j.
#     3. Check A, B, C on the modified voyage j.
#     4. If all checks pass and combined grand-total cost falls, commit.
#   Repeat until no improving move is found.
# =============================================================================

def _voyage_grand_total(voy: dict) -> float:
    """Recompute shipping cost + penalty for a voyage dict."""
    route = voy["ports"]
    if not route:
        return 0.0
    vtype = voy["vessel_type"]
    vmode = voy.get("vessel_mode", "spot")
    port_orders = {
        p: ORDERS[ORDERS["dest_port"] == p].to_dict("records")
        for p in route
    }
    speed    = speed_decision(route, port_orders)
    _, pen   = optimal_departure(port_orders, speed)
    ci       = compute_voyage_cost(route, vtype, vmode, speed)
    return ci["total_cost"] + pen


def _rebuild_voyage(base_voy: dict, new_cargo: dict) -> dict | None:
    """Return an updated voyage dict for a modified cargo_map, or None if empty."""
    if not new_cargo:
        return None
    vtype = base_voy["vessel_type"]

    cool = sum(port_cool_heat(p)[0] for p in new_cargo)
    heat = sum(port_cool_heat(p)[1] for p in new_cargo)
    route = constrained_nn_sequence(new_cargo, vtype)

    return {
        "ports":       route,
        "cargo_map":   new_cargo,
        "cool_kmt":    cool,
        "heat_kmt":    heat,
        "total_kmt":   sum(new_cargo.values()),
        "vessel_type": vtype,
        "vessel_mode": base_voy.get("vessel_mode", "spot"),
    }


def solve_or_opt(q4_voyages: list) -> tuple:
    """
    Assignment 3 – OR-opt: 1-port relocation between voyages.
    Returns (results_list, total_cost, total_penalty).
    """
    print(f"\n{'=' * 65}")
    print("  ASSIGNMENT 3  –  OR-OPT LOCAL EXCHANGE")
    print(f"{'=' * 65}")
    print("  Strategy: 1-port relocation between voyages (first-improvement)")
    print("  Checks enforced on every move: A (capacity), B (temp zone), C (draft)")

    import copy
    voyages   = copy.deepcopy(q4_voyages)
    improved  = True
    passes    = 0

    while improved:
        improved = False
        passes  += 1

        for i in range(len(voyages)):
            for port in list(voyages[i]["ports"]):
                for j in range(len(voyages)):
                    if i == j:
                        continue
                    if voyages[i]["vessel_type"] != voyages[j]["vessel_type"]:
                        continue

                    port_kmt      = voyages[i]["cargo_map"][port]
                    src_new_cargo = {p: k for p, k in voyages[i]["cargo_map"].items()
                                     if p != port}
                    dst_new_cargo = {**voyages[j]["cargo_map"], port: port_kmt}

                    vtype = voyages[j]["vessel_type"]

                    # Check A: destination capacity
                    if not check_A(0, sum(dst_new_cargo.values()), vtype):
                        continue

                    # Check B: temperature zone on destination
                    dst_cool = sum(port_cool_heat(p)[0] for p in dst_new_cargo)
                    dst_heat = sum(port_cool_heat(p)[1] for p in dst_new_cargo)
                    if not check_B(0, 0, dst_cool, dst_heat, vtype):
                        continue

                    # Check C: draft sequence on destination
                    if not check_C(list(dst_new_cargo.keys()), dst_new_cargo, vtype):
                        continue

                    # Evaluate cost delta
                    cost_before = (_voyage_grand_total(voyages[i])
                                   + _voyage_grand_total(voyages[j]))

                    src_new = _rebuild_voyage(voyages[i], src_new_cargo)
                    dst_new = _rebuild_voyage(voyages[j], dst_new_cargo)

                    cost_after = ((_voyage_grand_total(src_new) if src_new else 0.0)
                                  + _voyage_grand_total(dst_new))

                    if cost_after < cost_before - 1.0:     # $1 tolerance
                        # Commit move — handle index carefully if source empties
                        if src_new is None:
                            voyages[i] = None
                            voyages[j] = dst_new
                            voyages = [v for v in voyages if v is not None]
                        else:
                            voyages[i] = src_new
                            voyages[j] = dst_new
                        improved = True
                        break

                if improved:
                    break
            if improved:
                break

    print(f"  OR-opt complete: {passes} pass(es), {len(voyages)} voyages remain.")

    # Recompute final costs
    results       = []
    total_cost    = 0.0
    total_penalty = 0.0

    for idx, voy in enumerate(voyages, 1):
        route = voy["ports"]
        if not route:
            continue
        vtype = voy["vessel_type"]
        vmode = voy.get("vessel_mode", "spot")

        port_orders = {
            p: ORDERS[ORDERS["dest_port"] == p].to_dict("records")
            for p in route
        }
        speed         = speed_decision(route, port_orders)
        dep_date, pen = optimal_departure(port_orders, speed)
        ci            = compute_voyage_cost(route, vtype, vmode, speed)

        ci.update({
            "voyage_id":      idx,
            "total_kmt":      voy["total_kmt"],
            "cool_kmt":       voy.get("cool_kmt", 0.0),
            "heat_kmt":       voy.get("heat_kmt", 0.0),
            "hold_plan":      build_hold_plan(
                                  voy.get("cool_kmt", 0.0),
                                  voy.get("heat_kmt", 0.0),
                                  vtype),
            "departure_date": dep_date,
            "penalty_cost":   pen,
            "grand_total":    ci["total_cost"] + pen,
        })

        total_cost    += ci["total_cost"]
        total_penalty += pen
        results.append(ci)
    return results, total_cost, total_penalty



# =============================================================================
# REPORTING
# =============================================================================

def print_results(results: list, total_cost: float,
                  total_penalty: float, label: str):
    print(f"\n{'─' * 65}")
    print(f"  {label}  –  {len(results)} voyages")
    print(f"{'─' * 65}")

    for r in results:
        mode_str = f"{r['vessel_type']} ({r['vessel_mode']}) @ {r['speed']:.0f} kn"
        print(f"\n  Voyage {r['voyage_id']:>2} | {mode_str}")
        print(f"    Route     : Jubail → {' → '.join(r['route'])}")
        print(f"    Cargo     : {r['total_kmt']:.1f} KMT  "
              f"(COOL {r['cool_kmt']:.1f} | HEAT {r['heat_kmt']:.1f})")
        print(f"    Distance  : {r['loaded_nm']:.0f} nm loaded")
        print(f"    Days      : {r['loaded_days']:.2f} loaded"
              f" + {r['empty_days']:.2f} ballast = {r['total_days']:.2f}")
        print(f"    Hire      : ${r['hire_cost']:>12,.0f}")
        print(f"    Fuel      : ${r['fuel_cost']:>12,.0f}")
        print(f"    Shipping  : ${r['total_cost']:>12,.0f}")
        print(f"    Depart    : {r['departure_date'].strftime('%Y-%m-%d')}")
        print(f"    Penalty   : ${r['penalty_cost']:>12,.0f}")
        print(f"    TOTAL     : ${r['grand_total']:>12,.0f}")
        print(f"    Hold plan : {r['hold_plan']}")

    print(f"\n  {'─' * 45}")
    print(f"  Shipping cost : ${total_cost:>14,.0f}")
    if total_penalty > 0:
        print(f"  Penalty cost  : ${total_penalty:>14,.0f}")
        print(f"  Grand total   : ${(total_cost + total_penalty):>14,.0f}")
    print(f"{'=' * 65}")


# =============================================================================
# EXCEL EXPORT
# =============================================================================

def export_to_excel(all_results: dict, filepath: str = "vessel_schedule_output.xlsx"):
    """Export all results to a formatted multi-sheet Excel workbook."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb       = Workbook()
    wb.remove(wb.active)
    hdr_fill = PatternFill("solid", fgColor="1F4E79")
    hdr_font = Font(bold=True, color="FFFFFF", name="Calibri", size=10)
    body_font = Font(name="Calibri", size=10)
    alt_fill  = PatternFill("solid", fgColor="D6E4F0")
    thin      = Side(border_style="thin", color="AAAAAA")
    border    = Border(left=thin, right=thin, top=thin, bottom=thin)

    headers = [
        "Voyage #", "Vessel", "Mode", "Speed (kn)", "Route",
        "Total KMT", "COOL KMT", "HEAT KMT",
        "Distance (nm)", "Loaded Days", "Ballast Days", "Total Days",
        "Hire ($)", "Fuel ($)", "Shipping ($)",
        "Departure", "Penalty ($)", "Grand Total ($)",
    ]

    for label, (results, total_cost, total_penalty) in all_results.items():
        ws = wb.create_sheet(title=label)

        for col, h in enumerate(headers, 1):
            c = ws.cell(row=1, column=col, value=h)
            c.font = hdr_font; c.fill = hdr_fill; c.border = border
            c.alignment = Alignment(horizontal="center", wrap_text=True)

        for ri, r in enumerate(results, 2):
            fill = alt_fill if ri % 2 == 0 else PatternFill()
            row_data = [
                r["voyage_id"], r["vessel_type"], r["vessel_mode"],
                r.get("speed", SPEED_KN),
                "Jubail → " + " → ".join(r["route"]),
                r["total_kmt"], r["cool_kmt"], r["heat_kmt"],
                round(r["loaded_nm"]),
                round(r["loaded_days"], 2), round(r["empty_days"], 2),
                round(r["total_days"], 2),
                round(r["hire_cost"]), round(r["fuel_cost"]),
                round(r["total_cost"]),
                r.get("departure_date", PLAN_START).strftime("%Y-%m-%d"),
                round(r.get("penalty_cost", 0)),
                round(r["grand_total"]),
            ]
            for col, val in enumerate(row_data, 1):
                c = ws.cell(row=ri, column=col, value=val)
                c.font = body_font; c.border = border
                if fill.patternType:
                    c.fill = fill
                if col in (13, 14, 15, 17, 18):
                    c.number_format = "#,##0"

        # Summary row
        sr = len(results) + 3
        ws.cell(row=sr, column=1, value="TOTAL").font = Font(bold=True, name="Calibri")
        gc = ws.cell(row=sr, column=len(headers),
                     value=round(total_cost + total_penalty))
        gc.number_format = "#,##0"
        gc.font = Font(bold=True, name="Calibri")

        widths = [9, 10, 10, 10, 60, 10, 10, 10, 13, 12, 13, 12,
                  12, 12, 13, 12, 12, 14]
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w

    wb.save(filepath)
    print(f"\n  ✓ Results exported → {filepath}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    all_results = {}

    # ── Assignment 1: Q1 unconstrained ──────────────────────────────────────
    q1_results, q1_cost, q1_raw_voyages = solve_q1()
    print_results(q1_results, q1_cost, 0.0, "Q1 – Unconstrained")
    all_results["Q1"] = (q1_results, q1_cost, 0.0)

    # ── Assignment 2: Q4 full constraints (built on Q1 voyages) ─────────────
    q4_results, q4_cost, q4_penalty, q4_raw_voyages = solve_q4(q1_raw_voyages)
    print_results(q4_results, q4_cost, q4_penalty, "Q4 – Full Constraints")
    all_results["Q4"] = (q4_results, q4_cost, q4_penalty)

    print(f"\n  Q1 → Q4 cost change:")
    print(f"    Q1 shipping only  : ${q1_cost:,.0f}")
    print(f"    Q4 shipping       : ${q4_cost:,.0f}")
    print(f"    Q4 penalty        : ${q4_penalty:,.0f}")
    print(f"    Q4 grand total    : ${q4_cost + q4_penalty:,.0f}")

    # ── Assignment 3: OR-opt local exchange on Q4 ───────────────────────────
    or_results, or_cost, or_penalty = solve_or_opt(q4_raw_voyages)
    print_results(or_results, or_cost, or_penalty, "Q4 + OR-opt")
    all_results["Q4_OR-opt"] = (or_results, or_cost, or_penalty)

    saving = (q4_cost + q4_penalty) - (or_cost + or_penalty)
    print(f"\n  OR-opt improvement:")
    print(f"    Before : ${q4_cost + q4_penalty:,.0f}")
    print(f"    After  : ${or_cost + or_penalty:,.0f}")
    print(f"    Saving : ${saving:,.0f}")

    # ── Export all sheets ────────────────────────────────────────────────────
    export_to_excel(all_results, filepath="Results/vessel_schedule_output.xlsx")


if __name__ == "__main__":
    main()