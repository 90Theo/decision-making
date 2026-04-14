import numpy as np
import pyomo.environ as pyo
from sklearn.cluster import KMeans

import matplotlib

# matplotlib.use('Agg')  # Needed because PriceProcessRestaurant.py calls plt.show() at import time
# ^Now not needded because I commented out the matplot stuff from that file

import SystemCharacteristics
from PriceProcessRestaurant import price_model
from OccupancyProcessRestaurant import next_occupancy_levels

params = SystemCharacteristics.get_fixed_data()

def generate_scenario_tree(state, L=3, branching=3, n_samples=30):
    t0 = state["current_time"]
    # First we make the root node
    nodes = [{
        'id': 0,
        'stage': 0,
        'parent': None,
        'prob': 1.0,
        'price': state["price_t"],
        'price_prev': state["price_previous"],
        'occ1': state["Occ1"],
        'occ2': state["Occ2"],
        'time': t0
    }]

    # We make a dict to keep track of which node ids belong to each stage
    stage_nodes = {0: [0]}
    node_counter = 1

    for stage in range(1, L):
        stage_nodes[stage] = []

        for parent_id in stage_nodes[stage - 1]:
            parent = nodes[parent_id]

            # Draw n_samples possible next states from the given stochastic processes
            samples = []
            for _ in range(n_samples):
                p_next = price_model(parent['price'], parent['price_prev'])
                o1, o2 = next_occupancy_levels(parent['occ1'], parent['occ2'])
                samples.append([p_next, o1, o2])
            samples = np.array(samples)

            # Cluster samples into representative points. Argument branching decides how many points to cluster to
            n_clust = branching
            km = KMeans(n_clusters=n_clust, n_init=5, random_state=20) # random_state=20 because we are group 20 :)
            labels = km.fit_predict(samples)
            centroids = km.cluster_centers_
            counts = np.bincount(labels, minlength=n_clust)
            cprobs = counts / counts.sum()

            # Create one child node per cluster
            for k in range(len(centroids)):
                nodes.append({
                    'id': node_counter,
                    'stage': stage,
                    'parent': parent_id,
                    'prob': parent['prob'] * cprobs[k],
                    'price': float(centroids[k][0]),
                    'price_prev': parent['price'],
                    'occ1': float(centroids[k][1]),
                    'occ2': float(centroids[k][2]),
                    'time': t0 + stage
                })
                stage_nodes[stage].append(node_counter)
                node_counter += 1

    # Build scenario paths: walk from each leaf back to root, then reverse
    leaf_ids = stage_nodes[L - 1]
    scenarios = []
    for leaf_id in leaf_ids:
        path = []
        n = leaf_id
        while True:
            path.append(n)
            n = nodes[n]['parent']
            if n == None:
                break
        path.reverse()
        scenarios.append(path)
    return nodes, scenarios

def build_and_solve_sp(params, state, nodes, scenarios):
    model = pyo.ConcreteModel()

    # We make a list with each set: The rooms and the nodes. This will make it easy to loop thorugh them later
    rooms = [1, 2]
    N = list(range(len(nodes)))

    model.N = pyo.Set(initialize=N)
    model.R = pyo.Set(initialize=rooms)

    # Unpack parameters
    P_max = params['heating_max_power']
    P_vent = params['ventilation_power']
    z_exch = params['heat_exchange_coeff']
    z_loss = params['thermal_loss_coeff']
    z_conv = params['heating_efficiency_coeff']
    z_cool = params['heat_vent_coeff']
    z_occ = params['heat_occupancy_coeff']
    e_occ = params['humidity_occupancy_coeff']
    e_vent = params['humidity_vent_coeff']
    T_low  = params['temp_min_comfort_threshold']
    T_OK = params['temp_OK_threshold']
    T_high = params['temp_max_comfort_threshold']
    H_high = params['humidity_threshold']
    U_vent = params['vent_min_up_time']
    T_out = params['outdoor_temperature']
    nT = params['num_timeslots']

    # Big-M constants
    M_T = 50.0 # This one is for temperature so 50 C° seems safe
    M_H = 200.0 # This is for humidity so 200% should be completely safe

    # Decision variables
    model.p = pyo.Var(model.R, model.N, domain=pyo.NonNegativeReals, bounds=(0, P_max)) # heating power
    model.T = pyo.Var(model.R, model.N, bounds=(0, 50)) # room temperature
    model.H = pyo.Var(model.N, bounds=(0, 200)) # humidity
    model.v = pyo.Var(model.N, domain=pyo.Binary) # ventilation on/off
    model.s = pyo.Var(model.N, domain=pyo.Binary) # ventilation start (1 if switched on)
    model.yLow = pyo.Var(model.R, model.N, domain=pyo.Binary) # 1 if temp <= T_low
    model.yOK = pyo.Var(model.R, model.N, domain=pyo.Binary) # 1 if temp >= T_OK
    model.yHigh = pyo.Var(model.R, model.N, domain=pyo.Binary) # 1 if temp >= T_high
    model.u = pyo.Var(model.R, model.N, domain=pyo.Binary) # low-override active
    model.c_v = pyo.Var(model.N, domain=pyo.Integers, bounds=(0, nT)) # consecutive hours vent has been on

    # We define the objective function which is to minimize the expected electricity cost 
    def objective_function(m):
        return sum(
            nodes[n]['prob'] * nodes[n]['price'] * 
            (sum(m.p[r, n] for r in rooms) + P_vent * m.v[n])
            for n in N if nodes[n]['time'] < nT)
    model.obj = pyo.Objective(rule=objective_function, sense=pyo.minimize)

    model.cons = pyo.ConstraintList()

    #  Constraints for each node 
    for n_id in N:
        node  = nodes[n_id]
        stage = node['stage']
        t_idx = node['time']
        parent_id   = node['parent']

        if t_idx >= nT:
            continue

        if stage == 0:
            # Fix the current observed state at the root node
            model.T[1, 0].fix(state["T1"])
            model.T[2, 0].fix(state["T2"])
            model.H[0].fix(state["H"])
            model.c_v[0].fix(state["vent_counter"])
            model.u[1, 0].fix(state["low_override_r1"])
            model.u[2, 0].fix(state["low_override_r2"])

            # Make s = 1 if ventilation starts now and 0 otherwise
            if state["vent_counter"] > 0:
                v_prev = 1 
            else:
                v_prev = 0
            model.cons.add(model.s[0] >= model.v[0] - v_prev)
            model.cons.add(model.s[0] <= model.v[0])
            model.cons.add(model.s[0] <= 1 - v_prev)

            # Force ventilation to stay on if minimum uptime is not yet met
            if 0 < state["vent_counter"] < U_vent:
                remaining_uptime = U_vent - state["vent_counter"]
                for path in scenarios:
                    for fix_n in path[0:remaining_uptime]:
                        model.v[fix_n].fix(1)
                    
        else:
            # Propagate dynamics from parent node
            parent = nodes[parent_id]
            t_out_p = T_out[parent['time']] if parent['time'] < len(T_out) else T_out[-1]

            # Temperature dynamics for each room
            for r in rooms:
                if r == 1:
                    other_room = 2
                    occ_r = parent['occ1']
                else:
                    other_room = 1
                    occ_r = parent['occ2']
                model.cons.add(
                    model.T[r, n_id] == model.T[r, parent_id]
                    + z_exch * (model.T[other_room, parent_id] - model.T[r, parent_id])
                    - z_loss * (model.T[r, parent_id] - t_out_p)
                    + z_conv * model.p[r, parent_id]
                    - z_cool * model.v[parent_id]
                    + z_occ * occ_r)

            # Humidity dynamics
            model.cons.add(
                model.H[n_id] == model.H[parent_id]
                + e_occ * (parent['occ1'] + parent['occ2'])
                - e_vent * model.v[parent_id])

            # Consecutive ventilation counter
            model.cons.add(model.c_v[n_id] <= nT * model.v[parent_id])
            model.cons.add(model.c_v[n_id] >= model.c_v[parent_id] + 1 - nT * (1 - model.v[parent_id]))
            model.cons.add(model.c_v[n_id] <= model.c_v[parent_id] + 1)

            # Low-override controller logic
            for r in rooms:
                model.cons.add(model.u[r, n_id] >= model.yLow[r, n_id]) # Must turn on if yLow == 1
                model.cons.add(model.u[r, n_id] <= model.u[r, parent_id] + model.yLow[r, n_id]) # Must not turn on if yLow == 0
                model.cons.add(model.u[r, n_id] >= model.u[r, parent_id] - model.yOK[r, n_id]) # Must stay on if yOK == 0
                model.cons.add(model.u[r, n_id] <= 1 - model.yOK[r, n_id]) # Must turn off if yOK == 1

            # Ventilation start indicator
            model.cons.add(model.s[n_id] >= model.v[n_id] - model.v[parent_id]) # s is 1 if ventilation just started
            model.cons.add(model.s[n_id] <= model.v[n_id]) # s is 0 if ventilation is off
            model.cons.add(model.s[n_id] <= 1 - model.v[parent_id]) # s must be 0 if parent node had ventilation on

        # Now we are outside the else meaning the following applies to all nodes (including root)
        # Big-M constraints to set the binary variables (yLow, yOK, yHigh) based on temperature
        for r in rooms:
            model.cons.add(model.T[r, n_id] >= T_high - M_T * (1 - model.yHigh[r, n_id])) # If T < T_high yHigh is forced to 0
            model.cons.add(model.T[r, n_id] <= T_high + M_T * model.yHigh[r, n_id]) # If T > T_high yHigh is forced to 1
            model.cons.add(model.T[r, n_id] <= T_low + M_T * (1 - model.yLow[r, n_id])) # If T > T_low yLow is forced to 0
            model.cons.add(model.T[r, n_id] >= T_low - M_T * model.yLow[r, n_id]) # If T < T_low yLow is forced to 1
            model.cons.add(model.T[r, n_id] >= T_OK - M_T * (1 - model.yOK[r, n_id])) # If T < T_OK yOK is forcced to 0
            model.cons.add(model.T[r, n_id] <= T_OK + M_T * model.yOK[r, n_id]) # If T > T_OK yOK is forced to 1

        # Heater must be OFF if temperature is already too high
        # Heater must be ON (full power) if low-override is active
        for r in rooms:
            model.cons.add(model.p[r, n_id] <= P_max * (1 - model.yHigh[r, n_id])) # No power if yHigh == 1
            model.cons.add(model.p[r, n_id] >= P_max * model.u[r, n_id]) # Full power if u == 1 (i.e. low-override is active)

        # Force ventilation on if humidity exceeds threshold
        model.cons.add(model.H[n_id] <= H_high + M_H * model.v[n_id])

    # Minimum uptime constraint: once ventilation starts, it must stay on for U_vent hours
    for path in scenarios:
        for i, n_id in enumerate(path):
            remaining = path[i:]
            n_ahead = min(U_vent, len(remaining))
            model.cons.add(sum(model.v[remaining[j]] for j in range(n_ahead))>= n_ahead * model.s[n_id])

    # Solve 
    solver = pyo.SolverFactory('gurobi', solver_io='python')
    solver.options['TimeLimit'] = 10
    solver.options['MIPGap'] = 0.01
    solver.options['OutputFlag'] = 0
    result = solver.solve(model, tee=False)

    # Extract the here-and-now decisions from the root node (node 0)
    hp1  = float(pyo.value(model.p[1, 0]))
    hp2  = float(pyo.value(model.p[2, 0]))
    vent = int(pyo.value(model.v[0]))

    return hp1, hp2, vent

def select_action(state):
    current_time = state["current_time"]
    remaining = params['num_timeslots'] - current_time
    L = 3
    if remaining < L:
        L = remaining
    nodes, scenarios = generate_scenario_tree(state, L)
    hp1, hp2, vent = build_and_solve_sp(params, state, nodes, scenarios)
    
    HereAndNowActions = {
        "HeatPowerRoom1" : hp1,
        "HeatPowerRoom2" : hp2,
        "VentilationON" : vent
    }
    
    return HereAndNowActions