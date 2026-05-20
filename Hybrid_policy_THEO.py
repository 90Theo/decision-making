import time

import numpy as np
import pyomo.environ as pyo

import ADP_policy_20 as adp
import SP_policy_20 as sp


PARAMS = sp.params

# Keep the SP tree size close to SP_policy_20, but reserve the final stage as
# a terminal state where the ADP value function estimates the remaining cost.
TOTAL_BUDGET_SECONDS = 8.0
BUFFER_SECONDS = 0.5
LOOKAHEAD_STAGE_COSTS = 5
BRANCHING = 3
N_SAMPLES = 30
LOW_SHORTFALL_M = 10.0
TERMINAL_VALUE_WEIGHT = 1.0


def _theta_for_time(time_index):
    if 0 <= time_index < adp.T_SLOTS:
        return adp._THETA_LIST[time_index]
    return np.zeros(adp.N_FEATURES)


def _terminal_vent_counter_feature(model, path, state):
    """Linear proxy for ADP's capped vent_counter feature at a leaf state."""
    control_nodes = path[:-1]
    if not control_nodes:
        return min(int(state["vent_counter"]), 3) / 3.0

    recent_nodes = control_nodes[-3:]
    expr = sum(model.v[n] for n in recent_nodes)

    # If the terminal is close to the root, carry a little observed history
    # forward while keeping the expression linear in the SP decision variables.
    missing_history = 3 - len(recent_nodes)
    if missing_history > 0 and int(state["vent_counter"]) > 0:
        expr += min(int(state["vent_counter"]), missing_history) * model.v[control_nodes[0]]

    return expr / 3.0


def _add_terminal_shortfall_variables(model, leaf_ids):
    rooms = [1, 2]
    model.hybrid_leaf_ids = pyo.Set(initialize=leaf_ids)
    model.hybrid_low_shortfall = pyo.Var(
        rooms,
        model.hybrid_leaf_ids,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, LOW_SHORTFALL_M),
    )
    model.hybrid_low_shortfall_on = pyo.Var(
        rooms,
        model.hybrid_leaf_ids,
        domain=pyo.Binary,
    )
    model.hybrid_terminal_cons = pyo.ConstraintList()

    t_low = PARAMS["temp_min_comfort_threshold"]
    for n in leaf_ids:
        for r in rooms:
            gap = (t_low - model.T[r, n]) / 5.0
            z = model.hybrid_low_shortfall[r, n]
            b = model.hybrid_low_shortfall_on[r, n]
            model.hybrid_terminal_cons.add(z >= gap)
            model.hybrid_terminal_cons.add(z <= gap + LOW_SHORTFALL_M * (1 - b))
            model.hybrid_terminal_cons.add(z <= LOW_SHORTFALL_M * b)


def _terminal_value_expr(model, node, path, state):
    theta = _theta_for_time(int(node["time"]))
    n = node["id"]
    vent_counter_feature = _terminal_vent_counter_feature(model, path, state)

    return (
        theta[0] * 1.0
        + theta[1] * model.T[1, n] / 25.0
        + theta[2] * model.T[2, n] / 25.0
        + theta[3] * model.H[n] / 50.0
        + theta[4] * float(node["price"]) / 6.0
        + theta[5] * float(node["price_prev"]) / 6.0
        + theta[6] * float(node["occ1"]) / 35.0
        + theta[7] * float(node["occ2"]) / 35.0
        + theta[8] * int(node["time"]) / float(adp.T_SLOTS)
        + theta[9] * vent_counter_feature
        + theta[10] * model.hybrid_low_shortfall[1, n]
        + theta[11] * model.hybrid_low_shortfall[2, n]
    )


def _replace_sp_objective_with_hybrid(model, state, nodes, scenarios):
    leaf_paths = {path[-1]: path for path in scenarios}
    leaf_ids = sorted(leaf_paths)
    leaf_id_set = set(leaf_ids)
    rooms = [1, 2]

    _add_terminal_shortfall_variables(model, leaf_ids)

    ventilation_power = PARAMS["ventilation_power"]
    num_timeslots = PARAMS["num_timeslots"]
    stage_cost = sum(
        nodes[n]["prob"]
        * nodes[n]["price"]
        * (sum(model.p[r, n] for r in rooms) + ventilation_power * model.v[n])
        for n in model.N
        if n not in leaf_id_set and nodes[n]["time"] < num_timeslots
    )

    terminal_value = sum(
        nodes[n]["prob"] * _terminal_value_expr(model, nodes[n], leaf_paths[n], state)
        for n in leaf_ids
    )

    model.obj.deactivate()
    model.hybrid_obj = pyo.Objective(
        expr=stage_cost + TERMINAL_VALUE_WEIGHT * terminal_value,
        sense=pyo.minimize,
    )


def select_action(state):
    remaining_periods = PARAMS["num_timeslots"] - int(state["current_time"])

    # With no child state in the tree, the terminal VFA cannot depend on the
    # current action. Use ADP's native one-step MILP for the final period.
    if remaining_periods <= 1:
        return adp.select_action(state)

    t_start = time.time()
    tree_stages = min(LOOKAHEAD_STAGE_COSTS + 1, remaining_periods)
    nodes, scenarios = sp.generate_scenario_tree(
        state,
        L=tree_stages,
        branching=BRANCHING,
        n_samples=N_SAMPLES,
    )
    model = sp.build_sp(PARAMS, state, nodes, scenarios)
    _replace_sp_objective_with_hybrid(model, state, nodes, scenarios)

    solve_time = max(0.5, TOTAL_BUDGET_SECONDS - (time.time() - t_start) - BUFFER_SECONDS)
    hp1, hp2, vent = sp.solve_sp(model, time_limit=solve_time)

    return {
        "HeatPowerRoom1": hp1,
        "HeatPowerRoom2": hp2,
        "VentilationON": vent,
    }
