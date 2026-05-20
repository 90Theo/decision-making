import os
import sys
import numpy as np
import pyomo.environ as pyo
from SystemCharacteristics import get_fixed_data
from PriceProcessRestaurant import price_model
from OccupancyProcessRestaurant import next_occupancy_levels

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# Constants
EPSILON = 0.001
PARAMS   = get_fixed_data()
T_SLOTS  = int(PARAMS['num_timeslots'])

N_FEATURES     = 12
N_SCENARIOS    = 500
N_ITER         = 5
RIDGE_ALPHA    = 1
N_VFA_SAMPLES  = 50    # K samples for averaging VFA over stochastic outcomes


# Feature function φ(s) ∈ ℝ¹²
def compute_features(state):
    T_LOW = PARAMS['temp_min_comfort_threshold']
    return np.array([
        1.0,
        state['T1']             / 25.0,
        state['T2']             / 25.0,
        state['H']              / 50.0,
        state['price_t']        / 6.0,
        state['price_previous'] / 6.0,
        state['Occ1']           / 35.0,
        state['Occ2']           / 35.0,
        state['current_time']   / float(T_SLOTS),
        min(state['vent_counter'], 3) / 3.0,
        max(0.0, T_LOW - state['T1']) / 5.0,
        max(0.0, T_LOW - state['T2']) / 5.0,
    ], dtype=float)



# System dynamics
def _apply_overrule(state, action):
    d  = PARAMS
    p1 = action['HeatPowerRoom1']
    p2 = action['HeatPowerRoom2']
    v  = action['VentilationON']

    if state['H'] > d['humidity_threshold']:
        v = 1
    elif state['vent_counter'] > 0 and state['vent_counter'] < d['vent_min_up_time']:
        v = 1

    if state['T1'] > d['temp_max_comfort_threshold']:
        p1 = 0.0
    elif state['low_override_r1'] == 1:
        p1 = d['heating_max_power']

    if state['T2'] > d['temp_max_comfort_threshold']:
        p2 = 0.0
    elif state['low_override_r2'] == 1:
        p2 = d['heating_max_power']

    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}


def _step(state, action, occ1_next, occ2_next, price_next):
    d  = PARAMS
    t  = state['current_time']
    p1 = float(action['HeatPowerRoom1'])
    p2 = float(action['HeatPowerRoom2'])
    v  = float(action['VentilationON'])
    T_out = d['outdoor_temperature'][t]

    T1_next = (state['T1']
               + d['heat_exchange_coeff']      * (state['T2'] - state['T1'])
               + d['thermal_loss_coeff']       * (T_out - state['T1'])
               + d['heating_efficiency_coeff'] * p1
               - d['heat_vent_coeff']          * v
               + d['heat_occupancy_coeff']     * state['Occ1'])

    T2_next = (state['T2']
               + d['heat_exchange_coeff']      * (state['T1'] - state['T2'])
               + d['thermal_loss_coeff']       * (T_out - state['T2'])
               + d['heating_efficiency_coeff'] * p2
               - d['heat_vent_coeff']          * v
               + d['heat_occupancy_coeff']     * state['Occ2'])

    H_next = (state['H']
              + d['humidity_occupancy_coeff'] * (state['Occ1'] + state['Occ2'])
              - d['humidity_vent_coeff']      * v)

    vc_next = (state['vent_counter'] + 1) if v >= 0.5 else 0

    lo_r1_next = int(
        T1_next <= d['temp_min_comfort_threshold'] or
        (state['low_override_r1'] == 1 and T1_next < d['temp_OK_threshold'])
    )
    lo_r2_next = int(
        T2_next <= d['temp_min_comfort_threshold'] or
        (state['low_override_r2'] == 1 and T2_next < d['temp_OK_threshold'])
    )

    stage_cost = state['price_t'] * (p1 + p2 + d['ventilation_power'] * v)

    next_state = {
        'T1':             T1_next,
        'T2':             T2_next,
        'H':              H_next,
        'Occ1':           occ1_next,
        'Occ2':           occ2_next,
        'price_t':        price_next,
        'price_previous': state['price_t'],
        'vent_counter':   vc_next,
        'low_override_r1': lo_r1_next,
        'low_override_r2': lo_r2_next,
        'current_time':   t + 1,
    }
    return next_state, stage_cost


# MILP: min c_t(a) + (1/K) Σ_k θ_{t+1}ᵀ φ(s_{t+1}^k)
_SOLVER = pyo.SolverFactory('gurobi')
if not _SOLVER.available():
    raise RuntimeError("Gurobi solver not found. Install Gurobi and ensure it is on PATH.")


def _solve_MILP(state, theta_next):
    d   = PARAMS
    t   = int(state['current_time'])
    T1        = float(state['T1'])
    T2        = float(state['T2'])
    H         = float(state['H'])
    lam       = float(state['price_t'])
    lam_prev  = float(state['price_previous'])
    Occ1      = float(state['Occ1'])
    Occ2      = float(state['Occ2'])
    vc        = int(state['vent_counter'])
    lo_r1     = int(state['low_override_r1'])
    lo_r2     = int(state['low_override_r2'])
    T_out_t   = float(d['outdoor_temperature'][t])
    T_LOW     = float(d['temp_min_comfort_threshold'])

    K = N_VFA_SAMPLES
    scenarios = []
    for _ in range(K):
        price_k = price_model(lam, lam_prev)
        occ1_k, occ2_k = next_occupancy_levels(Occ1, Occ2)
        scenarios.append((price_k, occ1_k, occ2_k))

    a  = d['heat_exchange_coeff']
    b_ = d['thermal_loss_coeff']
    g  = d['heating_efficiency_coeff']
    c_ = d['heat_vent_coeff']
    e  = d['heat_occupancy_coeff']
    eh = d['humidity_occupancy_coeff']
    hv = d['humidity_vent_coeff']
    vc_next_coef = float(min(vc + 1, 3))

    T1_const = T1 + a*(T2-T1) + b_*(T_out_t-T1) + e*Occ1
    T2_const = T2 + a*(T1-T2) + b_*(T_out_t-T2) + e*Occ2
    H_const  = H  + eh*(Occ1+Occ2)

    m = pyo.ConcreteModel()
    m.p1 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, d['heating_max_power']))
    m.p2 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, d['heating_max_power']))
    m.v  = pyo.Var(domain=pyo.Binary)

    m.T1_next = pyo.Var(within=pyo.Reals)
    m.T2_next = pyo.Var(within=pyo.Reals)
    m.H_next  = pyo.Var(within=pyo.Reals)
    M_BIG     = 10.0
    m.z1      = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, M_BIG))
    m.z2      = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, M_BIG))
    m.b1      = pyo.Var(domain=pyo.Binary)
    m.b2      = pyo.Var(domain=pyo.Binary)

    # Deterministic transition: y_{t+1} = f(y_t, u_t)
    m.T1_dyn  = pyo.Constraint(expr=m.T1_next == T1_const + g*m.p1 - c_*m.v)
    m.T2_dyn  = pyo.Constraint(expr=m.T2_next == T2_const + g*m.p2 - c_*m.v)
    m.H_dyn   = pyo.Constraint(expr=m.H_next  == H_const  - hv*m.v)

    # Exact max(0, (T_LOW - T_next)/5) via Big-M binary formulation
    m.z1_lb  = pyo.Constraint(expr=m.z1 >= (T_LOW + EPSILON - m.T1_next) / 5.0)
    m.z1_ub1 = pyo.Constraint(expr=m.z1 <= (T_LOW + EPSILON - m.T1_next) / 5.0 + M_BIG * (1 - m.b1))
    m.z1_ub2 = pyo.Constraint(expr=m.z1 <= M_BIG * m.b1)

    m.z2_lb  = pyo.Constraint(expr=m.z2 >= (T_LOW + EPSILON - m.T2_next) / 5.0)
    m.z2_ub1 = pyo.Constraint(expr=m.z2 <= (T_LOW + EPSILON - m.T2_next) / 5.0 + M_BIG * (1 - m.b2))
    m.z2_ub2 = pyo.Constraint(expr=m.z2 <= M_BIG * m.b2)

    # Objective: c_t(u) + (1/K) Σ_k V̂((y_{t+1}, w_{k,t+1}); θ_{t+1})
    m.Scenarios = pyo.RangeSet(0, K - 1)

    def vfa_scenario(m, k):
        th = theta_next
        t_next = t + 1
        price_k, occ1_k, occ2_k = scenarios[k]
        return (
            th[0]  * 1.0
          + th[1]  * m.T1_next             / 25.0
          + th[2]  * m.T2_next             / 25.0
          + th[3]  * m.H_next              / 50.0
          + th[4]  * price_k               / 6.0
          + th[5]  * lam                   / 6.0
          + th[6]  * occ1_k                / 35.0
          + th[7]  * occ2_k                / 35.0
          + th[8]  * t_next                / float(T_SLOTS)
          + th[9]  * (vc_next_coef * m.v)  / 3.0
          + th[10] * m.z1
          + th[11] * m.z2
        )

    m.obj = pyo.Objective(
        expr = lam * (m.p1 + m.p2 + d['ventilation_power'] * m.v)
             + sum(vfa_scenario(m, k) for k in m.Scenarios) / K,
        sense = pyo.minimize
    )

    if 0 < vc < d['vent_min_up_time']:
        m.vent_inertia = pyo.Constraint(expr = m.v == 1)
    if H > d['humidity_threshold']:
        m.hum_overrule = pyo.Constraint(expr = m.v == 1)

    if T1 > d['temp_max_comfort_threshold']:
        m.p1_hi = pyo.Constraint(expr = m.p1 == 0.0)
    elif lo_r1 == 1 or T1 <= d['temp_min_comfort_threshold']:
        m.p1_lo = pyo.Constraint(expr = m.p1 == d['heating_max_power'])

    if T2 > d['temp_max_comfort_threshold']:
        m.p2_hi = pyo.Constraint(expr = m.p2 == 0.0)
    elif lo_r2 == 1 or T2 <= d['temp_min_comfort_threshold']:
        m.p2_lo = pyo.Constraint(expr = m.p2 == d['heating_max_power'])

    solver = _SOLVER
    result = solver.solve(m, tee=False)
    ok = (result.solver.termination_condition == pyo.TerminationCondition.optimal)
    assert ok, "1-step MILP failed to solve — should not happen."

    act = {
        'HeatPowerRoom1': float(np.clip(pyo.value(m.p1), 0.0, d['heating_max_power'])),
        'HeatPowerRoom2': float(np.clip(pyo.value(m.p2), 0.0, d['heating_max_power'])),
        'VentilationON':  int(round(float(pyo.value(m.v)))),
    }
    return act, float(pyo.value(m.obj))


# Training: forward-backward approximate backward induction
def _sample_initial_state():
    return {
        'T1':             21.0,
        'T2':             21.0,
        'H':              40.0,
        'Occ1':           np.random.uniform(25, 35),
        'Occ2':           np.random.uniform(15, 25),
        'price_t':        np.random.uniform(2, 8),
        'price_previous': np.random.uniform(2, 8),
        'vent_counter':   0,
        'low_override_r1': 0,
        'low_override_r2': 0,
        'current_time':   0,
    }


def train_ADP(n_scenarios=N_SCENARIOS, n_iter=N_ITER, ridge=RIDGE_ALPHA,
              verbose=True):
    np.random.seed(42)
    if verbose:
        print(f"[ADP] Training: {n_scenarios} scenarios × {n_iter} iterations …")

    theta_list = [np.zeros(N_FEATURES) for _ in range(T_SLOTS)]

    for iteration in range(n_iter):
        trajectories = []

        for _ in range(n_scenarios):
            state = _sample_initial_state()
            traj  = []

            for t in range(T_SLOTS):
                theta_next = theta_list[t + 1] if t + 1 < T_SLOTS \
                             else np.zeros(N_FEATURES)
                action, _ = _solve_MILP(state, theta_next)
                action = _apply_overrule(state, action)

                occ1_next, occ2_next = next_occupancy_levels(state['Occ1'], state['Occ2'])
                price_next = price_model(state['price_t'], state['price_previous'])

                next_state, cost = _step(state, action,
                                         occ1_next, occ2_next, price_next)
                traj.append((state.copy(), action.copy(), cost, next_state.copy()))
                state = next_state

            trajectories.append(traj)

        for t in range(T_SLOTS - 1, -1, -1):
            Phi = np.zeros((n_scenarios, N_FEATURES))
            y   = np.zeros(n_scenarios)

            theta_next_bp = theta_list[t + 1] if t + 1 < T_SLOTS \
                            else np.zeros(N_FEATURES)

            for j, traj in enumerate(trajectories):
                state_t = traj[t][0]
                Phi[j]  = compute_features(state_t)
                _, y[j] = _solve_MILP(state_t, theta_next_bp)

            A             = Phi.T @ Phi + ridge * np.eye(N_FEATURES)
            b             = Phi.T @ y
            theta_list[t] = np.linalg.solve(A, b)

        if verbose:
            s_ref = _sample_initial_state()
            V0 = theta_list[0] @ compute_features(s_ref)
            print(f"  Iteration {iteration+1}/{n_iter}  |  "
                  f"V̂_0(ref) = {V0:.3f} €")

    if verbose:
        print("[ADP] Training complete.")
    return theta_list


# Load pre-trained weights if available; otherwise train and save them.
_WEIGHTS_NPY = os.path.join(_DIR, 'adp_weights_20.npy')

_theta_matrix = None
if os.path.exists(_WEIGHTS_NPY):
    try:
        _theta_matrix = np.load(_WEIGHTS_NPY)
        if _theta_matrix.shape != (T_SLOTS, N_FEATURES):
            print(f"[ADP] Weight file shape mismatch: expected ({T_SLOTS},{N_FEATURES}), got {_theta_matrix.shape}. Retraining.")
            _theta_matrix = None
    except Exception as e:
        print(f"[ADP] Failed loading weights ({e}). Retraining.")
        _theta_matrix = None

if _theta_matrix is None:
    print(f"[ADP] Pre-trained weights not found or invalid at {_WEIGHTS_NPY}. Training ADP now (this may take a while)...")
    # Train using the defaults defined in this module
    theta_list = train_ADP(n_scenarios=N_SCENARIOS, n_iter=N_ITER, ridge=RIDGE_ALPHA, verbose=True)
    np.save(_WEIGHTS_NPY, np.array(theta_list))
    _theta_matrix = np.array(theta_list)

assert _theta_matrix.shape == (T_SLOTS, N_FEATURES), \
    f"Weight matrix shape mismatch after loading/training: expected ({T_SLOTS},{N_FEATURES}), got {_theta_matrix.shape}"
_THETA_LIST = [_theta_matrix[t] for t in range(T_SLOTS)]


# Policy entry point
def select_action(state):
    t = int(state['current_time'])
    theta_next = _THETA_LIST[t + 1] if t + 1 < T_SLOTS else np.zeros(N_FEATURES)
    action, _ = _solve_MILP(state, theta_next)
    return action
