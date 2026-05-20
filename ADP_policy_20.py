
import os
import sys
import numpy as np
import pyomo.environ as pyo
from SystemCharacteristics import get_fixed_data

try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# Constants
PARAMS   = get_fixed_data()
T_SLOTS  = int(PARAMS['num_timeslots'])

N_FEATURES     = 12
N_SCENARIOS    = 2000
N_ITER         = 5
RIDGE_ALPHA    = 1e-2
EPSILON_TEMP   = 0.01  # °C buffer above T_LOW inside the MILP
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


# Stochastic process models
def _sample_next_price(price_t, price_prev):
    mean_price         = 4.0
    reversion_strength = 0.12
    next_p = (price_t
              + 0.6 * (price_t - price_prev)
              + reversion_strength * (mean_price - price_t)
              + np.random.normal(0, 0.5))
    if next_p < 0 and np.random.rand() > 0.2:
        next_p = np.random.uniform(0, mean_price * 0.3)
    return float(np.clip(next_p, 0.0, 12.0))


def _sample_next_occupancy(occ1, occ2):
    mean_r1, mean_r2 = 35.0, 25.0
    rev, coupling    = 0.25, 0.10
    r1 = occ1 + rev*(mean_r1-occ1) + coupling*(occ2-occ1) + np.random.normal(0, 3.0)
    r2 = occ2 + rev*(mean_r2-occ2) + coupling*(occ1-occ2) + np.random.normal(0, 2.5)
    return float(np.clip(r1, 20, 50)), float(np.clip(r2, 10, 30))



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


def _greedy_action(state):
    d  = PARAMS
    v  = 1 if state['H'] > d['humidity_threshold'] * 0.75 else 0
    p1 = d['heating_max_power'] * 0.5
    p2 = d['heating_max_power'] * 0.5
    return _apply_overrule(state, {'HeatPowerRoom1': p1,
                                   'HeatPowerRoom2': p2,
                                   'VentilationON':  v})


# MILP: min c_t(a) + (1/K) Σ_k θ_{t+1}ᵀ φ(s_{t+1}^k)
def _get_solver():
    for name in ('gurobi', 'highs', 'cbc', 'glpk'):
        s = pyo.SolverFactory(name)
        if s.available():
            return s
    raise RuntimeError("No MILP solver found.")


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
    T_LOW     = float(d['temp_min_comfort_threshold']) + EPSILON_TEMP

    sampled_prices = [_sample_next_price(lam, lam_prev)
                      for _ in range(N_VFA_SAMPLES)]
    sampled_occs   = [_sample_next_occupancy(Occ1, Occ2)
                      for _ in range(N_VFA_SAMPLES)]
    lam_next_exp   = float(np.mean(sampled_prices))
    occ1_next_exp  = float(np.mean([o[0] for o in sampled_occs]))
    occ2_next_exp  = float(np.mean([o[1] for o in sampled_occs]))

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
    _Z_MAX = 1.0
    m.z1 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, _Z_MAX))
    m.z2 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, _Z_MAX))

    def T1_next(m): return T1_const + g*m.p1 - c_*m.v
    def T2_next(m): return T2_const + g*m.p2 - c_*m.v
    def H_next(m):  return H_const  - hv*m.v

    def vfa(m):
        th = theta_next
        t_next = t + 1
        return (
            th[0]  * 1.0
          + th[1]  * T1_next(m)            / 25.0
          + th[2]  * T2_next(m)            / 25.0
          + th[3]  * H_next(m)             / 50.0
          + th[4]  * lam_next_exp          / 6.0
          + th[5]  * lam                   / 6.0
          + th[6]  * occ1_next_exp         / 35.0
          + th[7]  * occ2_next_exp         / 35.0
          + th[8]  * t_next                / float(T_SLOTS)
          + th[9]  * (vc_next_coef * m.v)  / 3.0
          + th[10] * m.z1
          + th[11] * m.z2
        )

    m.obj = pyo.Objective(
        expr = lam * (m.p1 + m.p2 + d['ventilation_power'] * m.v) + vfa(m),
        sense = pyo.minimize
    )

    m.z1_nonneg = pyo.Constraint(expr = m.z1 >= 0.0)
    m.z1_cold   = pyo.Constraint(expr = m.z1 >= (T_LOW - T1_next(m)) / 5.0)
    m.z2_nonneg = pyo.Constraint(expr = m.z2 >= 0.0)
    m.z2_cold   = pyo.Constraint(expr = m.z2 >= (T_LOW - T2_next(m)) / 5.0)

    if 0 < vc < d['vent_min_up_time']:
        m.vent_inertia = pyo.Constraint(expr = m.v == 1)
    if H > d['humidity_threshold']:
        m.hum_overrule = pyo.Constraint(expr = m.v == 1)

    if T1 > d['temp_max_comfort_threshold']:
        m.p1_hi = pyo.Constraint(expr = m.p1 == 0.0)
    elif lo_r1 == 1:
        m.p1_lo = pyo.Constraint(expr = m.p1 == d['heating_max_power'])

    if T2 > d['temp_max_comfort_threshold']:
        m.p2_hi = pyo.Constraint(expr = m.p2 == 0.0)
    elif lo_r2 == 1:
        m.p2_lo = pyo.Constraint(expr = m.p2 == d['heating_max_power'])

    solver = _get_solver()
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
def _sample_training_day():
    price0 = np.random.uniform(2.0, 8.0)
    occ10  = np.random.uniform(25.0, 35.0)
    occ20  = np.random.uniform(15.0, 25.0)

    prices = [price0]
    occ1s  = [occ10]
    occ2s  = [occ20]

    for _ in range(T_SLOTS - 1):
        p_prev = prices[-2] if len(prices) > 1 else 6.0
        prices.append(_sample_next_price(prices[-1], p_prev))
        o1, o2 = _sample_next_occupancy(occ1s[-1], occ2s[-1])
        occ1s.append(o1)
        occ2s.append(o2)

    return prices, occ1s, occ2s


def _make_initial_state(prices, occ1s, occ2s):
    d = PARAMS
    T1_init = np.random.uniform(16.0, 24.0)
    T2_init = np.random.uniform(16.0, 24.0)
    H_init  = np.random.uniform(30.0, 60.0)
    return {
        'T1':             T1_init,
        'T2':             T2_init,
        'H':              H_init,
        'Occ1':           occ1s[0],
        'Occ2':           occ2s[0],
        'price_t':        prices[0],
        'price_previous': 4.0,
        'vent_counter':   0,
        'low_override_r1': int(T1_init <= d['temp_min_comfort_threshold']),
        'low_override_r2': int(T2_init <= d['temp_min_comfort_threshold']),
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
            prices, occ1s, occ2s = _sample_training_day()
            state = _make_initial_state(prices, occ1s, occ2s)
            traj  = []

            for t in range(T_SLOTS):
                if iteration == 0:
                    action = _greedy_action(state)
                else:
                    theta_next = theta_list[t + 1] if t + 1 < T_SLOTS \
                                 else np.zeros(N_FEATURES)
                    action, _ = _solve_MILP(state, theta_next)
                    action = _apply_overrule(state, action)

                occ1_next  = occ1s[t + 1]  if t + 1 < T_SLOTS else occ1s[-1]
                occ2_next  = occ2s[t + 1]  if t + 1 < T_SLOTS else occ2s[-1]
                price_next = prices[t + 1] if t + 1 < T_SLOTS else prices[-1]

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
            s_ref = _make_initial_state([4.0]*T_SLOTS,
                                        [30.0]*T_SLOTS,
                                        [20.0]*T_SLOTS)
            V0 = theta_list[0] @ compute_features(s_ref)
            print(f"  Iteration {iteration+1}/{n_iter}  |  "
                  f"V̂_0(s_ref) = {V0:.3f} €")

    if verbose:
        print("[ADP] Training complete.")
    return theta_list


# Load pre-trained weights
_WEIGHTS_NPY = os.path.join(_DIR, 'adp_weights_20.npy')

if os.path.exists(_WEIGHTS_NPY):
    _theta_matrix = np.load(_WEIGHTS_NPY)
    assert _theta_matrix.shape == (T_SLOTS, N_FEATURES), \
        f"Weight NPY shape mismatch: expected ({T_SLOTS},{N_FEATURES}), got {_theta_matrix.shape}"
    _THETA_LIST = [_theta_matrix[t] for t in range(T_SLOTS)]
    print(f"[ADP] Loaded pre-trained weights from {_WEIGHTS_NPY}")
else:
    print(f"[ADP] WARNING: {_WEIGHTS_NPY} not found. Training from scratch…")
    _THETA_LIST = train_ADP(n_scenarios=N_SCENARIOS, n_iter=N_ITER, verbose=True)
    np.save(_WEIGHTS_NPY, np.array(_THETA_LIST))
    print(f"[ADP] Saved weights to {_WEIGHTS_NPY}")


# Policy entry point
def select_action(state):
    t = int(state['current_time'])
    if t >= T_SLOTS - 1:
        return {'HeatPowerRoom1': 0.0, 'HeatPowerRoom2': 0.0, 'VentilationON': 0}

    theta_next = _THETA_LIST[t + 1]
    action, _ = _solve_MILP(state, theta_next)
    return action
