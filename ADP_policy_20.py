
#  Task 4: Approximate Dynamic Programming

#  ARCHITECTURE OVERVIEW
#  ─────────────────────
#  This file implements a two-phase ADP policy with a linear value function
#  approximation (linear VFA):
#
#    V̂_t(s ; θ_t) = θ_t ᵀ φ(s)      [linear in weights θ, nonlinear in s OK]


### Phase 1 — TRAINING  (runs once at module import, ~10-20 s) ###
#  Algorithm: Approximate Backward Induction with Forward-Backward passes
#
#    For each outer iteration i = 1 … I:
#      Forward pass  : simulate N days under a greedy policy → collect
#                      trajectories { (s_t^j, a_t^j, c_t^j, s_{t+1}^j) }
#      Backward pass : for t = T-1 down to 0, compute Bellman targets
#                        ŷ_t^j = c_t^j + θ_{t+1}ᵀ φ(s_{t+1}^j)
#                      then fit θ_t by ridge regression on those targets.



###  Phase 2 — DEPLOYMENT  (called every hour by the environment, ~0.1-0.5 s)###
#  At each hour t with observed state s_t, solve a tiny 1-step lookahead MILP:
#
#    min_{p1, p2, v}  λ_t (p1 + p2 + P^vent · v)  +  θ_{t+1}ᵀ φ(s̃_{t+1})
#
#  where s̃_{t+1} = f(s_t, p1, p2, v, E[ω_{t+1}]) uses dynamics equations and
#  expected next-period stochastic values (price, occupancy).  Because dynamics
#  are linear in decisions, and φ is linear in the state, the VFA term is
#  LINEAR in (p1, p2, v) → the problem stays a MILP with only 3 variables.



# Imports
import os
import sys
import numpy as np
import pyomo.environ as pyo
from SystemCharacteristics import get_fixed_data

# Directory
try:
    _DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DIR = os.getcwd()

if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# SECTION 1 — CONSTANTS & SYSTEM PARAMETERS


PARAMS   = get_fixed_data()
T_SLOTS  = int(PARAMS['num_timeslots'])   # 10 hours

# ── Linear VFA hyperparameters 
N_FEATURES   = 12     # size of φ(s) vector — see compute_features() below
N_SCENARIOS  = 200    # number of simulated training days per iteration
N_ITER       = 5      # number of forward-backward iterations
RIDGE_ALPHA  = 1e-2   # ridge regularisation λ in (ΦᵀΦ + λI)⁻¹Φᵀy


# SECTION 2 — FEATURE FUNCTION  φ(s)


#  The linear VFA is:   V̂_t(s ; θ_t) = θ_t ᵀ φ(s)
#
#  We choose 12 features that capture the key drivers of future cost.
#  All features are normalised to roughly [0, 1] for numerical stability.
#
#  Feature index  |  φ_k(s)                        |  Why it matters
#  ───────────────┼─────────────────────────────────┼──────────────────────────
#   0             |  1   (bias)                     |  baseline cost offset
#   1             |  T1 / 25                        |  room 1 temp → heating need
#   2             |  T2 / 25                        |  room 2 temp → heating need
#   3             |  H  / 50                        |  humidity → forced ventilation
#   4             |  λ  / 6                         |  current price → future trend
#   5             |  λ_prev / 6                     |  price momentum
#   6             |  κ1 / 35                        |  occupancy → heat & humidity load
#   7             |  κ2 / 35                        |  occupancy → heat & humidity load
#   8             |  t  / 10                        |  time-to-go (horizon awareness)
#   9             |  c_vent / 3                     |  ventilation lock-in state
#  10             |  max(0, T_low − T1) / 5         |  proximity to cold violation r1
#  11             |  max(0, T_low − T2) / 5         |  proximity to cold violation r2
#
#  Features 10–11 are piecewise linear in T1/T2.  Because they appear in the
#  MILP objective with a fixed coefficient θ_k, they can be reformulated as
#  a standard LP auxiliary variable: z_r ≥ 0,  z_r ≥ T_low − T_r_next.

def compute_features(state):
    """
    Returns φ(s) ∈ ℝ^12 for a given state dictionary.
    All values are normalised so that each feature lies in roughly [0, 1].
    """
    T_LOW = PARAMS['temp_min_comfort_threshold']   # 18 °C

    phi = np.array([
        1.0,                                                     # 0  bias
        state['T1']            / 25.0,                          # 1  temp r1
        state['T2']            / 25.0,                          # 2  temp r2
        state['H']             / 50.0,                          # 3  humidity
        state['price_t']       / 6.0,                           # 4  price
        state['price_previous']/ 6.0,                           # 5  prev price
        state['Occ1']          / 35.0,                          # 6  occupancy r1
        state['Occ2']          / 35.0,                          # 7  occupancy r2
        state['current_time']  / float(T_SLOTS),                # 8  time (0→1)
        min(state['vent_counter'], 3) / 3.0,                    # 9  vent counter (capped at 3 to match MILP)
        max(0.0, T_LOW - state['T1']) / 5.0,                    # 10 cold risk r1
        max(0.0, T_LOW - state['T2']) / 5.0,                    # 11 cold risk r2
    ], dtype=float)
    return phi


# SECTION 3 — STOCHASTIC PROCESS MODELS (inlined to avoid import side-effects)

def _sample_next_price(price_t, price_prev):
    """
    Sample next electricity price from the AR(2)-like process:
      λ_{t+1} = λ_t + 0.6(λ_t − λ_{t-1}) + 0.12(4 − λ_t) + ε,   ε~N(0, 0.5)
    """
    mean_price        = 4.0
    reversion_strength = 0.12
    next_p = (price_t
              + 0.6 * (price_t - price_prev)
              + reversion_strength * (mean_price - price_t)
              + np.random.normal(0, 0.5))
    if next_p < 0 and np.random.rand() > 0.2:
        next_p = np.random.uniform(0, mean_price * 0.3)
    return float(np.clip(next_p, 0.0, 12.0))


def _expected_next_price(price_t, price_prev):
    """
    Expected next price (mean of the process, ε = 0).
    Used in the deployment MILP so the stochastic state is approximated.
    """
    mean_price        = 4.0
    reversion_strength = 0.12
    exp_p = (price_t
             + 0.6 * (price_t - price_prev)
             + reversion_strength * (mean_price - price_t))
    return float(np.clip(exp_p, 0.0, 12.0))


def _sample_next_occupancy(occ1, occ2):
    """
    Sample next occupancy from the coupled mean-reverting Markov model.
    """
    mean_r1, mean_r2 = 35.0, 25.0
    rev      = 0.25
    coupling = 0.10
    r1 = occ1 + rev*(mean_r1-occ1) + coupling*(occ2-occ1) + np.random.normal(0, 3.0)
    r2 = occ2 + rev*(mean_r2-occ2) + coupling*(occ1-occ2) + np.random.normal(0, 2.5)
    return float(np.clip(r1, 20, 50)), float(np.clip(r2, 10, 30))


def _expected_next_occupancy(occ1, occ2):
    """
    Expected next occupancy (mean of the process, noise = 0).
    """
    mean_r1, mean_r2 = 35.0, 25.0
    rev      = 0.25
    coupling = 0.10
    r1 = occ1 + rev*(mean_r1-occ1) + coupling*(occ2-occ1)
    r2 = occ2 + rev*(mean_r2-occ2) + coupling*(occ1-occ2)
    return float(np.clip(r1, 20, 50)), float(np.clip(r2, 10, 30))

# SECTION 4 — SYSTEM DYNAMICS  (used during training simulation)

def _apply_overrule(state, action):
    """
    Enforce overrule controllers on top of a proposed action.
    This mirrors exactly the logic in Environment.py:apply_overrule().
    We call it during training to make trajectories realistic.
    """
    d  = PARAMS
    p1 = action['HeatPowerRoom1']
    p2 = action['HeatPowerRoom2']
    v  = action['VentilationON']

    # Ventilation overrules
    if state['H'] > d['humidity_threshold']:
        v = 1
    elif state['vent_counter'] > 0 and state['vent_counter'] < d['vent_min_up_time']:
        v = 1

    # Room 1 heating overrules
    if state['T1'] > d['temp_max_comfort_threshold']:
        p1 = 0.0
    elif state['low_override_r1'] == 1:
        p1 = d['heating_max_power']

    # Room 2 heating overrules
    if state['T2'] > d['temp_max_comfort_threshold']:
        p2 = 0.0
    elif state['low_override_r2'] == 1:
        p2 = d['heating_max_power']

    return {'HeatPowerRoom1': p1, 'HeatPowerRoom2': p2, 'VentilationON': v}


def _step(state, action, occ1_next, occ2_next, price_next):
    """
    Compute (next_state, stage_cost) given the current state, action, and the
    *already-revealed* next-period occupancy and price.

    This matches the dynamics in Environment.py:apply_dynamics() exactly.
    """
    d  = PARAMS
    t  = state['current_time']
    p1 = float(action['HeatPowerRoom1'])
    p2 = float(action['HeatPowerRoom2'])
    v  = float(action['VentilationON'])

    T_out = d['outdoor_temperature'][t]

    T1_next = (state['T1']
               + d['heat_exchange_coeff']     * (state['T2'] - state['T1'])
               + d['thermal_loss_coeff']      * (T_out - state['T1'])
               + d['heating_efficiency_coeff'] * p1
               - d['heat_vent_coeff']         * v
               + d['heat_occupancy_coeff']    * state['Occ1'])

    T2_next = (state['T2']
               + d['heat_exchange_coeff']     * (state['T1'] - state['T2'])
               + d['thermal_loss_coeff']      * (T_out - state['T2'])
               + d['heating_efficiency_coeff'] * p2
               - d['heat_vent_coeff']         * v
               + d['heat_occupancy_coeff']    * state['Occ2'])

    H_next  = (state['H']
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
        'T1':            T1_next,
        'T2':            T2_next,
        'H':             H_next,
        'Occ1':          occ1_next,
        'Occ2':          occ2_next,
        'price_t':       price_next,
        'price_previous': state['price_t'],
        'vent_counter':  vc_next,
        'low_override_r1': lo_r1_next,
        'low_override_r2': lo_r2_next,
        'current_time':  t + 1,
    }
    return next_state, stage_cost


def _greedy_action(state):
    """
    A simple baseline policy used for the initial forward pass.
    Heats at 50 % of max power; ventilates when humidity is high.
    The overrule controller is applied on top.
    """
    d  = PARAMS
    v  = 1 if state['H'] > d['humidity_threshold'] * 0.75 else 0
    p1 = d['heating_max_power'] * 0.5
    p2 = d['heating_max_power'] * 0.5
    return _apply_overrule(state, {'HeatPowerRoom1': p1,
                                   'HeatPowerRoom2': p2,
                                   'VentilationON':  v})


# SECTION 5 — TRAINING: APPROXIMATE BACKWARD INDUCTION


#  We use the Forward-Backward algorithm:
#
#  For I iterations:
#    FORWARD PASS:   simulate N days → collect {(s_t^j, a_t^j, c_t^j, s_{t+1}^j)}
#    BACKWARD PASS:  for t = T-1 down to 0:
#                      target ŷ_t^j = c_t^j + θ_{t+1}ᵀ φ(s_{t+1}^j)
#                      θ_t ← (ΦᵀΦ + αI)⁻¹ Φᵀ ŷ_t   [ridge regression]


def _sample_training_day():
    """
    Simulate one full day by sampling from the stochastic process models.
    Returns three lists of length T_SLOTS: prices, occ1s, occ2s.
    """
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
    # Sample initial temperatures from a wide range, not always from the fixed
    # initial value of 21 °C.  This is critical: if all trajectories start warm,
    # the regression never sees near-threshold states and learns the wrong sign
    # for the temperature weights (positive instead of the correct negative).
    # Sampling uniformly over [16, 24] ensures the regression sees states where
    # heating is urgently needed as well as states where it is not.
    T1_init = np.random.uniform(16.0, 24.0)
    T2_init = np.random.uniform(16.0, 24.0)
    H_init  = np.random.uniform(30.0, 60.0)
    return {
        'T1':            T1_init,
        'T2':            T2_init,
        'H':             H_init,
        'Occ1':          occ1s[0],
        'Occ2':          occ2s[0],
        'price_t':       prices[0],
        'price_previous': 4.0,
        'vent_counter':  0,
        'low_override_r1': int(T1_init <= d['temp_min_comfort_threshold']),
        'low_override_r2': int(T2_init <= d['temp_min_comfort_threshold']),
        'current_time':  0,
    }


def train_ADP(n_scenarios=N_SCENARIOS, n_iter=N_ITER, ridge=RIDGE_ALPHA,
              verbose=True):
    """
    Train the ADP value function weights.

    Returns
    -------
    theta_list : list of np.ndarray, shape (N_FEATURES,), length T_SLOTS
        theta_list[t] are the weights for V̂_t(s ; θ_t).
        theta_list[T_SLOTS-1] = 0 (terminal: no future costs from last slot).
    """
    np.random.seed(42)   # fix seed so training is reproducible regardless of import order
    if verbose:
        print(f"[ADP] Training: {n_scenarios} scenarios × {n_iter} iterations …")

    # Initialise all weight vectors to zero
    # initial policy is "greedy" (no value-function correction)
    theta_list = [np.zeros(N_FEATURES) for _ in range(T_SLOTS)]

    for iteration in range(n_iter):
        ### FORWARD PASS ###
        # Each trajectory is a list of T_SLOTS tuples:
        #   (state_t, action_t, cost_t, state_t+1)
        trajectories = []

        for _ in range(n_scenarios):
            prices, occ1s, occ2s = _sample_training_day()
            state = _make_initial_state(prices, occ1s, occ2s)
            traj  = []

            for t in range(T_SLOTS):
                # Choose action
                # Iteration 0: use greedy baseline (fast, seeds the state space)
                # Iteration 1+: use current ADP MILP policy (better samples)
                if iteration == 0:
                    action = _greedy_action(state)
                else:
                    theta_next = theta_list[t + 1] if t < T_SLOTS - 1 \
                                 else np.zeros(N_FEATURES)
                    action = _solve_lookahead_MILP(state, theta_next)
                    action = _apply_overrule(state, action)

                # Advance environment
                occ1_next  = occ1s[t + 1]  if t + 1 < T_SLOTS else occ1s[-1]
                occ2_next  = occ2s[t + 1]  if t + 1 < T_SLOTS else occ2s[-1]
                price_next = prices[t + 1] if t + 1 < T_SLOTS else prices[-1]

                next_state, cost = _step(state, action,
                                         occ1_next, occ2_next, price_next)
                traj.append((state.copy(), action.copy(), cost, next_state.copy()))
                state = next_state

            trajectories.append(traj)

        ### BACKWARD PASS ###
        # Work backwards t = T-1 → 0.
        # At each t, build the regression problem and solve it analytically.

        for t in range(T_SLOTS - 1, -1, -1):
            Phi = np.zeros((n_scenarios, N_FEATURES))
            y   = np.zeros(n_scenarios)

            for j, traj in enumerate(trajectories):
                state_t, _, cost_t, state_t1 = traj[t]

                # φ(s_t) — features of the state we're estimating value for
                Phi[j] = compute_features(state_t)

                # Bellman target:
                #   ŷ_t^j = c_t^j             (last step, V_T ≡ 0)
                #   ŷ_t^j = c_t^j + θ_{t+1}ᵀ φ(s_{t+1}^j)   (otherwise)
                if t == T_SLOTS - 1:
                    y[j] = cost_t
                else:
                    V_next = theta_list[t + 1] @ compute_features(state_t1)
                    y[j]   = cost_t + V_next

            # ── Ridge regression: θ_t = (ΦᵀΦ + αI)⁻¹ Φᵀ y ──────────────
            A             = Phi.T @ Phi + ridge * np.eye(N_FEATURES)
            b             = Phi.T @ y
            theta_list[t] = np.linalg.solve(A, b)

        if verbose:
            # Report estimated value of a "typical" initial state
            s_ref = _make_initial_state([4.0]*T_SLOTS,
                                        [30.0]*T_SLOTS,
                                        [20.0]*T_SLOTS)
            V0 = theta_list[0] @ compute_features(s_ref)
            print(f"  Iteration {iteration+1}/{n_iter}  |  "
                  f"V̂_0(s_ref) = {V0:.3f} €")

    if verbose:
        print("[ADP] Training complete.")
    return theta_list


# SECTION 6 — DEPLOYMENT: ONE-STEP LOOKAHEAD MILP


#  Given current state s_t and pre-trained weights θ_{t+1}, solve:
#
#    min_{p1, p2, v}  λ_t (p1 + p2 + P^vent · v)  +  θ_{t+1}ᵀ φ(s̃_{t+1})
#
#  where s̃_{t+1} is determined by the dynamics (linear in decisions) and
#  expected next-period stochastic components (constant in the MILP).
#
#  Decision variables:  p1, p2 ∈ [0, P̄]  (continuous),  v ∈ {0, 1}  (binary)
#  VFA term contribution per feature:
#    φ_0  →  constant                 (no decision variable dependence)
#    φ_1  →  T1_next / 25             (linear in p1, v)
#    φ_2  →  T2_next / 25             (linear in p2, v)
#    φ_3  →  H_next  / 50             (linear in v)
#    φ_4  →  E[λ_{t+1}] / 6          (constant)
#    φ_5  →  λ_t / 6                  (constant — prev price at t+1 = λ_t)
#    φ_6  →  E[κ1_{t+1}] / 35        (constant)
#    φ_7  →  E[κ2_{t+1}] / 35        (constant)
#    φ_8  →  (t+1) / 10               (constant)
#    φ_9  →  vc_next / 3              (linear in v: min(vc+1,3)·v)
#   φ_10  →  max(0, T_low−T1_next)/5  (piecewise-linear → auxiliary var z1)
#   φ_11  →  max(0, T_low−T2_next)/5  (piecewise-linear → auxiliary var z2)
#
#  Features 10–11 use LP auxiliary variables z_r:
#    z_r ≥ 0
#    z_r ≥ (T_low − T_r_next) / 5
#  In the objective: + θ_10 · z1 + θ_11 · z2
#  Since θ_10, θ_11 > 0 (being cold costs more in the future), the minimiser
#  will automatically set z_r = max(0, T_low − T_r_next) / 5. ✓

def _solve_lookahead_MILP(state, theta_next):
    """
    Solve the 1-step lookahead MILP for the current state.

    Parameters
    ----------
    state      : dict  — current environment state
    theta_next : array — value function weights for time t+1

    Returns
    -------
    action dict with keys HeatPowerRoom1, HeatPowerRoom2, VentilationON
    """
    d   = PARAMS
    t   = int(state['current_time'])

    # Extract current-state scalars
    T1        = float(state['T1'])
    T2        = float(state['T2'])
    H         = float(state['H'])
    lam       = float(state['price_t'])            # λ_t
    lam_prev  = float(state['price_previous'])     # λ_{t-1}
    Occ1      = float(state['Occ1'])
    Occ2      = float(state['Occ2'])
    vc        = int(state['vent_counter'])
    lo_r1     = int(state['low_override_r1'])
    lo_r2     = int(state['low_override_r2'])
    T_out_t   = float(d['outdoor_temperature'][t])
    T_LOW     = float(d['temp_min_comfort_threshold'])   # 18 °C

    # Expected next-period stochastic values (used as constants in MILP) 
    lam_next_exp          = _expected_next_price(lam, lam_prev)
    occ1_next_exp, occ2_next_exp = _expected_next_occupancy(Occ1, Occ2)

    #Coefficients for dynamics (keep notation clean)
    a  = d['heat_exchange_coeff']       # 0.6
    b_ = d['thermal_loss_coeff']        # 0.1   (b_ to avoid shadowing built-in)
    g  = d['heating_efficiency_coeff']  # 1.0
    c_ = d['heat_vent_coeff']           # 0.7
    e  = d['heat_occupancy_coeff']      # 0.02
    eh = d['humidity_occupancy_coeff']  # 0.18
    hv = d['humidity_vent_coeff']       # 15.0
    vc_next_coef = float(min(vc + 1, 3))  # vent_counter_next = vc_next_coef · v

    #Constant parts of next-state features (independent of decisions) 
    T1_const  = T1 + a*(T2-T1) + b_*(T_out_t-T1) + e*Occ1   # + g*p1 − c_*v
    T2_const  = T2 + a*(T1-T2) + b_*(T_out_t-T2) + e*Occ2   # + g*p2 − c_*v
    H_const   = H  + eh*(Occ1+Occ2)                           # − hv*v

    m = pyo.ConcreteModel()

    # Decision variables
    m.p1 = pyo.Var(domain=pyo.NonNegativeReals,
                   bounds=(0.0, d['heating_max_power']))   # p1 ∈ [0, P̄]
    m.p2 = pyo.Var(domain=pyo.NonNegativeReals,
                   bounds=(0.0, d['heating_max_power']))   # p2 ∈ [0, P̄]
    m.v  = pyo.Var(domain=pyo.Binary)                      # v  ∈ {0, 1}

    # Auxiliary variables for piecewise-linear features 10 & 11
    # z_r = max(0, T_low − T_r_next) / 5  →  enforced by constraints below.
    # Upper bound derived from dynamics: worst-case T drop in one step is
    # ~2.8 °C (T=18, T_out=−3, v=1, p=0), giving z ≤ (18−15.2)/5 ≈ 0.56.
    # We use 1.0 as a safe, tight bound (vs. the loose 2.6 used before).
    # This bound is CRITICAL: without it, if θ_10 < 0 the MILP is unbounded.
    _Z_MAX = 1.0
    m.z1 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, _Z_MAX))
    m.z2 = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, _Z_MAX))

    # Next-state expressions (linear in decisions)
    #   These appear in the VFA term of the objective.
    def T1_next(m): return T1_const + g*m.p1 - c_*m.v
    def T2_next(m): return T2_const + g*m.p2 - c_*m.v
    def H_next(m):  return H_const  - hv*m.v

    # VFA term: θ_{t+1}ᵀ φ(s̃_{t+1})
    # Each line corresponds to one feature φ_k(s_{t+1})
    def vfa(m):
        th = theta_next          # shorthand: th[k] is θ_{t+1,k}
        t_next = t + 1
        return (
            th[0]  * 1.0                                    # φ_0 : bias
          + th[1]  * T1_next(m)            / 25.0           # φ_1 : T1_next
          + th[2]  * T2_next(m)            / 25.0           # φ_2 : T2_next
          + th[3]  * H_next(m)             / 50.0           # φ_3 : H_next
          + th[4]  * lam_next_exp          / 6.0            # φ_4 : E[λ_{t+1}]
          + th[5]  * lam                   / 6.0            # φ_5 : λ_t (=prev at t+1)
          + th[6]  * occ1_next_exp         / 35.0           # φ_6 : E[κ1_{t+1}]
          + th[7]  * occ2_next_exp         / 35.0           # φ_7 : E[κ2_{t+1}]
          + th[8]  * t_next                / float(T_SLOTS) # φ_8 : time
          + th[9]  * (vc_next_coef * m.v)  / 3.0           # φ_9 : vent counter
          + th[10] * m.z1                                   # φ_10: cold risk r1
          + th[11] * m.z2                                   # φ_11: cold risk r2
        )


    #Objective function: immediate cost + estimated future cost 
    m.obj = pyo.Objective(
        expr = lam * (m.p1 + m.p2 + d['ventilation_power'] * m.v) + vfa(m),
        sense = pyo.minimize
    )

    # Constraints

    # [C1] Auxiliary variable z1 ≥ max(0, (T_low − T1_next)/5)
    #      LP linearisation of the piecewise-linear feature φ_10
    m.z1_nonneg = pyo.Constraint(expr = m.z1 >= 0.0)
    m.z1_cold   = pyo.Constraint(
        expr = m.z1 >= (T_LOW - T1_next(m)) / 5.0
    )

    # [C2] Same for room 2
    m.z2_nonneg = pyo.Constraint(expr = m.z2 >= 0.0)
    m.z2_cold   = pyo.Constraint(
        expr = m.z2 >= (T_LOW - T2_next(m)) / 5.0
    )

    # [C3] Ventilation inertia: if vent has been on for 1 or 2 hours, must stay ON
    if 0 < vc < d['vent_min_up_time']:
        m.vent_inertia = pyo.Constraint(expr = m.v == 1)

    # [C4] Humidity overrule: if H > H_high, ventilation must be ON
    if H > d['humidity_threshold']:
        m.hum_overrule = pyo.Constraint(expr = m.v == 1)

    # [C5/C6] Heater overrules — match Environment.py logic exactly
    if T1 > d['temp_max_comfort_threshold']:
        m.p1_hi = pyo.Constraint(expr = m.p1 == 0.0)
    elif lo_r1 == 1:
        m.p1_lo = pyo.Constraint(expr = m.p1 == d['heating_max_power'])

    if T2 > d['temp_max_comfort_threshold']:
        m.p2_hi = pyo.Constraint(expr = m.p2 == 0.0)
    elif lo_r2 == 1:
        m.p2_lo = pyo.Constraint(expr = m.p2 == d['heating_max_power'])

    # ── Solve ──────────────────────────────────────────────────────────────
    solver = pyo.SolverFactory('gurobi')
    result = solver.solve(m, tee=False)

    ok = (result.solver.termination_condition
          == pyo.TerminationCondition.optimal)

    if not ok:
        return _greedy_action(state)   # safe fallback

    return {
        'HeatPowerRoom1': float(np.clip(pyo.value(m.p1), 0.0, d['heating_max_power'])),
        'HeatPowerRoom2': float(np.clip(pyo.value(m.p2), 0.0, d['heating_max_power'])),
        'VentilationON':  int(round(float(pyo.value(m.v)))),
    }


# SECTION 7 — MODULE-LEVEL TRAINING  (runs once when module is imported)

#  The environment calls select_action() once per hour.
#  The 15-second per-call limit applies to select_action(), NOT to module
#  import.  Training runs once at import time so that select_action() only
#  performs a fast MILP solve (~1-2 ms).

_THETA_LIST = train_ADP(n_scenarios=N_SCENARIOS, n_iter=N_ITER, verbose=True)


# SECTION 9 — POLICY ENTRY POINT  (called by the environment every hour)

def select_action(state):
    """
    ADP policy: solve the 1-step lookahead MILP with pre-trained VFA weights.

    The environment calls this function once per hour with the current state.
    It must return within 15 seconds; in practice it takes < 1 second.

    Parameters
    ----------
    state : dict with keys
        T1, T2, H, Occ1, Occ2, price_t, price_previous,
        vent_counter, low_override_r1, low_override_r2, current_time

    Returns
    -------
    dict with keys  HeatPowerRoom1 (float), HeatPowerRoom2 (float),
                    VentilationON  (int ∈ {0, 1})
    """
    t = int(state['current_time'])

    # At the last time slot there is no future → θ_{T} = 0
    if t >= T_SLOTS - 1:
        theta_next = np.zeros(N_FEATURES)
    else:
        theta_next = _THETA_LIST[t + 1]

    return _solve_lookahead_MILP(state, theta_next)
