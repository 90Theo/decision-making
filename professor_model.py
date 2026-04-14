import pyomo.environ as pyo
import numpy as np

def professor_model(price_arr, occ_r1_arr, occ_r2_arr, data):
    model = pyo.ConcreteModel()
    num_t = data['num_timeslots']
    model.T = pyo.RangeSet(0, num_t - 1)
    model.R = pyo.Set(initialize=['r1', 'r2'])

    # Mapping occupancy arrays to a dictionary for easy indexing (r, t)
    occ_data = {}
    for t in range(num_t):
        occ_data[('r1', t)] = occ_r1_arr[t]
        occ_data[('r2', t)] = occ_r2_arr[t]

    # --- Variables ---
    model.p = pyo.Var(model.R, model.T, domain=pyo.NonNegativeReals) # Heating power [cite: 7]
    model.T_in = pyo.Var(model.R, model.T, domain=pyo.Reals)         # Indoor temp [cite: 8]
    model.H = pyo.Var(model.T, domain=pyo.NonNegativeReals)          # Humidity [cite: 9]
    model.v = pyo.Var(model.T, domain=pyo.Binary)                    # Ventilation active [cite: 10]
    model.s = pyo.Var(model.T, domain=pyo.Binary)                    # Startup [cite: 11]
    model.u = pyo.Var(model.R, model.T, domain=pyo.Binary)          # Overrule active [cite: 13]
    
    # Auxiliary variables [cite: 12, 46]
    model.y_low = pyo.Var(model.R, model.T, domain=pyo.Binary) 
    model.y_ok = pyo.Var(model.R, model.T, domain=pyo.Binary)
    model.y_high = pyo.Var(model.R, model.T, domain=pyo.Binary)

    M_temp = 100.0 # Big-M [cite: 25]
    M_hum = 100.0

    # --- Objective Function ---
    # Minimize total cost using the provided price array [cite: 31, 32, 33]
    def obj_rule(m):
        return sum(price_arr[t] * (data['ventilation_power'] * m.v[t] + 
                                   sum(m.p[r, t] for r in m.R)) for t in m.T)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- Constraints ---
    for r in model.R:
        model.T_in[r, 0].fix(data['initial_temperature'])
    model.H[0].fix(data['initial_humidity'])

    # Temperature Dynamics with dynamic occupancy [cite: 35, 36]
    def temp_dynamics_rule(m, r, t):
        if t == 0: return pyo.Constraint.Skip
        r_other = 'r2' if r == 'r1' else 'r1'
        return m.T_in[r, t] == (m.T_in[r, t-1] + 
                                data['heat_exchange_coeff'] * (m.T_in[r_other, t-1] - m.T_in[r, t-1]) - 
                                data['thermal_loss_coeff'] * (m.T_in[r, t-1] - data['outdoor_temperature'][t-1]) + 
                                data['heating_efficiency_coeff'] * m.p[r, t-1] - 
                                data['heat_vent_coeff'] * m.v[t-1] + 
                                data['heat_occupancy_coeff'] * occ_data[(r, t-1)])
    model.temp_dyn = pyo.Constraint(model.R, model.T, rule=temp_dynamics_rule)

    # Humidity Dynamics with dynamic occupancy [cite: 39, 40]
    def hum_dynamics_rule(m, t):
        if t == 0: return pyo.Constraint.Skip
        total_occ = sum(occ_data[(r, t-1)] for r in m.R)
        return m.H[t] == m.H[t-1] + data['humidity_occupancy_coeff'] * total_occ - data['humidity_vent_coeff'] * m.v[t-1]
    model.hum_dyn = pyo.Constraint(model.T, rule=hum_dynamics_rule)

    # High Temp Logic / Heater Deactivation [cite: 53, 54, 55]
    def high_temp_1(m, r, t):
        return m.T_in[r, t] >= data['temp_max_comfort_threshold'] - M_temp * (1 - m.y_high[r, t])
    def high_temp_2(m, r, t):
        return m.T_in[r, t] <= data['temp_max_comfort_threshold'] + M_temp * m.y_high[r, t]
    def heater_cutoff(m, r, t):
        return m.p[r, t] <= data['heating_max_power'] * (1 - m.y_high[r, t])
    model.hi_1 = pyo.Constraint(model.R, model.T, rule=high_temp_1)
    model.hi_2 = pyo.Constraint(model.R, model.T, rule=high_temp_2)
    model.hi_cut = pyo.Constraint(model.R, model.T, rule=heater_cutoff)

    # Low/OK Temp detection [cite: 58, 60, 64, 66]
    def low_temp_1(m, r, t):
        return m.T_in[r, t] <= data['temp_min_comfort_threshold'] + M_temp * (1 - m.y_low[r, t])
    def low_temp_2(m, r, t):
        return m.T_in[r, t] >= data['temp_min_comfort_threshold'] - M_temp * m.y_low[r, t]
    model.lo_1 = pyo.Constraint(model.R, model.T, rule=low_temp_1)
    model.lo_2 = pyo.Constraint(model.R, model.T, rule=low_temp_2)

    def ok_temp_1(m, r, t):
        return m.T_in[r, t] >= data['temp_OK_threshold'] - M_temp * (1 - m.y_ok[r, t])
    def ok_temp_2(m, r, t):
        return m.T_in[r, t] <= data['temp_OK_threshold'] + M_temp * m.y_ok[r, t]
    model.ok_1 = pyo.Constraint(model.R, model.T, rule=ok_temp_1)
    model.ok_2 = pyo.Constraint(model.R, model.T, rule=ok_temp_2)

    # Overrule Logic [cite: 72, 73, 75, 77, 78]
    def ovr_1(m, r, t): return m.u[r, t] >= m.y_low[r, t]
    def ovr_2(m, r, t):
        if t == 0: return pyo.Constraint.Skip
        return m.u[r, t] <= m.u[r, t-1] + m.y_low[r, t]
    def ovr_max(m, r, t): return m.p[r, t] >= data['heating_max_power'] * m.u[r, t]
    def ovr_de1(m, r, t):
        if t == 0: return pyo.Constraint.Skip
        return m.u[r, t] >= m.u[r, t-1] - m.y_ok[r, t]
    def ovr_de2(m, r, t): return m.u[r, t] <= 1 - m.y_ok[r, t]
    model.ov1 = pyo.Constraint(model.R, model.T, rule=ovr_1)
    model.ov2 = pyo.Constraint(model.R, model.T, rule=ovr_2)
    model.ovmax = pyo.Constraint(model.R, model.T, rule=ovr_max)
    model.ovde1 = pyo.Constraint(model.R, model.T, rule=ovr_de1)
    model.ovde2 = pyo.Constraint(model.R, model.T, rule=ovr_de2)

    # Ventilation Startup & Min Up-time [cite: 80, 83, 84]
    def vent_start(m, t):
        if t == 0: return m.s[t] >= m.v[t]
        return m.s[t] >= m.v[t] - m.v[t-1]
    model.v_start = pyo.Constraint(model.T, rule=vent_start)

    def vent_uptime(m, t):
        U = data['vent_min_up_time'] # U_vent [cite: 27]
        horizon = num_t # L [cite: 17]
        end_t = min(t + U, horizon)
        return sum(m.v[tau] for tau in range(t, end_t)) >= (min(U, horizon - t)) * m.s[t]
    model.v_up = pyo.Constraint(model.T, rule=vent_uptime)

    # Humidity Trigger [cite: 85, 86]
    def hum_trig(m, t):
        return m.H[t] <= data['humidity_threshold'] + M_hum * m.v[t]
    model.h_trig = pyo.Constraint(model.T, rule=hum_trig)

    # Solve
    solver = pyo.SolverFactory('gurobi')
    solver.solve(model)

    # Final result structure as requested
    HVAC_results = {
        "Temp_r1": [pyo.value(model.T_in['r1', t]) for t in model.T],
        "Temp_r2": [pyo.value(model.T_in['r2', t]) for t in model.T],
        "h_r1": [pyo.value(model.p['r1', t]) for t in model.T],
        "h_r2": [pyo.value(model.p['r2', t]) for t in model.T],
        "v": [pyo.value(model.v[t]) for t in model.T],
        "s": [pyo.value(model.s[t]) for t in model.T],
        "z_high": [pyo.value(model.y_high['r1', t]) for t in model.T],
        "z_low": [pyo.value(model.y_low['r1', t]) for t in model.T],
        "Hum": [pyo.value(model.H[t]) for t in model.T],
        "price": price_arr,
        "Occ_r1": occ_r1_arr,
        "Occ_r2": occ_r2_arr,
        "outdoor_temperature": data['outdoor_temperature'],
        "avg_cost": pyo.value(model.obj)
    }
    
    return HVAC_results