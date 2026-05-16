import pyomo.environ as pyo
from SystemCharacteristics import get_fixed_data
from pathlib import Path
from Environment import load_data

FIXED_DATA = get_fixed_data()
DIR = Path(__file__).parent

def create_professor_model(price_arr, occ_r1_arr, occ_r2_arr, data):
    
    num_t = data['num_timeslots']
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, num_t - 1)
    model.R = pyo.Set(initialize=['r1', 'r2'])

    # Mapping occupancy arrays to a dictionary for easy indexing (r, t)
    occ_data = {}
    for t in range(num_t):
        occ_data[('r1', t)] = occ_r1_arr[t]
        occ_data[('r2', t)] = occ_r2_arr[t]

    # Variables
    model.p = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals, bounds=(0,data['heating_max_power'])) 
    model.T_in = pyo.Var(model.R, model.T, within=pyo.Reals)         
    model.H = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0,100))          
    model.v = pyo.Var(model.T, within=pyo.Binary)
    model.s = pyo.Var(model.T, within=pyo.Binary)
    model.u = pyo.Var(model.R, model.T, within=pyo.Binary)
    
    # Auxiliary variables
    model.y_low = pyo.Var(model.R, model.T, within=pyo.Binary) 
    model.y_ok = pyo.Var(model.R, model.T, within=pyo.Binary)
    model.y_high = pyo.Var(model.R, model.T, within=pyo.Binary)

    # Params
    model.lambda_e = pyo.Param(model.T, initialize={t: price_arr[t] for t in model.T})
    model.kappa = pyo.Param(model.R, model.T, initialize={('r1', t): occ_r1_arr[t] for t in model.T} | {('r2', t): occ_r2_arr[t] for t in model.T})
    model.L = pyo.Param(initialize=data['num_timeslots'])
    model.T_out = pyo.Param(model.T, initialize={t: data['outdoor_temperature'][t] for t in model.T})
    model.P_vent = pyo.Param(initialize=data['ventilation_power'])
    model.P = pyo.Param(model.R, initialize={r: data['heating_max_power'] for r in model.R})
    model.T_low = pyo.Param(initialize=data['temp_min_comfort_threshold'])
    model.T_high = pyo.Param(initialize=data['temp_max_comfort_threshold'])
    model.T_OK = pyo.Param(initialize=data['temp_OK_threshold'])
    model.H_high = pyo.Param(initialize=data['humidity_threshold'])
    model.M_temp = pyo.Param(initialize=100)
    model.M_hum = pyo.Param(initialize=100)
    model.U_vent = pyo.Param(initialize=data['vent_min_up_time'])
    model.zeta_exch = pyo.Param(initialize=data['heat_exchange_coeff'])
    model.zeta_loss = pyo.Param(initialize=data['thermal_loss_coeff'])
    model.zeta_conv = pyo.Param(initialize=data['heating_efficiency_coeff'])
    model.zeta_cool = pyo.Param(initialize=data['heat_vent_coeff'])
    model.zeta_occ = pyo.Param(initialize=data['heat_occupancy_coeff'])
    model.eta_occ = pyo.Param(initialize=data['humidity_occupancy_coeff'])
    model.eta_vent = pyo.Param(initialize=data['humidity_vent_coeff'])



    # Objective Function
    def obj_rule(m):
        return sum(m.lambda_e[t] * (m.P_vent * m.v[t] + sum(m.p[r, t] for r in m.R)) for t in m.T)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints

    # INitial constraints
    model.T_in['r1', 0].fix(data['T1'])
    model.T_in['r2', 0].fix(data['T2'])
    model.H[0].fix(data['H'])
    model.u['r1', 0].fix(data['low_override_r1'])
    model.u['r2', 0].fix(data['low_override_r2'])
    
    

    # Temperature Dynamics with dynamic occupancy
    def temp_dynamics_rule(m, r, t):
        if t == 0: return pyo.Constraint.Skip
        r_other = 'r2' if r == 'r1' else 'r1'
        return m.T_in[r, t] == (m.T_in[r, t-1] +
                                m.zeta_exch * (m.T_in[r_other, t-1] - m.T_in[r, t-1]) -
                                m.zeta_loss * (m.T_in[r, t-1] - m.T_out[t-1]) +
                                m.zeta_conv * m.p[r, t-1] -
                                m.zeta_cool * m.v[t-1] +
                                m.zeta_occ * m.kappa[r, t-1])
    model.temp_dyn = pyo.Constraint(model.R, model.T, rule=temp_dynamics_rule)

    # Humidity Dynamics with dynamic occupancy
    def hum_dynamics_rule(m, t):
        if t == 0: return pyo.Constraint.Skip
        total_occ = sum(m.kappa[r, t-1] for r in m.R)
        return m.H[t] == m.H[t-1] + m.eta_occ * total_occ - m.eta_vent * m.v[t-1]
    model.hum_dyn = pyo.Constraint(model.T, rule=hum_dynamics_rule)

    # High Temp Logic / Heater Deactivation
    def high_temp_1(m, r, t):
        return m.T_in[r, t] >= m.T_high - m.M_temp * (1 - m.y_high[r, t])
    def high_temp_2(m, r, t):
        return m.T_in[r, t] <= m.T_high + m.M_temp * m.y_high[r, t]
    def heater_cutoff(m, r, t):
        return m.p[r, t] <= m.P[r] * (1 - m.y_high[r, t])
    model.hi_1 = pyo.Constraint(model.R, model.T, rule=high_temp_1)
    model.hi_2 = pyo.Constraint(model.R, model.T, rule=high_temp_2)
    model.hi_cut = pyo.Constraint(model.R, model.T, rule=heater_cutoff)

    # Low/OK Temp detection
    def low_temp_1(m, r, t):
        return m.T_in[r, t] <= m.T_low + m.M_temp * (1 - m.y_low[r, t])
    def low_temp_2(m, r, t):
        return m.T_in[r, t] >= m.T_low - m.M_temp * m.y_low[r, t]
    model.lo_1 = pyo.Constraint(model.R, model.T, rule=low_temp_1)
    model.lo_2 = pyo.Constraint(model.R, model.T, rule=low_temp_2)

    def ok_temp_1(m, r, t):
        return m.T_in[r, t] >= m.T_OK - m.M_temp * (1 - m.y_ok[r, t])
    def ok_temp_2(m, r, t):
        return m.T_in[r, t] <= m.T_OK + m.M_temp * m.y_ok[r, t]
    model.ok_1 = pyo.Constraint(model.R, model.T, rule=ok_temp_1)
    model.ok_2 = pyo.Constraint(model.R, model.T, rule=ok_temp_2)

    # Overrule Logic
    def ovr_1(m, r, t): 
        return m.u[r, t] >= m.y_low[r, t]
    def ovr_2(m, r, t):
        if t == 0: return pyo.Constraint.Skip
        return m.u[r, t] <= m.u[r, t-1] + m.y_low[r, t]
    
    def ovr_max(m, r, t): 
        return m.p[r, t] >= m.P[r] * m.u[r, t]
    
    def ovr_de1(m, r, t):
        if t == 0: return pyo.Constraint.Skip
        return m.u[r, t] >= m.u[r, t-1] - m.y_ok[r, t]
    def ovr_de2(m, r, t):
        if t == 0: return pyo.Constraint.Skip
        return m.u[r, t] <= 1 - m.y_ok[r, t]
    model.ov1 = pyo.Constraint(model.R, model.T, rule=ovr_1)
    model.ov2 = pyo.Constraint(model.R, model.T, rule=ovr_2)
    model.ovmax = pyo.Constraint(model.R, model.T, rule=ovr_max)
    model.ovde1 = pyo.Constraint(model.R, model.T, rule=ovr_de1)
    model.ovde2 = pyo.Constraint(model.R, model.T, rule=ovr_de2)

    # Ventilation Startup & Min Up-time
    def vent_start1(m, t):
        if t == 0: return m.s[t] >= m.v[t] - data['vent_counter']
        return m.s[t] >= m.v[t] - m.v[t-1]
    def vent_start2(m, t):
        return m.s[t] <= m.v[t]
    def vent_start3(m, t):
        if t == 0: return pyo.Constraint.Skip
        return m.s[t] <= 1 - m.v[t-1]
    model.v_start1 = pyo.Constraint(model.T, rule=vent_start1)
    model.v_start2 = pyo.Constraint(model.T, rule=vent_start2)
    model.v_start3 = pyo.Constraint(model.T, rule=vent_start3)

    def vent_uptime(m, t):
        horizon = num_t
        end_t = min(t + m.U_vent, horizon)
        return sum(m.v[tau] for tau in range(t, end_t)) >= (min(m.U_vent, horizon - t)) * m.s[t]
    model.v_up = pyo.Constraint(model.T, rule=vent_uptime)

    # Humidity Trigger
    def hum_trig(m, t):
        return m.H[t] <= m.H_high + m.M_hum * m.v[t]
    model.h_trig = pyo.Constraint(model.T, rule=hum_trig)

    return model



def solve_professor_model(price_arr, occ_r1_arr, occ_r2_arr, data):
    model = create_professor_model(price_arr, occ_r1_arr, occ_r2_arr, data)
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
        "cost_total": pyo.value(model.obj)
    }
    
    return HVAC_results


def evaluate_hindsight_model(days=100, file_price_data=DIR / "PriceData.csv", file_occupancy1=DIR / "OccupancyRoom1.csv", file_occupancy2=DIR / "OccupancyRoom2.csv"):
    all_results = []
    price_data, occupancy1_data, occupancy2_data = load_data(file_price_data, file_occupancy1, file_occupancy2)
    
    for day in range(days):
        price = price_data.iloc[day].values
        occupancy1 = occupancy1_data.iloc[day].values
        occupancy2 = occupancy2_data.iloc[day].values
       
        result =solve_professor_model(price, occupancy1, occupancy2, FIXED_DATA)
        all_results.append(result)
    
    return all_results

