import pyomo.environ as pyo
from SystemCharacteristics import get_fixed_data
from PriceProcessRestaurant import price_model
from OccupancyProcessRestaurant import next_occupancy_levels

def professor_model(price_arr, occ_r1_arr, occ_r2_arr, data):
    

    num_t = len(price_arr)
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
        return sum(model.lambda_e[t] * (model.P_vent * model.v[t] + sum(model.p[r, t] for r in model.R)) for t in model.T)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints
    model.T_in['r1', 0].fix(data['T1'])
    model.T_in['r2', 0].fix(data['T2'])
    model.H[0].fix(data['H'])

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
        if t == 0: return pyo.Constraint.Skip
        return m.s[t] >= m.v[t] - m.v[t-1]
    def vent_start2(m, t):
        return m.s[t] <= m.v[t]
    def vent_start3(m, t):
        if t == 0: return pyo.Constraint.Skip
        return m.s[t] <= 1 - m.v[t-1]
    model.v_start = pyo.Constraint(model.T, rule=vent_start1)
    model.v_start = pyo.Constraint(model.T, rule=vent_start2)
    model.v_start = pyo.Constraint(model.T, rule=vent_start3)

    def vent_uptime(m, t):
        horizon = num_t
        end_t = min(t + m.U_vent - 1, horizon - 1)
        return sum(m.v[tau] for tau in range(t, end_t)) >= (min(m.U_vent, horizon - t)) * m.s[t]
    model.v_up = pyo.Constraint(model.T, rule=vent_uptime)

    # Humidity Trigger
    def hum_trig(m, t):
        return m.H[t] <= m.H_high + m.M_hum * m.v[t]
    model.h_trig = pyo.Constraint(model.T, rule=hum_trig)
    return model


def lookahead_policy(state, data, lookaheads=3):

    # generate uncertainty trajecories for price and occupancy
    prices = []
    occ_r1 = []
    occ_r2 = []
    prices.append(state['price_t'])
    occ_r1.append(state['Occ1'])
    occ_r2.append(state['Occ2'])

    for t in range(1, lookaheads):
        price_t = price_model(prices[-1], prices[-2] if t > 1 else state['price_previous'])
        occ_r1_t, occ_r2_t = next_occupancy_levels(occ_r1[-1], occ_r2[-1])
        
        prices.append(price_t)
        occ_r1.append(occ_r1_t)
        occ_r2.append(occ_r2_t)

    # adjust data to reflect curent state
    data['T1'] = state['T1']
    data['T2'] = state['T2']
    data['H'] = state['H']
    data['vent_counter'] = state['vent_counter']
    data['low_override_r1'] = state['low_override_r1']
    data['low_override_r2'] = state['low_override_r2']

    model = professor_model(prices, occ_r1, occ_r2, data)
    solver = pyo.SolverFactory('gurobi')
    solver.solve(model)


    # Extract results for the first time step (t=0) model.P['r1',0], model.P['r2',0], model.v[0]
    heat_power_r1 = model.p['r1', 0]
    heat_power_r2 = model.p['r2', 0]
    ventilation_on = model.v[0]


    HereAndNowActions = {
    "HeatPowerRoom1" : heat_power_r1,
    "HeatPowerRoom2" : heat_power_r2, 
    "VentilationON" : ventilation_on
    }
    
    return HereAndNowActions



def select_action(state):
    data = get_fixed_data()
    HereAndNowActions = lookahead_policy(state, data, lookaheads=10)

    return HereAndNowActions