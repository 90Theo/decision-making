import pyomo.environ as pyo
from SystemCharacteristics import get_fixed_data
from PriceProcessRestaurant import price_model
from OccupancyProcessRestaurant import next_occupancy_levels
from professor_model import create_professor_model

NUM_LOOKAHEADS = 10
NUM_TRAJECORIES = 100

def lookahead_policy(state, data, lookaheads=3):
    lookaheads = min(lookaheads, data['num_timeslots'] - state['current_time'])

    # generate 100 uncertainty trajecories for price and occupancy and average over them to get expected values for the next time steps
    all_prices = []
    all_occ_r1 = []
    all_occ_r2 = []
    for i in range(NUM_TRAJECORIES):
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
        all_prices.append(prices)
        all_occ_r1.append(occ_r1)
        all_occ_r2.append(occ_r2)

    # average over the trajectories to get expected values
    prices = []
    occ_r1 = []
    occ_r2 = []
    for t in range(lookaheads):
        prices.append(sum(all_prices[i][t] for i in range(NUM_TRAJECORIES)) / NUM_TRAJECORIES)
        occ_r1.append(sum(all_occ_r1[i][t] for i in range(NUM_TRAJECORIES)) / NUM_TRAJECORIES)
        occ_r2.append(sum(all_occ_r2[i][t] for i in range(NUM_TRAJECORIES)) / NUM_TRAJECORIES)

    # adjust data to reflect curent state
    data['T1'] = state['T1']
    data['T2'] = state['T2']
    data['H'] = state['H']
    data['vent_counter'] = state['vent_counter']
    data['low_override_r1'] = state['low_override_r1']
    data['low_override_r2'] = state['low_override_r2']
    data['num_timeslots'] = lookaheads

    model = create_professor_model(prices, occ_r1, occ_r2, data)
    solver = pyo.SolverFactory('gurobi')
    solver.solve(model)


    # Extract results for the first time step (t=0) model.P['r1',0], model.P['r2',0], model.v[0]
    heat_power_r1 =pyo.value(model.p['r1', 0])
    heat_power_r2 = pyo.value(model.p['r2', 0])
    ventilation_on = pyo.value(model.v[0])


    HereAndNowActions = {
    "HeatPowerRoom1" : heat_power_r1,
    "HeatPowerRoom2" : heat_power_r2, 
    "VentilationON" : ventilation_on
    }
    
    return HereAndNowActions


import time
def select_action(state):
    data = get_fixed_data()
    HereAndNowActions = lookahead_policy(state, data, lookaheads=NUM_LOOKAHEADS)
    return HereAndNowActions