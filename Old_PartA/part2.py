import pandas as pd
import numpy as np
import pyomo.environ as pyo
from SystemCharacteristics import get_fixed_data

# Load the fixzed data
data = get_fixed_data()

# Load the CSV data (Price and Occupancy)
prices = pd.read_csv("PriceData.csv").values
occ_room1 = pd.read_csv("OccupancyRoom1.csv").values
occ_room2 = pd.read_csv("OccupancyRoom2.csv").values

for d in range(100):
    # Let us initialize a model
    model = pyo.ConcreteModel()

    # Defining the sets
    model.T = pyo.RangeSet(1,10) # The timesteps
    model.R = pyo.RangeSet(1,2) # The rooms

    # Defining the State variables
    model.Temp = pyo.Var(model.R, model.T, domain=pyo.Reals)
    model.Hum = pyo.Var(model.R, model.T, domain=pyo.Reals)

    # Defining the decision variables
    model.p = pyo.Var(model.R, model.T, domain=pyo.NonNegativeReals, bounds=(0,data['heating_max_power']))
    model.v = pyo.Var(model.T, domain=pyo.Binary)

    # Overrule controller indicators
    model.overrule_temp_low = pyo.Var(model.R, model.T, domain=pyo.Binary)
    model.overrule_temp_high = pyo.Var(model.R, model.T, domain=pyo.Binary)

    # Defining the objective function
    def cost_function(model):
        cost = 0
        for t in model.T:
            price_t = prices[d,t-1]
            power_usage = sum(model.p[r,t] for r in model.R)
            vent_usage =  model.v[t] * data['ventilation_power']
            cost += price_t * (power_usage + vent_usage)
        return cost
    model.Obj = pyo.Objective(rule=cost_function, sense=pyo.minimize)

    # Defining the rules for the temperature
    def temp_rule(model, r, t):
        if t == 1:
            temp_prev = data['initial_temperature']
        else:
            temp_prev = model.Temp[r, t-1]
            
        if r == 1:
            occ_prev = occ_room1[d,t-1] 
        elif r == 2:
            occ_prev = occ_room2[d,t-1]
            
        if r == 1:
            other_r = 2 
        else:
            other_r = 1
        
        if t == 1:
            temp_other_prev = data['initial_temperature']
        else:
            temp_other_prev = model.Temp[other_r, t-1]
        
        temp = (
            temp_prev
            + data['heat_exchange_coeff']*(temp_other_prev - temp_prev)
            + data['thermal_loss_coeff']*(data['outdoor_temperature'] - temp_prev)
            + data['heating_efficiency_coeff']*model.p[r,t]
            - data['heat_vent_coeff']*model.v[t]
            + data['heat_occupancy_coeff']*occ_prev
        )
        return model.Temp[r, t] == temp
    model.TempConstr = pyo.Constraint(model.R, model.T, rule=temp_rule)

    # Defining the rules for the humidity
    def hum_rule(model, t):
        if t == 1:
            hum_prev = data['initial_humidity']
        else:
            hum_prev = model.Hum[t-1]
            
        prev_total_occ = occ_room1[d,t-1] + occ_room2[d,t-1]
        
        hum = hum_prev + (data['humidity_occupancy_coeff'] * prev_total_occ) - (data['humidity_vent_coeff'] * model.v[t])
        return model.Hum[t] == hum
    model.HumConstr = pyo.Constraint(model.T, rule=hum_rule)

    M = 1000 
    def hum_overrule_rule(model, t):
        return model.Hum[t] <= data['humidity_threshold'] + M * model.v[t]
    model.HumOverrule = pyo.Constraint(model.T, rule=hum_overrule_rule)

    # Defining the rules for the temp overrule controller
    