from SystemCharacteristics import get_fixed_data
import pandas as pd
import importlib
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DIR = Path(__file__).parent
np.random.seed(20) # To be able to compare different runs, comment out if you want true randomness
FIXED_DATA = get_fixed_data()
DUMMY_POLICY = "dummy_policy_20"

# Loads the correct python file of the policy and the function select_action from that file
def load_policy(module_name, function_name):
    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    return func

#Loads data from the csv files specified
def load_data(file_price_data, file_occupancy1, file_occupancy2):
    price_data = pd.read_csv(file_price_data)
    occupancy1 = pd.read_csv(file_occupancy1)
    occupancy2 = pd.read_csv(file_occupancy2)

    return price_data, occupancy1, occupancy2


# Taken from the course content, slightly adjusted to fit our environment
def plot_HVAC_results(HVAC_results, axes=None):
    
    Temp_r1 = HVAC_results['Temp_r1']
    Temp_r2 = HVAC_results['Temp_r2']
    h_r1 = HVAC_results['h_r1']
    h_r2 = HVAC_results['h_r2']
    v = HVAC_results['v']
    Hum = HVAC_results['Hum']
    price = HVAC_results['price']
    Occ_r1 = HVAC_results['Occ_r1']
    Occ_r2 = HVAC_results['Occ_r2']
    
    T = range(len(Temp_r1))
    
    if axes is None:
        fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Room Temperatures
    axes[0].plot(T, Temp_r1, label='Room 1 Temp', marker='o')
    axes[0].plot(T, Temp_r2, label='Room 2 Temp', marker='s')
    axes[0].axhline(18, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(20, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Room Temperatures")
    axes[0].legend()
    axes[0].grid(True)
    
    # Heater consumption
    axes[1].bar(T, h_r1, width=0.4, label='Room 1 Heater', alpha=0.7)
    axes[1].bar(T, h_r2, width=0.4, label='Room 2 Heater', alpha=0.7)
    axes[1].set_ylabel("Heater Power (kW)")
    axes[1].set_title("Heater Consumption")
    axes[1].legend()
    axes[1].grid(True)
    
    # Ventilation and Humidity
    axes[2].step(T, v, where='mid', label='Ventilation ON', color='tab:blue')
    axes[2].plot(T, Hum, label='Humidity (%)', color='tab:orange', marker='o')
    axes[2].axhline(45, color='gray', linestyle='--', alpha=0.5)
    axes[2].axhline(60, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_ylabel("Ventilation / Humidity")
    axes[2].set_title("Ventilation Status and Humidity")
    axes[2].legend()
    axes[2].grid(True)
    
    # Electricity price and occupancy
    axes[3].plot(T, price, label='TOU Price (€/kWh)', color='tab:red', marker='x')
    axes[3].bar(T, Occ_r1, label='Occupancy Room 1', alpha=0.5)
    axes[3].bar(T, Occ_r2, bottom=Occ_r1, label='Occupancy Room 2', alpha=0.5)
    axes[3].set_ylabel("Price / Occupancy")
    axes[3].set_xlabel("Time (hours)")
    axes[3].set_title("Electricity Price and Occupancy")
    axes[3].legend()
    axes[3].grid(True)


# automatically checks if execution works, copied and slighly adhusted from course content
def check_and_sanitize_action(select_action, state, dummy_action):
    # ---------------------------------------
    # 1. Ask the policy & time it
    # ---------------------------------------
    t0 = time.time()
    try:
        action = select_action(state)
        elapsed = time.time() - t0

        # If policy is too slow → dummy
        if elapsed > 15.0:
            print(f"[WARNING] Policy too slow ({elapsed:.2f}s). Using dummy action.")
            return dummy_action(state)

    except Exception as e:
        print(f"[WARNING] Policy crashed: {e}. Using dummy action.")
        return dummy_action(state)

    

    # ---------------------------------------
    # 3. Clip actions to feasible bounds
    # ---------------------------------------
    # ---------------------------------------
    # 2. Clip to feasible set (or fail → dummy)
    # ---------------------------------------
    PowerMax = FIXED_DATA["heating_max_power"] 
    try:
        action["HeatPowerRoom1"] = float(np.clip(action["HeatPowerRoom1"], 0, PowerMax))
        action["HeatPowerRoom2"] = float(np.clip(action["HeatPowerRoom2"], 0, PowerMax))
    
        # ventilation: threshold to {0,1}
        action["VentilationON"] = int(float(action["VentilationON"]) > 0.5)

    except Exception as e:
        print(f"[WARNING] Action clipping failed: {e}. Using dummy action.")
        return dummy_action(state)

    # ---------------------------------------
    # Return sanitized action
    # ---------------------------------------
    return {"HeatPowerRoom1": action["HeatPowerRoom1"], "HeatPowerRoom2": action["HeatPowerRoom2"], "VentilationON": action["VentilationON"]}

def apply_overrule(state, decision):
    # HUmidity overrule
    if state['H'] > FIXED_DATA["humidity_threshold"]:
        decision['VentilationON'] = 1
    elif state['vent_counter'] > 0 and state['vent_counter'] < FIXED_DATA["vent_min_up_time"]:
        decision['VentilationON'] = 1

    # Heating overrule for room 1
    if state['T1'] > FIXED_DATA["temp_max_comfort_threshold"]:
        decision['HeatPowerRoom1'] = 0
    elif state['low_override_r1'] == 1:
        decision['HeatPowerRoom1'] = FIXED_DATA["heating_max_power"]
    
    # Heating overrule for room 2
    if state['T2'] > FIXED_DATA["temp_max_comfort_threshold"]:
        decision['HeatPowerRoom2'] = 0
    elif state['low_override_r2'] == 1:
        decision['HeatPowerRoom2'] = FIXED_DATA["heating_max_power"]
    return decision

# not needed, Overrule and check and sanitize cover this
def is_feasible(state, decision):
    return True

def apply_dynamics(state, decision, occupancy1, occupancy2, price):
    next_state = {}
    next_state["current_time"] = state['current_time'] + 1
    if next_state["current_time"] < FIXED_DATA["num_timeslots"]:   
        next_state["Occ1"] = occupancy1[next_state["current_time"]]
        next_state["Occ2"] = occupancy2[next_state["current_time"]]
        next_state["price_t"] = price[next_state["current_time"]]
    next_state["T1"] = state['T1'] \
        + FIXED_DATA["heat_exchange_coeff"] * (state['T2'] - state['T1']) \
        + FIXED_DATA["thermal_loss_coeff"] * (FIXED_DATA["outdoor_temperature"][state['current_time']] - state['T1']) \
        + FIXED_DATA["heating_efficiency_coeff"] * decision['HeatPowerRoom1'] \
        - FIXED_DATA["heat_vent_coeff"] * decision['VentilationON'] \
        + FIXED_DATA["heat_occupancy_coeff"] * state['Occ1']   
    next_state["T2"] = state['T2'] \
        + FIXED_DATA["heat_exchange_coeff"] * (state['T1'] - state['T2']) \
        + FIXED_DATA["thermal_loss_coeff"] * (FIXED_DATA["outdoor_temperature"][state['current_time']] - state['T2']) \
        + FIXED_DATA["heating_efficiency_coeff"] * decision['HeatPowerRoom2'] \
        - FIXED_DATA["heat_vent_coeff"] * decision['VentilationON'] \
        + FIXED_DATA["heat_occupancy_coeff"] * state['Occ2']
    next_state["H"] = state['H'] \
        + FIXED_DATA["humidity_occupancy_coeff"] * (state['Occ1'] + state['Occ2']) \
        - FIXED_DATA["humidity_vent_coeff"] * decision['VentilationON']
    next_state["price_previous"] = state['price_t']
    next_state["vent_counter"] = state['vent_counter'] + 1 if decision['VentilationON'] == 1 else 0
    next_state["low_override_r1"] = 1 if next_state['T1'] <= FIXED_DATA["temp_min_comfort_threshold"] or (state['low_override_r1'] == 1 and next_state['T1'] < FIXED_DATA["temp_OK_threshold"]) else 0
    next_state["low_override_r2"] = 1 if next_state['T2'] <= FIXED_DATA["temp_min_comfort_threshold"] or (state['low_override_r2'] == 1 and next_state['T2'] < FIXED_DATA["temp_OK_threshold"]) else 0
    return next_state


def evaluate_daily_performance(policy_file, price, occupancy1, occupancy2, initial_state):
    # Setup
    results = {
        'Temp_r1': [],
        'Temp_r2': [],
        'h_r1': [],
        'h_r2': [],
        'v': [],
        'Hum': [],
        'price': [],
        'Occ_r1': [],
        'Occ_r2': [],
        'cost': [],
    }

    state = initial_state.copy()
    select_action = load_policy(policy_file, "select_action")
    dummy_action = load_policy(DUMMY_POLICY, "select_action")
    while state['current_time'] < FIXED_DATA["num_timeslots"]:
        decision = check_and_sanitize_action(select_action, state, dummy_action)
        decision = apply_overrule(state, decision)

        # Log results
        results['Temp_r1'].append(state['T1'])
        results['Temp_r2'].append(state['T2'])
        results['h_r1'].append(decision['HeatPowerRoom1'])
        results['h_r2'].append(decision['HeatPowerRoom2'])
        results['v'].append(decision['VentilationON'])
        results['Hum'].append(state['H'])
        results['price'].append(state['price_t'])
        results['Occ_r1'].append(state['Occ1'])
        results['Occ_r2'].append(state['Occ2'])
        results['cost'].append(state['price_t'] * (decision['HeatPowerRoom1'] + decision['HeatPowerRoom2'] + decision['VentilationON'] * FIXED_DATA["ventilation_power"]))
        
        next_state = apply_dynamics(state, decision, occupancy1, occupancy2, price)
        state = next_state
    results['cost_total'] = sum(results['cost'])
    return results


def evaluate_performance(policy_file="dummy_policy_20.py", days=100, file_price_data=DIR / "PriceData.csv", file_occupancy1=DIR / "OccupancyRoom1.csv", file_occupancy2=DIR / "OccupancyRoom2.csv"):
    # Setup
    all_results = []
    price_data, occupancy1_data, occupancy2_data = load_data(file_price_data, file_occupancy1, file_occupancy2)
    


    for day in range(days):
        price = price_data.iloc[day].values
        occupancy1 = occupancy1_data.iloc[day].values
        occupancy2 = occupancy2_data.iloc[day].values
        initial_state = {
            "T1": FIXED_DATA["T1"], #initial temperature at room 1
            "T2": FIXED_DATA["T2"], #initial temperature at room 2
            "H": FIXED_DATA["H"], #initial humidity
            "Occ1": occupancy1[0], #initial occupancy at room 1
            "Occ2": occupancy2[0], #initial occupancy at room 2
            "price_t": price[0],  #initial price
            "price_previous": FIXED_DATA["price_previous"],  #initial previous price
            "vent_counter": FIXED_DATA["vent_counter"], # initial counter (the ventilation was not ON previously)
            "low_override_r1": FIXED_DATA["low_override_r1"],  #initial condition of the overrule controller in room 1 (OFF)
            "low_override_r2": FIXED_DATA["low_override_r2"], #initial condition of the overrule controller in room 2 (OFF)
            "current_time": 0 
        }

        result = evaluate_daily_performance(policy_file, price, occupancy1, occupancy2, initial_state)
        all_results.append(result)
    
    return all_results


def main():
    start_time = time.time()
    policy_file = "SP_Policy_Restaurant" # TODO replace with your policy file
    results = evaluate_performance(policy_file)
    plot_HVAC_results(results[0]) 
    plt.tight_layout()
    #plt.show()
    print("\nThe results are:")
    avg_daily_price = sum(day_result["cost_total"] for day_result in results) / len(results)
    print(f"Average daily price: {avg_daily_price:.2f}")
    print(f"It took: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()
