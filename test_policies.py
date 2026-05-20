from Environment import evaluate_performance
from Environment import plot_HVAC_results
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from professor_model import evaluate_hindsight_model
from SystemCharacteristics import get_fixed_data
import random
import time

FIXED_DATA = get_fixed_data()



policies = [
    #"Hindsight_policy_20",
    #"Dummy_policy_20",
    #"Lookahead_policy_20",
    #"SP2Stage_policy_20",
    #'Hybrid_ADP_policy_20',
    "Hybrid_policy_Fabian",
    # "OLD_multi_ADP_policy_20",
    #"SP_policy_20",
    #"SP_Price_LA_Occ_policy_20",
    #"ADP_policy_20",
    #"SP_Lookahead_policy_20",
    #'SP_Price_LA_Occ_policy_20',
    #'ADP_SP_policy_20',
    #"Hybrid_policy_20"
    ]

master_results = {}

# Data files
DIR = Path(os.getcwd())
price_file = DIR / "PriceData.csv"
occupancy1_file = DIR / "OccupancyRoom1.csv"
occupancy2_file = DIR / "OccupancyRoom2.csv"
num_days = 20
optimal = 127.20
end_strings = []

for i, policy in enumerate(policies):
    start_time = time.time()
    # random.seed(42)  # Set seed for reproducibility
    print(f"Evaluating policy: {policy}")
    if policy == "Hindsight_policy_20":
        results = evaluate_hindsight_model(num_days, file_price_data=price_file, file_occupancy1=occupancy1_file, file_occupancy2=occupancy2_file)
        optimal = np.mean([r['cost_total'] for r in results])
    else:
        results = evaluate_performance(policy, num_days, file_price_data=price_file, file_occupancy1=occupancy1_file, file_occupancy2=occupancy2_file)
    master_results[policy] = results
    avg_cost = np.mean([r['cost_total'] for r in master_results[policy]])
    end_time = time.time()
    end_string = f"Average daily cost over {len(results)} days for policy {policy}: {avg_cost:.2f} € ({(avg_cost/optimal-1)*100:.2f} % over optimal) after {end_time - start_time:.2f} seconds\n"
    end_strings.append(end_string)
    print(end_string)

for end_string in end_strings:
    print(end_string)