from Environment import evaluate_performance
from Environment import plot_HVAC_results
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from professor_model import evaluate_hindsight_model
from SystemCharacteristics import get_fixed_data
import time
import pandas as pd

FIXED_DATA = get_fixed_data()

SAVE_PICS = True

policies = [
    #"Hindsight_policy_20",
    #"Dummy_policy_20",
    #"Lookahead_policy_20",
    #"SP2Stage_policy_20",
    "SP_policy_20",
    #"ADP_policy_20",
    # "Hybrid_policy_Fabian"
    ]

master_results = {}

# Data files
DIR = Path(os.getcwd())
price_file = DIR / "PriceData.csv"
occupancy1_file = DIR / "OccupancyRoom1.csv"
occupancy2_file = DIR / "OccupancyRoom2.csv"
num_days = 1
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


print('------------RESULTS-----------------')
for end_string in end_strings:
    print(end_string)



###### Printing of graphs if wanted, as well as saving them ######

def pretty(policy):
    return policy.replace("_policy_20", "").replace("_", " ") + " policy"

if SAVE_PICS == True:
    all_costs = [res['cost_total'] for p in policies for res in master_results[p]]
    x_min, x_max = min(all_costs), max(all_costs)
    x_pad = (x_max - x_min) * 0.05
    x_lim = (x_min - x_pad, x_max + x_pad)

    all_counts, bin_edges = np.histogram(all_costs, bins=20, range=x_lim)
    y_max = 0
    for policy in policies:
        counts, _ = np.histogram([r['cost_total'] for r in master_results[policy]], bins=bin_edges)
        y_max = max(y_max, counts.max())
    y_lim = (0, y_max * 1.15)

    output_dir = DIR / "figures"
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, len(policies), figsize=(6 * len(policies), 4), squeeze=False)
    for i, policy in enumerate(policies):
        daily_costs = [res['cost_total'] for res in master_results[policy]]
        avg_cost = np.mean(daily_costs)
        label = pretty(policy)
        pd.DataFrame({'daily_cost': daily_costs}).to_csv(output_dir / f"{policy}_costs.csv", index=False)
        ax = axes[0][i]
        ax.hist(daily_costs, bins=bin_edges, edgecolor='black')
        ax.axvline(avg_cost, color='red', linestyle='--', label=f'Mean: {avg_cost:.2f} €')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(label)
        ax.set_xlabel("Daily Cost (€)")
        ax.set_ylabel("Number of Days")
        ax.legend()

        # Save individual histogram
        fig_single, ax_single = plt.subplots(figsize=(6, 4))
        ax_single.hist(daily_costs, bins=bin_edges, edgecolor='black')
        ax_single.axvline(avg_cost, color='red', linestyle='--', label=f'Mean: {avg_cost:.2f} €')
        ax_single.set_xlim(x_lim)
        ax_single.set_ylim(y_lim)
        ax_single.set_title(label)
        ax_single.set_xlabel("Daily Cost (€)")
        ax_single.set_ylabel("Number of Days")
        ax_single.legend()
        fig_single.tight_layout()
        fig_single.savefig(output_dir / f"{policy}_hist.png", dpi=150)
        plt.close(fig_single)

    fig.tight_layout()
    plt.show()

    # Bar chart: average cost per policy, sorted lowest to highest
    avg_costs = {p: np.mean([r['cost_total'] for r in master_results[p]]) for p in policies}
    sorted_policies = sorted(avg_costs, key=avg_costs.get)
    sorted_costs = [avg_costs[p] for p in sorted_policies]
    sorted_labels = [pretty(p) for p in sorted_policies]

    fig2, ax2 = plt.subplots(figsize=(max(6, len(policies) * 1.5), 5))
    bars = ax2.bar(sorted_labels, sorted_costs, edgecolor='black')
    ax2.bar_label(bars, fmt='%.2f €', padding=3)
    ax2.set_ylabel("Average Daily Cost (€)")
    ax2.set_title("Average Cost per Policy (lowest to highest)")
    ax2.set_xticklabels(sorted_labels, rotation=15, ha='right')
    fig2.tight_layout()
    fig2.savefig(output_dir / "avg_cost_bar.png", dpi=150)
    plt.show()