import pyomo.environ as pyo
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTask7 import fetch_data

FIXED_DATA = fetch_data()
DIR = Path(__file__).parent
n_stores = 15


def create_centralized_model(occ_r1_arr, occ_r2_arr, data):
    num_t = data['num_timeslots']
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, num_t - 1)
    model.R = pyo.Set(initialize=['r1', 'r2'])
    model.S = pyo.RangeSet(1, n_stores)

    occ_data = {}
    for t in range(num_t):
        occ_data[('r1', t)] = occ_r1_arr[t]
        occ_data[('r2', t)] = occ_r2_arr[t]

    model.p = pyo.Var(model.S, model.R, model.T, within=pyo.NonNegativeReals, bounds=(0, data['heating_max_power']))
    model.T_in = pyo.Var(model.S, model.R, model.T, within=pyo.Reals)

    model.v = pyo.Param(model.S, model.T, initialize=1)
    model.kappa = pyo.Param(model.S, model.R, model.T,
                             initialize={(s, 'r1', t): occ_r1_arr[t] for t in model.T for s in model.S}
                                       | {(s, 'r2', t): occ_r2_arr[t] for t in model.T for s in model.S})
    model.L = pyo.Param(initialize=data['num_timeslots'])
    model.T_out = pyo.Param(model.T, initialize={t: data['outdoor_temperature'][t] for t in model.T})
    model.P_vent = pyo.Param(initialize=0)
    model.P = pyo.Param(model.S, model.R, initialize={(s, r): data['heating_max_power'] for s in model.S for r in model.R})
    model.zeta_exch = pyo.Param(initialize=data['heat_exchange_coeff'])
    model.zeta_loss = pyo.Param(initialize=data['thermal_loss_coeff'])
    model.zeta_conv = pyo.Param(initialize=data['heating_efficiency_coeff'])
    model.zeta_cool = pyo.Param(initialize=data['heat_vent_coeff'])
    model.zeta_occ = pyo.Param(initialize=data['heat_occupancy_coeff'])
    model.w = pyo.Param(model.S, initialize={s: s + 1 for s in model.S})
    model.T_ref = pyo.Param(initialize=data['Temperature_reference'])

    def obj_rule(m):
        return sum(m.w[s] * (m.T_in[s, r, t] - m.T_ref)**2 for s in m.S for r in m.R for t in m.T)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def initial_temp_rule(m, s, r):
        return m.T_in[s, r, 0] == data['initial_temperature']
    model.initial_temp = pyo.Constraint(model.S, model.R, rule=initial_temp_rule)

    def temp_dynamics_rule(m, s, r, t):
        if t == 0: return pyo.Constraint.Skip
        r_other = 'r2' if r == 'r1' else 'r1'
        return m.T_in[s, r, t] == (m.T_in[s, r, t-1]
                                   + m.zeta_exch * (m.T_in[s, r_other, t-1] - m.T_in[s, r, t-1])
                                   - m.zeta_loss  * (m.T_in[s, r, t-1] - m.T_out[t-1])
                                   + m.zeta_conv  * m.p[s, r, t-1]
                                   - m.zeta_cool  * m.v[s, t-1]
                                   + m.zeta_occ   * m.kappa[s, r, t-1])
    model.temp_dyn = pyo.Constraint(model.S, model.R, model.T, rule=temp_dynamics_rule)

    def mall_limit(m, t):
        return sum(m.p[s, r, t] for s in m.S for r in m.R) <= data['P_mall']
    model.mall_limit = pyo.Constraint(model.T, rule=mall_limit)

    return model


def solve_centralized_model(occ_r1_arr, occ_r2_arr):
    model = create_centralized_model(occ_r1_arr, occ_r2_arr, FIXED_DATA)
    solver = pyo.SolverFactory('gurobi')
    solver.solve(model)
    print("Objective:", pyo.value(model.obj))

    num_t = FIXED_DATA['num_timeslots']
    times = np.arange(num_t)
    stores = list(range(1, n_stores + 1))

    T_avg = {s: np.array([(pyo.value(model.T_in[s, 'r1', t]) + pyo.value(model.T_in[s, 'r2', t])) / 2
                           for t in times]) for s in stores}
    p_total = {s: np.array([pyo.value(model.p[s, 'r1', t]) + pyo.value(model.p[s, 'r2', t])
                             for t in times]) for s in stores}

    return model, {'T_avg': T_avg, 'p_total': p_total, 'times': times, 'stores': stores, 'obj': pyo.value(model.obj)}

# This plotting function was created using Claude   
def plot_centralized_visulizations(results):
    T_avg   = results['T_avg']
    p_total = results['p_total']
    times   = results['times']
    stores  = results['stores']

    # red (store 1) → green (store 15)
    cmap = plt.cm.RdYlGn
    def store_color(s):
        return cmap((s - 1) / (n_stores - 1))

    # --- Plot 1: Temperature evolution ---
    _, ax = plt.subplots(figsize=(12, 6))
    for s in stores:
        ax.plot(times, T_avg[s], marker='o', markersize=4, linewidth=1.5,
                color=store_color(s), label=f'Store {s}')
    ax.axhline(FIXED_DATA['Temperature_reference'], color='black', ls='--', lw=1.5,
               label=f"T_ref = {FIXED_DATA['Temperature_reference']}°C", alpha=0.8)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Average Temperature (°C)', fontsize=12)
    ax.set_title('Temperature Evolution – All Stores (Centralized Solution)', fontsize=13)
    ax.set_xticks(times)
    ax.legend(fontsize=8, ncol=4, loc='lower right')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(DIR / 'figures' / 'Task7_temperatures.png', dpi=150)
    print("Saved Task7_temperatures.png")

    # --- Plot 2: Power distribution ---
    _, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(times))
    for s in stores:
        ax.bar(times, p_total[s], bottom=bottom,
               color=store_color(s), label=f'Store {s}', alpha=0.9)
        bottom += p_total[s]
    ax.axhline(FIXED_DATA['P_mall'], color='black', ls='--', lw=2,
               label=f"P_mall = {FIXED_DATA['P_mall']} kW")
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Heating Power (kW)', fontsize=12)
    ax.set_title('Heating Power Distribution Among Stores (Centralized Solution)', fontsize=13)
    ax.set_xticks(times)
    ax.legend(fontsize=8, ncol=4, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(DIR / 'figures' / 'Task7_power_distribution.png', dpi=150)
    print("Saved Task7_power_distribution.png")

    plt.show()


def create_store_model(occ_r1_arr, occ_r2_arr, data, lams, weight):
    num_t = data['num_timeslots']
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, num_t - 1)
    model.R = pyo.Set(initialize=['r1', 'r2'])

    occ_data = {}
    for t in range(num_t):
        occ_data[('r1', t)] = occ_r1_arr[t]
        occ_data[('r2', t)] = occ_r2_arr[t]

    model.p = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals, bounds=(0, data['heating_max_power']))
    model.T_in = pyo.Var(model.R, model.T, within=pyo.Reals)

    model.v = pyo.Param(model.T, initialize=1)
    model.kappa = pyo.Param(model.R, model.T,
                             initialize={('r1', t): occ_r1_arr[t] for t in model.T}
                                       | {('r2', t): occ_r2_arr[t] for t in model.T})
    model.L = pyo.Param(initialize=data['num_timeslots'])
    model.T_out = pyo.Param(model.T, initialize={t: data['outdoor_temperature'][t] for t in model.T})
    model.P_vent = pyo.Param(initialize=0)
    model.P = pyo.Param(model.R, initialize={r: data['heating_max_power'] for r in model.R})
    model.zeta_exch = pyo.Param(initialize=data['heat_exchange_coeff'])
    model.zeta_loss = pyo.Param(initialize=data['thermal_loss_coeff'])
    model.zeta_conv = pyo.Param(initialize=data['heating_efficiency_coeff'])
    model.zeta_cool = pyo.Param(initialize=data['heat_vent_coeff'])
    model.zeta_occ = pyo.Param(initialize=data['heat_occupancy_coeff'])
    model.w = pyo.Param(initialize=weight)
    model.T_ref = pyo.Param(initialize=data['Temperature_reference'])
    model.lam = pyo.Param(model.T, initialize=lams)

    # At t 9 nothing gets influenced so we fix it to 0 to avoid numerical issues with the dual variables
    for r in model.R:
        model.p[r, num_t - 1].fix(0)

    def obj_rule(m):
        return sum(m.w * (m.T_in[r, t] - m.T_ref)**2 + m.lam[t] * m.p[r,t] for r in m.R for t in m.T)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def initial_temp_rule(m, r):
        return m.T_in[r, 0] == data['initial_temperature']
    model.initial_temp = pyo.Constraint(model.R, rule=initial_temp_rule)


    def temp_dynamics_rule(m, r, t):
        if t == 0: return pyo.Constraint.Skip
        r_other = 'r2' if r == 'r1' else 'r1'
        return m.T_in[r, t] == (m.T_in[r, t-1]
                                   + m.zeta_exch * (m.T_in[r_other, t-1] - m.T_in[r, t-1])
                                   - m.zeta_loss  * (m.T_in[r, t-1] - m.T_out[t-1])
                                   + m.zeta_conv  * m.p[r, t-1]
                                   - m.zeta_cool  * m.v[t-1]
                                   + m.zeta_occ   * m.kappa[r, t-1])
    model.temp_dyn = pyo.Constraint(model.R, model.T, rule=temp_dynamics_rule)

    return model


def solve_master_problem(occ_r1_arr, occ_r2_arr, lam):
    all_p = []
    all_obj_val = []
    for i in range(1, n_stores+1):
        model = create_store_model(occ_r1_arr, occ_r2_arr, FIXED_DATA, lam, i+1)
        solver = pyo.SolverFactory('gurobi')
        solver.solve(model)
        # print("Objective:", pyo.value(model.obj))
        
        ps = []
        for t in range(FIXED_DATA['num_timeslots']):
            p_r1 = pyo.value(model.p['r1', t], exception=False) or 0.0
            p_r2 = pyo.value(model.p['r2', t], exception=False) or 0.0
            ps.append(p_r1 + p_r2)
        all_p.append(ps)

        weight = i + 1
        real_obj = weight * sum(
            (pyo.value(model.T_in[r, t]) - FIXED_DATA['Temperature_reference'])**2
            for r in ['r1', 'r2'] for t in range(FIXED_DATA['num_timeslots']))
        all_obj_val.append(real_obj)
    
    return all_obj_val, all_p




def optimize_lambda(occ_r1_arr, occ_r2_arr, alpha, iterations):
    lams = np.zeros(10)
    obj_values = []
    p_values = []
    lams_values = []

    for i in range(iterations):
        lams_values.append(lams.copy())
        obj, p = solve_master_problem(occ_r1_arr, occ_r2_arr, lams)
        obj_values.append(sum(obj))
        p_values.append(p)
        p_sum = np.array(p).sum(axis=0)
        lams = np.maximum(0, lams + alpha * (p_sum - FIXED_DATA['P_mall']))
        # lams = lams + alpha * (p_sum - FIXED_DATA['P_mall'])
    
    res = {}
    res['iterations'] = iterations
    res['obj_values'] = obj_values
    res['p_values'] = p_values
    res['lams_values'] = lams_values
    return res

def optimize_lambda_adaptive(occ_r1_arr, occ_r2_arr, alpha_zero, iterations):
    lams = np.zeros(10)
    obj_values = []
    p_values = []
    lams_values = []

    for i in range(iterations):
        lams_values.append(lams.copy())
        obj, p = solve_master_problem(occ_r1_arr, occ_r2_arr, lams)
        obj_values.append(sum(obj))
        p_values.append(p)
        p_sum = np.array(p).sum(axis=0)
        alpha = alpha_zero / (1 + i)
        lams = np.maximum(0, lams + alpha * (p_sum - FIXED_DATA['P_mall']))
        # lams = lams + alpha * (p_sum - FIXED_DATA['P_mall'])
    
    res = {}
    res['iterations'] = iterations
    res['obj_values'] = obj_values
    res['p_values'] = p_values
    res['lams_values'] = lams_values
    return res


# This plotting function was created using Claude
def plot_results(all_res, optimal):
    # below the dict keys of  each list element
    # all_res['alpha'] -> alpha value used for computation saved as string, adaptive step is saved as 'adaptive step'
    # all_res['obj'] -> optimal value
    # all_res[iterations] -> number of iterations (x-axis)
    # all_res[obj_values] -> objective value for each iteration
    # all_res[p_values] -> list of p-values for each timeslot of all iterations
    # all_res[lams_values] -> list of lambdas for each timeslot of all iterations

    # optimal contains the optimal solution using the mathematical model (serves as benchmark)

    figures_dir = DIR / 'figures'
    figures_dir.mkdir(exist_ok=True)

    num_t = FIXED_DATA['num_timeslots']
    P_mall = FIXED_DATA['P_mall']
    n_cases = len(all_res)
    t_colors = plt.cm.tab10(np.linspace(0, 1, num_t))

    def violation_per_slot(p_values):
        # p_values[k] may be (n_stores, num_t) or (num_t,); sum stores axis if 2-D
        viols = []
        for p in p_values:
            p_arr = np.array(p)
            p_sum = p_arr.sum(axis=0) if p_arr.ndim == 2 else p_arr
            viols.append(p_sum - P_mall)
        return np.array(viols)  # (iterations, num_t)

    # ----------------------------------------------------------------
    # Figure 1: Objective value vs iterations
    # ----------------------------------------------------------------
    _, ax = plt.subplots(figsize=(10, 5))
    for res in all_res:
        ax.plot(range(1, res['iterations'] + 1), res['obj_values'],
                label=res['label'], linewidth=1.8)
    ax.axhline(optimal, color='black', ls='--', lw=2,
               label=f'Centralized optimal ({optimal:.1f})')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title('Dual Decomposition – Objective vs Iterations', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(figures_dir / 'objective_vs_iterations.png', dpi=150)
    print("Saved figures/objective_vs_iterations.png")

    # ----------------------------------------------------------------
    # Figure 2: Lambda evolution per timeslot (one subplot per case)
    # ----------------------------------------------------------------
    ncols = 3
    nrows = int(np.ceil(n_cases / ncols))
    _, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for idx, res in enumerate(all_res):
        ax = axes[idx]
        lams_arr = np.array(res['lams_values'])  # (iterations, num_t)
        for t in range(num_t):
            ax.plot(range(res['iterations']), lams_arr[:, t],
                    lw=1.2, color=t_colors[t], label=f't={t}')
        ax.set_title(res['label'], fontsize=11)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('λ_t')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.4)

    for idx in range(n_cases, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Dual Variables λ_t Evolution', fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / 'lambda_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved figures/lambda_evolution.png")

    # ----------------------------------------------------------------
    # Figure 3: Constraint violation per timeslot (one subplot per case)
    # ----------------------------------------------------------------
    _, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for idx, res in enumerate(all_res):
        ax = axes[idx]
        viols = violation_per_slot(res['p_values'])  # (iterations, num_t)
        for t in range(num_t):
            ax.plot(range(1, res['iterations'] + 1), viols[:, t],
                    lw=1.0, alpha=0.85, color=t_colors[t], label=f't={t}')
        ax.axhline(0, color='black', ls='--', lw=1)
        ax.set_title(res['label'], fontsize=11)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Σ p_n,t − P_mall (kW)')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.4)

    for idx in range(n_cases, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Constraint Violation per Time Slot  (Σₙ pₙ,t − Pᵐᵃˡˡ)', fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / 'constraint_violation.png', dpi=150, bbox_inches='tight')
    print("Saved figures/constraint_violation.png")

    plt.show()



def main():
    occ = pd.read_csv(DIR / 'Task7Occupancies.csv')
    occ_r1_arr = occ.iloc[0, :10].to_list()
    occ_r2_arr = occ.iloc[1, :10].to_list()

    _, results = solve_centralized_model(occ_r1_arr, occ_r2_arr)
    plot_centralized_visulizations(results)

    #Now lets do the distributed algo
    iterations = 100
    alphas = [0.001, 0.01, 0.1, 1, 10]
    # alphas = [0.01]
    all_res = []
    for alpha in alphas:
        print('Running optimization for α =', alpha)
        res = optimize_lambda(occ_r1_arr, occ_r2_arr, alpha, iterations)
        res['label'] = f'α = {alpha}'
        all_res.append(res)

    print('Running optimization for adaptive α')
    res = optimize_lambda_adaptive(occ_r1_arr, occ_r2_arr, 5, iterations)
    res['label'] = 'Adaptive α'
    all_res.append(res)

    # plot results
    plot_results(all_res, results['obj'])
    


if __name__ == '__main__':
    main()
