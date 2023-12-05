# %%

import sys
from pathlib import Path

# To import functions from parent directory
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

import pickle, os
from datetime import datetime
from synthetic_data import *
from optimize import optimize, evaluate

# %% Generate Synthetic Data

# Material Parameters (MPa)
mat = "ubbi"

mu = 0.6
mu_v = 0.96
N = 8
N_v = 8
gamma0_taum = 7
c = -1
m = 4
params = [mu, mu_v, N, N_v, gamma0_taum, c, m]

mdata = []

#region # MDATA 1

# Strains
dt_linear = 10
dt_relax = 25
rate = 0.002
# Linear
t, strain = linear_loading(start_val=0, end_val=-0.3, start_time=0, rate=-rate, dt=dt_linear, plot=False)
# Relaxation
t_new, strain_new = relaxation_loading(val=-0.3, start_time=t[-1]+dt_relax, time_period=120, dt=dt_relax, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Linear
t_new, strain_new = linear_loading(start_val=-0.3, end_val=-0.6, start_time=t[-1]+dt_linear, rate=-rate, dt=dt_linear, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Relaxation
t_new, strain_new = relaxation_loading(val=-0.6, start_time=t[-1]+dt_relax, time_period=120, dt=dt_relax, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Linear
t_new, strain_new = linear_loading(start_val=-0.6, end_val=-0.8, start_time=t[-1]+dt_linear, rate=-rate, dt=dt_linear, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Linear
t_new, strain_new = linear_loading(start_val=-0.8, end_val=-0.6, start_time=t[-1]+dt_linear, rate=rate, dt=dt_linear, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Relaxation
t_new, strain_new = relaxation_loading(val=-0.6, start_time=t[-1]+dt_relax, time_period=120, dt=dt_relax, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Linear
t_new, strain_new = linear_loading(start_val=-0.6, end_val=-0.3, start_time=t[-1]+dt_linear, rate=rate, dt=dt_linear, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Relaxation
t_new, strain_new = relaxation_loading(val=-0.3, start_time=t[-1]+dt_relax, time_period=120, dt=dt_relax, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Linear
t_new, strain_new = linear_loading(start_val=-0.3, end_val=0, start_time=t[-1]+dt_linear, rate=rate, dt=dt_linear, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])

mdata.append(generate_mdata(t, strain, params, mat, plot=True, noise=0))

#endregion

#region # MDATA 2

# Strains
dt_linear = 0.2
dt_relax = 25
rate = 0.1
# Linear
t, strain = linear_loading(start_val=0, end_val=-0.3, start_time=0, rate=-rate, dt=dt_linear, plot=False)
# Relaxation
t_new, strain_new = relaxation_loading(val=-0.3, start_time=t[-1]+dt_relax, time_period=120, dt=dt_relax, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Linear
t_new, strain_new = linear_loading(start_val=-0.3, end_val=-0.6, start_time=t[-1]+dt_linear, rate=-rate, dt=dt_linear, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Relaxation
t_new, strain_new = relaxation_loading(val=-0.6, start_time=t[-1]+dt_relax, time_period=120, dt=dt_relax, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Linear
t_new, strain_new = linear_loading(start_val=-0.6, end_val=-0.8, start_time=t[-1]+dt_linear, rate=-rate, dt=dt_linear, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Linear
t_new, strain_new = linear_loading(start_val=-0.8, end_val=-0.6, start_time=t[-1]+dt_linear, rate=rate, dt=dt_linear, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Relaxation
t_new, strain_new = relaxation_loading(val=-0.6, start_time=t[-1]+dt_relax, time_period=120, dt=dt_relax, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Linear
t_new, strain_new = linear_loading(start_val=-0.6, end_val=-0.3, start_time=t[-1]+dt_linear, rate=rate, dt=dt_linear, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Relaxation
t_new, strain_new = relaxation_loading(val=-0.3, start_time=t[-1]+dt_relax, time_period=120, dt=dt_relax, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])
# Linear
t_new, strain_new = linear_loading(start_val=-0.3, end_val=0, start_time=t[-1]+dt_linear, rate=rate, dt=dt_linear, plot=False)
t = np.hstack([t, t_new])
strain = np.hstack([strain, strain_new])

mdata.append(generate_mdata(t, strain, params, mat, plot=True, noise=0))

#endregion

#%% Perform parameter identification

# Predicted material parameters
mat = "ubbi"

# Initial values
mu_pred = np.random.uniform(0.5, 1.5)
mu_v_pred = np.random.uniform(0.5, 1.5)
N_pred = 1/np.random.uniform(0.1, 0.2)
N_v_pred = 1/np.random.uniform(0.1, 0.2)
gamma0_taum_pred = np.random.uniform(10, 15)
c_pred = -np.random.uniform(0.9, 1.1)
m_pred = np.random.uniform(0.9, 1.1)

params_init = [mu_pred, mu_v_pred, N_pred, N_v_pred, gamma0_taum_pred, c_pred, m_pred]
params_names = ["mu", "mu_v", "N", "N_v", "gamma0_taum", "c", "m"]

# Optimize
params_out, params_hist, loss_hist = \
optimize(mdata, params_init, params_names, mat,
         max_iter=200, print_after=1, plot_after = 100, tol=10e-6,
         alpha=1e-1, b1=0.9, b2=0.999, e=1e-8, lambda_1 = 0,
         fix_param = [False, False, False, False, False, False, False],
         get_lowest=True)

# %% Plots

params_hist = np.array(params_hist)
lst = range(7)
for j in lst:
        plt.figure()
        plt.plot(params_hist[:, j])
        plt.plot([params[j] for i in range(200)], linestyle='dashed', color='red')
        plt.title(params_names[j])
plt.figure()
plt.plot(loss_hist)
plt.title('Loss')
evaluate(mdata, params_init, mat)
evaluate(mdata, params_out, mat)

# %% Store results

now = datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M-%S")

fname = f"saved_results_{now}.pkl"
fname = os.path.join("Results", fname)

with open(fname, 'wb') as file:
    pickle.dump([mdata, params_hist, loss_hist], file)