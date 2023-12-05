# %%

import sys
from pathlib import Path

# To import functions from parent directory
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

import pickle, os, numpy as np, matplotlib.pyplot as plt
from datetime import datetime
from synthetic_data import *
from optimize import optimize, evaluate

# %% Experimental data from paper

# Original data
data = np.loadtxt('visco1_2.dat')
P11 = data[:, 0]
stretch = data[:, 2]/100 + 1
t = data[:, 4]

plt.figure()
plt.scatter(stretch, P11, s=0.5)

# Change to true stress and true strain
true_strain = np.log(stretch)
true_stress = P11*stretch

plt.figure()
plt.scatter(true_strain, true_stress, s=0.5)

# Make it sparser
idx = range(0, len(t), 5)
t = t[idx]
true_strain = true_strain[idx]
true_stress = true_stress[idx]

plt.figure()
plt.scatter(true_strain, true_stress, s=0.5)

mdata = [(t, true_strain, true_stress)]

#%% Perform parameter identification

# Predicted material parameters
mat = "ubbmi"

# Initial values
coef = 4
mu_pred = np.random.uniform(0.5, 1.5)
mu_v_pred = np.random.uniform(0.5, 1.5)
N_pred = 1/np.random.uniform(0.1, 0.2)
N_v_pred = 1/np.random.uniform(0.1, 0.2)
tau_hat_pred = np.random.uniform(10, 15)
aj_pred = [np.random.uniform(0.5, 1.5) for i in range(coef)]

params_names = ['mu', 'mu_v', 'N', 'N_v', 'tau_hat'] + [f'a{j}'.format(j=j) for j in range(1, coef+1)]
params_init = [mu_pred, mu_v_pred, N_pred, N_v_pred, tau_hat_pred] + aj_pred

# Optimize
params_out, params_hist, loss_hist = \
optimize(mdata, params_init, params_names, mat,
         max_iter=5000, print_after=100, plot_after = 100, tol=10e-6,
         alpha=1e-1, b1=0.9, b2=0.999, e=1e-8, lambda_1 = 0,
         fix_param = False,
         get_lowest=True)

# %% Plots

evaluate(mdata, params_init, mat)
pred = evaluate(mdata, params_out, mat)

# %% Store results

now = datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M-%S")

fname = f"saved_results_{now}.pkl"
fname = os.path.join("Results", fname)

with open(fname, 'wb') as file:
    pickle.dump([pred, mdata, params_hist, loss_hist], file)
