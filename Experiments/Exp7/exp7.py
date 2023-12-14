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
# %% Get elastic parameters from Exp 6

fpath = "saved_results_elastic_2023-12-12_23-03-34.pkl"
with open(fpath, 'rb') as file:
    data = pickle.load(file)

params_hist = data[0]
loss_hist = data[1]

params_elastic = params_hist[np.argmin(loss_hist)]

# %% Viscoelastic data

data_data = [np.loadtxt('visco1_2.dat'), np.loadtxt('visco2_2.dat')]

mdata_data = []
mdata_data_sparse = []
for data in data_data:
    P11 = data[:, 0]
    stretch = data[:, 2]/100 + 1
    t = data[:, 4]

    # Change to true stress and true strain
    true_strain = np.log(stretch)
    true_stress = P11*stretch
    mdata_data.append((t, true_strain, true_stress))

    # Make it sparser
    idx = range(0, len(t), 5)
    t = t[idx]
    true_strain = true_strain[idx]
    true_stress = true_stress[idx]
    mdata_data_sparse.append((t, true_strain, true_stress))

# %% Fit viscoelastic parameters (Simple Bergstrom Boyce)

mat = 'ubbi'

# Initial values
mu_pred = params_elastic[0]
mu_v_pred = np.random.uniform(0.5, 1.5)
N_pred = params_elastic[2]
N_v_pred = 1/np.random.uniform(0.1, 0.2)
gamma0_taum_pred = np.random.uniform(0.5, 1.5)
c_pred = 0
m_pred = np.random.uniform(0.9, 1.1)

params_init = [mu_pred, mu_v_pred, N_pred, N_v_pred, gamma0_taum_pred, c_pred, m_pred]
params_names = ["mu", "mu_v", "N", "N_v", "gamma0_taum", "c", "m"]

# Optimize
params_out, params_hist, loss_hist = \
optimize(mdata_data_sparse, params_init, params_names, mat,
         max_iter=200, print_after=1, plot_after = 100, tol=10e-6,
         alpha=1e-1, b1=0.9, b2=0.999, e=1e-8, lambda_1 = 0,
         non_neg = [False, False, False, False, True, False, False],
         fix_param = [True, False, True, False, False, True, False],
         get_lowest=True)

# %% Plots

params_hist = np.array(params_hist)
lst = range(7)
for j in lst:
        plt.figure()
        plt.plot(params_hist[:, j])
        plt.title(params_names[j])
plt.figure()
plt.plot(loss_hist)
plt.title('Loss')
pred = evaluate(mdata_data, params_out, mat)

# %% Store Results

now = datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M-%S")

fname = f"saved_results_{now}.pkl"
fname = os.path.join("Results", fname)

with open(fname, 'wb') as file:
    pickle.dump({'params_out': params_out, 
                 'params_hist': params_hist, 
                 'loss_hist': loss_hist}, file)