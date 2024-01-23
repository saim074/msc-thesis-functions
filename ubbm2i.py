# %%
import numpy as np
from numpy import tensordot as td
import numpy.linalg as la

def ubbm2i_su(t, strain, histvars, params):
    """
    Update stress and history variables using homogenous uniaxial bergstrom boyce model
    with modified evolution equation and added second invariant term. Incompressible (J = 1)
    Inputs
        t       : list of timestamps upto the current timestep
        strain  : list of strain values upto the current timestep
        histvars: list of list of history variables upto the previous timestep
        params: list of material parameters
    Output
        stress_new: updated stress
        histvars  : updated list of list of history variables upto the
                    current timestep
    """

    # Maximum iterations
    count_max = 100

    # Material parameters
    num_branches = 2
    mu = params[0]
    mu_v_list = params[1: 1 + num_branches]
    N = params[1 + num_branches]
    N_v_list = params[2+num_branches: 2 + 2 * num_branches]
    c = params[2 + 2 * num_branches]
    c_v_list = params[3 + 2 * num_branches: 3 + 3 * num_branches]
    tau_hat_list = params[3 + 3 * num_branches: 3 + 4 * num_branches]
    aj_list = []
    num_aj = int((len(params) - (3 + 4 * num_branches))/num_branches)
    for branch in range(num_branches):
        aj_list.append(params[3 + 4 * num_branches + branch * num_aj: 3 + 4 * num_branches + (branch + 1) * num_aj])
    
    # Initialize history variables if needed
    if len(histvars) == 0:
        b11_init = 1
        histvars.append([b11_init]*num_branches)

    # History variables from previous timestep
    b11_prev_list = histvars[-1]

    #Fourth order identity tensor, I kron I, Projection Tensor
    I = np.eye(3)
    IxI = td(I, I, 0)
    II = I.reshape(3, 1, 3, 1)*I.reshape(1, 3, 1, 3)
    PP = II - (1/3)*IxI

    # Current deformation gradient
    F = np.eye(3)
    lambda1 = np.exp(strain[-1])
    F[0, 0] = lambda1
    F[1, 1] = 1/np.sqrt(lambda1)
    F[2, 2] = 1/np.sqrt(lambda1)
    dt = t[-1]-t[-2]

    # Necessary values
    Fbar = F.copy()
    bbar = Fbar@Fbar.T
    I1 = np.trace(bbar)
    I2 = 0.5*(I1**2 - np.trace(bbar@bbar))
    lambda_r = np.sqrt(I1/3/N)
    lang = (3-lambda_r**2)/(1-lambda_r**2)

    ## Unimodular elastic kirchoff stress
    taubar_e = (mu*lang/3)*bbar + 2*c*(I2*bbar - bbar@bbar)

    ## Unimodular viscous kirchoff stress (Newton update, each branch)
    F_prev = np.eye(3)
    F_prev[0, 0] = np.exp(strain[-2])
    F_prev[1, 1] = 1/np.sqrt(F_prev[0, 0])
    F_prev[2, 2] = 1/np.sqrt(F_prev[0, 0])
    Fbar_prev = F_prev.copy()

    b11_new_list = [0]*num_branches
    taubar_v_list = [np.eye(3)]*num_branches
    for branch, (mu_v, N_v, c_v, tau_hat, aj, b11_prev) in \
        enumerate(zip(mu_v_list, N_v_list, c_v_list, tau_hat_list, aj_list, b11_prev_list)):

        be_prev = np.eye(3)
        be_prev[0, 0] = b11_prev
        be_prev[1, 1] = 1/np.sqrt(b11_prev)
        be_prev[2, 2] = 1/np.sqrt(b11_prev)
        be_tr = Fbar@la.inv(Fbar_prev)@be_prev@la.inv(Fbar_prev).T@Fbar.T
        lambda_a_e_tr_sq, n_a = la.eig(be_tr)
        lambda_a_e_tr = np.sqrt(lambda_a_e_tr_sq).reshape(-1, 1)
        eps_a_tr = np.log(lambda_a_e_tr)

        # Initial values of the elastic logarithmic stretches
        eps_a = eps_a_tr
        res = np.ones(3)
        count = 0

        # Newton loop for unknown elastic logarithmic stretches
        while (abs(res) > 1.e-4).any() and (count < count_max):

            # Necessary values
            be = n_a@np.diag(np.exp(eps_a[:, 0])**2)@la.inv(n_a)
            I1_e = np.trace(be)
            I2_e = 0.5*(I1_e**2 - np.trace(be@be))
            lambda_r_e = np.sqrt(I1_e/3/N_v)
            lang_e = (3-lambda_r_e**2)/(1-lambda_r_e**2)
            lambda_e_a_sq, _ = la.eig(be)
            lambda_e_a = np.sqrt(lambda_e_a_sq).reshape(-1, 1)

            # Unimodular viscous kirchoff stress
            taubar_v = (mu_v*lang_e/3)*be + 2*c_v*(I2_e*be - be@be)
            tau_v_iso = td(PP, taubar_v, 2)
            tau_v = la.norm(tau_v_iso)/np.sqrt(2)
            devtau_a = la.eig(tau_v_iso)[0].reshape(-1, 1)

            # Effective creep rate
            gamma_dot = sum([aj[j-1]*(tau_v/tau_hat)**j for j in range(1, len(aj)+1)])

            res = eps_a + dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v - eps_a_tr

            # Local tangent
            beta1 = (dt/2/np.sqrt(2))*(1/tau_hat**3)*sum([aj[j-1]*(j-1)*(tau_v/tau_hat)**(j-3) for
                                                            j in range(2, len(aj)+1)])
            beta2 = dt*gamma_dot/np.sqrt(2)/tau_v
            T = (2/3)*mu_v*((3-lambda_r_e**2)/(1-lambda_r_e**2))*np.diag((lambda_e_a**2).reshape(3)) - \
                (4/9)*(mu_v/N_v)*(1/(1-lambda_r_e**2))*(lambda_e_a**2)*(lambda_e_a**2).reshape(3) + \
                4*c_v*I1_e*(lambda_e_a**2)*(lambda_e_a**2).reshape(3) - \
                4*c_v*(lambda_e_a**2)*(lambda_e_a**4).reshape(3) + \
                4*c_v*I2_e*np.diag((lambda_e_a**2).reshape(3)) - \
                8*c_v*np.diag((lambda_e_a**4).reshape(3))
            Tbar = T - (1/3)*np.sum(T, 0)
            D = (devtau_a.reshape(1, 3)@T).reshape(3, 1)
            K = I + beta1*devtau_a*D.reshape(3) + beta2*Tbar

            # Update
            K_inv = la.inv(K)
            eps_a = eps_a - K_inv@res

            # Update count
            count += 1

            # Store the branch values
            b11_new_list[branch] = be[0, 0]
            taubar_v_list[branch] = taubar_v

    ## Isochoric kirchoff stress and pressure
    taubar = taubar_e + sum(taubar_v_list)
    tau_iso = td(PP, taubar, 2)
    p = -tau_iso[1, 1]

    ## 1.5. Total kirchoff stress
    tau = p*I + tau_iso

    # Stress and history variables
    stress_new = tau[0, 0]
    histvars.append(b11_new_list)

    return stress_new, histvars