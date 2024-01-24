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
    num_branches = 1
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
    lambda_r = np.sqrt(I1/3/N)
    lang = (3-lambda_r**2)/(1-lambda_r**2)

    ## Unimodular elastic kirchoff stress
    taubar_e = (mu*lang/3)*bbar + 2*c*(I1*bbar - bbar@bbar)

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
            lambda_r_e = np.sqrt(I1_e/3/N_v)
            lang_e = (3-lambda_r_e**2)/(1-lambda_r_e**2)
            lambda_e_a_sq, _ = la.eig(be)
            lambda_e_a = np.sqrt(lambda_e_a_sq).reshape(-1, 1)

            # Unimodular viscous kirchoff stress
            taubar_v = (mu_v*lang_e/3)*be + 2*c_v*(I1_e*be - be@be)
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
                4*c_v*(lambda_e_a**2)*(lambda_e_a**2).reshape(3) + \
                4*c_v*I1_e*np.diag((lambda_e_a**2).reshape(3)) - \
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

def ubbmi_du(t, strain, histvars, stress_new, drecurr, params):
    """
    Update derivatives using homogenous uniaxial bergstrom boyce model
    with modified evolution equation and added second invariant term. Incompressible (J = 1)
    Inputs
        t         : list of timestamps upto the current timestep
        strain    : list of strain values upto the current timestep
        histvars  : list of list of history variables upto the current timestep
        stress_new: updated stress at current timestep
        drecurr   : list of list of recurrent derivatives upto the previous timestep
        params    : list of material parameters
    Output
        dsig   : list of derivatives of updated streess wrt each material parameter
        drecurr: updated list of list of recurrent derivatives upto the
                 current timestep
    """

    # Material parameters
    num_branches = 1
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

    # History variables from previous timestep
    b11_prev_list = histvars[-2]

    # History variables from current timestep
    b11_list = histvars[-1]

    # Initialize recurrent_derivatives if needed
    if len(drecurr) == 0:
        # All the viscous parameters with their respective b11s
        db11_init = [0 for j in range(len(params) - 3)]
        drecurr.append(db11_init)

    # Recurrent derivatives from previous timestep
    db11prev = drecurr[-1]

    # Fourth order identity tensor, I kron I, Projection Tensor
    I = np.eye(3)
    IxI = td(I, I, 0)
    II = I.reshape(3, 1, 3, 1)*I.reshape(1, 3, 1, 3)
    PP = II - (1/3)*IxI

    # Current deformation gradient
    F = np.eye(3)
    lambda1 = np.exp(strain[-1])
    F[0, 0] = lambda1
    lambda2 = 1/np.sqrt(lambda1)
    F[1, 1] = lambda2
    F[2, 2] = lambda2
    dt = t[-1]-t[-2]

    # Necessary values
    Fbar = F.copy()
    bbar = Fbar@Fbar.T
    I1 = np.trace(bbar)
    lambda_r = np.sqrt(I1/3/N)
    lang = (3-lambda_r**2)/(1-lambda_r**2)
    lambda1_prev = np.exp(strain[-2])
    lambda2_prev = 1/np.sqrt(lambda1_prev)

    ## 1. Elastic parameters
    # 1.1. dstress_mu
    dtaueiso_mu = td(PP,
                     (1/3)*lang*bbar,
                     2)
    df_mu = dtaueiso_mu[0, 0] - dtaueiso_mu[1, 1]
    dstress_mu = df_mu
    # 1.2. dstress_N
    dlambdar_N = -(1/(2*N))*np.sqrt(I1/(3*N))
    dlang_lambdar = 4*lambda_r/(lambda_r**2 - 1)**2
    dtaueiso_N = td(PP, (mu/3)*dlang_lambdar*dlambdar_N*bbar, 2)
    df_N = dtaueiso_N[0, 0] - dtaueiso_N[1, 1]
    dstress_N = df_N
    # 1.3. dstress_c
    dtaueiso_c = td(PP, 2*(I1*bbar - bbar@bbar), 2)
    df_c = dtaueiso_c[0, 0] - dtaueiso_c[1, 1]
    dstress_c = df_c

    # FOR EACH BRANCH
    dstress_muv_list = []
    dstress_Nv_list = []
    dstress_cv_list = []
    dstress_tauhat_list = []
    dstress_aj_list = []
    db11 = []
    for branch, (mu_v, N_v, c_v, tau_hat, aj, b11, b11_prev) in \
        enumerate(zip(mu_v_list, N_v_list, c_v_list, tau_hat_list, aj_list, b11_list, b11_prev_list)):

        # Necessary value for the branch
        be = np.array(
            [[b11, 0, 0],
             [0, 1/np.sqrt(b11), 0],
             [0, 0, 1/np.sqrt(b11)]]
        )
        I1_e = np.trace(be)
        lambda_r_e = np.sqrt(I1_e/3/N_v)
        lange = (3-lambda_r_e**2)/(1-lambda_r_e**2)

        tau_v_iso = td(PP, (mu_v/3)*lange*be + 2*c_v*(I1_e*bbar - bbar@bbar), 2)
        tau_v = la.norm(tau_v_iso)/np.sqrt(2)
        N11 = tau_v_iso[0, 0]/(tau_v*np.sqrt(2))
        gammadot = sum([aj[j-1]*(tau_v/tau_hat)**j for j in range(1, len(aj)+1)])
        expo = np.exp(-2*gammadot*N11*dt)

        ## 2. Matrix A

        # 2.1. df_b11
        dlange_lambdare = 4*lambda_r_e/(lambda_r_e**2 - 1)**2
        dlambdare_I1e = 1/(2*np.sqrt(3*N_v*I1_e))
        dI1e_b11 = 1 - 1/(b11**(3/2))
        dlambdare_b11 = dlambdare_I1e*dI1e_b11
        dbe_b11 = np.array(
            [[1, 0, 0],
             [0, -1/(2*b11**(3/2)), 0],
             [0, 0, -1/(2*b11**(3/2))]]
        )
        dbesq_b11 = np.array(
            [[2*b11, 0, 0],
             [0, -1/b11**2, 0],
             [0, 0, -1/b11**2]]
        )
        dtauviso_b11 = td(PP,
                          ((mu_v/3)*dlange_lambdare*dlambdare_b11*be + (mu_v/3)*lange*dbe_b11 + \
                           2*c_v*I1_e*dbe_b11 + 2*c_v*be*dI1e_b11 - 2*c_v*dbesq_b11),
                          2)
        df_b11 = dtauviso_b11[0, 0] - dtauviso_b11[1, 1]

        # 2.2. dg_b11
        dtauv_tauviso = tau_v_iso/(2*tau_v)
        dtauv_b11 = td(dtauv_tauviso, dtauviso_b11, 2)
        dgammadot_b11 = (1/tau_hat)*sum([j*aj[j-1]*(tau_v/tau_hat)**(j-1) for j in range(1, len(aj)+1)])*dtauv_b11
        dexpo_N11 = -2*gammadot*dt*expo
        dN11_b11 = \
        (la.norm(tau_v_iso)*dtauviso_b11[0, 0] - (np.sqrt(2)*dtauv_b11)*(tau_v_iso[0, 0]))/(la.norm(tau_v_iso)**2)
        dexpo_gammadot = -2*N11*dt*expo
        dexpo_b11 = dexpo_gammadot*dgammadot_b11 + dexpo_N11*dN11_b11
        betr11 = ((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)*b11_prev
        dbe11_b11 = dexpo_b11*betr11
        dg_b11 = dbe11_b11
    
        # 2.3. Construct the matrix A and A_inv
        A = np.array(
            [[1, -df_b11],
             [0, 1-dg_b11]]
        )
        A_inv = la.inv(A)

        # Number of branch params
        num_branch_params = 4 + len(aj)

        ## 3. mu_v

        # 3.1. b_muv
        dbe11_b11prev = expo*((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)
        db11prev_muv = db11prev[num_branch_params*branch]
        dtauviso_muv = td(PP,
                          (1/3)*lange*be,
                          2)
        dgammadot_tauviso = (1/tau_hat)*sum([j*aj[j-1]*(tau_v/tau_hat)**(j-1) for
                                             j in range(1, len(aj)+1)])*dtauv_tauviso
        dtauviso11_tauviso = np.zeros((3, 3))
        dtauviso11_tauviso[0, 0] = 1
        dN11_tauviso = \
         (la.norm(tau_v_iso)*dtauviso11_tauviso - tau_v_iso[0, 0]*(np.sqrt(2)*dtauv_tauviso))/(la.norm(tau_v_iso)**2)
        dexpo_tauviso = dexpo_gammadot*dgammadot_tauviso + dexpo_N11*dN11_tauviso
        dbe11_tauviso = dexpo_tauviso*betr11
        dbe11_muv = td(dbe11_tauviso, dtauviso_muv, 2)
        df_muv = dtauviso_muv[0, 0] - dtauviso_muv[1, 1]
        dg_b11prev = dbe11_b11prev
        dg_muv = dbe11_muv
        b_muv = np.array(
            [[df_muv],
             [dg_muv + dg_b11prev*db11prev_muv]]
        )

        # 3.2. solve for d_muv
        d_muv = A_inv@b_muv
        dstress_muv_list.append(d_muv[0, 0])
        db11.append(d_muv[1, 0])

        ## 4. N_v

        # 4.1. b_Nv
        db11prev_Nv = db11prev[num_branch_params*branch + 1]
        dlambdare_Nv = -(1/(2*N_v))*np.sqrt(I1_e/(3*N_v))
        dtauviso_Nv = td(PP, (mu_v/3)*dlange_lambdare*dlambdare_Nv*be)
        dbe11_Nv = td(dbe11_tauviso, dtauviso_Nv, 2)
        df_Nv = dtauviso_Nv[0, 0] - dtauviso_Nv[1, 1]
        dg_Nv = dbe11_Nv
        b_Nv = np.array(
            [[df_Nv],
             [dg_Nv + dg_b11prev*db11prev_Nv]]
        )

        # 4.2. solve for d_Nv
        d_Nv = A_inv@b_Nv
        dstress_Nv_list.append(d_Nv[0, 0])
        db11.append(d_Nv[1, 0])

        ## 5. c_v

        # 5.1. b_cv
        db11prev_cv = db11prev[num_branch_params*branch + 2]
        # dlambdare_Nv = -(1/(2*N_v))*np.sqrt(I1_e/(3*N_v))
        dtauviso_cv = td(PP, 2*(I1_e*be - be@be))
        dbe11_cv = td(dbe11_tauviso, dtauviso_cv, 2)
        df_cv = dtauviso_cv[0, 0] - dtauviso_cv[1, 1]
        dg_cv = dbe11_cv
        b_cv = np.array(
            [[df_cv],
             [dg_cv + dg_b11prev*db11prev_cv]]
        )

        # 5.2. solve for d_cv
        d_cv = A_inv@b_cv
        dstress_cv_list.append(d_cv[0, 0])
        db11.append(d_cv[1, 0])

        ## 6. tau_hat

        # 6.1. b_tauhat
        db11prev_tauhat = db11prev[num_branch_params*branch + 3]
        dgammadot_tauhat = -(1/tau_hat)*sum([j*aj[j-1]*(tau_v/tau_hat)**j for j in range(1, len(aj)+1)])
        dbe11_tauhat = dexpo_gammadot*dgammadot_tauhat*betr11
        df_tauhat = 0
        dg_tauhat = dbe11_tauhat
        b_tauhat = np.array(
            [[df_tauhat],
             [dg_tauhat + dg_b11prev*db11prev_tauhat]]
        )

        # 6.2. solve for d_tauhat
        d_tauhat = A_inv@b_tauhat
        dstress_tauhat_list.append(d_tauhat[0, 0])
        db11.append(d_tauhat[1, 0])

        ## 7. aj

        dstress_aj = []
        db11_aj = []
        for j in range(1, len(aj)+1):

            # 7.1. b_aj
            db11prev_aj = db11prev[num_branch_params*branch + 3 + j]
            dgammadot_aj = (tau_v/tau_hat)**j
            dbe11_aj = dexpo_gammadot*dgammadot_aj*betr11
            df_aj = 0
            dg_aj = dbe11_aj
            b_aj = np.array(
                [[df_aj],
                [dg_aj + dg_b11prev*db11prev_aj]]
            )

            # 7.2. solve for d_aj
            d_aj = A_inv@b_aj
            dstress_aj.append(d_aj[0, 0])
            db11_aj.append(d_aj[1, 0])

        dstress_aj_list.append(dstress_aj)
        db11.extend(db11_aj)

    # Stress and Recurrent derivatives
    dstress = [dstress_mu] + dstress_muv_list + [dstress_N] + dstress_Nv_list + dstress_cv_list + \
               dstress_tauhat_list + [item for row in dstress_aj_list for item in row]
    drecurr_new = db11
    drecurr.append(drecurr_new)

    return dstress, drecurr