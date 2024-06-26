# %%
import numpy as np
from numpy import tensordot as td
import numpy.linalg as la

def stress_update(t, strain, histvars, params, mat):

    """
    Update stress and history variables
    Inputs
        t       : list of timestamps upto the current timestep
        strain  : list of strain values upto the current timestep
        histvars: list of list of history variables upto the previous timestep
        params  : list of material parameters
        mat     : string identifying which material model to use
    Output
        stress_new: updated stress
        histvars  : updated list of list of history variables upto the
                    current timestep
    """
    # Uniaxial bergstrom boyce
    if mat == "ubb":
        stress_new, histvars = \
        ubb_su(t, strain, histvars, params)
    # Uniaxial bergstrom boyce (incompressible)
    elif mat == "ubbi":
        stress_new, histvars = \
        ubbi_su(t, strain, histvars, params)
    # Uniaxial bergstrom boyce modified
    elif mat == "ubbm":
        stress_new, histvars = \
        ubbm_su(t, strain, histvars, params)
    # Uniaxial bergstrom boyce modified (incompressible)
    elif mat == "ubbmi":
        stress_new, histvars = \
        ubbmi_su(t, strain, histvars, params)
    elif mat == "ubbm2i":
        stress_new, histvars = \
        ubbm2i_su(t, strain, histvars, params)
    elif mat == "umri":
        stress_new, histvars = \
        umri_su(t, strain, histvars, params)

    return stress_new, histvars

def derivative_update(t, strain, histvars, stress_new, drecurr, params, mat):
    """
    Update stress and recurrent derivatives
    Inputs
        t         : list of timestamps upto the current timestep
        strain    : list of strain values upto the current timestep
        histvars  : list of list of history variables upto the current timestep
        stress_new: updated stress at current timestep
        drecurr   : list of list of recurrent derivatives upto the previous timestep
        params    : list of material parameters
        mat       : string identifying which material model to use
    Output
        dstress: list of derivatives of updated streess wrt each material parameter
        drecurr: updated list of list of recurrent derivatives upto the
                 current timestep
    """

    # Uniaxial bergstrom boyce
    if mat == "ubb":
        dstress, drecurr = \
        ubb_du(t, strain, histvars, stress_new, drecurr, params)
    # Uniaxial bergstrom boyce (incompressible)
    elif mat == "ubbi":
        dstress, drecurr = \
        ubbi_du(t, strain, histvars, stress_new, drecurr, params)
    # Uniaxial bergstrom boyce modified
    elif mat == "ubbm":
        dstress, drecurr = \
        ubbm_du(t, strain, histvars, stress_new, drecurr, params)
    # Uniaxial bergstrom boyce modified (incompressible)
    elif mat == "ubbmi":
        dstress, drecurr = \
        ubbmi_du(t, strain, histvars, stress_new, drecurr, params)
    elif mat == "ubbm2i":
        dstress, drecurr = \
        ubbm2i_du(t, strain, histvars, stress_new, drecurr, params)
    elif mat == "umri":
        dstress, drecurr = \
        umri_du(t, strain, histvars, stress_new, drecurr, params)

    return dstress, drecurr

#region ##--UNIAXIAL BERGSTROM BOYCE (INCOMPRESSIBLE)--##

def ubbi_su(t, strain, histvars, params):
    """
    Update stress and history variables using homogenous uniaxial bergstrom boyce model
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
    mu = params[0]
    mu_v = params[1]
    N = params[2]
    N_v = params[3]
    gamma0_taum = params[4]
    c = params[5]
    m = params[6]

    # Perturbation parameter
    perturbation_param = 0.02

    # Initialize history variables if needed
    if len(histvars) == 0:
        b11_init = 1
        histvars.append([b11_init])

    # History variables from previous timestep
    b11_prev = histvars[-1][0]

    #Fourth order identity tensor, I kron I, Projection Tensor
    I = np.eye(3)
    IxI = td(I, I, 0)
    II = I.reshape(3, 1, 3, 1)*I.reshape(1, 3, 1, 3)
    PP = II - (1/3)*IxI

    # Current deformation gradient
    F = np.eye(3)
    lambda1 = np.exp(strain[-1])
    F[0, 0] = lambda1
    F[1, 1] = lambda1**(-1/2)
    F[2, 2] = lambda1**(-1/2)
    dt = t[-1]-t[-2]

    # Necessary values
    J = F[0, 0]*F[1, 1]*F[2, 2] #det
    Fbar = J**(-1/3)*F
    Cbar = Fbar.T@Fbar
    I1 = np.trace(Cbar)
    lambda_r = np.sqrt(I1/3/N)
    lang = (3-lambda_r**2)/(1-lambda_r**2)
    bbar = Fbar@Fbar.T
    C = F.T@F
    lambda_a_sq, N_a = la.eig(C) # !!!!!!!!!
    lambda_a = np.sqrt(lambda_a_sq).reshape(-1, 1)

    # Unimodular elastic kirchoff stress
    taubar_e = (mu*lang/3)*bbar

    # Unimodular viscous kirchoff stress (Newton Update)
    F_prev = np.eye(3)
    F_prev[0, 0] = np.exp(strain[-2])
    F_prev[1, 1] = F_prev[0, 0]**(-1/2)
    F_prev[2, 2] = F_prev[0, 0]**(-1/2)
    be_prev = np.eye(3)
    be_prev[0, 0] = b11_prev
    be_prev[1, 1] = 1/np.sqrt(b11_prev)
    be_prev[2, 2] = 1/np.sqrt(b11_prev)
    J_prev = F_prev[0, 0]*F_prev[1, 1]*F_prev[2, 2] #det
    Fbar_prev = J_prev**(-1/3)*F_prev
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
        Ci = Fbar.T@la.inv(be)@Fbar
        I1_i = np.trace(Ci)
        lambda_chain_i = np.sqrt(I1_i/3)
        I1_e = np.trace(be)
        lambda_r_e = np.sqrt(I1_e/3/N_v)
        lang_e = (3-lambda_r_e**2)/(1-lambda_r_e**2)

        # Unimodular viscous kirchoff stress
        taubar_v = (mu_v*lang_e/3)*be
        tau_v_iso = td(PP, taubar_v, 2)
        tau_v = la.norm(tau_v_iso)/np.sqrt(2)
        devtau_a = la.eig(tau_v_iso)[0].reshape(-1, 1)

        # Effective creep rate
        gamma_dot = gamma0_taum*(lambda_chain_i-1+perturbation_param)**c*(tau_v)**m

        # Residual
        # if tau_v < 1e-8:
        #     tau_v = 1e-8
        res = eps_a + dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v - eps_a_tr

        # Local tangent
        beta1 = dt*(m-1)*gamma0_taum*(lambda_chain_i**c-1+perturbation_param)*(tau_v)**(m-3)/2/np.sqrt(2)
        beta2 = dt*gamma_dot/np.sqrt(2)/tau_v
        T = (2/3)*mu_v*((3-lambda_r_e**2)/(1-lambda_r_e**2))*np.diag((lambda_a**2).reshape(3)) - \
        (4/9)*(mu_v/N_v)*(1/(1-lambda_r_e**2))*(lambda_a**2)*(lambda_a**2).reshape(3)
        Tbar = T - (1/3)*np.sum(T, 0)
        D = (devtau_a.reshape(1, 3)@T).reshape(3, 1)
        K = I + beta1*devtau_a*D.reshape(3) + beta2*Tbar

        # Update
        K_inv = la.inv(K)
        eps_a = eps_a - K_inv@res

        # Update count
        count += 1

    # Isochoric kirchoff stress and pressure
    tau_iso = taubar_e + taubar_v
    p = -tau_iso[1, 1]

    # Total Kirchoff stress
    tau = p*I + tau_iso

    # Stress and history variables
    stress_new = tau[0, 0]
    b11_new = be[0, 0]
    histvars_new = [b11_new]
    histvars.append(histvars_new)

    return stress_new, histvars

def ubbi_du(t, strain, histvars, stress_new, drecurr, params):
    """
    Update derivatives using homogenous uniaxial bergstrom boyce model
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
    mu = params[0]
    mu_v = params[1]
    N = params[2]
    N_v = params[3]
    gamma0_taum = params[4]
    c = params[5]
    m = params[6]

    # Perturbation parameter
    perturbation_param = 0.02

    # History variables from previous timestep
    b11_prev = histvars[-2][0]

    # History variables from current timestep
    b11 = histvars[-1][0]

    # Initialize recurrent_derivatives if needed
    if len(drecurr) == 0:
        db11_init = [0 for j in range(len(params))]
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
    J = F[0, 0]*F[1, 1]*F[2, 2] #det
    Fbar = J**(-1/3)*F
    Cbar = Fbar.T@Fbar
    I1 = np.trace(Cbar)
    lambda_r = np.sqrt(I1/3/N)
    lang = (3-lambda_r**2)/(1-lambda_r**2)
    bbar = Fbar@Fbar.T

    be = np.array(
        [[b11, 0, 0],
         [0, 1/np.sqrt(b11), 0],
         [0, 0, 1/np.sqrt(b11)]]
    )
    I1_e = np.trace(be)
    lambda_r_e = np.sqrt(I1_e/3/N_v)
    lange = (3-lambda_r_e**2)/(1-lambda_r_e**2)

    Ci = Fbar.T@la.inv(be)@Fbar
    I1_i = np.trace(Ci)
    lambda_chain_i = np.sqrt(I1_i/3)
    tau_v_iso = td(PP, (mu_v/3)*lange*be, 2)
    tau_v = la.norm(tau_v_iso)/np.sqrt(2)
    # if tau_v < 1e-8:
    #     tau_v = 1e-8
    N11 = tau_v_iso[0, 0]/(tau_v*np.sqrt(2))
    gammadot = gamma0_taum*(lambda_chain_i-1+perturbation_param)**c*(tau_v)**m
    expo = np.exp(-2*gammadot*N11*dt)
    lambda1_prev = np.exp(strain[-2])
    lambda2_prev = 1/np.sqrt(lambda1_prev)


    ### 1. Matrix A

    ## 1.1. df_b11
    dlange_lambdare = 4*lambda_r_e/(lambda_r_e**2 - 1)**2
    dlambdare_I1e = 1/(2*np.sqrt(3*N_v*I1_e))
    dI1e_b11 = 1 - 1/(b11**(3/2))
    dlambdare_b11 = dlambdare_I1e*dI1e_b11
    dbe_b11 = np.array(
        [[1, 0, 0],
         [0, -1/(2*b11**(3/2)), 0],
         [0, 0, -1/(2*b11**(3/2))]]
    )
    dtauviso_b11 = td(PP,
                      ((mu_v/3)*dlange_lambdare*dlambdare_b11*be + (mu_v/3)*lange*dbe_b11),
                      2)
    df_b11 = dtauviso_b11[0, 0] - dtauviso_b11[1, 1]

    ## 1.2. dh_b11
    dI1i_b11 = \
    -(lambda1**(4/3)*lambda2**(-4/3))/(b11**2) + (lambda1**(-2/3)*lambda2**(2/3))/(np.sqrt(b11))
    dlambdaichain_I1i = 1/(2*np.sqrt(3*I1_i))
    dlambdaichain_b11 = dlambdaichain_I1i*dI1i_b11
    dtauv_tauviso = tau_v_iso/(2*tau_v)
    dtauv_b11 = td(dtauv_tauviso, dtauviso_b11, 2)
    dgammadot_b11 = \
    gamma0_taum*c*(lambda_chain_i-1+perturbation_param)**(c-1)*(tau_v)**m*dlambdaichain_b11 + \
    gamma0_taum*m*(lambda_chain_i-1+perturbation_param)**c*(tau_v)**(m-1)*dtauv_b11
    dexpo_N11 = -2*gammadot*dt*expo
    dN11_b11 = \
    (la.norm(tau_v_iso)*dtauviso_b11[0, 0] - (np.sqrt(2)*dtauv_b11)*(tau_v_iso[0, 0]))/(2*tau_v**2)
    dexpo_gammadot = -2*N11*dt*expo
    dexpo_b11 = dexpo_gammadot*dgammadot_b11 + dexpo_N11*dN11_b11
    betr11 = ((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)*b11_prev
    dbe11_b11 = dexpo_b11*betr11
    dh_b11 = dbe11_b11

    ## 1.3. Construct the matrix A and A_inv
    A = np.array(
        [[1, -df_b11],
         [0, 1-dh_b11]]
    )
    A_inv = la.inv(A)

    ### 2. mu

    ## 2.1. b_mu
    db11prev_mu = db11prev[0]
    dtaueiso_mu = td(PP,
                     (1/3)*lang*bbar,
                     2)
    dbe11_b11prev = expo*((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)
    dg_b11prev = dbe11_b11prev
    df_mu = dtaueiso_mu[0, 0] - dtaueiso_mu[1, 1]
    dg_mu = 0
    b_mu = np.array(
        [[df_mu],
         [dg_mu + dg_b11prev*db11prev_mu]]
    )

    ## 2.2. solve for d_mu
    d_mu = A_inv@b_mu
    dstress_mu = d_mu[0, 0]
    db11_mu = d_mu[1, 0]

    ### 3. mu_v

    ## 3.1. b_muv
    db11prev_muv = db11prev[1]
    dtauviso_muv = td(PP,
                      (1/3)*lange*be,
                      2)
    dgammadot_tauviso = gamma0_taum*m*(lambda_chain_i-1+perturbation_param)**c*tau_v**(m-1)*dtauv_tauviso
    dtauviso11_tauviso = np.zeros((3, 3))
    dtauviso11_tauviso[0, 0] = 1
    dN11_tauviso = \
     (la.norm(tau_v_iso)*dtauviso11_tauviso - tau_v_iso[0, 0]*(np.sqrt(2)*dtauv_tauviso))/(2*tau_v**2)
    dexpo_tauviso = dexpo_gammadot*dgammadot_tauviso + dexpo_N11*dN11_tauviso
    dbe11_tauviso = dexpo_tauviso*betr11
    dbe11_muv = td(dbe11_tauviso, dtauviso_muv, 2)
    df_muv = dtauviso_muv[0, 0] - dtauviso_muv[1, 1]
    dg_muv = dbe11_muv
    b_muv = np.array(
        [[df_muv],
         [dg_muv + dg_b11prev*db11prev_muv]]
    )

    ## 3.2. solve for d_muv
    d_muv = A_inv@b_muv
    dstress_muv = d_muv[0, 0]
    db11_muv = d_muv[1, 0]

    ### 4. N

    ## 4.1. b_N
    db11prev_N = db11prev[2]
    dlambdar_N = -(1/(2*N))*np.sqrt(I1/(3*N))
    dlang_lambdar = 4*lambda_r/(lambda_r**2 - 1)**2
    dtaueiso_N = td(PP, (mu/3)*dlang_lambdar*dlambdar_N*bbar, 2)
    df_N = dtaueiso_N[0, 0] - dtaueiso_N[1, 1]
    # dg_N = dtaueiso_N[1, 1]
    dg_N = 0
    b_N = np.array(
        [[df_N],
         [dg_N + dg_b11prev*db11prev_N]]
    )

    ## 4.2. solve for d_N
    d_N = A_inv@b_N
    dstress_N = d_N[0, 0]
    db11_N = d_N[1, 0]

    ### 5. N_v

    ## 5.1. b_Nv
    db11prev_Nv = db11prev[3]
    dlambdare_Nv = -(1/(2*N_v))*np.sqrt(I1_e/(3*N_v))
    dtauviso_Nv = td(PP, (mu_v/3)*dlange_lambdare*dlambdare_Nv*be)
    dbe11_Nv = td(dbe11_tauviso, dtauviso_Nv, 2)
    df_Nv = dtauviso_Nv[0, 0] - dtauviso_Nv[1, 1]
    dg_Nv = dbe11_Nv
    b_Nv = np.array(
        [[df_Nv],
         [dg_Nv + dg_b11prev*db11prev_Nv]]
    )

    ## 5.2. solve for d_Nv
    d_Nv = A_inv@b_Nv
    dstress_Nv = d_Nv[0, 0]
    db11_Nv = d_Nv[1, 0]

    ### 6. gamma0_taum

    ## 6.1. b_gamma0taum
    db11prev_gamma0taum = db11prev[4]
    dgammadot_gamma0taum = (lambda_chain_i-1+perturbation_param)**c*tau_v**m
    dbe11_gamma0taum = dexpo_gammadot*dgammadot_gamma0taum*betr11
    df_gamma0taum = 0
    dg_gamma0taum = dbe11_gamma0taum
    b_gamma0taum = np.array(
        [[df_gamma0taum],
         [dg_gamma0taum + dg_b11prev*db11prev_gamma0taum]]
    )

    ## 6.2. solve for d_gamma0taum
    d_gamma0taum = A_inv@b_gamma0taum
    dstress_gamma0taum = d_gamma0taum[0, 0]
    db11_gamma0taum = d_gamma0taum[1, 0]

    ### 7. c

    ## 7.1. b_c
    db11prev_c = db11prev[5]
    dgammadot_c = \
    gamma0_taum*(lambda_chain_i-1+perturbation_param)**c*np.log(lambda_chain_i-1+perturbation_param)*tau_v**m
    dbe11_c = dexpo_gammadot*dgammadot_c*betr11
    df_c = 0
    dg_c = dbe11_c
    b_c = np.array(
        [[df_c],
         [dg_c + dg_b11prev*db11prev_c]]
    )

    ## 7.2. solve for d_c
    d_c = A_inv@b_c
    dstress_c = d_c[0, 0]
    db11_c = d_c[1, 0]

    ### 8. m

    ## 8.1. b_m
    db11prev_m = db11prev[6]
    dgammadot_m = \
    gamma0_taum*(lambda_chain_i-1+perturbation_param)**c*tau_v**m*np.log(tau_v)
    dbe11_m = dexpo_gammadot*dgammadot_m*betr11
    df_m = 0
    dg_m = dbe11_m
    b_m = np.array(
        [[df_m],
         [dg_m + dg_b11prev*db11prev_m]]
    )

    ## 8.2. solve for d_m
    d_m = A_inv@b_m
    dstress_m = d_m[0, 0]
    db11_m = d_m[1, 0]

    # Stress and Recurrent derivatives
    dstress = [dstress_mu, dstress_muv, dstress_N, dstress_Nv,
               dstress_gamma0taum, dstress_c, dstress_m]
    db11 = [db11_mu, db11_muv, db11_N, db11_Nv,
            db11_gamma0taum, db11_c, db11_m]
    drecurr_new = db11
    drecurr.append(drecurr_new)

    return dstress, drecurr

#endregion

#region ##--UNIAXIAL MOONEY-RIVLIN INELASTIC (INCOMPRESSIBLE)--##

def umri_su(t, strain, histvars, params):
    """
    Update stress and history variables using homogenous uniaxial mooney-rivlin + viscoelastic model
    Incompressible (J = 1)
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
    num_branches = 5
    c1 = params[0]
    c1_v_list = params[1: 1 + num_branches]
    c2 = params[1 + num_branches]
    c2_v_list = params[2+num_branches: 2 + 2 * num_branches]
    tauhinv_list = params[2 + 2 * num_branches: 2 + 3 * num_branches]
    aj_list = []
    num_aj = int((len(params) - (2 + 3 * num_branches))/num_branches)
    for branch in range(num_branches):
        aj_list.append(params[2 + 3 * num_branches + branch * num_aj: 2 + 3 * num_branches + (branch + 1) * num_aj])
    
    # Initialize history variables if needed
    if len(histvars) == 0:
        b11_init = 1
        histvars.append([b11_init]*num_branches)

    # History variables from previous timestep
    b11_prev_list = histvars[-1]

    #Fourth order identity tensor, I kron I, Projection Tensor
    I = np.eye(3)
    PP = np.einsum('ij,kl->ikjl', I, I) - (1/3) * np.einsum('ij,kl', I, I)

    # Current deformation gradient
    lambda1 = np.exp(strain[-1])
    F = np.diag([lambda1, 1/np.sqrt(lambda1), 1/np.sqrt(lambda1)])
    dt = t[-1]-t[-2]

    # Necessary values
    Fbar = F.copy() # Incompressible
    bbar = Fbar@Fbar.T
    I1 = np.trace(bbar)

    ## Unimodular elastic kirchoff stress
    taubar_e = 2*(c1 + I1*c2)*bbar - 2*c2*bbar@bbar

    ## Unimodular viscous kirchoff stress (Newton update, each branch)
    lambda1_prev = np.exp(strain[-2])
    F_prev = np.diag([lambda1_prev, 1/np.sqrt(lambda1_prev), 1/np.sqrt(lambda1_prev)])
    Fbar_prev = F_prev.copy()

    b11_new_list = [0]*num_branches
    taubar_v_list = [np.eye(3)]*num_branches
    for branch, (c1_v, c2_v, tauhinv, aj, b11_prev) in \
        enumerate(zip(c1_v_list, c2_v_list, tauhinv_list, aj_list, b11_prev_list)):

        be_prev = np.diag([b11_prev, 1/np.sqrt(b11_prev), 1/np.sqrt(b11_prev)])
        be_tr = Fbar@la.inv(Fbar_prev)@be_prev@la.inv(Fbar_prev).T@Fbar.T
        n_a = np.eye(3)
        eps_a_tr = 0.5 * np.log(np.diag(be_tr)).reshape(-1, 1)

        # Initial values of the elastic logarithmic stretches
        eps_a = eps_a_tr
        res = np.ones(3)
        count = 0

        # Newton loop for unknown elastic logarithmic stretches
        while (abs(res) > 1.e-4).any() and (count < count_max):

            # Necessary values
            be = n_a@np.diag(np.exp(eps_a[:, 0])**2)@la.inv(n_a)
            I1_e = np.trace(be)
            lambda_e_a = np.sqrt(np.diag(be)).reshape(-1, 1)

            # Unimodular viscous kirchoff stress
            taubar_v = 2*(c1_v + I1_e*c2_v)*be - 2*c2_v*be@be
            tau_v_iso = td(PP, taubar_v, 2)
            tau_v = la.norm(tau_v_iso)/np.sqrt(2) + 1e-16
            devtau_a = np.diag(tau_v_iso).reshape(-1, 1)

            # Effective creep rate
            gamma_dot = sum([aj[j-1]*(tau_v*tauhinv)**j for j in range(1, len(aj)+1)])

            res = eps_a + dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v - eps_a_tr

            # Local tangent
            tauhinv += 1e-16
            beta1 = (dt/2/np.sqrt(2))*(tauhinv**3)*sum([aj[j-1]*(j-1)*(tau_v*tauhinv)**(j-3) for
                                                            j in range(2, len(aj)+1)])
            beta2 = dt*gamma_dot/np.sqrt(2)/tau_v
            T = 4*c2_v*(lambda_e_a**2)*(lambda_e_a**2).reshape(3) + \
                4*(c1_v + I1_e*c2_v)*np.diag((lambda_e_a**2).reshape(3)) - \
                8*c2_v*np.diag((lambda_e_a**4).reshape(3))
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

        if count == count_max:
            print('Iteration overflow')

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

def umri_du(t, strain, histvars, stress_new, drecurr, params):
    """
    Update stress and history variables using homogenous uniaxial mooney-rivlin + viscoelastic model
    Incompressible (J = 1)
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
    num_branches = 5
    c1 = params[0]
    c1_v_list = params[1: 1 + num_branches]
    c2 = params[1 + num_branches]
    c2_v_list = params[2+num_branches: 2 + 2 * num_branches]
    tauhinv_list = params[2 + 2 * num_branches: 2 + 3 * num_branches]
    aj_list = []
    num_aj = int((len(params) - (2 + 3 * num_branches))/num_branches)
    for branch in range(num_branches):
        aj_list.append(params[2 + 3 * num_branches + branch * num_aj: 2 + 3 * num_branches + (branch + 1) * num_aj])

    # History variables from previous timestep
    b11_prev_list = histvars[-2]

    # History variables from current timestep
    b11_list = histvars[-1]

    # Initialize recurrent_derivatives if needed
    if len(drecurr) == 0:
        # All the viscous parameters with their respective b11s
        db11_init = [0 for j in range(len(params) - 2)]
        drecurr.append(db11_init)

    # Recurrent derivatives from previous timestep
    db11prev = drecurr[-1]

    # Fourth order identity tensor, I kron I, Projection Tensor
    I = np.eye(3)
    PP = np.einsum('ij,kl->ikjl', I, I) - (1/3) * np.einsum('ij,kl', I, I)

    # Current deformation gradient
    lambda1 = np.exp(strain[-1])
    lambda2 = 1/np.sqrt(lambda1)
    F = np.diag([lambda1, lambda2, lambda2])
    dt = t[-1]-t[-2]

    # Necessary values
    Fbar = F.copy()
    bbar = Fbar@Fbar.T
    I1 = np.trace(bbar)
    lambda1_prev = np.exp(strain[-2])
    lambda2_prev = 1/np.sqrt(lambda1_prev)

    ## 1. Elastic parameters
    # 1.1. dstress_c1
    dtaueiso_c1 = td(PP,
                     2*bbar,
                     2)
    df_c1 = dtaueiso_c1[0, 0] - dtaueiso_c1[1, 1]
    dstress_c1 = df_c1
    # 1.2. dstress_c2
    dtaueiso_c2 = td(PP,
                     2*I1*bbar - 2*bbar@bbar,
                     2)
    df_c2 = dtaueiso_c2[0, 0] - dtaueiso_c2[1, 1]
    dstress_c2 = df_c2

    # FOR EACH BRANCH
    dstress_c1v_list = []
    dstress_c2v_list = []
    dstress_tauhinv_list = []
    dstress_aj_list = []
    db11 = []
    for branch, (c1_v, c2_v, tauhinv, aj, b11, b11_prev) in \
        enumerate(zip(c1_v_list, c2_v_list, tauhinv_list, aj_list, b11_list, b11_prev_list)):

        # Necessary value for the branch
        be = np.array(
            [[b11, 0, 0],
             [0, 1/np.sqrt(b11), 0],
             [0, 0, 1/np.sqrt(b11)]]
        )
        I1_e = np.trace(be)

        tau_v_iso = td(PP, 2*(c1_v + I1_e*c2_v)*be - 2*c2_v*be@be, 2)
        tau_v = la.norm(tau_v_iso)/np.sqrt(2)
        N11 = tau_v_iso[0, 0]/(tau_v*np.sqrt(2))
        gammadot = sum([aj[j-1]*(tau_v*tauhinv)**j for j in range(1, len(aj)+1)])
        expo = np.exp(-2*gammadot*N11*dt)

        ## 2. Matrix A

        # 2.1. df_b11
        dI1e_b11 = 1 - 1/(b11**(3/2))
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
                          2*(c1_v + I1_e*c2_v)*dbe_b11 + 2*c2_v*be*dI1e_b11 - 2*c2_v*dbesq_b11,
                          2)
        df_b11 = dtauviso_b11[0, 0] - dtauviso_b11[1, 1]

        # 2.2. dg_b11
        dtauv_tauviso = tau_v_iso/(2*tau_v)
        dtauv_b11 = td(dtauv_tauviso, dtauviso_b11, 2)
        dgammadot_b11 = (tauhinv)*sum([j*aj[j-1]*(tau_v*tauhinv)**(j-1) for j in range(1, len(aj)+1)])*dtauv_b11
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
        num_branch_params = 3 + len(aj)

        ## 3. c1_v

        # 3.1. c1_muv
        dbe11_b11prev = expo*((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)
        db11prev_c1v = db11prev[num_branch_params*branch]
        dtauviso_c1v = td(PP,
                          2*be,
                          2)
        dgammadot_tauviso = (tauhinv)*sum([j*aj[j-1]*(tau_v*tauhinv)**(j-1) for
                                             j in range(1, len(aj)+1)])*dtauv_tauviso
        dtauviso11_tauviso = np.zeros((3, 3))
        dtauviso11_tauviso[0, 0] = 1
        dN11_tauviso = \
            (la.norm(tau_v_iso)*dtauviso11_tauviso - tau_v_iso[0, 0]*(np.sqrt(2)*dtauv_tauviso))/(la.norm(tau_v_iso)**2)
        dexpo_tauviso = dexpo_gammadot*dgammadot_tauviso + dexpo_N11*dN11_tauviso
        dbe11_tauviso = dexpo_tauviso*betr11
        dbe11_c1v = td(dbe11_tauviso, dtauviso_c1v, 2)
        df_c1v = dtauviso_c1v[0, 0] - dtauviso_c1v[1, 1]
        dg_b11prev = dbe11_b11prev
        dg_c1v = dbe11_c1v
        b_c1v = np.array(
            [[df_c1v],
             [dg_c1v + dg_b11prev*db11prev_c1v]]
        )

        # 3.2. solve for d_c1v
        d_c1v = A_inv@b_c1v
        dstress_c1v_list.append(d_c1v[0, 0])
        db11.append(d_c1v[1, 0])

        ## 4. c2_v

        # 4.1. b_c2v
        db11prev_c2v = db11prev[num_branch_params*branch + 1]
        dtauviso_c2v = td(PP, 2*I1_e*be - 2*be@be)
        dbe11_c2v = td(dbe11_tauviso, dtauviso_c2v, 2)
        df_c2v = dtauviso_c2v[0, 0] - dtauviso_c2v[1, 1]
        dg_c2v = dbe11_c2v
        b_c2v = np.array(
            [[df_c2v],
             [dg_c2v + dg_b11prev*db11prev_c2v]]
        )

        # 4.2. solve for d_c2v
        d_c2v = A_inv@b_c2v
        dstress_c2v_list.append(d_c2v[0, 0])
        db11.append(d_c2v[1, 0])

        ## 5. tauhinv

        # 5.1. b_tauhinv
        db11prev_tauhinv = db11prev[num_branch_params*branch + 2]
        dgammadot_tauhinv = (1/tauhinv)*sum([j*aj[j-1]*(tauhinv*tau_v)**j for j in range(1, len(aj)+1)])
        dbe11_tauhinv = dexpo_gammadot*dgammadot_tauhinv*betr11
        df_tauhinv = 0
        dg_tauhinv = dbe11_tauhinv
        b_tauhinv = np.array(
            [[df_tauhinv],
             [dg_tauhinv + dg_b11prev*db11prev_tauhinv]]
        )

        # 5.2. solve for d_tauhinv
        d_tauhinv = A_inv@b_tauhinv
        dstress_tauhinv_list.append(d_tauhinv[0, 0])
        db11.append(d_tauhinv[1, 0])

        ## 6. aj

        dstress_aj = []
        db11_aj = []
        for j in range(1, len(aj)+1):

            # 6.1. b_aj
            db11prev_aj = db11prev[num_branch_params*branch + 2 + j]
            dgammadot_aj = (tau_v*tauhinv)**j
            dbe11_aj = dexpo_gammadot*dgammadot_aj*betr11
            df_aj = 0
            dg_aj = dbe11_aj
            b_aj = np.array(
                [[df_aj],
                [dg_aj + dg_b11prev*db11prev_aj]]
            )

            # 6.2. solve for d_aj
            d_aj = A_inv@b_aj
            dstress_aj.append(d_aj[0, 0])
            db11_aj.append(d_aj[1, 0])

        dstress_aj_list.append(dstress_aj)
        db11.extend(db11_aj)

    # Stress and Recurrent derivatives
    dstress = [dstress_c1] + dstress_c1v_list + [dstress_c2] + dstress_c2v_list + dstress_tauhinv_list + \
        [item for row in dstress_aj_list for item in row]
    drecurr_new = db11
    drecurr.append(drecurr_new)

    return dstress, drecurr

#endregion

#region ##-- UNIAXIAL BERGSTROM BOYCE --##

def ubb_su(t, strain, histvars, params):
    """
    Update stress and history variables using homogenous uniaxial bergstrom boyce model
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
    count_a_max = 100

    # Material parameters
    kappa = params[0]
    mu = params[1]
    mu_v = params[2]
    N = params[3]
    N_v = params[4]
    gamma0_taum = params[5]
    c = params[6]
    m = params[7]

    # Perturbation parameter
    perturbation_param = 0.02

    # Initialize history variables if needed
    if len(histvars) == 0:
        lambda2_init = 1
        b11_init = 1
        histvars.append([lambda2_init, b11_init])

    # History variables from previous timestep
    lambda2_prev = histvars[-1][0]
    b11_prev = histvars[-1][1]

    #Fourth order identity tensor, I kron I, Projection Tensor
    I = np.eye(3)
    IxI = td(I, I, 0)
    II = I.reshape(3, 1, 3, 1)*I.reshape(1, 3, 1, 3)
    PP = II - (1/3)*IxI

    # Current deformation gradient
    F = np.eye(3)
    lambda1 = np.exp(strain[-1])
    F[0, 0] = lambda1
    F[1, 1] = lambda2_prev
    F[2, 2] = lambda2_prev
    dt = t[-1]-t[-2]

    # Residual and count initialisation
    res = np.ones([2, 1])
    count = 0

    # Newton update loop for unknown stretches
    while (abs(res) > 1.e-4).any() and (count < count_max):

        # Necessary values
        J = F[0, 0]*F[1, 1]*F[2, 2] #det
        p = kappa*(J-1)
        Fbar = J**(-1/3)*F
        Cbar = Fbar.T@Fbar
        I1 = np.trace(Cbar)
        lambda_r = np.sqrt(I1/3/N)
        ##########
        if abs(1-lambda_r**2)<1e-8:
            print("1-lambda_r**2", 1-lambda_r**2)
        lang = (3-lambda_r**2)/(1-lambda_r**2)
        bbar = Fbar@Fbar.T
        C = F.T@F
        lambda_a_sq, N_a = la.eig(C) # !!!!!!!!!
        lambda_a = np.sqrt(lambda_a_sq).reshape(-1, 1)
        s = kappa

        ### 1. Stresses

        ## 1.1. Volumetric kirchoff stress
        tau_vol = p*I

        ## 1.2. Unimodular elastic kirchoff stress
        taubar_e = (mu*lang/3)*bbar

        ## 1.3. Unimodular viscous kirchoff stress (Newton update)
        F_prev = np.eye(3)
        F_prev[0, 0] = np.exp(strain[-2])
        F_prev[1, 1] = lambda2_prev
        F_prev[2, 2] = lambda2_prev
        be_prev = np.eye(3)
        be_prev[0, 0] = b11_prev
        #####################################
        if abs(np.sqrt(b11_prev))<1e-8:
            print("np.sqrt(b11_prev)", np.sqrt(b11_prev))
        be_prev[1, 1] = 1/np.sqrt(b11_prev)
        be_prev[2, 2] = 1/np.sqrt(b11_prev)
        J_prev = F_prev[0, 0]*F_prev[1, 1]*F_prev[2, 2] #det
        Fbar_prev = J_prev**(-1/3)*F_prev
        be_tr = Fbar@la.inv(Fbar_prev)@be_prev@la.inv(Fbar_prev).T@Fbar.T
        lambda_a_e_tr_sq, n_a = la.eig(be_tr)
        lambda_a_e_tr = np.sqrt(lambda_a_e_tr_sq).reshape(-1, 1)
        eps_a_tr = np.log(lambda_a_e_tr)

        # Initial values of the elastic logarithmic stretches
        eps_a = eps_a_tr
        res_a = np.ones(3)
        count_a = 0

        # Newton loop for unknown elastic logarithmic stretches
        while (abs(res_a) > 1.e-4).any() and (count_a < count_a_max):

            # Necessary values
            be = n_a@np.diag(np.exp(eps_a[:, 0])**2)@la.inv(n_a)
            Ci = Fbar.T@la.inv(be)@Fbar
            I1_i = np.trace(Ci)
            lambda_chain_i = np.sqrt(I1_i/3)
            I1_e = np.trace(be)
            lambda_r_e = np.sqrt(I1_e/3/N_v)
            ######################################
            if abs(1-lambda_r_e**2)<1e-8:
                print("1-lambda_r_e**2", 1-lambda_r_e**2)
            lang_e = (3-lambda_r_e**2)/(1-lambda_r_e**2)

            # Unimodular viscous kirchoff stress
            taubar_v = (mu_v*lang_e/3)*be
            tau_v_iso = td(PP, taubar_v, 2)
            tau_v = la.norm(tau_v_iso)/np.sqrt(2)
            devtau_a = la.eig(tau_v_iso)[0].reshape(-1, 1)

            # Effective creep rate
            gamma_dot = gamma0_taum*(lambda_chain_i-1+perturbation_param)**c*(tau_v)**m

            # Residual
            ##########################
            # if abs(tau_v)<1e-8:
            #     print("dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v", dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v)
            #     print("gamma0_taum", gamma0_taum)
            #     print("lambda_chain_i", lambda_chain_i)
            #     print("gamma_dot", gamma_dot)
            #     print("devtau_a", devtau_a)
            #     print("tau_v", tau_v)
            #     tau_v = 1e-8
            res_a = eps_a + dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v - eps_a_tr

            # Local tangent
            beta1 = dt*(m-1)*gamma0_taum*(lambda_chain_i**c-1+perturbation_param)*(tau_v)**(m-3)/2/np.sqrt(2)
            beta2 = dt*gamma_dot/np.sqrt(2)/tau_v
            T = (2/3)*mu_v*((3-lambda_r_e**2)/(1-lambda_r_e**2))*np.diag((lambda_a**2).reshape(3)) - \
            (4/9)*(mu_v/N_v)*(1/(1-lambda_r_e**2))*(lambda_a**2)*(lambda_a**2).reshape(3)
            Tbar = T - (1/3)*np.sum(T, 0)
            D = (devtau_a.reshape(1, 3)@T).reshape(3, 1)
            K = I + beta1*devtau_a*D.reshape(3) + beta2*Tbar

            # Update
            K_inv = la.inv(K)
            eps_a = eps_a - K_inv@res_a

            # Update count
            count_a += 1

            # Print
            # print(f"\t\tResidual {count}_{count_a} :", np.max(res_a))

        ## 1.4. Isochoric kirchoff stress
        taubar = taubar_e + taubar_v
        tau_iso = td(PP, taubar, 2)

        ## 1.5. Total kirchoff stress
        tau = tau_vol + tau_iso


        ### 2. Tangent Modulus

        ## 2.1. Volumetric Tangent Modulus
        CC_vol = (p+s)*IxI - 2*p*II

        ## 2.2. Unimodular elastic Tangent Modulus
        CCbar_e = (4/9)*(mu/N)*(1/((1-lambda_r**2)**2))*td(bbar, bbar, 0)

        ## 2.3. Unimodular viscous Tangent Modulus
        c_ab = T@K_inv
        tau_a = la.eig(taubar_v)[0].reshape(-1, 1)
        ###########################
        if any(abs(lambda_a_e_tr**2)<1e-8):
            print("lambda_a_e_tr**2", lambda_a_e_tr**2)
        stilde_a = tau_a/lambda_a_e_tr**2
        M = N_a.T.reshape(3, 3, 1)*N_a.T.reshape(3, 1, 3)
        G = M.reshape(3, 1, 3, 1, 3, 1)*M.reshape(1, 3, 1, 3, 1, 3) + \
        M.reshape(3, 1, 3, 1, 1, 3) + M.reshape(1, 3, 1, 3, 3, 1)
        CCbar_v_algo = 0
        for a_it in range(3):
            for b_it in range(3):
                CCbar_v_algo += (c_ab[a_it, b_it]-2*tau_a[a_it]*(a_it==b_it))/(lambda_a_e_tr[a_it]**2*lambda_a_e_tr[b_it]**2)*td(M[a_it], M[b_it], 0)
        for a_it in range(3):
            for b_it in range(3):
                if b_it != a_it:
                    if abs(lambda_a_e_tr[a_it]**2 - lambda_a_e_tr[b_it]**2) < 10**-8:
                        CCbar_v_algo += (1/4)*((c_ab[a_it, a_it] - 2*tau_a[a_it])/lambda_a_e_tr[a_it]**4)*(G[a_it][b_it]+G[b_it][a_it])
                    else:
                        CCbar_v_algo += (1/2)*((stilde_a[a_it] - stilde_a[b_it])/(lambda_a_e_tr[a_it]**2 - lambda_a_e_tr[b_it]**2))*(G[a_it][b_it]+G[b_it][a_it])
        F_e_tr = np.diag(lambda_a_e_tr.reshape(3))
        CCbar_v_algo = np.einsum('iI,jJ,kK,lL,IJKL->ijkl', F_e_tr, F_e_tr, F_e_tr, F_e_tr, CCbar_v_algo)

        ## 2.4. Isochoric Tangent Modulus
        CCbar = CCbar_e + CCbar_v_algo
        CC_iso = CCbar + (2/3)*td(taubar, I, 2)*II-(2/3)*(td(taubar, I, 0) + td(I, taubar, 0))
        CC_iso = td(td(PP, CC_iso, 2), PP, 2)

        ## 2.5. Total Tangent Modulus
        CC = CC_vol + CC_iso

        ## 2.6. Convert to Eulerian then first
        F_inv = la.inv(F)
        CC_eul = np.einsum('Aa,Bb,Cc,Dd,abcd->ABCD', F_inv, F_inv, F_inv, F_inv, CC)
        S = F_inv@tau@F_inv.T
        AA = np.einsum('aA,cC,ABCD->aBcD', F, F, CC_eul) + np.einsum('ac, DB->aBcD', I, S)

        # Update unknown deformations
        F_u = np.array([[F[1, 1]], [F[2, 2]]])
        P = tau@F_inv.T
        res = np.array([[P[1, 1]], [P[2, 2]]])
        AA_red_inv = np.zeros([2, 2])
        AA_red_inv[0, 0] = AA[2, 2, 2, 2]
        AA_red_inv[0, 1] = -AA[1, 1, 2, 2]
        AA_red_inv[1, 0] = -AA[2, 2, 1, 1]
        AA_red_inv[1, 1] = AA[1, 1, 1, 1]
        AA_red_inv = AA_red_inv/(AA[1, 1, 1, 1]*AA[2, 2, 2, 2] - AA[1, 1, 2, 2]*AA[2, 2, 1, 1])
        # print(CC_red_inv)
        F_u = F_u - AA_red_inv@res
        F[1, 1] = F_u[0, 0]
        F[2, 2] = F_u[1, 0]

        # Update count
        count += 1

        # Print some stuff
        # print("\tResidual ", count, "      :", np.max(res))
        # print("\tIterations Inside : ", count_a)
        if count_a == count_a_max:
            print("Inside iterations didn't converge, Time=", t[-1], ", Strain=", strain[-1])
            if abs(tau_v)<1e-8:
                print(params)
                print("dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v", dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v)
                print("gamma0_taum", gamma0_taum)
                print("lambda_chain_i", lambda_chain_i)
                print("gamma_dot", gamma_dot)
                print("devtau_a", devtau_a)
                print("tau_v", tau_v)
                tau_v = 1e-8

    # Print some stuff
    # print("\nIterations : ", count)
    # print("b_11       :", be[0, 0])
    # print("F_11       :", F[0, 0])
    # print("F_22       :", F[1, 1])
    # print("P_11       :", P[0, 0], "\n")
    # print("%-------------------------------%\n")
    if count == count_max:
        print("Outside iterations didn't converge, Time=", t[-1], ", Strain=", strain[-1])

    # Stress and history variables
    stress_new = tau[0, 0]
    lambda2_new = F[1, 1]
    b11_new = be[0, 0]
    histvars_new = [lambda2_new, b11_new]
    histvars.append(histvars_new)

    return stress_new, histvars


def ubb_du(t, strain, histvars, stress_new, drecurr, params):
    """
    Update derivatives using homogenous uniaxial bergstrom boyce model
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
    kappa = params[0]
    mu = params[1]
    mu_v = params[2]
    N = params[3]
    N_v = params[4]
    gamma0_taum = params[5]
    c = params[6]
    m = params[7]

    # Perturbation parameter
    perturbation_param = 0.02

    # History variables from previous timestep
    lambda2_prev = histvars[-2][0]
    b11_prev = histvars[-2][1]

    # History variables from current timestep
    lambda2 = histvars[-1][0]
    b11 = histvars[-1][1]
    ######################
    # print("b11:", b11)

    # Initialize recurrent_derivatives if needed
    if len(drecurr) == 0:
        dlambda2_init = [0 for j in range(len(params))]
        db11_init = [0 for j in range(len(params))]
        drecurr.append(dlambda2_init+db11_init)

    # Recurrent derivatives from previous timestep
    dlambda2prev = drecurr[-1][:8]
    db11prev = drecurr[-1][8:]

    # Fourth order identity tensor, I kron I, Projection Tensor
    I = np.eye(3)
    IxI = td(I, I, 0)
    II = I.reshape(3, 1, 3, 1)*I.reshape(1, 3, 1, 3)
    PP = II - (1/3)*IxI

    # Current deformation gradient
    F = np.eye(3)
    lambda1 = np.exp(strain[-1])
    F[0, 0] = lambda1
    F[1, 1] = lambda2
    F[2, 2] = lambda2
    dt = t[-1]-t[-2]

    # Necessary values
    J = F[0, 0]*F[1, 1]*F[2, 2] #det
    Fbar = J**(-1/3)*F
    Cbar = Fbar.T@Fbar
    I1 = np.trace(Cbar)
    lambda_r = np.sqrt(I1/3/N)
    lang = (3-lambda_r**2)/(1-lambda_r**2)
    bbar = Fbar@Fbar.T

    be = np.array(
        [[b11, 0, 0],
         [0, 1/np.sqrt(b11), 0],
         [0, 0, 1/np.sqrt(b11)]]
    )
    I1_e = np.trace(be)
    lambda_r_e = np.sqrt(I1_e/3/N_v)
    lange = (3-lambda_r_e**2)/(1-lambda_r_e**2)

    Ci = Fbar.T@la.inv(be)@Fbar
    I1_i = np.trace(Ci)
    lambda_chain_i = np.sqrt(I1_i/3)
    tau_v_iso = td(PP, (mu_v/3)*lange*be, 2)
    tau_v = la.norm(tau_v_iso)/np.sqrt(2)
    N11 = tau_v_iso[0, 0]/(tau_v*np.sqrt(2))
    gammadot = gamma0_taum*(lambda_chain_i-1+perturbation_param)**c*(tau_v)**m
    expo = np.exp(-2*gammadot*N11*dt)
    lambda1_prev = np.exp(strain[-2])


    ### 1. Matrix A

    ## 1.1. df_lambda2 and dg_lambda2
    dtauvol_lambda2 = kappa*(2*lambda1*lambda2)*np.eye(3)
    dlang_lambdar = 4*lambda_r/(lambda_r**2 - 1)**2
    dlambdar_I1 = 1/(2*np.sqrt(3*N*I1))
    dI1_lambda2 = (4/3)*(-lambda1**(4/3)*lambda2**(-7/3) + lambda1**(-2/3)*lambda2**(-1/3))
    dbbar_lambda2 = (1/3)*np.array(
        [[-4*lambda1**(4/3)*lambda2**(-7/3), 0, 0],
         [0, 2*lambda1**(-2/3)*lambda2**(-1/3), 0],
         [0, 0, 2*lambda1**(-2/3)*lambda2**(-1/3)]]
    )
    dlambdar_lambda2 = dlambdar_I1*dI1_lambda2
    dtaueiso_lambda2 = td(PP,
                          ((mu/3)*dlang_lambdar*dlambdar_lambda2*bbar + (mu/3)*lang*dbbar_lambda2),
                          2)
    df_lambda2 = dtauvol_lambda2[0, 0] + dtaueiso_lambda2[0, 0]
    dg_lambda2 = dtauvol_lambda2[1, 1] + dtaueiso_lambda2[1, 1]

    ## 1.2. df_b11 and dg_b11
    dlange_lambdare = 4*lambda_r_e/(lambda_r_e**2 - 1)**2
    dlambdare_I1e = 1/(2*np.sqrt(3*N_v*I1_e))
    dI1e_b11 = 1 - 1/(b11**(3/2))
    dlambdare_b11 = dlambdare_I1e*dI1e_b11
    dbe_b11 = np.array(
        [[1, 0, 0],
         [0, -1/(2*b11**(3/2)), 0],
         [0, 0, -1/(2*b11**(3/2))]]
    )
    dtauviso_b11 = td(PP,
                      ((mu_v/3)*dlange_lambdare*dlambdare_b11*be + (mu_v/3)*lange*dbe_b11),
                      2)
    df_b11 = dtauviso_b11[0, 0]
    dg_b11 = dtauviso_b11[1, 1]

    ## 1.3. dh_lambda2
    dexpo_gammadot = -2*N11*dt*expo
    dlambdaichain_I1i = 1/(2*np.sqrt(3*I1_i))
    dI1i_lambda2 = \
    -(4/3)*(lambda1**(4/3)*lambda2**(-7/3)/b11) + (4/3)*lambda1**(-2/3)*lambda2**(-1/3)*np.sqrt(b11)
    dlambdaichain_lambda2 = dlambdaichain_I1i*dI1i_lambda2
    dgammadot_lambda2 = \
    gamma0_taum*c*(lambda_chain_i-1+perturbation_param)**(c-1)*(tau_v)**m*dlambdaichain_lambda2
    dexpo_lambda2 = dexpo_gammadot*dgammadot_lambda2
    betr11 = ((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)*b11_prev
    dbetr11_lambda2 = \
    -(4/(3*lambda2))*((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)*b11_prev
    dbe11_lambda2 = dexpo_lambda2*betr11 + expo*dbetr11_lambda2
    dh_lambda2 = dbe11_lambda2

    ## 1.4. dh_b11
    dI1i_b11 = \
    -(lambda1**(4/3)*lambda2**(-4/3))/(b11**2) + (lambda1**(-2/3)*lambda2**(2/3))/(np.sqrt(b11))
    dlambdaichain_b11 = dlambdaichain_I1i*dI1i_b11
    dtauv_tauviso = tau_v_iso/(np.sqrt(2)*la.norm(tau_v_iso))
    dtauv_b11 = td(dtauv_tauviso, dtauviso_b11, 2)
    dgammadot_b11 = \
    gamma0_taum*c*(lambda_chain_i-1+perturbation_param)**(c-1)*(tau_v)**m*dlambdaichain_b11 + \
    gamma0_taum*m*(lambda_chain_i-1+perturbation_param)**c*(tau_v)**(m-1)*dtauv_b11
    dexpo_N11 = -2*gammadot*dt*expo
    dN11_b11 = \
    (la.norm(tau_v_iso)*dtauviso_b11[0, 0] - (np.sqrt(2)*dtauv_b11)*(tau_v_iso[0, 0]))/(la.norm(tau_v_iso)**2)
    dexpo_b11 = dexpo_gammadot*dgammadot_b11 + dexpo_N11*dN11_b11
    dbe11_b11 = dexpo_b11*betr11
    dh_b11 = dbe11_b11

    ## 1.5. Construct the matrix A and A_inv
    A = np.array(
        [[1, -df_lambda2, -df_b11],
         [0, -dg_lambda2, -dg_b11],
         [0, -dh_lambda2, 1-dh_b11]]
    )
    A_inv = la.inv(A)
    ########################
    # print("A: ", A)

    ### 2. kappa

    ## 2.1. b_kappa
    db11prev_kappa = db11prev[0]
    dlambda2prev_kappa = dlambda2prev[0]
    dtauvol_kappa = (J-1)*np.eye(3)
    dbe11_b11prev = expo*((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)
    dbe11_lambda2prev = \
    expo*(4/(3*lambda2_prev))*((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)*b11_prev
    df_kappa = dtauvol_kappa[0, 0]
    dg_kappa = dtauvol_kappa[1, 1]
    dh_kappa = 0
    dh_b11prev = dbe11_b11prev
    dh_lambda2prev = dbe11_lambda2prev
    b_kappa = np.array(
        [[df_kappa],
         [dg_kappa],
         [dh_kappa + dh_b11prev*db11prev_kappa + dh_lambda2prev*dlambda2prev_kappa]]
    )

    ## 2.2. solve for d_kappa
    d_kappa = A_inv@b_kappa
    dstress_kappa = d_kappa[0, 0]
    dlambda2_kappa = d_kappa[1, 0]
    db11_kappa = d_kappa[2, 0]

    ### 3. mu

    ## 3.1. b_mu
    db11prev_mu = db11prev[1]
    dlambda2prev_mu = dlambda2prev[1]
    dtaueiso_mu = td(PP,
                     (1/3)*lang*bbar,
                     2)
    df_mu = dtaueiso_mu[0, 0]
    dg_mu = dtaueiso_mu[1, 1]
    dh_mu = 0
    b_mu = np.array(
        [[df_mu],
         [dg_mu],
         [dh_mu + dh_b11prev*db11prev_mu + dh_lambda2prev*dlambda2prev_mu]]
    )

    ## 3.2. solve for d_mu
    d_mu = A_inv@b_mu
    dstress_mu = d_mu[0, 0]
    dlambda2_mu = d_mu[1, 0]
    db11_mu = d_mu[2, 0]
    ###########################
    # print("lang: ", lang)
    # print("bbar: ", bbar)
    # print("dtaueiso_mu:", dtaueiso_mu)
    # print("b_mu: ", b_mu)
    # print("d_mu: ", d_mu)

    ### 4. mu_v

    ## 4.1. b_muv
    db11prev_muv = db11prev[2]
    dlambda2prev_muv = dlambda2prev[2]
    dtauviso_muv = td(PP,
                      (1/3)*lange*be,
                      2)
    dgammadot_tauviso = gamma0_taum*m*(lambda_chain_i-1+perturbation_param)**c*tau_v**(m-1)*dtauv_tauviso
    dtauviso11_tauviso = np.zeros((3, 3))
    dtauviso11_tauviso[0, 0] = 1
    dN11_tauviso = \
     (la.norm(tau_v_iso)*dtauviso11_tauviso - tau_v_iso[0, 0]*(np.sqrt(2)*dtauv_tauviso))/(la.norm(tau_v_iso)**2)
    dexpo_tauviso = dexpo_gammadot*dgammadot_tauviso + dexpo_N11*dN11_tauviso
    dbe11_tauviso = dexpo_tauviso*betr11
    dbe11_muv = td(dbe11_tauviso, dtauviso_muv, 2)
    df_muv = dtauviso_muv[0, 0]
    dg_muv = dtauviso_muv[1, 1]
    dh_muv = dbe11_muv
    b_muv = np.array(
        [[df_muv],
         [dg_muv],
         [dh_muv + dh_b11prev*db11prev_muv + dh_lambda2prev*dlambda2prev_muv]]
    )

    ## 4.2. solve for d_muv
    d_muv = A_inv@b_muv
    dstress_muv = d_muv[0, 0]
    dlambda2_muv = d_muv[1, 0]
    db11_muv = d_muv[2, 0]
    ######################
    # print("lange: ", lange)
    # print("be: ", be)
    # print("dtauviso_muv:", dtauviso_muv)
    # print("b_muv: ", b_muv)
    # print("d_muv: ", d_muv)

    ### 5. N

    ## 5.1. b_N
    db11prev_N = db11prev[3]
    dlambda2prev_N = dlambda2prev[3]
    dlambdar_N = -(1/(2*N))*np.sqrt(I1/(3*N))
    dtaueiso_N = td(PP, (mu/3)*dlang_lambdar*dlambdar_N*bbar, 2)
    df_N = dtaueiso_N[0, 0]
    dg_N = dtaueiso_N[1, 1]
    dh_N = 0
    b_N = np.array(
        [[df_N],
         [dg_N],
         [dh_N + dh_b11prev*db11prev_N + dh_lambda2prev*dlambda2prev_N]]
    )

    ## 5.2. solve for d_N
    d_N = A_inv@b_N
    dstress_N = d_N[0, 0]
    dlambda2_N = d_N[1, 0]
    db11_N = d_N[2, 0]

    ### 6. N_v

    ## 6.1. b_Nv
    db11prev_Nv = db11prev[4]
    dlambda2prev_Nv = dlambda2prev[4]
    dlambdare_Nv = -(1/(2*N_v))*np.sqrt(I1_e/(3*N_v))
    dtauviso_Nv = td(PP, (mu_v/3)*dlange_lambdare*dlambdare_Nv*be)
    dbe11_Nv = td(dbe11_tauviso, dtauviso_Nv, 2)
    df_Nv = dtauviso_Nv[0, 0]
    dg_Nv = dtauviso_Nv[1, 1]
    dh_Nv = dbe11_Nv
    b_Nv = np.array(
        [[df_Nv],
         [dg_Nv],
         [dh_Nv + dh_b11prev*db11prev_Nv + dh_lambda2prev*dlambda2prev_Nv]]
    )

    ## 6.2. solve for d_Nv
    d_Nv = A_inv@b_Nv
    dstress_Nv = d_Nv[0, 0]
    dlambda2_Nv = d_Nv[1, 0]
    db11_Nv = d_Nv[2, 0]

    ### 7. gamma0_taum

    ## 7.1. b_gamma0taum
    db11prev_gamma0taum = db11prev[5]
    dlambda2prev_gamma0taum = dlambda2prev[5]
    dgammadot_gamma0taum = (lambda_chain_i-1+perturbation_param)**c*tau_v**m
    dbe11_gamma0taum = dexpo_gammadot*dgammadot_gamma0taum*betr11
    df_gamma0taum = 0
    dg_gamma0taum = 0
    dh_gamma0taum = dbe11_gamma0taum
    b_gamma0taum = np.array(
        [[df_gamma0taum],
         [dg_gamma0taum],
         [dh_gamma0taum + dh_b11prev*db11prev_gamma0taum + dh_lambda2prev*dlambda2prev_gamma0taum]]
    )

    ## 7.2. solve for d_gamma0taum
    d_gamma0taum = A_inv@b_gamma0taum
    dstress_gamma0taum = d_gamma0taum[0, 0]
    dlambda2_gamma0taum = d_gamma0taum[1, 0]
    db11_gamma0taum = d_gamma0taum[2, 0]

    ### 8. c

    ## 8.1. b_c
    db11prev_c = db11prev[6]
    dlambda2prev_c = dlambda2prev[6]
    dgammadot_c = \
    gamma0_taum*(lambda_chain_i-1+perturbation_param)**c*np.log(lambda_chain_i-1+perturbation_param)*tau_v**m
    dbe11_c = dexpo_gammadot*dgammadot_c*betr11
    df_c = 0
    dg_c = 0
    dh_c = dbe11_c
    b_c = np.array(
        [[df_c],
         [dg_c],
         [dh_c + dh_b11prev*db11prev_c + dh_lambda2prev*dlambda2prev_c]]
    )

    ## 8.2. solve for d_c
    d_c = A_inv@b_c
    dstress_c = d_c[0, 0]
    dlambda2_c = d_c[1, 0]
    db11_c = d_c[2, 0]

    ### 9. m

    ## 9.1. b_m
    db11prev_m = db11prev[7]
    dlambda2prev_m = dlambda2prev[7]
    dgammadot_m = \
    gamma0_taum*(lambda_chain_i-1+perturbation_param)**c*tau_v**m*np.log(tau_v)
    dbe11_m = dexpo_gammadot*dgammadot_m*betr11
    df_m = 0
    dg_m = 0
    dh_m = dbe11_m
    b_m = np.array(
        [[df_m],
         [dg_m],
         [dh_m + dh_b11prev*db11prev_m + dh_lambda2prev*dlambda2prev_m]]
    )

    ## 9.2. solve for d_m
    d_m = A_inv@b_m
    dstress_m = d_m[0, 0]
    dlambda2_m = d_m[1, 0]
    db11_m = d_m[2, 0]

    # Stress and Recurrent derivatives
    dstress = [dstress_kappa, dstress_mu, dstress_muv, dstress_N, dstress_Nv,
               dstress_gamma0taum, dstress_c, dstress_m]
    dlambda2 = [dlambda2_kappa, dlambda2_mu, dlambda2_muv, dlambda2_N, dlambda2_Nv,
                dlambda2_gamma0taum, dlambda2_c, dlambda2_m]
    db11 = [db11_kappa, db11_mu, db11_muv, db11_N, db11_Nv,
            db11_gamma0taum, db11_c, db11_m]
    drecurr_new = dlambda2 + db11
    drecurr.append(drecurr_new)

    return dstress, drecurr

#endregion

#region ##--UNIAXIAL BERGSTROM BOYCE MODIFIED--##

def ubbm_su(t, strain, histvars, params):
    """
    Update stress and history variables using homogenous uniaxial bergstrom boyce model
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
    count_a_max = 100

    # Material parameters
    kappa = params[0]
    mu = params[1]
    mu_v = params[2]
    N = params[3]
    N_v = params[4]
    tau_hat = params[5]
    aj = params[6:]

    # Initialize history variables if needed
    if len(histvars) == 0:
        lambda2_init = 1
        b11_init = 1
        histvars.append([lambda2_init, b11_init])

    # History variables from previous timestep
    lambda2_prev = histvars[-1][0]
    b11_prev = histvars[-1][1]

    #Fourth order identity tensor, I kron I, Projection Tensor
    I = np.eye(3)
    IxI = td(I, I, 0)
    II = I.reshape(3, 1, 3, 1)*I.reshape(1, 3, 1, 3)
    PP = II - (1/3)*IxI

    # Current deformation gradient
    F = np.eye(3)
    lambda1 = np.exp(strain[-1])
    F[0, 0] = lambda1
    F[1, 1] = lambda2_prev
    F[2, 2] = lambda2_prev
    dt = t[-1]-t[-2]

    # Residual and count initialisation
    res = np.ones([2, 1])
    count = 0

    # Newton update loop for unknown stretches
    while (abs(res) > 1.e-4).any() and (count < count_max):

        # Necessary values
        J = F[0, 0]*F[1, 1]*F[2, 2] #det
        p = kappa*(J-1)
        Fbar = J**(-1/3)*F
        Cbar = Fbar.T@Fbar
        I1 = np.trace(Cbar)
        lambda_r = np.sqrt(I1/3/N)
        ##########
        if abs(1-lambda_r**2)<1e-8:
            print("1-lambda_r**2", 1-lambda_r**2)
        lang = (3-lambda_r**2)/(1-lambda_r**2)
        bbar = Fbar@Fbar.T
        C = F.T@F
        lambda_a_sq, N_a = la.eig(C) # !!!!!!!!!
        lambda_a = np.sqrt(lambda_a_sq).reshape(-1, 1)
        s = kappa

        ### 1. Stresses

        ## 1.1. Volumetric kirchoff stress
        tau_vol = p*I

        ## 1.2. Unimodular elastic kirchoff stress
        taubar_e = (mu*lang/3)*bbar

        ## 1.3. Unimodular viscous kirchoff stress (Newton update)
        F_prev = np.eye(3)
        F_prev[0, 0] = np.exp(strain[-2])
        F_prev[1, 1] = lambda2_prev
        F_prev[2, 2] = lambda2_prev
        be_prev = np.eye(3)
        be_prev[0, 0] = b11_prev
        #####################################
        if abs(np.sqrt(b11_prev))<1e-8:
            print("np.sqrt(b11_prev)", np.sqrt(b11_prev))
        be_prev[1, 1] = 1/np.sqrt(b11_prev)
        be_prev[2, 2] = 1/np.sqrt(b11_prev)
        J_prev = F_prev[0, 0]*F_prev[1, 1]*F_prev[2, 2] #det
        Fbar_prev = J_prev**(-1/3)*F_prev
        be_tr = Fbar@la.inv(Fbar_prev)@be_prev@la.inv(Fbar_prev).T@Fbar.T
        lambda_a_e_tr_sq, n_a = la.eig(be_tr)
        lambda_a_e_tr = np.sqrt(lambda_a_e_tr_sq).reshape(-1, 1)
        eps_a_tr = np.log(lambda_a_e_tr)

        # Initial values of the elastic logarithmic stretches
        eps_a = eps_a_tr
        res_a = np.ones(3)
        count_a = 0

        # Newton loop for unknown elastic logarithmic stretches
        while (abs(res_a) > 1.e-4).any() and (count_a < count_a_max):

            # Necessary values
            be = n_a@np.diag(np.exp(eps_a[:, 0])**2)@la.inv(n_a)
            I1_e = np.trace(be)
            lambda_r_e = np.sqrt(I1_e/3/N_v)
            ######################################
            if abs(1-lambda_r_e**2)<1e-8:
                print("1-lambda_r_e**2", 1-lambda_r_e**2)
            lang_e = (3-lambda_r_e**2)/(1-lambda_r_e**2)

            # Unimodular viscous kirchoff stress
            taubar_v = (mu_v*lang_e/3)*be
            tau_v_iso = td(PP, taubar_v, 2)
            tau_v = la.norm(tau_v_iso)/np.sqrt(2)
            devtau_a = la.eig(tau_v_iso)[0].reshape(-1, 1)

            # Effective creep rate
            gamma_dot = sum([aj[j-1]*(tau_v/tau_hat)**j for j in range(1, len(aj)+1)])

            # Residual
            ##########################
            # if abs(tau_v)<1e-8:
            #     print("dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v", dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v)
            #     print("gamma0_taum", gamma0_taum)
            #     print("lambda_chain_i", lambda_chain_i)
            #     print("gamma_dot", gamma_dot)
            #     print("devtau_a", devtau_a)
            #     print("tau_v", tau_v)
            #     tau_v = 1e-8
            res_a = eps_a + dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v - eps_a_tr

            # Local tangent
            beta1 = (dt/2/np.sqrt(2))*(1/tau_hat**3)*sum([aj[j-1]*(j-1)*(tau_v/tau_hat)**(j-3) for
                                                          j in range(2, len(aj)+1)])
            beta2 = dt*gamma_dot/np.sqrt(2)/tau_v
            T = (2/3)*mu_v*((3-lambda_r_e**2)/(1-lambda_r_e**2))*np.diag((lambda_a**2).reshape(3)) - \
            (4/9)*(mu_v/N_v)*(1/(1-lambda_r_e**2))*(lambda_a**2)*(lambda_a**2).reshape(3)
            Tbar = T - (1/3)*np.sum(T, 0)
            D = (devtau_a.reshape(1, 3)@T).reshape(3, 1)
            K = I + beta1*devtau_a*D.reshape(3) + beta2*Tbar

            # Update
            K_inv = la.inv(K)
            eps_a = eps_a - K_inv@res_a

            # Update count
            count_a += 1

            # Print
            # print(f"\t\tResidual {count}_{count_a} :", np.max(res_a))

        ## 1.4. Isochoric kirchoff stress
        taubar = taubar_e + taubar_v
        tau_iso = td(PP, taubar, 2)

        ## 1.5. Total kirchoff stress
        tau = tau_vol + tau_iso


        ### 2. Tangent Modulus

        ## 2.1. Volumetric Tangent Modulus
        CC_vol = (p+s)*IxI - 2*p*II

        ## 2.2. Unimodular elastic Tangent Modulus
        CCbar_e = (4/9)*(mu/N)*(1/((1-lambda_r**2)**2))*td(bbar, bbar, 0)

        ## 2.3. Unimodular viscous Tangent Modulus
        c_ab = T@K_inv
        tau_a = la.eig(taubar_v)[0].reshape(-1, 1)
        ###########################
        if any(abs(lambda_a_e_tr**2)<1e-8):
            print("lambda_a_e_tr**2", lambda_a_e_tr**2)
        stilde_a = tau_a/lambda_a_e_tr**2
        M = N_a.T.reshape(3, 3, 1)*N_a.T.reshape(3, 1, 3)
        G = M.reshape(3, 1, 3, 1, 3, 1)*M.reshape(1, 3, 1, 3, 1, 3) + \
        M.reshape(3, 1, 3, 1, 1, 3) + M.reshape(1, 3, 1, 3, 3, 1)
        CCbar_v_algo = 0
        for a_it in range(3):
            for b_it in range(3):
                CCbar_v_algo += (c_ab[a_it, b_it]-2*tau_a[a_it]*(a_it==b_it))/(lambda_a_e_tr[a_it]**2*lambda_a_e_tr[b_it]**2)*td(M[a_it], M[b_it], 0)
        for a_it in range(3):
            for b_it in range(3):
                if b_it != a_it:
                    if abs(lambda_a_e_tr[a_it]**2 - lambda_a_e_tr[b_it]**2) < 10**-8:
                        CCbar_v_algo += (1/4)*((c_ab[a_it, a_it] - 2*tau_a[a_it])/lambda_a_e_tr[a_it]**4)*(G[a_it][b_it]+G[b_it][a_it])
                    else:
                        CCbar_v_algo += (1/2)*((stilde_a[a_it] - stilde_a[b_it])/(lambda_a_e_tr[a_it]**2 - lambda_a_e_tr[b_it]**2))*(G[a_it][b_it]+G[b_it][a_it])
        F_e_tr = np.diag(lambda_a_e_tr.reshape(3))
        CCbar_v_algo = np.einsum('iI,jJ,kK,lL,IJKL->ijkl', F_e_tr, F_e_tr, F_e_tr, F_e_tr, CCbar_v_algo)

        ## 2.4. Isochoric Tangent Modulus
        CCbar = CCbar_e + CCbar_v_algo
        CC_iso = CCbar + (2/3)*td(taubar, I, 2)*II-(2/3)*(td(taubar, I, 0) + td(I, taubar, 0))
        CC_iso = td(td(PP, CC_iso, 2), PP, 2)

        ## 2.5. Total Tangent Modulus
        CC = CC_vol + CC_iso

        ## 2.6. Convert to Eulerian then first
        F_inv = la.inv(F)
        CC_eul = np.einsum('Aa,Bb,Cc,Dd,abcd->ABCD', F_inv, F_inv, F_inv, F_inv, CC)
        S = F_inv@tau@F_inv.T
        AA = np.einsum('aA,cC,ABCD->aBcD', F, F, CC_eul) + np.einsum('ac, DB->aBcD', I, S)

        # Update unknown deformations
        F_u = np.array([[F[1, 1]], [F[2, 2]]])
        P = tau@F_inv.T
        res = np.array([[P[1, 1]], [P[2, 2]]])
        AA_red_inv = np.zeros([2, 2])
        AA_red_inv[0, 0] = AA[2, 2, 2, 2]
        AA_red_inv[0, 1] = -AA[1, 1, 2, 2]
        AA_red_inv[1, 0] = -AA[2, 2, 1, 1]
        AA_red_inv[1, 1] = AA[1, 1, 1, 1]
        AA_red_inv = AA_red_inv/(AA[1, 1, 1, 1]*AA[2, 2, 2, 2] - AA[1, 1, 2, 2]*AA[2, 2, 1, 1])
        # print(CC_red_inv)
        F_u = F_u - AA_red_inv@res
        F[1, 1] = F_u[0, 0]
        F[2, 2] = F_u[1, 0]

        # Update count
        count += 1

        # Print some stuff
        # print("\tResidual ", count, "      :", np.max(res))
        # print("\tIterations Inside : ", count_a)
        if count_a == count_a_max:
            print("Inside iterations didn't converge, Time=", t[-1], ", Strain=", strain[-1])

    # Print some stuff
    # print("\nIterations : ", count)
    # print("b_11       :", be[0, 0])
    # print("F_11       :", F[0, 0])
    # print("F_22       :", F[1, 1])
    # print("P_11       :", P[0, 0], "\n")
    # print("%-------------------------------%\n")
    if count == count_max:
        print("Outside iterations didn't converge, Time=", t[-1], ", Strain=", strain[-1])

    # Stress and history variables
    stress_new = tau[0, 0]
    lambda2_new = F[1, 1]
    b11_new = be[0, 0]
    histvars_new = [lambda2_new, b11_new]
    histvars.append(histvars_new)

    return stress_new, histvars

def ubbm_du(t, strain, histvars, stress_new, drecurr, params):
    """
    Update derivatives using homogenous uniaxial bergstrom boyce model
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
    kappa = params[0]
    mu = params[1]
    mu_v = params[2]
    N = params[3]
    N_v = params[4]
    tau_hat = params[5]
    aj = params[6:]

    # History variables from previous timestep
    lambda2_prev = histvars[-2][0]
    b11_prev = histvars[-2][1]

    # History variables from current timestep
    lambda2 = histvars[-1][0]
    b11 = histvars[-1][1]

    # Initialize recurrent_derivatives if needed
    if len(drecurr) == 0:
        dlambda2_init = [0 for j in range(len(params))]
        db11_init = [0 for j in range(len(params))]
        drecurr.append(dlambda2_init+db11_init)

    # Recurrent derivatives from previous timestep
    dlambda2prev = drecurr[-1][:len(params)]
    db11prev = drecurr[-1][len(params):]

    # Fourth order identity tensor, I kron I, Projection Tensor
    I = np.eye(3)
    IxI = td(I, I, 0)
    II = I.reshape(3, 1, 3, 1)*I.reshape(1, 3, 1, 3)
    PP = II - (1/3)*IxI

    # Current deformation gradient
    F = np.eye(3)
    lambda1 = np.exp(strain[-1])
    F[0, 0] = lambda1
    F[1, 1] = lambda2
    F[2, 2] = lambda2
    dt = t[-1]-t[-2]

    # Necessary values
    J = F[0, 0]*F[1, 1]*F[2, 2] #det
    Fbar = J**(-1/3)*F
    Cbar = Fbar.T@Fbar
    I1 = np.trace(Cbar)
    lambda_r = np.sqrt(I1/3/N)
    lang = (3-lambda_r**2)/(1-lambda_r**2)
    bbar = Fbar@Fbar.T

    be = np.array(
        [[b11, 0, 0],
         [0, 1/np.sqrt(b11), 0],
         [0, 0, 1/np.sqrt(b11)]]
    )
    I1_e = np.trace(be)
    lambda_r_e = np.sqrt(I1_e/3/N_v)
    lange = (3-lambda_r_e**2)/(1-lambda_r_e**2)

    tau_v_iso = td(PP, (mu_v/3)*lange*be, 2)
    tau_v = la.norm(tau_v_iso)/np.sqrt(2)
    N11 = tau_v_iso[0, 0]/(tau_v*np.sqrt(2))
    gammadot = sum([aj[j-1]*(tau_v/tau_hat)**j for j in range(1, len(aj)+1)])
    expo = np.exp(-2*gammadot*N11*dt)
    lambda1_prev = np.exp(strain[-2])


    ### 1. Matrix A

    ## 1.1. df_lambda2 and dg_lambda2
    dtauvol_lambda2 = kappa*(2*lambda1*lambda2)*np.eye(3)
    dlang_lambdar = 4*lambda_r/(lambda_r**2 - 1)**2
    dlambdar_I1 = 1/(2*np.sqrt(3*N*I1))
    dI1_lambda2 = (4/3)*(-lambda1**(4/3)*lambda2**(-7/3) + lambda1**(-2/3)*lambda2**(-1/3))
    dbbar_lambda2 = (1/3)*np.array(
        [[-4*lambda1**(4/3)*lambda2**(-7/3), 0, 0],
         [0, 2*lambda1**(-2/3)*lambda2**(-1/3), 0],
         [0, 0, 2*lambda1**(-2/3)*lambda2**(-1/3)]]
    )
    dlambdar_lambda2 = dlambdar_I1*dI1_lambda2
    dtaueiso_lambda2 = td(PP,
                          ((mu/3)*dlang_lambdar*dlambdar_lambda2*bbar + (mu/3)*lang*dbbar_lambda2),
                          2)
    df_lambda2 = dtauvol_lambda2[0, 0] + dtaueiso_lambda2[0, 0]
    dg_lambda2 = dtauvol_lambda2[1, 1] + dtaueiso_lambda2[1, 1]

    ## 1.2. df_b11 and dg_b11
    dlange_lambdare = 4*lambda_r_e/(lambda_r_e**2 - 1)**2
    dlambdare_I1e = 1/(2*np.sqrt(3*N_v*I1_e))
    dI1e_b11 = 1 - 1/(b11**(3/2))
    dlambdare_b11 = dlambdare_I1e*dI1e_b11
    dbe_b11 = np.array(
        [[1, 0, 0],
         [0, -1/(2*b11**(3/2)), 0],
         [0, 0, -1/(2*b11**(3/2))]]
    )
    dtauviso_b11 = td(PP,
                      ((mu_v/3)*dlange_lambdare*dlambdare_b11*be + (mu_v/3)*lange*dbe_b11),
                      2)
    df_b11 = dtauviso_b11[0, 0]
    dg_b11 = dtauviso_b11[1, 1]

    ## 1.3. dh_lambda2
    betr11 = ((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)*b11_prev
    dbetr11_lambda2 = \
    -(4/(3*lambda2))*((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)*b11_prev
    dbe11_lambda2 = expo*dbetr11_lambda2
    dh_lambda2 = dbe11_lambda2

    ## 1.4. dh_b11
    dtauv_tauviso = tau_v_iso/(np.sqrt(2)*la.norm(tau_v_iso))
    dtauv_b11 = td(dtauv_tauviso, dtauviso_b11, 2)
    dgammadot_b11 = (1/tau_hat)*sum([j*aj[j-1]*(tau_v/tau_hat)**(j-1) for j in range(1, len(aj)+1)])*dtauv_b11
    dexpo_N11 = -2*gammadot*dt*expo
    dN11_b11 = \
    (la.norm(tau_v_iso)*dtauviso_b11[0, 0] - (np.sqrt(2)*dtauv_b11)*(tau_v_iso[0, 0]))/(la.norm(tau_v_iso)**2)
    dexpo_gammadot = -2*N11*dt*expo
    dexpo_b11 = dexpo_gammadot*dgammadot_b11 + dexpo_N11*dN11_b11
    dbe11_b11 = dexpo_b11*betr11
    dh_b11 = dbe11_b11

    ## 1.5. Construct the matrix A and A_inv
    A = np.array(
        [[1, -df_lambda2, -df_b11],
         [0, -dg_lambda2, -dg_b11],
         [0, -dh_lambda2, 1-dh_b11]]
    )
    A_inv = la.inv(A)
    ########################
    # print("A: ", A)

    ### 2. kappa

    ## 2.1. b_kappa
    db11prev_kappa = db11prev[0]
    dlambda2prev_kappa = dlambda2prev[0]
    dtauvol_kappa = (J-1)*np.eye(3)
    dbe11_b11prev = expo*((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)
    dbe11_lambda2prev = \
    expo*(4/(3*lambda2_prev))*((lambda1*lambda2_prev)/(lambda2*lambda1_prev))**(4/3)*b11_prev
    df_kappa = dtauvol_kappa[0, 0]
    dg_kappa = dtauvol_kappa[1, 1]
    dh_kappa = 0
    dh_b11prev = dbe11_b11prev
    dh_lambda2prev = dbe11_lambda2prev
    b_kappa = np.array(
        [[df_kappa],
         [dg_kappa],
         [dh_kappa + dh_b11prev*db11prev_kappa + dh_lambda2prev*dlambda2prev_kappa]]
    )

    ## 2.2. solve for d_kappa
    d_kappa = A_inv@b_kappa
    dstress_kappa = d_kappa[0, 0]
    dlambda2_kappa = d_kappa[1, 0]
    db11_kappa = d_kappa[2, 0]

    ### 3. mu

    ## 3.1. b_mu
    db11prev_mu = db11prev[1]
    dlambda2prev_mu = dlambda2prev[1]
    dtaueiso_mu = td(PP,
                     (1/3)*lang*bbar,
                     2)
    df_mu = dtaueiso_mu[0, 0]
    dg_mu = dtaueiso_mu[1, 1]
    dh_mu = 0
    b_mu = np.array(
        [[df_mu],
         [dg_mu],
         [dh_mu + dh_b11prev*db11prev_mu + dh_lambda2prev*dlambda2prev_mu]]
    )

    ## 3.2. solve for d_mu
    d_mu = A_inv@b_mu
    dstress_mu = d_mu[0, 0]
    dlambda2_mu = d_mu[1, 0]
    db11_mu = d_mu[2, 0]
    ###########################
    # print("lang: ", lang)
    # print("bbar: ", bbar)
    # print("dtaueiso_mu:", dtaueiso_mu)
    # print("b_mu: ", b_mu)
    # print("d_mu: ", d_mu)

    ### 4. mu_v

    ## 4.1. b_muv
    db11prev_muv = db11prev[2]
    dlambda2prev_muv = dlambda2prev[2]
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
    df_muv = dtauviso_muv[0, 0]
    dg_muv = dtauviso_muv[1, 1]
    dh_muv = dbe11_muv
    b_muv = np.array(
        [[df_muv],
         [dg_muv],
         [dh_muv + dh_b11prev*db11prev_muv + dh_lambda2prev*dlambda2prev_muv]]
    )

    ## 4.2. solve for d_muv
    d_muv = A_inv@b_muv
    dstress_muv = d_muv[0, 0]
    dlambda2_muv = d_muv[1, 0]
    db11_muv = d_muv[2, 0]
    ######################
    # print("lange: ", lange)
    # print("be: ", be)
    # print("dtauviso_muv:", dtauviso_muv)
    # print("b_muv: ", b_muv)
    # print("d_muv: ", d_muv)

    ### 5. N

    ## 5.1. b_N
    db11prev_N = db11prev[3]
    dlambda2prev_N = dlambda2prev[3]
    dlambdar_N = -(1/(2*N))*np.sqrt(I1/(3*N))
    dtaueiso_N = td(PP, (mu/3)*dlang_lambdar*dlambdar_N*bbar, 2)
    df_N = dtaueiso_N[0, 0]
    dg_N = dtaueiso_N[1, 1]
    dh_N = 0
    b_N = np.array(
        [[df_N],
         [dg_N],
         [dh_N + dh_b11prev*db11prev_N + dh_lambda2prev*dlambda2prev_N]]
    )

    ## 5.2. solve for d_N
    d_N = A_inv@b_N
    dstress_N = d_N[0, 0]
    dlambda2_N = d_N[1, 0]
    db11_N = d_N[2, 0]

    ### 6. N_v

    ## 6.1. b_Nv
    db11prev_Nv = db11prev[4]
    dlambda2prev_Nv = dlambda2prev[4]
    dlambdare_Nv = -(1/(2*N_v))*np.sqrt(I1_e/(3*N_v))
    dtauviso_Nv = td(PP, (mu_v/3)*dlange_lambdare*dlambdare_Nv*be)
    dbe11_Nv = td(dbe11_tauviso, dtauviso_Nv, 2)
    df_Nv = dtauviso_Nv[0, 0]
    dg_Nv = dtauviso_Nv[1, 1]
    dh_Nv = dbe11_Nv
    b_Nv = np.array(
        [[df_Nv],
         [dg_Nv],
         [dh_Nv + dh_b11prev*db11prev_Nv + dh_lambda2prev*dlambda2prev_Nv]]
    )

    ## 6.2. solve for d_Nv
    d_Nv = A_inv@b_Nv
    dstress_Nv = d_Nv[0, 0]
    dlambda2_Nv = d_Nv[1, 0]
    db11_Nv = d_Nv[2, 0]

    ### 7. tau_hat

    ## 7.1. b_tauhat
    db11prev_tauhat = db11prev[5]
    dlambda2prev_tauhat = dlambda2prev[5]
    dgammadot_tauhat = -(1/tau_hat)*sum([j*aj[j-1]*(tau_v/tau_hat)**j for j in range(1, len(aj)+1)])
    dbe11_tauhat = dexpo_gammadot*dgammadot_tauhat*betr11
    df_tauhat = 0
    dg_tauhat = 0
    dh_tauhat = dbe11_tauhat
    b_tauhat = np.array(
        [[df_tauhat],
         [dg_tauhat],
         [dh_tauhat + dh_b11prev*db11prev_tauhat + dh_lambda2prev*dlambda2prev_tauhat]]
    )

    ## 7.2. solve for d_tauhat
    d_tauhat = A_inv@b_tauhat
    dstress_tauhat = d_tauhat[0, 0]
    dlambda2_tauhat = d_tauhat[1, 0]
    db11_tauhat = d_tauhat[2, 0]

    ### 8. aj

    dstress_aj = []
    dlambda2_aj = []
    db11_aj = []
    for j in range(1, len(aj)+1):

        ## 8.1. b_aj
        db11prev_aj = db11prev[5+j]
        dlambda2prev_aj = dlambda2prev[5+j]
        dgammadot_aj = (tau_v/tau_hat)**j
        dbe11_aj = dexpo_gammadot*dgammadot_aj*betr11
        df_aj = 0
        dg_aj = 0
        dh_aj = dbe11_aj
        b_aj = np.array(
            [[df_aj],
            [dg_aj],
            [dh_aj + dh_b11prev*db11prev_aj + dh_lambda2prev*dlambda2prev_aj]]
        )

        ## 8.2. solve for d_aj
        d_aj = A_inv@b_aj
        dstress_aj.append(d_aj[0, 0])
        dlambda2_aj.append(d_aj[1, 0])
        db11_aj.append(d_aj[2, 0])

    # Stress and Recurrent derivatives
    dstress = [dstress_kappa, dstress_mu, dstress_muv, dstress_N, dstress_Nv,
               dstress_tauhat] + dstress_aj
    dlambda2 = [dlambda2_kappa, dlambda2_mu, dlambda2_muv, dlambda2_N, dlambda2_Nv,
                dlambda2_tauhat] + dlambda2_aj
    db11 = [db11_kappa, db11_mu, db11_muv, db11_N, db11_Nv,
            db11_tauhat] + db11_aj
    drecurr_new = dlambda2 + db11
    drecurr.append(drecurr_new)

    return dstress, drecurr

#endregion

#region ##--UNIAXIAL BERGSTROM BOYCE MODIFIED (INCOMPRESSIBLE)--##

def ubbmi_su(t, strain, histvars, params):
    """
    Update stress and history variables using homogenous uniaxial bergstrom boyce model
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
    tau_hat_list = params[2 + 2 * num_branches: 2 + 3 * num_branches]
    aj_list = []
    num_aj = int((len(params) - (2 + 3 * num_branches))/num_branches)
    for branch in range(num_branches):
        aj_list.append(params[2 + 3 * num_branches + branch * num_aj: 2 + 3 * num_branches + (branch + 1) * num_aj])

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
    J = F[0, 0]*F[1, 1]*F[2, 2] #det
    Fbar = J**(-1/3)*F
    Cbar = Fbar.T@Fbar
    I1 = np.trace(Cbar)
    lambda_r = np.sqrt(I1/3/N)
    lang = (3-lambda_r**2)/(1-lambda_r**2)
    bbar = Fbar@Fbar.T
    C = F.T@F
    lambda_a_sq, N_a = la.eig(C) # !!!!!!!!!
    lambda_a = np.sqrt(lambda_a_sq).reshape(-1, 1)

    ## Unimodular elastic kirchoff stress
    taubar_e = (mu*lang/3)*bbar

    ## Unimodular viscous kirchoff stress (Newton update, each branch)
    F_prev = np.eye(3)
    F_prev[0, 0] = np.exp(strain[-2])
    F_prev[1, 1] = 1/np.sqrt(F_prev[0, 0])
    F_prev[2, 2] = 1/np.sqrt(F_prev[0, 0])
    J_prev = F_prev[0, 0]*F_prev[1, 1]*F_prev[2, 2] #det
    Fbar_prev = J_prev**(-1/3)*F_prev

    b11_new_list = [0]*num_branches
    taubar_v_list = [np.eye(3)]*num_branches
    for branch, (mu_v, N_v, tau_hat, aj, b11_prev) in enumerate(zip(mu_v_list, N_v_list, tau_hat_list, aj_list, b11_prev_list)):

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

            # Unimodular viscous kirchoff stress
            taubar_v = (mu_v*lang_e/3)*be
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
            T = (2/3)*mu_v*((3-lambda_r_e**2)/(1-lambda_r_e**2))*np.diag((lambda_a**2).reshape(3)) - \
            (4/9)*(mu_v/N_v)*(1/(1-lambda_r_e**2))*(lambda_a**2)*(lambda_a**2).reshape(3)
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
    tau_hat_list = params[2 + 2 * num_branches: 2 + 3 * num_branches]
    aj_list = []
    num_aj = int((len(params) - (2 + 3 * num_branches))/num_branches)
    for branch in range(num_branches):
        aj_list.append(params[2 + 3 * num_branches + branch * num_aj: 2 + 3 * num_branches + (branch + 1) * num_aj])

    # History variables from previous timestep
    b11_prev_list = histvars[-2]

    # History variables from current timestep
    b11_list = histvars[-1]

    # Initialize recurrent_derivatives if needed
    if len(drecurr) == 0:
        # All the viscous parameters with their respective b11s
        db11_init = [0 for j in range(len(params) - 2)]
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
    J = F[0, 0]*F[1, 1]*F[2, 2] #det
    Fbar = J**(-1/3)*F
    Cbar = Fbar.T@Fbar
    I1 = np.trace(Cbar)
    lambda_r = np.sqrt(I1/3/N)
    lang = (3-lambda_r**2)/(1-lambda_r**2)
    bbar = Fbar@Fbar.T
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

    # FOR EACH BRANCH
    dstress_muv_list = []
    dstress_Nv_list = []
    dstress_tauhat_list = []
    dstress_aj_list = []
    db11 = []
    for branch, (mu_v, N_v, tau_hat, aj, b11, b11_prev) in \
        enumerate(zip(mu_v_list, N_v_list, tau_hat_list, aj_list, b11_list, b11_prev_list)):

        # Necessary value for the branch
        be = np.array(
            [[b11, 0, 0],
             [0, 1/np.sqrt(b11), 0],
             [0, 0, 1/np.sqrt(b11)]]
        )
        I1_e = np.trace(be)
        lambda_r_e = np.sqrt(I1_e/3/N_v)
        lange = (3-lambda_r_e**2)/(1-lambda_r_e**2)

        tau_v_iso = td(PP, (mu_v/3)*lange*be, 2)
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
        dtauviso_b11 = td(PP,
                          ((mu_v/3)*dlange_lambdare*dlambdare_b11*be + (mu_v/3)*lange*dbe_b11),
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
        num_branch_params = 3 + len(aj)

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

        ## 5. tau_hat

        # 5.1. b_tauhat
        db11prev_tauhat = db11prev[num_branch_params*branch + 2]
        dgammadot_tauhat = -(1/tau_hat)*sum([j*aj[j-1]*(tau_v/tau_hat)**j for j in range(1, len(aj)+1)])
        dbe11_tauhat = dexpo_gammadot*dgammadot_tauhat*betr11
        df_tauhat = 0
        dg_tauhat = dbe11_tauhat
        b_tauhat = np.array(
            [[df_tauhat],
             [dg_tauhat + dg_b11prev*db11prev_tauhat]]
        )

        # 5.2. solve for d_tauhat
        d_tauhat = A_inv@b_tauhat
        dstress_tauhat_list.append(d_tauhat[0, 0])
        db11.append(d_tauhat[1, 0])

        ## 6. aj

        dstress_aj = []
        db11_aj = []
        for j in range(1, len(aj)+1):

            # 6.1. b_aj
            db11prev_aj = db11prev[num_branch_params*branch + 2 + j]
            dgammadot_aj = (tau_v/tau_hat)**j
            dbe11_aj = dexpo_gammadot*dgammadot_aj*betr11
            df_aj = 0
            dg_aj = dbe11_aj
            b_aj = np.array(
                [[df_aj],
                [dg_aj + dg_b11prev*db11prev_aj]]
            )

            # 6.2. solve for d_aj
            d_aj = A_inv@b_aj
            dstress_aj.append(d_aj[0, 0])
            db11_aj.append(d_aj[1, 0])

        dstress_aj_list.append(dstress_aj)
        db11.extend(db11_aj)

    # Stress and Recurrent derivatives
    dstress = [dstress_mu] + dstress_muv_list + [dstress_N] + dstress_Nv_list + \
               dstress_tauhat_list + [item for row in dstress_aj_list for item in row]
    drecurr_new = db11
    drecurr.append(drecurr_new)

    return dstress, drecurr

#endregion

#region ##--UNIAXIAL BERGSTROM BOYCE MODIFIED 2 (INCOMPRESSIBLE)--##

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
    num_branches = 10
    mu = params[0]
    mu_v_list = params[1: 1 + num_branches]
    N = params[1 + num_branches]
    N_v_list = params[2+num_branches: 2 + 2 * num_branches]
    c = params[2 + 2 * num_branches]
    c_v_list = params[3 + 2 * num_branches: 3 + 3 * num_branches]
    tauhinv_list = params[3 + 3 * num_branches: 3 + 4 * num_branches]
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
    for branch, (mu_v, N_v, c_v, tauhinv, aj, b11_prev) in \
        enumerate(zip(mu_v_list, N_v_list, c_v_list, tauhinv_list, aj_list, b11_prev_list)):

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
            gamma_dot = sum([aj[j-1]*(tauhinv*tau_v)**j for j in range(1, len(aj)+1)])

            res = eps_a + dt*gamma_dot*devtau_a/np.sqrt(2)/tau_v - eps_a_tr

            # Local tangent
            beta1 = (dt/2/np.sqrt(2))*(tauhinv**3)*sum([aj[j-1]*(j-1)*(tauhinv*tau_v)**(j-3) for
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

def ubbm2i_du(t, strain, histvars, stress_new, drecurr, params):
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
    num_branches = 10
    mu = params[0]
    mu_v_list = params[1: 1 + num_branches]
    N = params[1 + num_branches]
    N_v_list = params[2+num_branches: 2 + 2 * num_branches]
    c = params[2 + 2 * num_branches]
    c_v_list = params[3 + 2 * num_branches: 3 + 3 * num_branches]
    tauhinv_list = params[3 + 3 * num_branches: 3 + 4 * num_branches]
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
    dstress_tauhinv_list = []
    dstress_aj_list = []
    db11 = []
    for branch, (mu_v, N_v, c_v, tauhinv, aj, b11, b11_prev) in \
        enumerate(zip(mu_v_list, N_v_list, c_v_list, tauhinv_list, aj_list, b11_list, b11_prev_list)):

        # Necessary value for the branch
        be = np.array(
            [[b11, 0, 0],
             [0, 1/np.sqrt(b11), 0],
             [0, 0, 1/np.sqrt(b11)]]
        )
        I1_e = np.trace(be)
        lambda_r_e = np.sqrt(I1_e/3/N_v)
        lange = (3-lambda_r_e**2)/(1-lambda_r_e**2)

        tau_v_iso = td(PP, (mu_v/3)*lange*be + 2*c_v*(I1_e*be - be@be), 2)
        tau_v = la.norm(tau_v_iso)/np.sqrt(2)
        N11 = tau_v_iso[0, 0]/(tau_v*np.sqrt(2))
        gammadot = sum([aj[j-1]*(tauhinv*tau_v)**j for j in range(1, len(aj)+1)])
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
        dgammadot_b11 = (tauhinv)*sum([j*aj[j-1]*(tauhinv*tau_v)**(j-1) for j in range(1, len(aj)+1)])*dtauv_b11
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
        dgammadot_tauviso = (tauhinv)*sum([j*aj[j-1]*(tauhinv*tau_v)**(j-1) for
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

        ## 6. tauhinv

        # 6.1. b_tauhinv
        db11prev_tauhinv = db11prev[num_branch_params*branch + 3]
        dgammadot_tauhinv = (1/tauhinv)*sum([j*aj[j-1]*(tauhinv*tau_v)**j for j in range(1, len(aj)+1)])
        dbe11_tauhinv = dexpo_gammadot*dgammadot_tauhinv*betr11
        df_tauhinv = 0
        dg_tauhinv = dbe11_tauhinv
        b_tauhinv = np.array(
            [[df_tauhinv],
             [dg_tauhinv + dg_b11prev*db11prev_tauhinv]]
        )

        # 6.2. solve for d_tauhinv
        d_tauhinv = A_inv@b_tauhinv
        dstress_tauhinv_list.append(d_tauhinv[0, 0])
        db11.append(d_tauhinv[1, 0])

        ## 7. aj

        dstress_aj = []
        db11_aj = []
        for j in range(1, len(aj)+1):

            # 7.1. b_aj
            db11prev_aj = db11prev[num_branch_params*branch + 3 + j]
            dgammadot_aj = (tauhinv*tau_v)**j
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
    dstress = [dstress_mu] + dstress_muv_list + [dstress_N] + dstress_Nv_list + [dstress_c] + \
        dstress_cv_list + dstress_tauhinv_list + [item for row in dstress_aj_list for item in row]
    drecurr_new = db11
    drecurr.append(drecurr_new)

    return dstress, drecurr

#endregion
