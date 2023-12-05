#%%

from mat import stress_update, derivative_update

import matplotlib.pyplot as plt, numpy as np

def evaluate(mdata, params, mat):

    """
    Plots the prediction's fit to actual stress-strain data
    Inputs
        mdata : list of data with each element a tuple of lists
                [(t, strain, stress), ...]
        params: list of material parameters
        mat   : string identifying which material model to use
    """

    # Does the result fit the stress-strain curve well?
    for t, strain, stress in mdata:

        # Initialize empty list for history variables
        histvars = []

        # Initialize list for predicted stress
        stress_hat = [0]

        for n in range(1, len(t)):
            # Stress Update
            stress_new, histvars = stress_update(t[:n+1], strain[:n+1],
                                                 histvars, params, mat)
            stress_hat.append(stress_new)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        # plt.plot(t, stress_hat, label = "Predicted")
        # plt.scatter(t, stress, label = "True", s=15, c="orange")
        # plt.xlabel("Time")
        # plt.ylabel("Stress")
        plt.plot(t, stress_hat/np.exp(strain), label = "Predicted")
        plt.scatter(t, stress/np.exp(strain), label = "True", s=15, c="orange")
        plt.xlabel("Time")
        plt.xlabel("P11")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(strain, stress_hat, label = "Predicted")
        plt.scatter(strain, stress, label = "True", s=15, c="orange")
        # plt.plot(np.exp(strain), stress_hat/np.exp(strain), label = "Predicted")
        # plt.scatter(np.exp(strain), stress/np.exp(strain), label = "True", s=15, c="orange")
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.legend()

    return

def optimize(mdata, params_init, params_names, mat, max_iter=5000, print_after=100, plot_after = 100, tol=10e-6,
             alpha=1e-3, b1=0.9, b2=0.999, e=1e-8, lambda_1 = 0,
             fix_param = False, get_lowest=True):

    """
    Finds the optimum value for the parameters using gradient descent
    Inputs
        mdata       : list of data with each element a tuple of lists
                      [(t, strain, stress), ...]
        params_init : initial values for the material parameters, a list
        params_names: names of the material parameters for printing, a list
        mat         : string identifying which material model to use
        max_iter    : maximum number of iterations to run
        print_after : print the values after this many iterations
        plot_after  : plot the fit after this many iterations
        tol         : params change tolerance (params_new/params_prev - 1)<tol for stopping
        alpha       : learning rate (Adam)
        b1          : decay constant 1 (Adam)
        b2          : decay constant 2 (Adam)
        e           : small positive constant to avoid division by zero (Adam)
        lambda_1    : a single or a list of values of L1 regularization for each parameter
        fix_param   : a single or a list of booleans to fix parameters
        get_lowest  : if True, get the parameters that correspond to the lowest loss value
    Outputs
        params_out : final values of the parameters
        params_hist: history of parameters
        loss_hist  : history of loss values
    """

    # Initial guesses
    params_hist = [params_init]

    # Print initial values
    print("Initial values")
    for j in range(len(params_init)):
        print(params_names[j], " :", params_hist[-1][j])

    # Initialize values for v and w (Adam)
    v = [0 for j in range(len(params_init))]
    m = [0 for j in range(len(params_init))]

    loss_hist = []
    for i in range(max_iter):

        # Current and new parameters
        params_curr = params_hist[-1]
        params_new = [0 for j in range(len(params_init))]

        #Calculate loss and derivatives
        L, dL = calc_loss_and_derivatives(mdata, params_curr, mat)

        #Update parameters
        for j in range(len(params_init)):

            # Check if we should update the parameter
            if type(fix_param) == list:
                fix_param_j = fix_param[j]
            else:
                fix_param_j = fix_param

            if type(lambda_1) == list:
                lambda_1_j = lambda_1[j]
            else:
                lambda_1_j = lambda_1

            # Update parameter
            if fix_param_j:
                params_new[j] = params_curr[j]
            else:
                # dL[j] = dL[j] + lambda_1_j*np.sign(params_curr[j])
                m[j] = b1*m[j] + (1-b1)*dL[j]
                v[j] = b2*v[j] + (1-b2)*dL[j]**2
                m_hat = m[j]/(1-b1**(i+1))
                v_hat = v[j]/(1-b2**(i+1))
                # params_new[j] = params_curr[j] - alpha*(m_hat/(np.sqrt(v_hat)+e))
                params_new[j] = params_curr[j] - alpha*((m_hat/(np.sqrt(v_hat)+e)) + lambda_1_j*np.sign(params_curr[j]))

        # Append to histories
        params_hist.append(params_new)
        loss_hist.append(L)

        # Print
        if (i+1) % print_after == 0:
            print("Loss: ", loss_hist[-1], "\n")
            print("iteration number: ", i)
            for j in range(len(params_init)):
                print(params_names[j], " :", params_hist[-1][j])

        # Plot
        if (i+1) % plot_after == 0:
            evaluate(mdata, params_hist[-1], mat)
            plt.show()

        # Diverged
        if (np.isnan(params_hist[-1])).any():
            print("\n\nFAILED\n\n")
            print("Loss: ", loss_hist[-1], "\n")
            print("iteration number: ", i)
            for j in range(len(params_init)):
                print(params_names[j], " :", params_hist[-1][j])
            break

        # Converged
        if len(loss_hist)>2:
            if np.max(abs(np.array(params_hist[-1])/np.array(params_hist[-2]) - 1))<tol:
                print("\n\nCONVERGED\n\n")
                print("Loss: ", loss_hist[-1], "\n")
                print("iteration number: ", i)
                for j in range(len(params_init)):
                    print(params_names[j], " :", params_hist[-2][j])
                break

    # Should we get the value with the lowest loss
    if get_lowest:
        print("\n\nLOWEST LOSS")
        iter_low = np.argmin(np.array(loss_hist))
        params_out = params_hist[iter_low]
        print("iteration number: ", iter_low)
        for j in range(len(params_init)):
            print(params_names[j], " :", params_out[j])
        print("Loss: ", loss_hist[iter_low], "\n")
    else:
        print("\n\nLAST VALUE")
        params_out = params_hist[-1]
        print("iteration number: ", i)
        for j in range(len(params_init)):
            print(params_names[j], " :", params_out[j])
        print("Loss: ", loss_hist[-1], "\n")


    return params_out, params_hist, loss_hist

def calc_loss_and_derivatives(mdata, params, mat):

    """
    Calculate loss and its derivatives wrt to the learnable parameters
    Inputs
        mdata : list of data with each element a tuple of lists
                      [(t, strain, stress), ...]
        params: list of material parameters
        mat   : string identifying which material model to use
    Output
        L : loss at current iteration
        dL: list of derivatives of loss wrt each material parameter
    """

    #Total number of samples
    num_samples = 0

    #Calculate the Loss and its derivatives
    L = 0
    dL = [0 for j in range(len(params))]

    #For each loading set in material data
    for t, strain, stress in mdata:

        # Initialize empty lists for history variables and recurrent derivatives
        histvars = []
        drecurr = []

        for n in range(1, len(t)):

            # Stress Update
            stress_new, histvars= stress_update(t[:n+1], strain[:n+1],
                                                histvars, params, mat)

            # Derivatives Update
            dstress, drecurr = derivative_update(t[:n+1], strain[:n+1],
                                              histvars, stress_new,
                                              drecurr, params, mat)

            #Add to the loss term
            L += (stress_new - stress[n])**2

            #Loss derivatives
            for j in range(len(params)):
                dL[j] += (stress_new - stress[n])*dstress[j]


        #Num of samples
        num_samples += len(t)

    #Divide by number of samples
    L = L/(2*num_samples)
    for j in range(len(params)):
        dL[j] = dL[j]/num_samples

    return L, dL