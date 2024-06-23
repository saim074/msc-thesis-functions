# %%

from mat import stress_update

import matplotlib.pyplot as plt, numpy as np

def generate_mdata(t, strain, params, mat,
                   plot=True, noise=0):

    """
    Generate stress vector for the given loading
    Inputs
        t       : list of timestamps upto the current timestep
        strain  : list of strain values upto the current timestep
        params  : list of material parameters
        mat     : string identifying which material model to use
        plot    : if True, plot the time vs strain, time vs stress,
                  strain vs stress plots
        noise   : parameter to add noise
    Output
        mdata  : a tuple with time, strain and calculated stress
                 (t, strain, stress)
    """

    # Initialize empty list for history variables
    histvars = []

    # Initialize list for stress
    stress = [0]

    for n in range(1, len(t)):
        # Stress Update
        stress_new, histvars = stress_update(t[:n+1], strain[:n+1],
                                             histvars, params, mat)
        stress_new += np.random.normal(0, noise)
        stress.append(stress_new)

    # Plot
    if plot:
        plt.figure(figsize = (20, 7))

        plt.subplot(1, 3, 1)
        plt.scatter(t, strain, s = 1)
        plt.xlabel("Time", fontsize = 15)
        plt.ylabel("Strain", fontsize = 15)

        plt.subplot(1, 3, 2)
        plt.scatter(t, stress, s=1)
        plt.xlabel("Time", fontsize = 15)
        plt.ylabel("Stress", fontsize = 15)

        plt.subplot(1, 3, 3)
        plt.scatter(strain, stress, s=1)
        plt.xlabel("Strain", fontsize = 15)
        plt.ylabel("Stress", fontsize = 15)

    return (t, strain, stress)

def sawtooth_loading(max_val, min_val, start, rate,
                     cycles = 1, dt = 1, 
                     plot=True):
    """
    Generates time and y values for a sawtooth loading
    Inputs
        max_val: maximum value
        min_val: minimum value
        start  : start value (0 for strain, 1 for stretch)
        rate   : absolute rate of loading
        cycles : number of cycles
        dt     : approximate delta time
        plot   : if True, plot time vs y
    Outputs
        t: time values
        y: cyclic load values, either stretch or strain
    """

    # Loading
    diff = max_val - start
    diff_t = diff/rate
    steps = int(np.ceil(diff_t/dt))
    t1 = np.linspace(0, diff_t, steps+1)
    y1 = start + rate * t1
    # Unloading
    diff = max_val - min_val
    diff_t = diff/rate
    steps = int(np.ceil(diff_t/dt))
    t2 = (np.linspace(t1[-1], diff_t + t1[-1], steps+1))[1:]
    y2 = max_val - rate * (t2 - t1[-1])
    # Loading
    diff = start - min_val
    diff_t = diff/rate
    steps = int(np.ceil(diff_t/dt))
    t3 = (np.linspace(t2[-1], diff_t + t2[-1], steps+1))[1:]
    y3 = min_val + rate * (t3 - t2[-1])

    # Number of cycles
    t = np.array([])
    y = np.array([])
    t_cycle = t3[-1]
    for c in range(cycles):
        t = np.concatenate([t, 
                            t1 + t_cycle*c, 
                            t2 + t_cycle*c, 
                            t3 + t_cycle*c])
        y = np.concatenate([y, y1, y2, y3])


    # Plot
    if plot:
        plt.plot(t, y)

    return t, y

def cyclic_loading(max_val, min_val, start, freq,
                   cycles = 1, dt = 1,
                   plot=True):
    """
    Generates time and y values for a cyclic loading
    Inputs
        max_val: maximum value
        min_val: minimum value
        start  : start value (0 for strain, 1 for stretch)
        freq   : frequency of the cycles
        cycles : list of material parameters
        dt     : delta time
        plot   : if True, plot time vs y
    Output
        t: time values
        y: cyclic load values, either stretch or strain
    """

    # Time and stretch array
    mid = (max_val+min_val)/2
    amp = (max_val-min_val)/2
    offset = np.arcsin((start-mid)/amp)/2/np.pi/freq
    datapoints = int(cycles/freq/dt)
    t = np.linspace(0, cycles/freq, datapoints)
    y = mid + amp*np.sin(2*np.pi*freq*(t+offset))

    # Plot
    if plot:
        plt.plot(t, y)

    return t, y

def linear_loading(start_val, end_val, start_time, rate, dt = 1, plot=True):
    """
    Generates time and y values for a linear loading
    Inputs
        start_val : maximum value
        end_val   : minimum value
        start_time: start value (0 for strain, 1 for stretch)
        rate      : frequency of the cycles
        dt        : delta time
        plot      : if True, plot time vs y
    Output
        t: time values
        y: cyclic load values, either stretch or strain
    """

    # Time and stretch array
    time_taken = (end_val - start_val)/rate
    datapoints = int(time_taken/dt)
    t = np.linspace(start_time, start_time+time_taken, datapoints)
    y = np.linspace(start_val, end_val, datapoints)

    # Plot
    if plot:
        plt.plot(t, y)

    return t, y

def relaxation_loading(val, start_time, time_period, dt = 1, plot=True):
    """
    Generates time and y values for a cyclic loading
    Inputs
        val        : the constant value
        time_period: how long to relax
        start_time : start value (0 for strain, 1 for stretch)
        dt         : delta time
        plot       : if True, plot time vs y
    Output
        t: time values
        y: cyclic load values, either stretch or strain
    """

    # Time and stretch array
    datapoints = int(time_period/dt)
    t = np.linspace(start_time, start_time+time_period, datapoints)
    y = np.array([val for i in range(datapoints)])

    # Plot
    if plot:
        plt.plot(t, y)

    return t, y