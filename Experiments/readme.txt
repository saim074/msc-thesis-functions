ubbi: Uniaxial Bergstrom Boyce (Incompressible)

- Experiment 1: Data from dal+kaliske paper fitted on same model
- Experiment 2: Data from goktepe+miehe05 paper Fig7b (slowest rate) fitted on modified bergstrom boyce
- Experiment 3: Data from goktepe+miehe05 paper Fig7c and 7d (intermediate and fastest) fitted on modified bergstrom boyce
- Experiment 4: Fig7d (Highest rate, largest hysteresis) fitted to increasing number of power terms

TO-DO:
    - Make animations
    - Fit Fig7d to simple bergstrom boyce

##--EXPERIMENT 1--##

    Data:
        Synthetic data generated using ubbi parameters from dal+kaliske09 paper
        Two sets of compression mdata with true strain rates -0.002/s and -0.1/s
            with relaxation of 120 s at true strain = 0.3 and 0.6
        Used large dt_relaxation and dt_linear to make data sparse

    Fitting:
        Fitted to ubbi perfectly

##--EXPERIMENT 2--##

    Data:
        Experimental data from paper
        Fig 7b
        Data made sparser by taking every fifth point

    Fitting:
        No parameters fixed, aj to the power of 4
        Evaluated on the same data, plot in results for P11 vs stretch and true strain vs true stress

##--EXPERIMENT 3--##

    Data:
        Experimental data from paper
        Fig 7c (visco3_2) rate = 0.5/min and d(visco2_2) rate=5/min
        Data made sparser by taking every fifth point

    Fitting:
        No parameters fixed, aj to the power of 4
        Evaluated on the same data, plot in results for P11 vs stretch and true strain vs true stress

    Comments:
        So it seems that the highest nonlinearity and worst fit comes from Fig7d where the rate is highest

##--EXPERIMENT 4--##

    Data:
        Experimental data from paper
        Fig 7c (visco3_2) rate = 0.5/min and d(visco2_2) rate=5/min
        Data made sparser by taking every fifth point

    Fitting:
        No parameters fixed, aj increasing from 1 to 9
        Evaluated on the same data, plot in results for P11 vs stretch and true strain vs true stress

    Comments:
        The increase in the power terms did not help much with the plot