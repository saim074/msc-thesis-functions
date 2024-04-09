# RNN Based Model Discovery of Finite Viscoelasticity

This repository holds the code and data that is used for the article **RNN Based Model Discovery of Finite Viscoelasticity** (currently in preparation)

## Overview

We develop a framework that automatically discovers an accurate model of finite viscoelasticity and identifies a set of physically meaningful parameters in the process. We use a recurrent neural network (RNN) architecture to represent the stress-update that a priori satisfies necessary physical constraints. For solving the nonlinear Ordinary Differential Equations (ODEs) that govern the internal variables' evolution, our approach utilizes an implicit backward Euler scheme, enabling model training with very sparse data.

## File Structure

* Python Scripts
    * `optimize.py` : Contains functions for the 
        * calculation of loss and its derivatives
        * parameter update step
        * evaluation of the fit of the trained model
    * `mat.py` : Functions for the stress and derivative update of all the material models
    * `synthetic_data.py` : Functions to generate synthetic data
* Experiment Folders
    * **Synthetic Data** : Contains  a Jupyter notebook detailing the generation of synthetic data and training of the model.
    * **VHB4910 Data**: Contains data from [[1]](#1) for the VHB4910 polymer and a Jupyter notebook detailing the training and evaluation of the model.

## Usage
Please refer to the Jupyter Notebooks in the experiments folder.

## Authors

* Saim Masood - M.Sc. student - Dept. of Civil Engg. - METU, Ankara (saim.masood@metu.edu.tr)
* Serdar Göktepe - Associate Professor - Dept. of Civil Engg. - METU, Ankara (sgoktepe@metu.edu.tr)

## References

<a id="1">[1]</a>
    Hossain, Mokarram; Vu, D. K.; Steinmann, P. [2012]: *Experimental study and numerical
    modelling of VHB 4910 polymer.* Computational Materials Science, 59: 65–74.

