# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:53:52 2025

Adapted from DANTE.py

It creates, saves and plots the Equation of State of a degenerate free Fermi gass of mass 'm'

@author: Diego García Tejada
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines
import pandas as pd

# Define the functions

def calc_CDIFG (m):
    """
    Function that calculates the numerical values of the Relativistic Equation of State for a
    Completley Degenerate, Ideal Fermi Gas of particle mass = m.
    
    Due to numerical issues for values of P lower than 10^-12 it uses the 21st order taylor expansion.
    
    Parameters
    ----------
    m : float
        Mass of the particle mass in GeV.
        
    Returns
    -------
    Pandas DataFrame
        DF containing all the datapoints of EoS.
    """
    def phi(x):
        return np.sqrt(1 + x**2) * ((2/3) * x**3 - x) + np.log(x + np.sqrt(1 + x**2))
    
    def psi(x):
        return np.sqrt(1 + x**2) * (2 * x**3 + x) - np.log(x + np.sqrt(1 + x**2))
        
    def phi_21st(x):
        return (8/15) * x**5 - (4/21) * x**7 + (1/9) * x**9 - (5/66) * x**11 + (35/624) * x**13 - (7/160) * x**15 + (77/2176) * x**17 - (143/4864) * x**19 + (715/28672) * x**21
    
    def psi_21st(x):
        return (8/3) * x**3 + (4/5) * x**5 - (1/7) * x**7 + (1/18) * x**9 - (5/176) * x**11 + (7/416) * x**13 - (7/640) * x**15 + (33/4352) * x**17 - (429/77824) * x**19 + (715/172032) * x**21
    
    k = (m)**(4) * 1.4777498161008e-3
        
    x_vals = np.geomspace(1e-6, 10, 500)
    
    pressures = []
    densities = []
    
    for x in x_vals:
        P_analytical = k * phi(x)
        
        if P_analytical >= 1e-12:
            P = P_analytical
            rho = k * psi(x)
        else:
            P = k * phi_21st(x)
            rho = k * psi_21st(x)
            
        pressures.append(P)
        densities.append(rho)
        
    df = pd.DataFrame({'rho': densities,'P': pressures})
    
    df.to_excel(rf'data_eos\eos_cdifg_{m}.xlsx', index=False)
        
    return df

def read_create_dm_eos (m):
    """
    Reads an equation of state (EOS) file in Excel format and extracts density 
    and pressure data of the dark matter.
    
    If the file is not found it creates one using the calc_CDIFG function.
    
    Parameters
    ----------
    m : float
        mass of the dark matter particle in GeV.
        
    Returns
    -------
    p_data : numpy.ndarray
        Array containing pressure values extracted from the EOS file.
    rho_data : numpy.ndarray
        Array containing density values extracted from the EOS file.
    """
    try:
        eos_data = pd.read_excel(f'data_eos\eos_cdifg_{m}.xlsx')
        rho_data = eos_data['rho'].values
        p_data = eos_data['P'].values
    except FileNotFoundError:
        eos_data = calc_CDIFG(m)
        rho_data = eos_data['rho'].values
        p_data = eos_data['P'].values
        
    return p_data, rho_data

# Compute the EoS



