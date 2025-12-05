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
from scipy.interpolate import interp1d 

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

G = 1.4765679173556 # in units of km / solar masses
m = 0.939565 # Neutron mass in GEV
p, rho = read_create_dm_eos(m)

# Plot the EoS

fig, ax = plt.subplots(figsize=(9.71, 6))

ax.plot(rho, p, label='Fermi Gas Eos', color = 'r', linewidth=1.5, linestyle='-')
ax.set_xlabel(r'$\rho$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center')
ax.set_ylabel(r'$p$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center')
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
ax.set_xscale('log')
ax.set_yscale('log')

# Configure ticks
ax.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
ax.minorticks_on()

# Set thicker axes
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# Set limits
ax.set_xlim(1e-13, 1e2)
ax.set_ylim(1e-20, 1e2)

# Set ticks
ax.set_xticks(np.geomspace(1e-13, 1e1, 15))
ax.set_yticks(np.geomspace(1e-19, 1e1, 11))

# Write the legend
plt.legend(fontsize=15, loc = "upper left", bbox_to_anchor=(0.009, 0.99), frameon=True, fancybox=False, ncol = 1,
           edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)

# Save the figure
plt.tight_layout()
plt.savefig(rf"figures\EoS_{m}.pdf", format="pdf", bbox_inches="tight")
plt.show()