# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 23:11:15 2025

Comparison between the stellar structure equations of Newtonian gravity and General Relativity. Both of them solve a stellar model of a neutron star made of an incompressible fluid.

@author: Diego García Tejada
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines

# Physical parameters (solar mass = 1.98847e30 kg)
G = 1.4765679173556 # G in units of km / solar masses
rho_0 = 5e-4 # Representative density for a neutron star

# Define functions for ease of generalization
def newtonian_structure_profiles(rho, M):
    """
    Compute the radial, mass, and pressure profiles of a Newtonian
    constant–density star.

    Parameters
    ----------
    rho : float
        Constant mass density of the star.
    M : float
        Total mass of the star.

    Returns
    -------
    r : ndarray
        Radial coordinates from the center (0) to the stellar radius R,
        sampled with 1000 points.
    m : ndarray
        Enclosed mass profile m(r) = (4/3)πρ r³.
    p : ndarray
        Pressure profile p(r) = (2/3)πρ² (R² − r²), obtained from the
        Newtonian hydrostatic equilibrium equation.
    """
    R = (3/4 * M/(rho*np.pi))**(1/3)
    r = np.linspace(0, R, 1000)
    m = 4/3 * np.pi * rho * r**3
    p = 2/3 * np.pi * rho**2 * (R**2-r**2)

    return (r, m, p)

def GR_structure_profiles(rho, M):
    """
       Compute the radial, mass, and pressure profiles of a Newtonian
    constant–density star.

    Parameters
    ----------
    rho : float
        Constant mass density of the star.
    M : float
        Total mass of the star.

    Returns
    -------
    r : ndarray
        Radial coordinates from the center (0) to the stellar radius R,
        sampled with 1000 points.
    m : ndarray
        Enclosed mass profile m(r) = (4/3)πρ r³.
    p : ndarray
        Pressure profile p(r) = (2/3)πρ² (R² − r²), obtained from the
        Newtonian hydrostatic equilibrium equation.
    """
    R = (3/4 * M/(rho*np.pi))**(1/3)
    r = np.linspace(0, R, 1000)
    m = 4/3 * rho * np.pi * r**3
    term1 = np.sqrt(R**3 - 2 * G * M * R**2)
    term2 = np.sqrt(R**3 - 2 * G * M * r**2)
    numerator = term1 - term2
    denominator = term2 - 3 * term1
    p = rho * (numerator / denominator)
    
    return (r, m, p)

def build_solution_dict(M_list, rho):
    """
    Compute Newtonian and GR (TOV) structure profiles for a list of masses
    and store them in a nested dictionary.

    Parameters
    ----------
    M_list : iterable of float
        List of stellar masses to evaluate.
    rho : float
        Constant density of the star.

    Returns
    -------
    dict
        Dictionary with entries:
            sols[M]["r"]["N" or "E"]  → radial profiles
            sols[M]["m"]["N" or "E"]  → mass profiles
            sols[M]["p"]["N" or "E"]  → pressure profiles
        where "N" is Newtonian and "E" is GR.
    """
    sols = {}
    
    for M in M_list:
        # Newtonian solution
        r_N, m_N, p_N = newtonian_structure_profiles(rho, M)

        # GR (TOV) solution
        r_E, m_E, p_E = GR_structure_profiles(rho, M)

        # Store in nested dictionary
        sols[M] = {
            "r": {"N": r_N, "E": r_E},
            "m": {"N": m_N, "E": m_E},
            "p": {"N": p_N, "E": p_E},
        }

    return sols

# Compute the solutions
M_list = [1, 2, 3, 3.6]
rho = 5e-4
data = build_solution_dict(M_list, rho)

# Plot the solutions
fig, ax = plt.subplots(figsize=(9.71, 6))
colors = ['r', 'b', 'g', 'm']

c=0
for M in M_list: 
    ax.plot(data[M]['r']['E'], data[M]['p']['E'], color = colors[c], linewidth=1.5, linestyle='-')
    ax.plot(data[M]['r']['N'], data[M]['p']['N'], color = colors[c], linewidth=1.5, linestyle=(c*2, [6, 4]))
    c+=1
ax.set_xlabel(r'$r$ $\left[km\right]$', fontsize=15, loc='center')
ax.set_ylabel(r'$p$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center', color='k')
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
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
ax.set_xlim(0, 12.05)
ax.set_ylim(1e-8, 1e-1)

# Set ticks
ax.set_xticks(np.arange(0, 12.01, 1))

# Write the legend
handles_list = [
    mlines.Line2D([], [], color='k', linestyle='-', linewidth=1.5, label=r"TOV"),
    mlines.Line2D([], [], color='k', linestyle=(1, [6, 4]), linewidth=1.5, label=r"Newton"),
    mlines.Line2D([], [], color=colors[0], linestyle='-', linewidth=1.5, label=r"$M=1$"),
    mlines.Line2D([], [], color=colors[1], linestyle='-', linewidth=1.5, label=r"$M=2$"),
    mlines.Line2D([], [], color=colors[2], linestyle='-', linewidth=1.5, label=r"$M=3$"),
    mlines.Line2D([], [], color=colors[3], linestyle='-', linewidth=1.5, label=r"$M=3.6$")
    ]
    
plt.legend(handles=handles_list, fontsize=15, loc = "upper right", bbox_to_anchor=(0.994, 0.99), frameon=True, fancybox=False,
           ncol = 3,edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)

plt.tight_layout()
plt.savefig(rf"comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()