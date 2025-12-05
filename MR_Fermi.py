# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 18:08:30 2025

From the data obtained from DANTE.py we construct the mass radius relation for a neutron star made up of a free Fermi gas of neutrons

@author: Diego García Tejada
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
import scipy.optimize as opt
import multiprocessing as mp
from tqdm import tqdm

# Physical parameters (solar mass = 1.98847e30 kg)
G = 1.4765679173556 # G in units of km / solar masses

s_type = 2
d_type = 1
dm_m = 0.93956542

# Read the data
df = pd.read_csv(rf"data\{s_type}_{d_type}_{dm_m}.csv")

# Configure the plot
fig, ax1 = plt.subplots(figsize=(9.71, 6))
colors = ['r', 'b', 'g', 'm']

plt.plot(df["R"], df["M"], label = rf'$M(R)$', color = colors[0], linewidth = 1.5, linestyle = '-', marker = "*",  mfc="k", mec = "k", ms = 5)

plt.xlabel(r'$R$ $\left[km\right]$', fontsize=15, loc='center')
plt.ylabel(r'$M$ $\left[ M_{\odot} \right]$', fontsize=15, loc='center')

# Set limits
plt.xlim(0, 50)
plt.ylim(0, 0.8)

# Configure ticks for all four sides
plt.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
plt.minorticks_on()

# Customize tick spacing 
plt.gca().set_xticks(np.arange(0, 50.1, 5))  # Major x ticks 
plt.gca().set_yticks(np.arange(0, 0.81, 0.1))  # Major y ticks 

# Set thicker axes
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Add a legend
plt.legend(fontsize=15, loc = "upper right", bbox_to_anchor=(0.99, 0.99), frameon=True, fancybox=False, ncol = 1, edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)

# Save plot as PDF
plt.tight_layout()
plt.savefig("figures\MR_Fermi.pdf", format="pdf", bbox_inches="tight")

plt.show()