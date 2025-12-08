# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 14:28:56 2025

Adapted from DANTE.py

It creates, saves and plots the Equation of State of Hebeler et al.

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

data = {}

eos_list = ['soft', 'middle', 'stiff']
for eos in eos_list:
    eos_data = pd.read_excel(f"data_eos\eos_{eos}.xlsx")
    data[f"{eos}"] = eos_data

fig, ax = plt.subplots(figsize=(6, 6))
colors = sns.color_palette("Set1", 10)
eos_colors = {"soft": 0, "middle": 1, "stiff": 2}
eos_markers = {"soft": "o", "middle": "s", "stiff": "D"}

for eos in eos_list:
    ax.plot(data[f"{eos}"]["Density"], data[f"{eos}"]["Pressure"], 
            label=f"{eos}", color=colors[eos_colors[f"{eos}"]], linewidth=1.5, linestyle='-',
           marker = eos_markers[f'{eos}'],  mfc=colors[eos_colors[f"{eos}"]], mec = 'k', ms = 5)

#plt.title(r'Equation of State for Baryonic Matter', fontsize=15, loc='left', fontweight='bold')
ax.set_xlabel(r'$\rho$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center')
ax.set_ylabel(r'$p$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center')
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
ax.set_xscale('log')
ax.set_yscale('log')

ax.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right = True)
ax.minorticks_on()

ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_color('k')
ax.spines['right'].set_color('k')
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')

#ax.set_xlim(1e-18, 2.5e-5)
#ax.set_ylim(1e-30, 9e-4)
ax.set_xticks(np.geomspace(1e-17, 1e-3, 8))
#ax.set_xticks(np.arange(0, 9.6, 0.2), minor=True)
ax.set_yticks(np.geomspace(1e-31, 1e-3, 15))
#ax.set_yticks(np.arange(0, 8.1e-5, 0.2e-5), minor=True)

plt.legend(fontsize=15, loc = "upper left", bbox_to_anchor=(0.009, 0.99), frameon=True, fancybox=False, ncol = 1,
           edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)

plt.tight_layout()
plt.savefig(f"figures\EoS_NS_Low.svg", format="svg", bbox_inches="tight")

plt.show()

###############################################################################

fig, ax = plt.subplots(figsize=(9.71, 6))
colors = sns.color_palette("Set1", 10)
eos_colors = {"soft": 0, "middle": 1, "stiff": 2}
eos_markers = {"soft": "o", "middle": "s", "stiff": "D"}

for eos in eos_list:
    ax.plot(data[f"{eos}"]["Density"], data[f"{eos}"]["Pressure"], 
            label=f"{eos}", color=colors[eos_colors[f"{eos}"]], linewidth=1.5, linestyle='-',
           marker = eos_markers[f'{eos}'],  mfc=colors[eos_colors[f"{eos}"]], mec = 'k', ms = 5)

#plt.title(r'Equation of State for Baryonic Matter', fontsize=15, loc='left', fontweight='bold')
ax.set_xlabel(r'$\rho$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center')
ax.set_ylabel(r'$p$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center')
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
ax.set_xscale('log')
ax.set_yscale('log')

ax.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right = True)
ax.minorticks_on()

ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_color('k')
ax.spines['right'].set_color('k')
ax.spines['bottom'].set_color('k')
ax.spines['left'].set_color('k')

ax.set_xlim(2.5e-5, 1.5e-3)
ax.set_ylim(1e-7, 1e-3)

plt.legend(fontsize=15, loc = "upper left", bbox_to_anchor=(0.009, 0.99), frameon=True, fancybox=False, ncol = 1,
           edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)

plt.tight_layout()
plt.savefig(f"figures\EoS_NS_High.svg", format="svg", bbox_inches="tight")

plt.show()