# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 16:08:51 2025

Using data calculated through DANTE.py it plots the preassures and mass profiles of a Fermi model Neutron Star

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

data = {}

s_type = 2
d_type = 0
eos_c = 'soft'
dm_m = 0.93956542
p1_c = 'M'
p1_v_list = [0.2, 0.3, 0.4, 0.5]

for p1_v in p1_v_list:
    df = pd.read_csv(rf"data\{s_type}_{d_type}_{dm_m}_{p1_c}_{p1_v}.csv")
    data[f'{p1_v}'] = df

# Configure the plot
fig, ax1 = plt.subplots(figsize=(9.8, 6))
colors = ['r', 'b', 'g', 'm']
c=0

# Set the axis.
ax1.set_xlabel(r'$r$ $\left[km\right]$', fontsize=15, loc='center')
ax1.set_ylabel(r'$p$ $\left[ M_{\odot} / km^3 \right]$', fontsize=15, loc='center', color='k')
ax1.tick_params(axis='y', colors='k')
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
ax1.set_yscale('log')
ax2 = ax1.twinx()
ax2.set_ylabel(r'$m$ $\left[ M_{\odot} \right]$', fontsize=15, loc='center', color='k')
ax2.tick_params(axis='y', colors='k')

# The plot thickenss
for p1_v in p1_v_list:
    ax1.plot(data[f"{p1_v}"]['r'], data[f"{p1_v}"]['p_B'], color = colors[c], linewidth=1.5, linestyle='-')
    ax2.plot(data[f"{p1_v}"]['r'], data[f"{p1_v}"]['m'], color = colors[c], linewidth=1.5, linestyle='--')
    c+=1

if True == True:
    ax1.set_xlim(0, 25)
    ax1.set_ylim(3e-15, 2e-5)
    ax2.set_ylim(0, 0.75)

ax1.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True)
ax1.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True)
ax1.minorticks_on()
ax2.tick_params(axis='both', which='major', direction='in', length=8, width=1.2, labelsize=12, top=True, right=True)
ax2.tick_params(axis='both', which='minor', direction='in', length=4, width=1, labelsize=12, top=True, right=True)
ax2.minorticks_on()

if True == True:
    ax1.set_xticks(np.arange(0, 25.5, 2))
    #ax1.set_xticks(np.arange(0, 9.6, 0.2), minor=True)
    ax1.set_yticks(np.geomspace(1e-14, 1e-5, 10))
    #ax1.set_yticks(np.arange(0, 8.1e-5, 0.2e-5), minor=True)
    ax2.set_yticks(np.arange(0, 0.75, 0.1))
    ax2.set_yticks(np.arange(0, 0.75, 0.02), minor=True)
    
for ax in [ax1, ax2]:
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_color('k')
    ax.spines['right'].set_color('k')
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')

    
handles_list = [
    mlines.Line2D([], [], color=colors[0], linestyle='-', label=r"$M=0.2$"),
    mlines.Line2D([], [], color=colors[1], linestyle='-', label=r"$M=0.3$"),
    mlines.Line2D([], [], color='k', linestyle='-', label=r"$p(r)$"),
    mlines.Line2D([], [], color=colors[2], linestyle='-', label=r"$M=0.4$"),
    mlines.Line2D([], [], color=colors[3], linestyle='-', label=r"$M=0.5$"),
    mlines.Line2D([], [], color='k', linestyle='--', label=r"$m(r)$")]

plt.legend(handles=handles_list, loc = "upper right", bbox_to_anchor=(0.99, 0.99), fontsize=15, frameon=True, fancybox=False, ncol = 2, edgecolor="black", framealpha=1, labelspacing=0.2, handletextpad=0.3, handlelength=1.4, columnspacing=1)

# Save plot as PDF
#plt.title(rf'DM Stars, Relativity check. $m_{{\chi}} = {dm_m}$', loc='left', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"figures\TOV_Fermi.pdf", format="pdf", bbox_inches="tight")

plt.show()