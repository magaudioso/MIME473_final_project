#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from scipy.stats import linregress
from sklearn.metrics import r2_score
import plotly.graph_objects as go 
from scipy.signal import butter, lfilter, freqz
from matplotlib import style
style.use('default')
from textwrap import wrap
import scipy as sp


# # Cummulative Stress Strain Plots

# In[20]:


ss_df = pd.read_csv(r"D:\McGill\Year4\Winter\MIME 473\Project\cummulative_stress_strain_df.csv", header = [0,1,2])
ss_df


# In[79]:


plt.plot(ss_df['S1']['Filtered']['Strain'], ss_df['S1']['Filtered']['Stress (GPa)'], label = '0.7nm', color = 'tab:blue')
plt.plot(ss_df['S2']['Filtered']['Strain'], ss_df['S2']['Filtered']['Stress (GPa)'], label = '1.4nm', color = 'tab:orange')
plt.plot(ss_df['S4']['Filtered']['Strain'], ss_df['S4']['Filtered']['Stress (GPa)'], label = '3.5nm', color = 'tab:green')
plt.plot(ss_df['S5']['Filtered']['Strain'], ss_df['S5']['Filtered']['Stress (GPa)'], label = '7.0nm', color = 'tab:purple')
plt.legend()
plt.title('Stress (GPa) vs. Strain w/Differences in Interatomic Twin Spacing',pad = 20)
plt.ylabel('Stress (GPa)')
plt.xlabel('Strain')
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\cummulative_stress_strain_twinning.png',
            dpi=300, bbox_inches='tight')
plt.show()


# In[22]:


plt.plot(ss_df['D0']['Filtered']['Strain'], ss_df['D0']['Filtered']['Stress (GPa)'], label = '2.3nm', color = 'tab:blue')
plt.plot(ss_df['D1']['Filtered']['Strain'], ss_df['D1']['Filtered']['Stress (GPa)'], label = '3.5nm', color = 'tab:orange')
plt.plot(ss_df['D2']['Filtered']['Strain'], ss_df['D2']['Filtered']['Stress (GPa)'], label = '4.6nm', color = 'tab:green')
plt.plot(ss_df['D3']['Filtered']['Strain'], ss_df['D3']['Filtered']['Stress (GPa)'], label = '5.8nm', color = 'tab:purple')
plt.plot(ss_df['D4']['Filtered']['Strain'], ss_df['D4']['Filtered']['Stress (GPa)'], label = '6.9nm', color = 'tab:red')
plt.legend()
plt.title('Stress (GPa) vs. Strain w/Differences in Nanowire Diameter',pad = 20)
plt.ylabel('Stress (GPa)')
plt.xlabel('Strain')
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\cummulative_stress_strain_diameter.png',
            dpi=300, bbox_inches='tight')
plt.show()


# ## Comparitive Analysis Charts

# In[80]:


conf_mat_diameter = pd.read_csv(r"D:\McGill\Year4\Winter\MIME 473\Project\cummulative_output_mat_diameter.csv", header = [0,1])
conf_mat_spacing = pd.read_csv(r"D:\McGill\Year4\Winter\MIME 473\Project\ouput_mat_twin_spacig.csv",header = [0,1])


# ### Modulus

# In[81]:


conf_mat_diameter['Modulus'].plot.bar(align='center', cmap = 'viridis')
plt.title('Moduli of different Diameter Nanowires')
plt.ylabel('Modulus (GPa)')
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\moduli_spacing_diameter.png', dpi=300)


# In[82]:


conf_mat_spacing['Modulus'].plot.bar(align='center', cmap = 'inferno'
)
plt.title('Moduli of different Twin-Spaced Nanowires')
plt.ylabel('Modulus (GPa)')
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\moduli_spacing_twin_spacing.png', dpi=300)


# ### Max Stress

# In[83]:


conf_mat_diameter['Max Stress'].plot.bar(align='center',cmap = 'viridis')
plt.title('Moduli of different Diameter Nanowires')
plt.ylabel('Stress (GPa)')
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\max_stress_spacing_diameter.png', dpi=300)


# In[84]:


conf_mat_spacing['Max Stress'].plot.bar(align='center', cmap='inferno')
plt.title('Max Stress of different Twin-Spaced Nanowires')
plt.ylabel('Stress (GPa)')
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\max_stress_spacing.png', dpi=300)


# ### CRSS

# In[85]:


conf_mat_diameter['CRSS'].plot.bar(align='center',cmap='viridis')
plt.title('CRSS of different Diameter Nanowires')
plt.ylabel('Critical Resolved Shear Stress (CRSS) (GPa)')
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\CRSS_diameter.png', dpi=300)


# In[86]:


conf_mat_spacing['CRSS'].plot.bar(align='center', cmap='inferno')
plt.title('CRSS of different Twin Spaced Nanowires')
plt.ylabel('Critical Resolved Shear Stress (CRSS) (GPa)')
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\CRSS_spacing.png', dpi=300)


# In[ ]:




