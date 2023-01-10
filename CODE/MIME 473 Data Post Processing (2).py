#!/usr/bin/env python
# coding: utf-8

# ## MIME 473 Data Post Processing

# In[475]:


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


# In[476]:


df = pd.read_csv(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\S5_df.csv')


# # Parameters

# In[477]:


a = 35 #diameter or the circle at its widest point(rectangular prism length)
b = np.pi*(a/2)**2 #comes from the area of the cylinder (pi*(17.2675476 Angstrom)^2)
deltaL = 0.000027


# ## Unfiltered Data Plots

# In[478]:


plt.plot(df['Step'], df['Temperature'], label="Temperature")
plt.xlabel('Step')
plt.ylabel('Temperature (K)')
plt.legend()
plt.title('Temperature (K) vs. Step (s)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\temperature.png', dpi=300)
plt.show()


# In[479]:


plt.plot(df['Step'], df['TotEnergy'], label ="Total Energy" )
plt.plot(df['Step'], df['PotEnergy'], label = "Potential Energy")
plt.xlabel('Step')
plt.ylabel('Energy (J)')
plt.legend()
plt.title('Potential and Total Energy (J) vs. Step (s)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\energy.png', dpi=300)
plt.show()


# In[480]:


plt.plot(df['Step'], df['Pressure'], label="Pressure")
plt.xlabel('Step')
plt.ylabel('Pressure (bar)')
plt.legend()
plt.title('Pressure (bar) vs. Step (s)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\pressure.png', dpi = 300)
plt.show()


# In[481]:


plt.plot(df['Step'], df['PressureX'], label = "PressureX", color = "blue")
plt.plot(df['Step'], df['PressureY'], label = "PressureY", color = 'green')
plt.plot(df['Step'], df['PressureZ'], label = "PressureZ", color = "red")
plt.xlabel('Step')
plt.ylabel('Pressure')
plt.legend()
plt.title('Pressure in 3D (bar) vs. Step (s)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\presure3d.png', dpi = 300)
plt.show()


# In[482]:


def preprocess(df):
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    # Filter requirements.
    order = 1
    fs = 200     # sample rate, Hz
    cutoff = 1.2  # desired cutoff frequency of the filter, Hz
    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)
    #parameters
    T = 5.0         # seconds
    n = int(T * fs) # total number of samples

    
    # Filter the data, and plot both the original and filtered signals.
    f_temperature = butter_lowpass_filter(df['Temperature'], cutoff, fs, order)
    f_pressure = butter_lowpass_filter(df['Pressure'], cutoff, fs, order)
    f_potE = butter_lowpass_filter(df['PotEnergy'], cutoff, fs, order)
    f_totE = butter_lowpass_filter(df['TotEnergy'], cutoff, fs, order)
    f_PressX = butter_lowpass_filter(df['PressureX'], cutoff, fs, order)
    f_PressY = butter_lowpass_filter(df['PressureY'], cutoff, fs, order)
    f_PressZ = butter_lowpass_filter(df['PressureZ'], cutoff, fs, order)
    fdata = {'Step' : df['Step'], 
             'Temperature' : f_temperature,
            'Pressure' : f_pressure, 
             'PotEnergy' : f_potE, 
             'TotEnergy' : f_totE,
            'PressureX' : f_PressX,
             'PressureY': f_PressY,
             'PressureZ': f_PressZ}
    filtered = pd.DataFrame(data=fdata)
    return filtered


# In[483]:


f_df = preprocess(df)


# In[484]:


plt.plot(f_df['Step'], f_df['Temperature'], label="Temperature")
plt.xlabel('Step')
plt.ylabel('Temperature (K)')
plt.legend()
plt.title('Filtered Temperature (K) vs. Step (s)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\f_temperature.png', dpi=300)
plt.show()


# In[485]:


plt.plot(f_df['Step'], f_df['TotEnergy'], label ="Total Energy" )
plt.plot(f_df['Step'], f_df['PotEnergy'], label = "Potential Energy")
plt.xlabel('Step')
plt.ylabel('Energy (J)')
plt.legend()
plt.title('Filtered Potential and Total Energy (J) vs. Step (s)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\f_energy.png', dpi=300)
plt.show()


# In[486]:


plt.plot(f_df['Step'], f_df['Pressure'], label="Pressure")
plt.xlabel('Step')
plt.ylabel('Pressure (bar)')
plt.legend()
plt.title('Filtered Pressure (bar) vs. Step (s)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\f_pressure.png', dpi = 300)
plt.show()


# In[487]:


plt.plot(f_df['Step'], f_df['PressureX'], label = "PressureX", color = "blue")
plt.plot(f_df['Step'], f_df['PressureY'], label = "PressureY", color = 'green')
plt.plot(f_df['Step'], f_df['PressureZ'], label = "PressureZ", color = "red")
plt.xlabel('Step')
plt.ylabel('Pressure')
plt.legend()
plt.title('Filtered Pressure in 3D (bar) vs. Step (s)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\f_presure3d.png', dpi = 300)
plt.show()


# In[488]:


fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(df['Step'], df['Temperature'], label='Temperature (K)', color = 'tab:blue')
axs[0, 0].set_title('Temperature (K) vs. Time Step (s)')
axs[0, 0].set(ylabel='Temperature (K)')
axs[0,0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[0,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[0,0].set_title('Temperature (K) vs. Step (s)', pad=20)

axs[1, 0].plot(df['Step'], df['Pressure'], label ='Pressure (bar)', color = 'tab:orange')
axs[1, 0].set_title("Pressure (bar) vs. Step (s)")
axs[1, 0].set(ylabel='Pressure (bar)')
axs[1,0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[1,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[1,0].set_title('Pressure (bar) vs. Step (s)', pad=20)


axs[0, 1].plot(df['Step'], df['PotEnergy'], label='Potential Energy (J)', color='tab:green')
axs[0, 1].plot(df['Step'], df['TotEnergy'], label='Total Energy (J)', color='tab:red')
# axs[0, 1].set_title("Potential and Total Energy (J) vs. Step (s)")
axs[0, 1].set(ylabel='Energy (J)')
axs[0,1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[0,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[0,1].set_title("\n".join(wrap("Potential and Total Energy (J) vs. Step (s)", 60)), pad=20)


axs[1, 1].plot(df['Step'], df['PressureX'], label='Pressure in X (bar)', color='tab:purple')
axs[1, 1].plot(df['Step'], df['PressureY'], label='Pressure in Y (bar)', color='tab:brown')
axs[1, 1].plot(df['Step'], df['PressureZ'], label='Pressure in Z (bar)', color='tab:pink')
axs[1, 1].set_title("Pressure in 3D vs. Step (s)")

axs[1, 1].set(xlabel = 'Time Step (s)', ylabel='Pressure (bar)')
axs[1,1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[1,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[1,1].set_title('Pressure in 3D vs. Step (s)', pad=20)
fig.tight_layout(pad = 2)
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\temp_press_energy_pressxyz.png', dpi = 300)
plt.show()


# In[489]:


fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(f_df['Step'], f_df['Temperature'], label='Temperature (K)', color = 'tab:blue')
axs[0, 0].set_title('Temperature (K) vs. Time Step (s)')
axs[0, 0].set(ylabel='Temperature (K)')
axs[0,0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[0,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[0,0].set_title('Temperature (K) vs. Step (s)', pad=20)

axs[1, 0].plot(f_df['Step'], f_df['Pressure'], label ='Pressure (bar)', color = 'tab:orange')
axs[1, 0].set_title("Pressure (bar) vs. Step (s)")
axs[1, 0].set(ylabel='Pressure (bar)')
axs[1,0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[1,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[1,0].set_title('Pressure (bar) vs. Step (s)', pad=20)


axs[0, 1].plot(f_df['Step'], f_df['PotEnergy'], label='Potential Energy (J)', color='tab:green')
axs[0, 1].plot(f_df['Step'], f_df['TotEnergy'], label='Total Energy (J)', color='tab:red')
# axs[0, 1].set_title("Potential and Total Energy (J) vs. Step (s)")
axs[0, 1].set(ylabel='Energy (J)')
axs[0,1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[0,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[0,1].set_title("\n".join(wrap("Potential and Total Energy (J) vs. Step (s)", 60)), pad=20)


axs[1, 1].plot(f_df['Step'], f_df['PressureX'], label='Pressure in X (bar)', color='tab:purple')
axs[1, 1].plot(f_df['Step'], f_df['PressureY'], label='Pressure in Y (bar)', color='tab:brown')
axs[1, 1].plot(f_df['Step'], f_df['PressureZ'], label='Pressure in Z (bar)', color='tab:pink')
axs[1, 1].set_title("Pressure in 3D vs. Step (s)")

axs[1, 1].set(xlabel = 'Time Step (s)', ylabel='Pressure (bar)')
axs[1,1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
axs[1,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[1,1].set_title('Pressure in 3D vs. Step (s)', pad=20)
fig.tight_layout(pad = 2)
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\filtered_temp_press_energy_pressxyz.png', dpi = 300)
plt.show()


# # Tensile Data

# ### Parameters (UNFILTERED DATA)

# In[490]:


strain_data = []
for i in range(len(df)):
    strain_data.append(i*deltaL/40)
strain_data


# In[491]:


T_data = {'Stress (GPa)': (-df['PressureZ']/10000)*(a**2/b), 
         'Strain' : strain_data}
stress_strain_df = pd.DataFrame(data=T_data)
stress_strain_df.to_csv('D:\McGill\Year4\Winter\MIME 473\Project\S5\stress_strain.csv')


# In[492]:


plt.plot(stress_strain_df['Strain'], stress_strain_df['Stress (GPa)'])
plt.xlabel('Strain')
plt.ylabel('Stress (GPa)')
plt.title('Stress (GPa) vs. Strain')
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\stress_strain.png', dpi = 300)
plt.show()


# In[493]:


f_strain_data = []
for i in range(len(f_df)):
    f_strain_data.append(i*deltaL/40)
f_strain_data


# In[494]:


f_T_data = {'Stress (GPa)': (-f_df['PressureZ']/10000)*(a**2/b), 
         'Strain' : f_strain_data}
f_stress_strain_df = pd.DataFrame(data=f_T_data)
f_stress_strain_df.to_csv(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\f_stress_strain.csv')


# In[495]:


plt.plot(f_stress_strain_df['Strain'], f_stress_strain_df['Stress (GPa)'])
plt.xlabel('Strain')
plt.ylabel('Stress (GPa)')
plt.title('Filtered Stress (GPa) vs. Strain')
plt.savefig(r'D:\McGill\Year4\Winter\MIME 473\Project\S5\f_stress_strain.png', dpi = 300)
plt.show()


# # Need to find moduli, UTS, Fail Strain, Max Temp, Max Tot/Pot Energy, Max Pressure

# ## find modulus

# ## max stress/strain, temperature, energies, pressures

# In[496]:


f_stress_strain_df['Stress (GPa)']


# In[497]:


slopes = []

for i in range(len(f_stress_strain_df)):
    m = ((f_stress_strain_df['Stress (GPa)'][i])-0)/((f_stress_strain_df['Strain'][i])-0)
    slopes.append(m)
    if (slopes[i]-slopes[i-1]) < 25 and i > (len(f_stress_strain_df))/2:
        break
slopes


# In[498]:


max_strain, max_stress = max(f_stress_strain_df['Strain']), max(f_stress_strain_df['Stress (GPa)'])
max_temperature, max_pressure = max(f_df['Temperature']), max(f_df['Pressure'])
max_tot_energy, max_pot_energy = max(f_df['TotEnergy']), max(f_df['PotEnergy'])
max_pressureX, max_pressureY, max_pressureZ = max(f_df['PressureX']),max(f_df['PressureY']), max(f_df['PressureZ'])


# In[499]:


conf_mat = {'Modulus': slopes[-2], 'Max Strain': float(max_strain), 
            'Yield Stress' : 0,
            'Max Stress' : float(max_stress), 
           'Max Temperature': float(max_temperature), 'Max Pressure': float(max_pressure), 
            'Max TotEnergy' : float(max_tot_energy), 'Max Potential Energy': float(max_pot_energy),
            'Max PressureX': float(max_pressureX), 'Max PressureY' : float(max_pressureY),
            'Max PressureZ' : float(max_pressureZ) }
conf_mat


# In[500]:


conf_df = pd.DataFrame(data=conf_mat,index = [0])
conf_df.to_csv(r"D:\McGill\Year4\Winter\MIME 473\Project\S5\output_mat.csv")


# In[ ]:





# In[ ]:




