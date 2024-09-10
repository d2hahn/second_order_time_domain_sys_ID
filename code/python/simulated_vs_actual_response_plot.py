"""
simulated_vs_actual_response_plot.py

Summary:
Code intakes output of form

[input_vec [mbar], measured_output [mbar], simulated_output [mbar], time [s]]

from ../matlab/outputs/simulated_data/pressure_case/ output from lsim_prgrm.m

Using this data, a triple plot of the simulated and actual response, error betweeen the two responses (error = simulated
- actual), and the input signal as a fn of time is presented. As well, the RMS error ( RMSE = sqrt(sum((y_sim - y)^2)/N))
is shown on the error plot. These plots are to be saved in ./plots/.. for comparision of the identified
models from the model set. Note proper sysID would also need to use a validation data set. Uncertainties in the measured
values are also plotted and every fifth point is shown.

Dependencies:
1. pandas
2. numpy
3. matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import output data and simulation data from ../matlab/outputs/simulation_data/pressure_case
#recall data form [meas_outputs sim_outputs time]
data = pd.read_csv('../matlab/outputs/simulation_data/25_mbar/25_mbar_3_18_2024.csv', names = ['Input [mbar]','Measured [mbar]', 'Simulated [mbar]', 'Time [s]'])

#calculate RMS error
meas_vec = data['Measured [mbar]'].values
sim_vec = data['Simulated [mbar]'].values

#error = estimated-measured
error_vec = sim_vec-meas_vec

#RMS error = (sum(error_vec^2)/N)^1/2
RMS_error = np.sqrt(np.sum(np.square(error_vec))/len(error_vec))
print(RMS_error)
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 30})

#plot 1) actual and estimated response, 2) input
props = dict(facecolor='white')

mksize = 10
mkewidth = 2
cpsize = 9
lw = 3
lstyle = '--'


fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True, figsize=(15, 9) )
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.5
#errorbars
ax1.errorbar(data['Time [s]'][::5], data['Measured [mbar]'][::5],yerr = 1.11007, marker = "o", color= "red", fmt=' ', capsize=cpsize,
            fillstyle = 'none', markersize=mksize, zorder=1, markeredgewidth = mkewidth)

#ax1.plot(data['Time [s]'], data['Measured [mbar]'], color = 'red', linewidth = 2)
ax1.plot(data['Time [s]'],data['Simulated [mbar]'], color = 'black', linestyle= '--', linewidth=lw, zorder=2)
ax1.set_ylabel('Output Pressure [mbar]', labelpad=17)
#ax1.set(ylabel = 'Output Pressure [mbar]')
ax1.set(ylim = [data['Measured [mbar]'].min()-2,data['Measured [mbar]'].max()+2])
ax1.set(yticks = np.arange(0,26, step = 5))
ax1.legend(['Simulated', 'Measured'], framealpha=1, edgecolor='black', fancybox=False)
ax1.xaxis.set_tick_params(width=1.5, length= 5,which='major')
ax1.yaxis.set_tick_params(width=1.5, length = 5 ,which='both')


ax2.plot(data['Time [s]'], (data['Simulated [mbar]']-data['Measured [mbar]']), color = 'red', linewidth = lw)
ax2.set(ylim = [-4,4])
ax2.set(yticks = np.arange(-4,5, step = 2))
ax2.set_ylabel('Error [mbar]', labelpad=7)
#ax2.set(ylabel = 'Error [mbar]')
rms_plt = ax2.text(0.85, .94,
                       'RMSE' + ' = ' + str(round(RMS_error, 2)),
                       horizontalalignment='left', verticalalignment='top', fontsize=12,
                       fontfamily='Times New Roman', transform=ax2.transAxes)
ax2.xaxis.set_tick_params(width=1.5, length= 5,which='major')
ax2.yaxis.set_tick_params(width=1.5, length = 5 ,which='both')

ax3.plot(data['Time [s]'], data['Input [mbar]'], color = 'black', linewidth = lw)
ax3.set_ylabel('Input Pressure [mbar]', labelpad=17)
#ax3.set(ylabel = 'Input Pressure [mbar]')
ax3.set(xlabel = 'Time [s]')
ax3.set(yticks = np.arange(0,26, step = 5))
ax3.xaxis.set_tick_params(width=1.5, length= 5,which='major')
ax3.yaxis.set_tick_params(width=1.5, length = 5 ,which='both')


plt.xlim(data['Time [s]'].min(),data['Time [s]'].max())
plt.show()

#give user option to save the figure
check = input('save figure ?(y/n): ')
if check == 'y':
    fig.savefig('./plots/sim_v_meas/25_sim_uncert_erkorkmaz_3_18_2024_tr.svg')
else:
    print('not saved')
