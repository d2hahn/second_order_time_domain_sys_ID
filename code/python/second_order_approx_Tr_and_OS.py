"""
second_order_approx_Tr_and_OS.py

Summary:
Code intakes cropped data from csv_crop_prgrm.py (./outputs/cropped_data/) and from this data estimates the
overshoot and rise time of the step response of a general underdamped (0 < dr < 1) second order system of the form:

G(s) = wn^2/(s^2 + 2*dr*wn*s +wn^2)

Where for a general 2nd order systems response to a step input,
Peak Time = T_p = Time required to reach the first (maximum) peak
Overshoot = OS = The amount that the output waveform overshoots the steady-state (final) value at peak time, T_p
Rise Time = T_r = Time for system response to go from 10-90% of final value (10-90% of initial value for decrease step)

The estimate of the step is set such that the initial value of the step is the average of pressure values
at the steady state before the balloon is popped. The upper portion of the step is said to be up until the time that
the last  pressure reading before the pressure readings begin to decrease, the lower portion of the step is said to
be at the first time that decrease is noted. This is important for the uncertainty calculations and plots.

From the determination of overshoot the damping ratio, dr, is determined
,where

dr = -ln(OS)/sqrt(pi^2+ln(OS)^2)

From the estimate of the damping ratio , the natural frequency, wn, is determined from interpolation
of the empirical table relating damping ratio to the normalized rise time, NTr = wn*Tr (Nise,2018)
and the natural frequency is calculated from NTr/Tr_measured.

Uncertainty in each measurement and calculated value is also performed, please see Thesis and/or analysis binder for
reference on these calculations.

A .csv file of the form ['Peak Time [s]', 'Rise Time [s]' 'Overshoot', 'DR', 'nat_freq_tr [rad/s]', 'nat_freq_tr [rad/s]'] is output to ./outputs/dr_and_wn/ and a
is output to ./outputs/dr_and_wn/ and a plot of the system response with the identified peak and rise time is shown.
Outputs are to be used in second_order_response_from_dr_and_wn.m to evaluate the step response and output
the coeffecients for a generalization of the general underdamped second order system.

G(s) = b/(s^2 + a*s + b) ; a = 2*dr*wn, b=wn^2

References:
    Control Systems Engineering, Nise (2018)
    Modern Control Engineering, Ogata (2016)
    Feedback Control of Dynamic Systems, Franklin (2015)

Dependencies:
1. pandas
2. numpy
3. matplotlib.pyplot
4. math

Notes:
    1. change the input file path for different cases. (see line 66)
    2. Ensure you look at cropped output data to select appropriate time to simulate step input. (see lines 73,74,114-116 )
    3. code is meant for a decreasing step input, slight modification to peak time, rise time, and overshoot calculations needed if
        increasing step input considered. Inherent to this, is the assumption that the final (steady_state) output value is = 0.
    4. Need to add the ability to export the step, output, and time data like in second_order_approx_w_LS.py
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
import scipy.stats as stats

#reading in cropped data from ./outpus/cropped_data/, change path on each run
step_plot = pd.read_csv('./outputs/cropped_data/3_18_2024/250_mbar_3_18_2024.csv', index_col =0)

#create input vector (simulate since our true input is not a measurable signal)
time_vec = step_plot['Time [s]'].values
press_vec = step_plot['Pressure [mbar]'].values

input_vec_size = np.ones(len(time_vec))
upper_input_vec = input_vec_size[0:52]*np.average(press_vec[0:52]) #index determined by looking at excel and graph data
lower_input_vec = input_vec_size[52:len(input_vec_size)]*0 #index determined by looking at excel and graph data
input_vec = np.concatenate((upper_input_vec, lower_input_vec))

#appending input vector to step_plot dataframe
step_plot['Input [mbar]'] = input_vec

#setting value of initial and final value of y
y_final = 0
y_initial = input_vec[0]

#determining peak time (Time required to meet the first (or maximum) peak)
peak_time_index = step_plot[['Pressure [mbar]']].idxmin()
peak_time_point = step_plot.loc[peak_time_index]
peak_time = peak_time_point['Time [s]'].values[0]

#determining overshoot abs(c_max-c_final/c_final) in our case: (y_Tp/y_initial)
os = abs(((y_initial+abs(peak_time_point['Pressure [mbar]'].values[0]))-y_initial)/y_initial)

"""
Calculating uncertainty in overshoot:

del_u_OS = +-sqrt((d_OS/d_y_Tp*del_u_y_Tp)^2+(d_OS/d_y_initial*del_u_y_initial)^2)

Where,
d_OS/d_y_Tp = 1/y_initial

d_OS/d_y_initial = -y_Tp(y_initial)^2

del_u_y_Tp = del_u_pressure_sensor_o_w_c

del_u_y_initial = del_u_pressure_sensor_1_w_c = +-sqrt(del_u_ps_o^2+del_u_ps_t^2); del_u_ps_t = t_(N-1, 1- alpha/2)*sigma_y/sqrt(N)
since y_initial is an average.
"""
d_os_d_y_Tp = 1/y_initial
d_os_d_y_i = peak_time_point['Pressure [mbar]'].values[0]/((y_initial)**2)

# del_u_y_Tp = del_u_pressure_sensor_1_w_c
del_u_ps_o_w_c = 1.01480

#del_u_y_initial = del_u_pressure_sensor_1_w_c = +-sqrt(del_u_ps_o^2+del_u_ps_t^2); del_u_ps_t = t_(N-1, 1- alpha/2)*sigma_y/sqrt(N)
sigma_y_i = np.std(press_vec[0:52],ddof = 1) #need to change index for different cases
t_val = stats.t.ppf(1-0.025, len(press_vec[0:52])-1) #need to change index for different cases
del_u_ps_t = t_val*(sigma_y_i/m.sqrt(len(press_vec[0:52]))) #need to change index for different cases
del_u_ps_1_w_c = m.sqrt(del_u_ps_o_w_c**2+del_u_ps_t**2)

#del_u_OS = +-sqrt((d_OS/d_y_Tp*del_u_y_Tp)^2+(d_OS/d_y_initial*del_u_y_initial)^2)
del_u_os = m.sqrt((d_os_d_y_Tp*del_u_ps_o_w_c)**2+(d_os_d_y_i*del_u_ps_1_w_c)**2)

#calculating damping ratio, dr = (-ln(os)/sqrt(pi^2+ln^2(os)) (Nise, 2018)
dr = -1*m.log(os)/(m.sqrt(m.pi**2+m.log(os)**2))

"""
Calculating uncertainty in damping ratio

del_u_dr = +- sqrt((d_dr/d_os*del_u_os)^2)

d_dr/d_os = -(1/os)*[pi^2+ln^2(os)]^(-1/2)+(1/os)*ln^2(os)*[pi^2+ln^2(os)]^(-3/2)
"""
#d_dr_d_os = -(1/os)*[pi^2+ln^2(os)]^(-1/2)+(1/os)*ln^2(os)*[pi^2+ln^2(os)]^(-3/2)
d_dr_d_os = -1*(1/os)*((m.pi**2)+(m.log(os)**2))**(-0.5)+(1/os)*(m.log(os)**2)*((m.pi**2)+(m.log(os)**2))**(-3/2)
del_u_dr = abs(d_dr_d_os*del_u_os)


#determining rise time, (time required for waveform to go from 0.1 to 0.9 of the final value)
point1_initial = 0.1*y_initial
point9_initial = 0.9*y_initial
points_satisfying_tr = step_plot.loc[(step_plot['Pressure [mbar]']<=point9_initial) & (step_plot['Pressure [mbar]']>=point1_initial)]
tr_beginning_index = points_satisfying_tr.index.min()
tr_end_index = points_satisfying_tr.index.max()
tr_start = points_satisfying_tr.loc[tr_beginning_index]
tr_end = points_satisfying_tr.loc[tr_end_index]
tr = tr_end['Time [s]'] - tr_start['Time [s]']

"""
Determinining uncertainty in rise time
"""
#upper bound
y_initial_ub = y_initial+del_u_ps_1_w_c
point1_initial_ub = 0.1*y_initial_ub
point9_initial_ub = 0.9*y_initial_ub
points_satisfying_tr_ub = step_plot.loc[((step_plot['Pressure [mbar]']+1.11007)<=point9_initial_ub) & ((step_plot['Pressure [mbar]']+1.11007)>=point1_initial_ub)]
tr_beginning_index_ub = points_satisfying_tr_ub.index.min()
tr_end_index_ub = points_satisfying_tr_ub.index.max()
tr_start_ub = points_satisfying_tr_ub.loc[tr_beginning_index_ub]
tr_end_ub = points_satisfying_tr_ub.loc[tr_end_index_ub]
tr_ub = tr_end_ub['Time [s]'] - tr_start_ub['Time [s]']

#lower bound
y_initial_lb = y_initial-del_u_ps_1_w_c
point1_initial_lb = 0.1*y_initial_lb
point9_initial_lb = 0.9*y_initial_lb
points_satisfying_tr_lb = step_plot.loc[((step_plot['Pressure [mbar]']-1.11007)<=point9_initial_lb) & ((step_plot['Pressure [mbar]']-1.11007)>=point1_initial_lb)]
tr_beginning_index_lb = points_satisfying_tr_lb.index.min()
tr_end_index_lb = points_satisfying_tr_lb.index.max()
tr_start_lb = points_satisfying_tr_lb.loc[tr_beginning_index_lb]
tr_end_lb = points_satisfying_tr_lb.loc[tr_end_index_lb]
tr_lb = tr_end_lb['Time [s]'] - tr_start_lb['Time [s]']

#sequential perturbation
del_tr_pos = tr_ub-tr
del_tr_neg = tr_lb-tr
del_tr = (abs(del_tr_pos)+abs(del_tr_neg))/2

#del_u_t_meas (will need to change accuracy and resolution for different daq)
sample_rate = 10000 #hz
usb_6003_res = 12.5*10**(-9) #s (need to change for different daq)
usb_6003_acc = 100 #ppm (need to change for different daq)
del_u_fs = usb_6003_acc/1000000*sample_rate
d_T_d_fs = -(sample_rate)**(-2)
del_u_T_A_o = abs(d_T_d_fs*del_u_fs)
del_u_T_meas_o = m.sqrt(del_u_T_A_o**2+(0.5*usb_6003_res)**2+(0.5*0.0001)**2)

#uncertainty in tr
del_u_tr = m.sqrt(del_tr**2+del_u_T_meas_o**2)

#determining rise time (see Nise 2018, pg.359 (Chapter 4 section 4.6 evaluation of Rist Time)
#inputting data from Fig 4.16 (Nise, 2018) [dr vs. normalized_tr] ; normalized_tr = w_n*t_r
normalized_tr_lookup = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                 [1.104, 1.203, 1.321, 1.463, 1.638, 1.854, 2.126, 2.467, 2.883]])

#interpolating normalized_tr from lookup table and the dr calculated from the overshoot
normalized_tr = np.interp([dr], normalized_tr_lookup[0], normalized_tr_lookup[1])

#interpolating upper bound of normalized_tr from upper bound of dr
normalized_tr_ub = np.interp([dr+del_u_dr], normalized_tr_lookup[0], normalized_tr_lookup[1])

#interpolating lower bound of normalized_tr from lower bound of dr
normalized_tr_lb = np.interp([dr-del_u_dr], normalized_tr_lookup[0], normalized_tr_lookup[1])

#calculating w_n from normalized_tr from lookup table and tr calculated from data
w_n_tr = (normalized_tr/tr)[0]
print(w_n_tr)

"""
Calculating uncertainty in w_n_tr
"""
d_w_n_d_N_tr = 1/tr
d_w_n_d_tr = -normalized_tr*(tr)**(-2)
del_u_N_tr = (abs(normalized_tr_ub-normalized_tr)+abs(normalized_tr_lb-normalized_tr))/2
del_u_w_n_tr = m.sqrt((d_w_n_d_N_tr*del_u_N_tr)**2+(d_w_n_d_tr*del_u_tr)**2)

#from dr and peak_time calculate undamped natural frequency, w_n = pi/(peak_time*sqrt(1-dr^2)) (Nise, 2018)
w_n_tp = m.pi/(peak_time*m.sqrt(1-dr**2))
print(w_n_tp)


#plot of step response with estimation of true input and peak time/rise time as check
#setting font
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 12})

#creating interval for rise time plots
p_tr_start = np.linspace(0, tr_start['Pressure [mbar]'])
p_tr_end = np.linspace(0,tr_end['Pressure [mbar]'])
location_start = tr_start['Time [s]']*np.ones(len(p_tr_start))
location_end = tr_end['Time [s]']*np.ones(len(p_tr_end))



fig,ax = plt.subplots()

line1, = ax.plot(step_plot['Time [s]'], step_plot['Pressure [mbar]'], 'o', markersize= 6, color='blue')
line2, = ax.plot(step_plot['Time [s]'], input_vec, color="red")
line3, = ax.plot(location_start, p_tr_start, color='black')
line4, = ax.plot(location_end, p_tr_end, color='black')
ax.axvline(peak_time_point['Time [s]'][peak_time_index.values[0]], color='black', linestyle = '--')
ax.set(xlim = (step_plot["Time [s]"].min(), step_plot["Time [s]"].max()))
ax.set(xlabel = "Time [s]")
ax.set(ylabel = "Pressure [mbar]")
ax.legend([line1, line2],["Measurement","Input"], framealpha=1, edgecolor='black', fancybox=False)
tp_plt = ax.text(0.28, .9,
                       r'$\mathdefault{T_p}$' + ' = ' + str(round(peak_time, 5)) ,
                       horizontalalignment='left', verticalalignment='top', fontsize=12,
                       fontfamily='Times New Roman', transform=ax.transAxes)
plt.show()

#outputting peak time, rise time, overshoot, damping ratio, and natural frequencies as .csv file ( along with uncertainties
df_output = pd.DataFrame(data={'Peak Time [s]': [peak_time],'Rise Time [s]': [tr], 'Overshoot': [os], 'DR': [dr], 'w_n_tr [rad/s]': [w_n_tr]
                               , 'w_n_tp [rad/s]': [w_n_tp]})
#outputting uncertainties (actual and rel) for rise time, overshoot, damping ratio, and natural frequency (from Tr)
df_uncert_output = pd.DataFrame(data = {'del_u_tr [s]': [del_u_tr], 'del_u_tr_rel [%]':[del_u_tr/tr*100],
                                        'del_u_os': [del_u_os], 'del_u_os_rel [%]': [del_u_os/os*100],
                                        'del_u_dr': [del_u_dr], 'del_u_dr_rel [%]': [del_u_dr/dr*100],
                                        'del_u_wn_tr [rad/s]': [del_u_w_n_tr],'del_u_wn_tr_rel [%]': [del_u_w_n_tr/w_n_tr*100]})


#outputting dataframe to .csv file
output_to_csv = input("output dataframe [Peak Time [s], Rise Time [s],  OS , DR, nat_freq] to .csv file? (y/n): ")
if output_to_csv == 'y':
    name_of_test  = input('what is the name of the test for which the parameters were estimated? : ')
    df_output.to_csv('./outputs/dr_and_wn/'+ name_of_test +'.csv')
    df_uncert_output.to_csv('./outputs/dr_and_wn/'+ name_of_test +'_uncert.csv')
else:
    print('not output to .csv')


# 'settling time code, may want to look into later as of now data is too noisy for a 2% threshold, but a 5% threshold'
# 'may be doable, but note that in arbitrarily setting the settling time, there is more uncertainty to the true nature' \
# 'of the system.'
# # #determining settling time (threshold being the first value within +- 2% of steady-state value)
# # #appending input vector to step_plot dataframe
# # step_plot['Input [mbar]'] = input_vec
# #
# # #setting value of initial and final value of y
# # y_final = 0
# # y_initial = input_vec[0]
# #
# # #setting settling time threshold and steady state value
# # settling_time_threshold = 0.02 #2%
# # steady_state_value = abs(y_final-y_initial)
# #
# # #obtain the times where the output is less than or equal to 2% of the steady-state value
# # points_satisfying_threshold = step_plot.loc[abs(step_plot['Pressure [mbar]']-y_final)<=settling_time_threshold*steady_state_value]
# # print(points_satisfying_threshold)
# #
# # #from each time that satisfies the output, check if the outputs from that time, to the last time measured satisfy the
# #
# # settling_time_index = points_satisfying_threshold[['Time [s]']].idxmin()
# #
# # settling_time_point = points_satisfying_threshold.loc[settling_time_index]
# # print(settling_time_point)