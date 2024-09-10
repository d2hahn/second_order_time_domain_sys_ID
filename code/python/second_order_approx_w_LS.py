"""
second_order_approx_w_LS.py

Summary:
Code intakes the cropped .csv file output from csv_crop_prgrm.py. Creates an estimate of the input vector based off a
qualitative judgement from the user on when the true step occurred by looking at the cropped .csv data.
The estimate of the step is set such that the initial value of the step is the average of pressure values
at the steady state before the balloon is popped. The upper portion of the step is said to be up until the time that
the last  pressure reading before the pressure readings begin to decrease, the lower portion of the step is said to
be at the first time that decrease is noted.

The simulated input and the measured output are then plotted on the same graph. Following this, OLS estimation is used to estimate the
parameters of a general second order system in discrete time (most pressure measurement junctions can be reasonably
approximated using a general second order system see Measurement in Fluid Mechanics ~ Tavoularis and/or Measurement
Systems: Application and Design ~ Doeblin). The estimated parameters as well as their associated uncertainty are output
as .csv files to ./outputs/estimated_params/ by the users input. This output is then to be used in
ct_param_est_from_dt_param_est.m to convert the discrete time model to continuous time and estimate the natural
frequency and damping ratio of the system.  Followed by obtiaining the FR from the identified TD model to see the
passband frequencies of the system. General 2nd order model in the DT domain,is of the form:

 H(z) = Y(z)/U(z) =(b_1*z+b_2)/(z^2+a_1*z+a_2) (Astrom and Wittenmark Computer Control Systems 2nd Edition)

Program also outputs the input, measured output, and time data used for the OLS fitting to ./estimated_params/folder/
exp_i_o_data/ as well as a plot of data used for OLS (input and output) and a plot of success of the OLS estimation based
on the data used for the estimation (not a comparision of model vs. system response, but an assessment of the accuracy
of the OLS estimation.)

General form of outputs:
parameter outputs [[a_1, a_2, b_1, b_2, b_3]^T, uncertainties]
i/o outputs [Time [s], Output [mbar], Input [mbar]]

Dependencies:
1. pandas
2. numpy
3. matplotlib.pyplot

Notes:
    1. User has to qualitativley discern where true step took place from looking at the data in the cropped .csv file
    used as an input for the rest of the code. Use of code previously was to say that the step occurred at the time point
    that the first decrease in pressure was noted.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import OLS_param_estimate_w_est_variance
from scipy import stats

#type in name of output folder
output_folder = './outputs/estimated_params/3_18_2024/'

#reading in cropped data from ./outpus/cropped_data/, change path on each run
step_plot = pd.read_csv('./outputs/cropped_data/3_18_2024/250_mbar_3_18_2024.csv', index_col =0) #change path on each run

#create input vector (simulate since our true input is not a measurable signal)
time_vec = step_plot['Time [s]'].values
press_vec = step_plot['Pressure [mbar]'].values

input_vec_size = np.ones(len(time_vec))
upper_input_vec = input_vec_size[0:52]*np.average(press_vec[0:52]) #index determined by looking at excel and graph data
lower_input_vec = input_vec_size[52:len(input_vec_size)]*0 #index determined by looking at excel and graph data
input_vec = np.concatenate((upper_input_vec, lower_input_vec))

press_vec = step_plot['Pressure [mbar]'].values

#plotting data and simulated step
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 12})
csfont = {'fontname':'Times New Roman'}
plt.errorbar((step_plot['Time [s]']), (step_plot['Pressure [mbar]']), xerr= None, yerr=None, fmt='o', markersize= 6, color='blue')
plt.errorbar(step_plot['Time [s]'], input_vec, color="red")
plt.xlim(step_plot["Time [s]"].min(), step_plot["Time [s]"].max())
plt.xlabel("Time [s]")
plt.ylabel("Pressure [mbar]")
plt.legend(["Measurement", "Input"], framealpha=1, edgecolor='black', fancybox=False)
plt.show()

'''
General Second Order system in Discrete Time:

y(t+2) = -a_1*y(t+1) -a_2*y(t) + b_1*u(t+1) + b_2*u(t)

fit in OLS using GMMNE

y = XB+e

y = [y(t+2)...y(N)]^T

'''

#determining output vectors from data
y_t_plus_two = press_vec[2:len(press_vec)]
y_t_plus_one = -1*press_vec[1:len(press_vec)-1]
y_t = -1*press_vec[0:len(press_vec)-2]

#determing input vectors from simulated input
u_t_plus_one = input_vec[1:len(input_vec)-1]
u_t = input_vec[0:len(input_vec)-2]

#creating transpose of known constant matrix from known responses, X^T
x_mat_t = np.array([y_t_plus_one, y_t, u_t_plus_one, u_t])

#creating response vector for known output from data
y_vec = y_t_plus_two

#estimating parameters and associated covariance matrix of estimate using OLS estimation (see functions.py)
beta_hat, est_cov_mat = OLS_param_estimate_w_est_variance(y_vec,x_mat_t)

#calculating expectation of y_vec from estimated parameters
expect_y = np.matmul(np.transpose(x_mat_t),beta_hat)

#plot 1) actual and estimated response, 2) input
fig, (ax1,ax2,ax3) = plt.subplots(3, sharex=True )
ax1.plot(time_vec[2:len(time_vec)], y_vec, color = 'red', linewidth = 2)
ax1.plot(time_vec[2:len(time_vec)],expect_y, color = 'black', linestyle= '--', linewidth=2)
ax1.set(ylabel = 'Output Pressure [mbar]')
ax1.set(ylim = [y_vec.min()-1,y_vec.max()+1])
ax1.set(yticks = np.arange(0,26, step = 5))
ax1.legend(['Measured', 'Estimated'], framealpha=1, edgecolor='black', fancybox=False)

ax2.plot(time_vec[2:len(time_vec)], (expect_y-y_vec), color = 'blue')
ax2.set(ylim = [-0.6,0.6])
ax2.set(yticks = np.arange(-0.6,0.7, step = 0.2))
ax2.set(ylabel = 'Error [mbar]')

ax3.plot(time_vec[2:len(time_vec)], input_vec[2:len(time_vec)], color = 'black')
ax3.set(ylabel = 'Input Pressure [mbar]')
ax3.set(xlabel = 'Time [s]')
ax3.set(yticks = np.arange(0,26, step = 5))

plt.xlim(time_vec[2:len(time_vec)].min(),time_vec[2:len(time_vec)].max())
plt.show()

#gathering estimated parameter variance estimate from estimated covariance matrix (variances are diag of cov_mat)
est_param_est_var = est_cov_mat.diagonal()

#determining std_dev_hat of B_hat
est_std_dev_hat = np.sqrt(est_param_est_var)

#determing uncertainty of parameter estiamte at 95% CI
#DOF (n-r)
n = np.shape(y_vec)[0]
r = np.linalg.matrix_rank(np.transpose(x_mat_t))
dof = n-r

#determining critical t-value
crit_t_value = stats.t.ppf(1-0.025, dof)

#calculating uncertainty (95% confidence interval of parameters) u_del_beta_hat = t_(n-r, 1-0.025)*std_dev(betat_hat)
uncert_hat = crit_t_value*est_std_dev_hat

#converting parameter estimates and uncertainty estimates from numpy arrays to dataframe
row_names= ['a_1_hat', 'a_2_hat', 'b_1_hat', 'b_2_hat']
params_df = pd.DataFrame(beta_hat, index=row_names, columns=['beta_hat'])
uncert_df = pd.DataFrame(uncert_hat, index=row_names, columns=['uncertainty (95% CI)'])

#combining the parameter estimates and estimates of uncertaitny at the 95% CI into one dataframe
combined_df = pd.concat([params_df,uncert_df], axis=1)

#adding input and output data to dataframe
# [Time [s], Output [mbar], Input [mbar]]
dict_io = {'Time [s]': time_vec, 'Output [mbar]': press_vec, 'Input [mbar]': input_vec}
io_df = pd.DataFrame(data=dict_io,columns=['Time [s]', 'Output [mbar]', 'Input [mbar]'])

#outputting dataframe to .csv file
output_to_csv = input("output dataframes (estimated parameters and uncertainty) to .csv file? (y/n): ")
if output_to_csv == 'y':
    name_of_test  = input('what is the name of the test for which the parameters were estimated? : ')
    combined_df.to_csv(output_folder+name_of_test+'.csv')
    io_df.to_csv(output_folder+'exp_i_o_data/'+name_of_test+'.csv')
else:
    print('not output to .csv')