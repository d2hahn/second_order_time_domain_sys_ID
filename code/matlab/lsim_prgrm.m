%{
lsim_prgrm.m

Summary:
Code intakes input, measured output, and time data from second_order_approx_w_LS.py
 or 2_order_LS_w_2_zero.py in
 ../python/outputs/estimated_params/exp_i_o_data as well as the continuous
 (Should add this funcitonality to second_order_approx_Tr_and_OS.py) 
 time parameters from ct_param_est_from_dt_param_est.m or
 ct_param_est_from_dt_param_est_2_zeroes.m or
 second_order_response_from_dr_and_wn.m. 

Note form of parameters .csv file is 

[0 0 b; 1 a b]

From this data a simulation of
 the response of the model with the estimated parameters to the same input is 
performed, from which the output of the system 
output of the simulation, time, and input data are placed in a matrix of
the form:

[input_vec [mbar], measured_output [mbar], simulated_output [mbar], time [s]]

and this matrix is output as a .csv file to
./outputs/simulation_data/pressure_case/ for use in
simulated_vs_actual_response_plot.py.

Notes:
1. Function used to obtain simulation data from input, lsim(), see MATLAB
    documentation for the use of this function.
2. Need to change path of read i_o_data and ct_params see lines 34 and 65.

%}

%{
import input and time data from exp_i_o_data_folder from python outputs 
data is of form: [index time[s] output[mbar] input[mbar]]
%}
i_o_data = readmatrix(['../python/outputs/estimated_params/3_18_2024/exp_i_o_data' ...
    '/250_mbar_3_18_2024.csv']);

pressure_case = '250_mbar';

%edit matrix (remove NAN line at top)
i_o_data_crop = i_o_data(2:end,:);

%obtaining time, input, and output vectors
time_vec = i_o_data_crop(:,2);
output_vec = i_o_data_crop(:,3);
input_vec = i_o_data_crop(:,4);

%adding additional time values and input values to back of measured vectors
%since lsim() does not allow you to start the simulation at steady state
%i.e. I want to compare the transient response to a step decrease in
%pressure as was done in experiments, hence I needed to have lsim provide
%the steady state response which requires simulating an initial increase in
%pressure to steady state, followed by the true input (step decrease, from steady state)
%that was used in the actual experiment.
back_half_of_t = linspace(-0.1,0,1000);
back_half_of_t = back_half_of_t(1,1:end-1);
t_back_vec = transpose(back_half_of_t);
input_back_vec = input_vec(1,1)*ones(size(t_back_vec));

%appending the simulated time and input to back of measured data. True
%input data starts at t=0s (appended data is negative time)
full_input_vec = [input_back_vec;input_vec];
full_time_vec = [t_back_vec; time_vec];

%obtaining output ct params (of form [num; den])
ct_params = readmatrix('./outputs/continuous_time_params/250_mbar/w_n_and_dr/250_mbar_3_18_2024_tr.csv');
system = tf(ct_params(1,:), ct_params(2,:))

%getting output simulation data and time vector
lsim(system,full_input_vec, full_time_vec)
[output,tout] = lsim(system,full_input_vec, full_time_vec);
sim_mat = [output tout];


%getting only simulation data for the step decrease using conditional
%indexing
flag_t = tout>=0;
output_sim_mat  = sim_mat(flag_t,:); 

%creating matrix of form [input [mbar] measured_output[mbar] sim_output[mbar] time[s]] for output to
%./outputs/simulation_data/pressure_case/
output_sim_mat_w_meas = [input_vec output_vec output_sim_mat]

%outputting simulation data to ./outputs/simulation_data/pressure_case/
test = input('output simulation data as .csv file ? (y/n): ','s');
if test == 'y'
    file_name = input('file name for output?: ', 's')
    output_file = strcat('./outputs/simulation_data/',pressure_case,'/', file_name,'.csv');
    writematrix(output_sim_mat_w_meas, output_file)
else
    else_string =  'output simulation data not output to csv'
end





