%{
ct_param_est_from_dt_param_est.m

Summary:
Code is to intake the output estimated parameters from
second_order_approx_w_LS.py (in ../python/outputs/estimated_params/
for ZOH equivalent transfer function of general
2nd order continuous time system.

H(z)  = (b_1 +b_2*z)/(z^2 + a_1*z + a_2) 

Code then converts from dt to ct and
plots the step response. Parameters from CT representation are output to
./outputs/continuous_time_params for use in lsim_prgrm.m 
%}


%setting path of the estimated parameters for given test
est_params_folder = '../python/outputs/estimated_params/erkorkmaz_2_13_2024/'; %change on each run
pressure = '200_mbar' %change for different pressures
test_file = '200_mbar.csv';%change depending on estimated_params_folder
read_matrix_path = strcat(est_params_folder,test_file);

%{
Reading in data at path set above to a matrix of the form:

[a_1_hat a_1_hat_uncert; a_2_hat a_2_hat_uncert; b_1_hat b_1_hat_uncert;
b_2_hat b_2_hat_uncert]

%}

params_and_uncert_mat = readmatrix(read_matrix_path);
params_and_uncert_mat = params_and_uncert_mat(:,2:3);

%{
Defining discrete time transfer function of form 

H(z) = b_1*z+b_2/(z^2+a_1*z+a_2)

%}

num = [params_and_uncert_mat(3,1) params_and_uncert_mat(4,1)]; %numerator parameters (from estimate)
den = [1 params_and_uncert_mat(1,1) params_and_uncert_mat(2,1)]; %denominator parameters (from estimate)
ts = 1/10000; %sample time [s]

dt_sys = tf(num,den,ts)

%converting DT TF to CT
ct_sys = d2c(dt_sys)

%obtaining numerator and denominator coeffecients of CT estimate
[ct_num,ct_den] = tfdata(ct_sys,'v');

%used roots to check if we could approximate higher order model aas second order
%(could not)
zeroes = roots(ct_num);
poles = roots(ct_den);

%obtaining step data (for plots) and information (rise time and settling time)
step(ct_sys)
[output, time] = step(ct_sys);
info = stepinfo(ct_sys); %put 'RiseTimeLimits',[0 0.63] for first order approx (not used here)

%outputting ct parameters to file
ct_params = [ct_num; ct_den];
test = input(strcat('output continuous time parameter values .csv file for ', test_file, ' ? (y/n): '),'s');
if test == 'y'
    file_name = input('file name for output?: ', 's')
    output_file = strcat('./outputs/continuous_time_params/',pressure,'/', file_name,'.csv');
    writematrix(ct_params, output_file)
else
    else_string = strcat(test_file, ' continuous time parameter values not output to csv')
end

% %repeating for upper bound of uncertainty
% num_ub = [params_and_uncert_mat(3,1)+params_and_uncert_mat(3,2) params_and_uncert_mat(4,1)+params_and_uncert_mat(4,2)]; %numerator parameters (from estimate)
% den_ub = [1 params_and_uncert_mat(1,1)+params_and_uncert_mat(1,2) params_and_uncert_mat(2,1)+params_and_uncert_mat(2,2)]; %denominator parameters (from estimate)
% 
% dt_sys_ub = tf(num_ub,den_ub,ts)
% 
% %converting DT TF to CT
% ct_sys_ub = d2c(dt_sys_ub)
% 
% %obtaining numerator and denominator coeffecients of CT estimate
% [ct_num_ub,ct_den_ub] = tfdata(ct_sys_ub,'v');
% 
% %used roots to check if we could approximate higher order model aas second order
% %(could not)
% roots(ct_num_ub);
% roots(ct_den_ub);
% 
% %obtaining step data (for plots) and information (rise time and settling time)
% [output_ub, time_ub] = step(ct_sys_ub);
% info_ub = stepinfo(ct_sys_ub) %put 'RiseTimeLimits',[0 0.63] for first order approx (not used here)
% 
% %repeating for lower bound of uncertainty
% num_lb = [params_and_uncert_mat(3,1)-params_and_uncert_mat(3,2) params_and_uncert_mat(4,1)-params_and_uncert_mat(4,2)]; %numerator parameters (from estimate)
% den_lb = [1 params_and_uncert_mat(1,1)-params_and_uncert_mat(1,2) params_and_uncert_mat(2,1)-params_and_uncert_mat(2,2)]; %denominator parameters (from estimate)
% 
% dt_sys_lb = tf(num_lb,den_lb,ts)
% 
% %converting DT TF to CT
% ct_sys_lb = d2c(dt_sys_lb)
% 
% %obtaining numerator and denominator coeffecients of CT estimate
% [ct_num_lb,ct_den_lb] = tfdata(ct_sys_lb,'v');
% 
% %used roots to check if we could approximate higher order model aas second order
% %(could not)
% roots(ct_num_lb);
% roots(ct_den_lb);
% 
% %obtaining step data (for plots) and information (rise time and settling time)
% [output_lb, time_lb] = step(ct_sys_lb);
% info_lb = stepinfo(ct_sys_lb) %put 'RiseTimeLimits',[0 0.63] for first order approx (not used here)

% %organizing data
% %ct_params = [actual upper_bound lower_bound]
% ct_num_params = [transpose(ct_num) transpose(ct_num_ub) transpose(ct_num_lb)];
% ct_den_params = [transpose(ct_den) transpose(ct_den_ub) transpose(ct_num_lb)];
% 
% %transient characterstics
% %rise_time = [actual_rt rt_ub rt_lb]
% rise_time = [info.RiseTime info_ub.RiseTime info_lb.RiseTime];
% 
% %set_time = [actual_st st_ub st_lb]
% set_time = [info.SettlingTime info_ub.SettlingTime info_lb.SettlingTime];
% 
% %step response
% %step_data = [actual time  ub time_ub lb time_lb]
% step_data = [output time output_ub time_ub output_lb time_lb];
% 
% %output data to files

