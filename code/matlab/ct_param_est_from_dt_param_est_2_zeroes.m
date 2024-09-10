%{
ct_param_est_from_dt_param_est_2_zeroes.m

Summary:
Code is to intake the output estimated parameters from
s2_order_LS_w_2_zero.py (in
../python/outputs/estimated_params/second_order_zeroes/
for ZOH equivalent transfer function of general
2nd order continuous time system.

H(z)  = (b_1*z^2+b_2*z+b_3)/(z^2 + a_1*z + a_2) 

Code then converts from dt to ct and
plots the step response. Parameters from CT representation are output to
./outputs/continuous_time_params for use in lsim_prgrm.m 
%}

%setting path of the estimated parameters for given test
est_params_folder = '../python/outputs/estimated_params/second_order_zeroes/2_25_24/';
pressure = '250_mbar'
test_file = '250_mbar_so_zero.csv';%only line you need to change (unless considering two_phase_tests)
read_matrix_path = strcat(est_params_folder,test_file);

%{
Reading in data at path set above to a matrix of the form:

[a_1_hat a_1_hat_uncert; a_2_hat a_2_hat_uncert; b_1_hat b_1_hat_uncert;
b_2_hat b_2_hat_uncert; b_3_hat b_3_hat_uncert]

%}

params_and_uncert_mat = readmatrix(read_matrix_path);
params_and_uncert_mat = params_and_uncert_mat(:,2:3);

%{
Defining discrete time transfer function of form 

H(z) = b_1*z^2+b_2*z+b_3/(z^2+a_1*z+a_2)

%}

num = [params_and_uncert_mat(3,1) params_and_uncert_mat(4,1) params_and_uncert_mat(5,1)]; %numerator parameters (from estimate)
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
info = stepinfo(ct_sys) %put 'RiseTimeLimits',[0 0.63] for first order approx (not used here)

%outputting ct parameters to file
ct_params = [ct_num; ct_den];
test = input(strcat('output continuous time parameter values .csv file for ', test_file(1,1:end-4), ' ? (y/n): '),'s');
if test == 'y'
    file_name = input('file name for output?: ', 's')
    output_file = strcat('./outputs/continuous_time_params/',pressure,'/', file_name,'.csv');
    writematrix(ct_params, output_file)
else
    else_string = strcat(test_file(1,1:end-4), ' continuous time parameter values not output to csv')
end
