%{
second_order_response_from_dr_and_wn.m

Summary:
Code intakes the damping ratio,dr, and natural frequency,wn, [rad/s] estimated
from second_order_approx_Tr_and_OS.py (../python/outputs/dr_and_wn/) and
evaluates the parameters for the general second order underdamped transfer
function of the form:

G(s) = b/(s^2 + a*s + b) ; a = 2*dr*wn, b=wn^2

Also calculates the uncertainty in each parameter, as well as determines
the upper and lower bound bode plot of the FR of the estimated system. Code
outputs the parameters calculated from rise time and overshoot (_tr) and
peak time and overshoot (_tp), the uncertainties (as well and upper and
lower bound) are only output for the _tr files. Output to ./outputs/continuous
 _time_params/w_n_and_dr/pressure/ in the form
 
[0 0 b; 1 a b]

[u_a rel_u_a u_b rel_u_b]

As well, the bode data for
the system (and upper and lower bound for _tr) is output to
./outputs/simulation_data/bode/pressure/ in the form:

[mag [dB] phase [deg] time [s]]

over the frequency range 10^-2 <-> 10^6 [rad/s] (can change)

For uncertainty in parameters
equations please see Thesis and/or analysis binder.

Notes:
1. Change value of pressure, ensure you have a directory named to that
pressure (line 40)
2. Read in appropriate file (line 45)
%}

pressure = '250_mbar'; %fill out before each run (for output)

%reading in .csv file from ../python/outputs/dr_and_wn of form:
% [index, peak_time [s], rise time [s],  overshoot, dr, w_n_tr [rad/s], w_n_tp [rad/s]]
dr_wn_folder = '../python/outputs/dr_and_wn/';
test_file = '250_mbar_3_18_2024';
dr_and_wn_mat = readmatrix(strcat(dr_wn_folder,test_file,'.csv'));

%reading in .csv file frpm ../python/outputs/dr_and_wn of form:
%[index del_u_tr, del_u_os, del_u_dr, del_u_wn] (every second column after index is rel_u)
uncert_mat = readmatrix(strcat(dr_wn_folder,test_file,'_uncert.csv'));


%cleaning up matrix to [peak_time [s], rise time [s],  overshoot, dr, w_n_tr [rad/s], w_n_tp [rad/s]]
dr_and_wn_mat =dr_and_wn_mat(2,2:end);

%cleaning up uncertainty matrix to [del_u_tr, del_u_os, del_u_dr, del_u_wn]
uncert_mat = uncert_mat(2,2:2:end);

%{
creating general second order tf of form:

G(s) = b/(s^2 + a*s+ b); a = 2*dr*wn, b=wn^2

for both the w_n estimated from rise time and peak time
%}

w_n_tr = dr_and_wn_mat(1,5);
w_n_tp = dr_and_wn_mat(1,6);
dr = dr_and_wn_mat(1,4);

num_tr = [0 0 w_n_tr^2];
den_tr = [1 2*dr*w_n_tr w_n_tr^2];
num_tp = [0 0 w_n_tp^2];
den_tp = [1 2*dr*w_n_tp w_n_tp^2];

system_tr = tf(num_tr,den_tr);
system_tp = tf(num_tp, den_tp);

%calculating uncertainty in parameters (a and b)
%del_u_a
d_a_d_dr  = 2*w_n_tr;
d_a_d_wn = 2*dr;
del_u_a = sqrt((d_a_d_dr*uncert_mat(1,3))^2+(d_a_d_wn*uncert_mat(1,4))^2);

%del_u_b
del_u_b = 2*w_n_tr*uncert_mat(1,4);

%calculating upper and lower bound numerator and denominator of system
num_tr_ub = num_tr+[0 0 del_u_b];
num_tr_lb = num_tr-[0 0 del_u_b];

den_tr_ub = den_tr + [0 del_u_a del_u_b];
den_tr_lb = den_tr - [0 del_u_a del_u_b];

%obtaining upper and lower bound system
system_tr_ub = tf(num_tr_ub,den_tr_ub);
system_tr_lb = tf(num_tr_lb, den_tr_lb);

%plotting step response 
figure(1)
step(system_tr)
figure(2)
step(system_tp)

%plotting bode 
w_range = {10^-2, 10^6}; %setting range of frequencies to encompass frequencies considered in experiments

figure(3)
bode(system_tr,w_range)


figure(4)
bode(system_tp,w_range)
%obtaining the magnitude, phase, and frequency data for use in nicer plotting
%programs
[mag_tr,phase_tr,wout_tr] = bode(system_tr,w_range);
[mag_tr_ub, phase_tr_ub, wout_tr_ub] = bode(system_tr_ub,w_range);
[mag_tr_lb, phase_tr_lb, wout_tr_lb] = bode(system_tr_lb,w_range);


[mag_tp,phase_tp,wout_tp] = bode(system_tp,w_range);

%converting magnitude to dB (see bode() documentation)
mag_vec_tr = squeeze(mag_tr);
mag_vec_dB_tr = 20*log10(mag_vec_tr);

mag_vec_tr_ub = squeeze(mag_tr_ub);
mag_vec_dB_tr_ub = 20*log10(mag_vec_tr_ub);

mag_vec_tr_lb = squeeze(mag_tr_lb);
mag_vec_dB_tr_lb = 20*log10(mag_vec_tr_lb);

mag_vec_tp = squeeze(mag_tp);
mag_vec_dB_tp = 20*log10(mag_vec_tp);

%obtaining value of phase FR in deg
phase_vec_tr = squeeze(phase_tr);
phase_vec_tr_ub = squeeze(phase_tr_ub);
phase_vec_tr_lb = squeeze(phase_tr_lb);


phase_vec_tp = squeeze(phase_tp);

%organizing bode data [mag [dB], phase [deg], omega [rad/s]]
bode_output_data_tr = [mag_vec_dB_tr phase_vec_tr wout_tr];
bode_output_data_tr_ub = [mag_vec_dB_tr_ub phase_vec_tr_ub wout_tr_ub];
bode_output_data_tr_lb = [mag_vec_dB_tr_lb phase_vec_tr_lb wout_tr_lb];

bode_output_data_tp = [mag_vec_dB_tp phase_vec_tp wout_tp];

%organizing parameters [num ; den] ~ [0  0 b ; 1 a b] 
params_output_form_tr  = [num_tr ; den_tr];
params_output_form_tr_ub  = [num_tr_ub ; den_tr_ub];
params_output_form_tr_lb  = [num_tr_lb ; den_tr_lb];

params_output_form_tp  = [num_tp ; den_tp];

%organizing uncertainty [del_u_a[rad/s] del_u_a_rel [%] del_u_b[rad/s]
%del_u_b_rel [%]]
uncert_output_form_tr = [del_u_a del_u_a/den_tr(1,2)*100 del_u_b del_u_b/num_tr(1,3)*100];


% % figure(5)
% % semilogx(wout_tr,mag_vec_dB_tr)
% % yticks([-320:40:0])
% % grid on

%outputting ct parameters to./continuous_time_params/press_case/w_n_and_dr/
test = input(strcat('output continuous time parameter values .csv file for ', test_file, ' ? (y/n): '),'s');
if test == 'y'
    file_name = input('file name for output?: ', 's')
    output_file_tr = strcat('./outputs/continuous_time_params/',pressure,'/w_n_and_dr/', file_name,'_tr.csv');
    output_file_tr_ub = strcat('./outputs/continuous_time_params/',pressure,'/w_n_and_dr/', file_name,'_tr_ub.csv');
    output_file_tr_lb = strcat('./outputs/continuous_time_params/',pressure,'/w_n_and_dr/', file_name,'_tr_lb.csv');
    output_file_tr_uncert = strcat('./outputs/continuous_time_params/',pressure,'/w_n_and_dr/', file_name,'_tr_uncert.csv');
    
    output_file_tp = strcat('./outputs/continuous_time_params/',pressure,'/w_n_and_dr/', file_name,'_tp.csv');
    writematrix(params_output_form_tr, output_file_tr)
    writematrix(params_output_form_tr_ub, output_file_tr_ub)
    writematrix(params_output_form_tr_lb, output_file_tr_lb)
    writematrix(uncert_output_form_tr, output_file_tr_uncert)

    writematrix(params_output_form_tp, output_file_tp)
else
    else_string = strcat(test_file, ' continuous time parameter values not output to csv')
end

%outputting bode data to ./outputs/simulation_data/bode/pressure/
test = input('output bode data as .csv file ? (y/n): ','s');
if test == 'y'
    file_name = input('file name for output?: ', 's')
    output_file_tr = strcat('./outputs/simulation_data/bode/',pressure,'/', file_name,'_tr.csv');
    output_file_tr_ub = strcat('./outputs/simulation_data/bode/',pressure,'/', file_name,'_tr_ub.csv');
    output_file_tr_lb = strcat('./outputs/simulation_data/bode/',pressure,'/', file_name,'_tr_lb.csv');

    output_file_tp = strcat('./outputs/simulation_data/bode/',pressure,'/', file_name,'_tp.csv');
    writematrix(bode_output_data_tr, output_file_tr)
    writematrix(bode_output_data_tr_ub, output_file_tr_ub)
    writematrix(bode_output_data_tr_lb, output_file_tr_lb)
    writematrix(bode_output_data_tp, output_file_tp)
else
    else_string =  'output bode data not output to csv'
end


