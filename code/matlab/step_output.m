%{
step_output.m

Summary:
Program is used to output step response data for a given general underdamped second
order system of the form:

G(s) = b/(s^2 + a*s + b) ; a = 2*dr*wn, b=wn^2

Where the intook ct_params come in a .csv file of the form:

[0 0 b; 1 a b]

Outputs a .csv file of the form

[y_out, t_out] 
to

./outputs/step_data/..

for use in python or matlab plotting.
%}

ct_params = readmatrix('./outputs/continuous_time_params/250_mbar/w_n_and_dr/250_mbar_3_18_2024_tr.csv')

system = tf(ct_params(1,3),ct_params(2,:) );

time_vec = linspace(0,0.02);

[y_out,t_out] = step(system, time_vec);

output_data = [y_out t_out];

writematrix(output_data, './outputs/step_data/3_18_2024/250_mbar_3_18_2024_tr.csv');



