# second_order_time_domain_sys_ID


## Summary
Python and MATLAB codebase for performing second-order system identification from time-domain data of a systems dynamic response to a decreasing step input function. Both discrete and continuous time estimation methods are available. Uncertainties given at the 95% confidence level. Code was originally developed to characterize dynamics of pressure sensor apparatuses through decreasing step response testing.

## Dependencies
### Python
1. functions.py
2. matplotlib.pyplot
3. matplotlib.SpanSelector
4. numpy
5. pandas
6. scipy

### MATLAB
1. Control Systems Toolbox

## Order of Use of Code
1. csv_crop_prgrm.py (crop time-domain data to obtain portion of response that carries the dynamic chracteristics.)

### Discrete-Time OLS Identification
2. second_order_approx_w_LS.py or 2_order_LS_w_2_zero.py (Perform OLS fit of discrete-time 2nd order model to cropped time-domain data considering either a single zero or two zeroes, outputs estimated parameters.)
3. ct_param_est_from_dt_param_est.m or ct_param_est_from_dt_param_est_2_zeroes.m (Convert discrete-time transfer function estimated in 2. to continuous-time, also obtain step response data of simulated transfer function for comparison to experimental data. Use for either one or two zeroes.)
4. lsim_prgrm.m (Obtain simulation data of the response of the estimated model to the same input applied to the real system.)

### Continuous-Time Identification
2. second_order_approx_Tr_and_OS.py (Estimate natural frequency and damping ratio from response charactersitics obtained from cropped time-domain data.)
3. second_order_response_from_dr_and_wn.m (Calculating general second-order TF parameters from estimated nat. freq and damping ratio. Also obtaining frequency response data.)
4. lsim_prgrm.m (Obtain simulation data of the response of the estimated model to the same input applied to the real system.)


