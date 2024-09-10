import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
"""
Title: OLS_param_estimate_w_est_variance(y_vec, x_mat_t)

Summary:
Function performs ordinary least squares (OLS) estimation of the parameters a Gauss Markov Model with Normal Errors 
(GMMNE) General Linear Model (GLM) of the form:
 
 y = XB + e ; e~N(0,sigma^2)
 
Where:
B_hat = X*(X^T*X)^-1*X^T*y is the BLUE (Best Linear Unbiased Estimator)

sigma^2_hat = (y^T*(I-P_x)*y)/(n-r)= SSE/(n-r) = sum((y_i - y_hat_i)^2)/(n-r) is the error variance
Note: 
n = number of observations = number of rows in y_vec
r = rank(X)

cov(B_hat)_hat = sigma^2_hat*(X^T*X)^-1

var(B_hat)_hat = diag(cov(B_hat)_hat)

Inputs:
1) vector of responses, y_vec = y (Nx1 vector)
2) transpose of matrix of known constants, x_mat_t = X^T (note the transpose is input as it is easier to setup the transpose
    in numpy) (MxN matrix)
    
Outputs:
1) vector of estimated parameters, B_hat (Mx1 numpy array)
2) estimated covariance matrix of estimated parameters, cov(B_hat)_hat (MxM numpy array)
"""

def OLS_param_estimate_w_est_variance(y_vec,x_mat_t):

    # parameter estimation using OLS
    x_mat = np.transpose(x_mat_t)
    x_t_mul_x = np.matmul(x_mat_t, x_mat)
    x_t_mul_x_inv = np.linalg.inv(x_t_mul_x)
    x_t_mul_x_inv_mul_x_t = np.matmul(x_t_mul_x_inv, x_mat_t)
    beta_hat = np.matmul(x_t_mul_x_inv_mul_x_t, y_vec)

    # estimation of variance
    y_vec_shape = np.shape(y_vec)
    num_rows = y_vec_shape[0]
    rank_x_mat = np.linalg.matrix_rank(x_mat)
    expect_y = np.matmul(x_mat, beta_hat)
    residuals = y_vec - expect_y
    squared_residuals = np.square(residuals)
    sse = np.sum(squared_residuals)
    var_est = sse / (num_rows - rank_x_mat)

    # estimation of covariance matrix of estimated parameters
    est_cov_mat = var_est * x_t_mul_x_inv
    return beta_hat, est_cov_mat
"""
**********************************************END OF FUNCTION***********************************************************
"""

'''
Title: cropped_plot (df)

Below function uses SpanSelector widget to select data we want to use for system identificaiton using transient data 
from experiments step-pressure experiments. Formulation of code is custom built for our case, but the general idea of the 
code closely resembles that of the SpanSelector example from the matplotlib documentation, which can be found at the 
below link:

    https://matplotlib.org/stable/gallery/widgets/span_selector.html

Summary: 
Function intakes a dataframe of pressure and time data and allows the user to select a region of that data that they want to 
use for future analysis, upon selection the user is prompted to either crop the new data region, or use the current region
for future analysis. The user can continually crop the new region, where each crop sets dict_of_df['original'] = df_
data_for_crop and dict_of_df['cropped] = df_newly_cropped. Once the user no longer wants to refine the data region the 
dictionary with the most recent cropped df pair is output.

Inputs: 
1) df, a dataframe of the form [Time [s], Pressure [mbar]]

Outputs:
1) dict_of_df = {original:df_old , cropped:df_new}

'''

def cropped_plot(df):
    n=0

    #creating dictionary of cropped and original dataframe
    dict_of_dfs = {}
    dict_of_dfs["original"] = df

    while n != 1:
        # set font and fontsize
        plt.rc('font', family='Times New Roman')
        plt.rcParams.update({'font.size': 12})

        #creating subplot figure (top is original plot bottom is cropped plot)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

        #setting x and y limits
        ax1.plot(dict_of_dfs["original"]["Time [s]"], dict_of_dfs["original"]["Pressure [mbar]"])
        ax1.set_xlim(dict_of_dfs["original"]["Time [s]"].min(), dict_of_dfs["original"]["Time [s]"].max())
        ax1.set_ylim(dict_of_dfs["original"]["Pressure [mbar]"].min(), dict_of_dfs["original"]["Pressure [mbar]"].max())

        #instructions on what to do
        ax1.set_title('Press left mouse button and drag '
                      'to select a region in the top graph')

        # setting legend and ylabel for full plot
        ax1.set(ylabel="P [mbar]")

        line2, = ax2.plot([], [])

        # setting x and y labels of cropped plot and grid
        ax2.set(xlabel="Time [s]", ylabel="P [mbar]")
        ax2.grid()

        #callback fn, invoked after the SpanSelector fn finishes (the range of the values is selected in the graph)
        def onselect(xmin, xmax):
            indmin, indmax = np.searchsorted(dict_of_dfs["original"]["Time [s]"], (xmin, xmax))
            indmax = min(len(dict_of_dfs["original"]["Time [s]"]) - 1, indmax)

            #x and y values of selected region
            region_x = dict_of_dfs["original"]["Time [s]"][indmin:indmax]
            region_y = dict_of_dfs["original"]["Pressure [mbar]"][indmin:indmax]

            #creating dataframe of region selected and adding to dictionary
            cropped_df = pd.DataFrame({"Time [s]": region_x, "Pressure [mbar]": region_y})
            dict_of_dfs["cropped"] = cropped_df

            if len(region_x) >= 2:
                line2.set_data(region_x, region_y)
                ax2.set_xlim(region_x.min(), region_x.max())
                ax2.set_ylim(region_y.min(), region_y.max())
                fig.canvas.draw_idle()

        span = SpanSelector(
            ax1,
            onselect,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True
        )
        # Set useblit=True on most backends for enhanced performance.
        plt.show()
        question = input("Do you want to edit the cropped plot? (y/n): ")
        while question != 'y' and question != 'n':
            question = input('not a valid response, try again: ')
        if question == 'y':
            cropped_dict_values = dict_of_dfs['cropped']
            dict_of_dfs['original'] = cropped_dict_values
        elif question =='n':
            n=1
    return dict_of_dfs