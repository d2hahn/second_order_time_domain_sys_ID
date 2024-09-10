"""
csv_crop_prgrm.py

Summary:
Program is intended to take the transient data obtained from step response testing of the pressure junctions with the
input files being .csv's obtained using DAQExpress/LabVIEW software in ../../junction_dynamics_data/. Program allows user to
choose the file they want to crop and from which folder. The input file is then cropped using an edited version of
SpanSelector widget from matplotlib. This edited version prompts the user after an initial crop if they want to crop the
newly cropped data, until a reasonable portion of output time domain data is obtained.The user can then choose to output
the cropped file to .csv for use in second_order_approx_Tr_and_OS.py, second_order_approx_w_LS.py, and/or 2_order_LS_w
_2_zero.py.

Dependencies:
    1. pandas
    2. matplotlib.pyplot
    3. numpy
    4. matplotlib.widgets (SpanSelector)

Notes:
    1. Change output folder, input folder, and test (lines 31, 34, 35 ) for different folders/files you want to read in
    2. May have to change pressure sensor calibration curve equation (see line 56 )
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector
from functions import cropped_plot

#type in name of output folder
output_folder = './outputs/cropped_data/3_18_2024/'

#give parameters for folder and what test in folder to read
folder = '../../junction_dynamics_data/'
test = '250_mbar_3_18_2024'
type_of_file = '.csv'


#concatenate folder, test, and file type into a single string representing the relative path of the datafile
path = folder+test+type_of_file

#read data into dataframe from .csv file
df = pd.read_csv(path)

#reorganizing dataframe into form [Time [s] Pressure [V]], and ensuring values are float
df = df.drop([0,1,2])
df.columns = ["Time [s]", "Pressure [V]"]
df = df.reset_index()
df = df.drop(columns=['index'])
df = df.astype('float') #need to convert values from txt to float

#calculating pressure from voltage using calibration curve (P1 used for present analysis)
#P1 calibration curve: P[mbar] = 201.09*E[V] -1.9532[V]
#P2 calibration curve: P[mbar] = 200.75*E[V] -1.3837[V]

df['Pressure [mbar]'] = 201.09*df["Pressure [V]"] - 1.9532


#selecting portion of data to use using span selector
#below code is slightly modified version of span_selector example given on matplotlib documentation
#https://matplotlib.org/stable/gallery/widgets/span_selector.html

dict_of_dfs = cropped_plot(df)
# 'Allow for finer data selection with while loop'
# # set font and fontsize
# plt.rc('font', family='Times New Roman')
# plt.rcParams.update({'font.size': 12})
#
# #creating subplot figure (top is original plot bottom is cropped plot)
# fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
#
# #setting x and y limits
# ax1.plot(df["Time [s]"], df["Pressure [mbar]"])
# ax1.set_xlim(df["Time [s]"].min(), df["Time [s]"].max())
# ax1.set_ylim(df["Pressure [mbar]"].min(), df["Pressure [mbar]"].max())
#
# #instructions on what to do
# ax1.set_title('Press left mouse button and drag '
#               'to select a region in the top graph')
#
# # setting legend and ylabel for full plot
# ax1.set(ylabel="P [mbar]")
#
# line2, = ax2.plot([], [])
#
# # setting x and y labels of cropped plot and grid
# ax2.set(xlabel="Time [s]", ylabel="P [mbar]")
# ax2.grid()
#
# #creating dictionary of cropped and original dataframe
# dict_of_dfs = {}
# dict_of_dfs["original"] = df
#
# #callback fn, invoked after the SpanSelector fn finishes (the range of the values is selected in the graph)
# def onselect(xmin, xmax):
#     indmin, indmax = np.searchsorted(df["Time [s]"], (xmin, xmax))
#     indmax = min(len(df["Time [s]"]) - 1, indmax)
#
#     #x and y values of selected region
#     region_x = df["Time [s]"][indmin:indmax]
#     region_y = df["Pressure [mbar]"][indmin:indmax]
#
#     #creating dataframe of region selected and adding to dictionary
#     cropped_df = pd.DataFrame({"Time [s]": region_x, "Pressure [mbar]": region_y})
#     dict_of_dfs["cropped"]  = cropped_df
#
#     if len(region_x) >= 2:
#         line2.set_data(region_x, region_y)
#         ax2.set_xlim(region_x.min(), region_x.max())
#         ax2.set_ylim(region_y.min(), region_y.max())
#         fig.canvas.draw_idle()
#
#
#
# span = SpanSelector(
#     ax1,
#     onselect,
#     "horizontal",
#     useblit=True,
#     props=dict(alpha=0.5, facecolor="tab:blue"),
#     interactive=True,
#     drag_from_anywhere=True
# )
# # Set useblit=True on most backends for enhanced performance.
# plt.show()

#set cropped df first index to zero
dict_of_dfs["cropped"]= dict_of_dfs["cropped"].reset_index()
dict_of_dfs["cropped"] = dict_of_dfs["cropped"].drop(columns=['index'])

#set cropped df first time to zero
dict_of_dfs["cropped"]["Time [s]"] = dict_of_dfs["cropped"]["Time [s]"] - dict_of_dfs["cropped"]["Time [s]"][0]

question = input("output to csv? (y/n): ")
if question == 'y':
    second_question = input("for 1: OLS or 2: Plot? (input 3 to cancel) (input 1 or 2 or 3) : ")
    while second_question != '1' and second_question != '2' and second_question != '3':
        print("not a valid entry, retry")
        second_question = input("for 1: OLS or 2: Plot? (input 3 to cancel) (input 1 or 2 or 3) : ")
    if second_question =='1':
        #output cropped dataframe to .csv
        test_name = input("type in name of test: ")
        dict_of_dfs["cropped"].to_csv(output_folder+test_name+type_of_file)
        print('output cropped .csv to '+ output_folder)
    elif second_question == '2':
        test_name = input("type in name of test: ")
        dict_of_dfs["cropped"].to_csv('./outputs/cropped_data/plots/'+test_name+type_of_file)
        print('output cropped .csv to ./outputs/cropped_data/plots/')
    else:
        print("not output to .csv")
else:
    print("not output to .csv")