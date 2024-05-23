# compare ABC_basic with ABC_pseudostates

import importlib
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

path_basic = 'comparison_ABC/ABC_basic/'
path_pseudo = 'comparison_ABC/ABC/'


    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# read in pseudo irf files (not within loop)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
list_of_dfs_pseudo = []

all_files_pseudo = os.listdir(path_pseudo)
irf_files = [file for file in all_files_pseudo if "irf_" in file]
for irf_file in irf_files:
    file_path = os.path.join(path_pseudo, irf_file)
    df = pd.read_csv(file_path)  
    list_of_dfs_pseudo.append(df)




# read in pseudo moments files (not within loop)
list_of_dfs_pseudo_sim = []

# also read the pseudo files
all_files = os.listdir(path_pseudo)
sim_files = [file for file in all_files if "simulated_" in file]
for f in sim_files:
    file_path = os.path.join(path_pseudo, f)
    df = pd.read_csv(file_path)  # Adjust the reading function if your files are in a different format
    list_of_dfs_pseudo_sim.append(df)

df_pseudo_sim_all = pd.concat(list_of_dfs_pseudo_sim, axis=1)
pseudo_summary = df_pseudo_sim_all.describe()

list_of_irf_dfs = []
list_of_moment_dfs = []
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# loop over the folders
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for folder in range(1,11):


# List all files in the folder
    all_files_basic_path = path_basic + str(folder) + '/'
    all_files_basic = os.listdir(all_files_basic_path)

    # Filter files that contain "irf_"
    irf_files = [file for file in all_files_basic if "irf_" in file]

    # Initialize an empty list to store DataFrames
    list_of_dfs = []

    # Read each irf_file into a DataFrame and append to the list
    for irf_file in irf_files:
        file_path = os.path.join(all_files_basic_path, irf_file)
        df = pd.read_csv(file_path)  
        list_of_dfs.append(df)

    # create one dataframe with all the irfs
    df_basic_all = pd.concat(list_of_dfs, axis=1)

    # select the columns from iteration folder 
    # Initialize an empty list to store DataFrames
    list_of_dfs_pseudo_folder = []

    for df in list_of_dfs_pseudo:
        df_folder = df.filter(regex=f'_{folder}$', axis=1)
        df_folder.columns = df_folder.columns.str.replace(f'_{folder}$', '', regex=True)


        list_of_dfs_pseudo_folder.append(df_folder)
    # create one dataframe with all the irfs
    df_pseudo_all = pd.concat(list_of_dfs_pseudo_folder, axis=1)

    # plot irfs
    fig, axs = plt.subplots(2, 4, figsize=(10, 10))
    axs[0, 0].plot(df_basic_all['K_x_ax'], label='basic', color='blue')
    axs[0, 0].plot(df_pseudo_all['K_x_ax'], label='pseudo', color='red')
    axs[0, 0].set_xlabel('Index')
    axs[0, 0].set_ylabel('Values')
    axs[0, 0].set_title('K_x (ax)')
    #axs[0, 0].legend(loc='upper right')

    axs[0, 1].plot(df_basic_all['a_x_ax'], label='basic', color='blue')
    axs[0, 1].plot(df_pseudo_all['a_x_ax'], label='pseudo', color='red')
    axs[0, 1].set_xlabel('Index')
    axs[0, 1].set_ylabel('Values')
    axs[0, 1].set_title('a_x')
    #axs[0, 1].legend(loc='upper right')

    axs[0, 2].plot(df_basic_all['Ishare_y_ax'], label='basic', color='blue')
    axs[0, 2].plot(df_pseudo_all['Ishare_y_ax'], label='pseudo', color='red')
    axs[0, 2].set_xlabel('Index')
    axs[0, 2].set_ylabel('Values')
    axs[0, 2].set_title('Ishare_y (ax)')
    #axs[0, 2].legend(loc='upper right')

    axs[0, 3].plot(df_basic_all['Lamda_y_ax'], label='basic', color='blue')
    axs[0, 3].plot(df_pseudo_all['Lamda_y_ax'], label='pseudo', color='red')
    axs[0, 3].set_xlabel('Index')
    axs[0, 3].set_ylabel('Values')
    axs[0, 3].set_title('Lamda_y (ax)')
    #axs[0, 3].legend(loc='upper right')

    axs[1, 0].plot(df_basic_all['K_x_dx'], label='basic', color='blue')
    axs[1, 0].plot(df_pseudo_all['K_x_dx'], label='pseudo', color='red')
    axs[1, 0].set_xlabel('Index')
    axs[1, 0].set_ylabel('Values')
    axs[1, 0].set_title('K_x (dx)')
    #axs[1, 0].legend(loc='upper right')

    axs[1, 1].plot(df_basic_all['d_x_dx'], label='basic', color='blue')
    axs[1, 1].plot(df_pseudo_all['d_x_dx'], label='pseudo', color='red')
    axs[1, 1].set_xlabel('Index')
    axs[1, 1].set_ylabel('Values')
    axs[1, 1].set_title('d_x')
    #axs[1, 1].legend(loc='upper right')

    axs[1, 2].plot(df_basic_all['Ishare_y_dx'], label='basic', color='blue')
    axs[1, 2].plot(df_pseudo_all['Ishare_y_dx'], label='pseudo', color='red')
    axs[1, 2].set_xlabel('Index')
    axs[1, 2].set_ylabel('Values')
    axs[1, 2].set_title('Ishare_y (dx)')
    #axs[1, 2].legend(loc='upper right')

    axs[1, 3].plot(df_basic_all['Lamda_y_dx'], label='basic', color='blue')
    axs[1, 3].plot(df_pseudo_all['Lamda_y_dx'], label='pseudo', color='red')
    axs[1, 3].set_xlabel('Index')
    axs[1, 3].set_ylabel('Values')
    axs[1, 3].set_title('Lamda_y (dx)')
    #axs[1, 3].legend(loc='upper right')

    # Create a single legend for both subplots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    fig_path = '../notebooks/ABC_model/latex/tables/irf_comparison/irf_comparison_' + str(folder) + '.png'
    fig.savefig(fig_path, bbox_inches='tight')




    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # initialize a list to store the results
    df_diffs = pd.DataFrame()
    # loop over the columns of df_basic_all
    for column in df_basic_all.columns:

        # calculate the percentage difference ERROR
        df_diff = abs(df_pseudo_all[column] - df_basic_all[column]) / abs(df_basic_all[column]) 
        # add the column name
        df_diff.name = column
        # append the result to the list
        df_diffs = pd.concat([df_diffs,df_diff],axis=1)

    #print(df_diffs)
    irf_export_file = 'comparison_ABC/results/irf_diffs_' + str(folder) + '.csv'
    df_diffs.to_csv(irf_export_file)

    # append to list of irf dfs
    list_of_irf_dfs.append(df_diffs)


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # compare moments
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # Filter files that contain "simulated_"
    sim_files = [file for file in all_files_basic if "simulated_" in file]

    # Initialize an empty list to store DataFrames
    list_of_dfs = []
    

    # Read each sim_file into a DataFrame and append to the list
    for f in sim_files:
        file_path = os.path.join(all_files_basic_path, f)
        df = pd.read_csv(file_path)  # Adjust the reading function if your files are in a different format
        if 'euler' in f:
            df = df.abs() * 1000
            #df = df.add_suffix('\n(1E3)')
        list_of_dfs.append(df)

    # create one dataframe with all the sims
    df_basic_all = pd.concat(list_of_dfs, axis=1)

    

    # for each column of df_basic_all calculate the descriptive statistics and calculate the percentage difference to the descriptive value of df_pseudo_all that has the exact same name
    basic_summary = df_basic_all.describe()

    # save for latex
    basic_name = "basic_sim_summary_" + str(folder) + ".csv" 
    basic_export = basic_summary
    basic_export.columns = basic_export.columns.str.replace('_', '')
    basic_export = basic_export.round(3)
    basic_export.to_csv("../notebooks/ABC_model/latex/tables/post_process/" +basic_name) 
    
    # # initialize a list to store the results
    # df_diffs = pd.DataFrame()

    # # loop over the columns of df_basic_all
    # for column in basic_summary.columns:

    #     # column name pseudo
    #     column_pseudo = column + '_' + str(folder)
    #     # calculate the percentage difference ERROR
    #     df_diff = abs(pseudo_summary[column_pseudo] - basic_summary[column]) / abs(basic_summary[column])
    #     # add the column name
    #     df_diff.name = column
    #     # append the result to the list
    #     df_diffs = pd.concat([df_diffs,df_diff],axis=1)

    # #print(df_diffs)
    # sim_export_file = 'comparison_ABC/results/moment_diffs_' + str(folder) + '.csv'
    # df_diffs.to_csv(sim_export_file)
    # #df_diffs.to_csv('comparison_ABC/df_diffs.csv')

    # # append to list
    # list_of_moment_dfs.append(df_diffs)

    # # initialize a list to store the results
    # df_diffs = pd.DataFrame()

    # # loop over the columns of df_basic_all
    # for column in df_basic_all.columns:

    #     # column name pseudo
    #     column_pseudo = column + '_' + str(folder)
    #     # calculate the percentage difference ERROR
    #     df_diff = abs(df_pseudo_sim_all[column_pseudo] - df_basic_all[column]) / abs(df_basic_all[column])
    #     # add the column name
    #     df_diff.name = column
    #     # append the result to the list
    #     df_diffs = pd.concat([df_diffs,df_diff],axis=1)
    
    # # export csv for latex
    # df_diffs.columns = df_diffs.columns.str.replace('_', '')
    # df_diffs.describe().round(3).to_csv("../notebooks/ABC_model/latex/tables/post_process/sim_diffs_summary_" + str(folder) + ".csv")

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # compare policies along simulated path
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # read in the policy file
    ABC_sim = pd.read_csv('comparison_ABC/ABC/' +str(folder) + '_results_simpath.csv')

    # drop columns that are not needed
    df_basic_all = df_basic_all.drop(['eq_0','eq_1','eq_2', 'IC_y'], axis=1)
    # initialize a list to store the results
    df_diffs = pd.DataFrame()

    for column in df_basic_all.columns:
        df_diff = abs(ABC_sim[column] - df_basic_all[column]) / abs(df_basic_all[column]) * 100
        df_diff.name = column
        df_diffs = pd.concat([df_diffs,df_diff],axis=1)
    
    # export pointwise csv for check
    df_diffs.to_csv('comparison_ABC/results/policy_comparison/policy_diffs_' + str(folder) + '.csv', index=False)

    # export csv for latex
    df_diffs.columns = df_diffs.columns.str.replace('_', '')
    df_diffs.describe().round(3).to_csv("../notebooks/ABC_model/latex/tables/policy_comparison/pol_diffs_summary_" + str(folder) + ".csv")   

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # compare steady state
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    steady_basic = pd.read_csv('comparison_ABC/ABC_basic/' +str(folder) + '/steady_state.csv').drop(['eq_0','eq_1','eq_2', 'IC_y'], axis=1)
    steady_pseudo = pd.read_csv('comparison_ABC/ABC/' +str(folder) + '_steadystate.csv').drop(['eq_0','eq_1','eq_2', 'IC_y'], axis=1)

    steady_basic.set_index([['basic']*len(steady_basic)], inplace=True)
    steady_pseudo.set_index([['pseudo']*len(steady_pseudo)], inplace=True)

    # save steady state to latex
    steady_export = pd.concat([steady_basic, steady_pseudo])
    steady_export.columns = steady_export.columns.str.replace('_', '')
    steady_export.to_csv("../notebooks/ABC_model/latex/tables/steady_state/"  + str(folder) + "_steadystate.csv")



# irf_description = pd.concat(list_of_irf_dfs, axis=0)
# moments_description = pd.concat(list_of_moment_dfs, axis=0)

# irf_description.describe().to_csv('comparison_ABC/results/irf_diffs_description.csv')
# irf_description.columns = irf_description.columns.str.replace('_', '')
# irf_description.describe().drop('count').round(3).to_csv('../notebooks/ABC_model/latex/tables/irf_diffs_description.csv')
# moments_description.index.name = 'moment'



# average_moment_diff = moments_description.drop(['eq_1','eq_2'], axis=1).groupby('moment').mean()
# max_moment_diff = moments_description.drop(['eq_1','eq_2'], axis=1).groupby('moment').max()

# average_moment_diff.to_csv('comparison_ABC/results/av_moment_diffs.csv')

# average_moment_diff.columns =  average_moment_diff.columns.str.replace('_', '')
# max_moment_diff.columns =  max_moment_diff.columns.str.replace('_', '')
# average_moment_diff.round(3).to_csv('../notebooks/ABC_model/latex/tables/av_moment_diffs.csv')
# max_moment_diff.round(3).to_csv('../notebooks/ABC_model/latex/tables/max_moment_diffs.csv')
