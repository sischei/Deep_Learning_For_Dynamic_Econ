import importlib
import tensorflow as tf
import Parameters
from Graphs import do_random_step
from Graphs import run_episode
import sys
import pandas as pd
import State
import PolicyState
import Definitions
import matplotlib.pyplot as plt
import numpy as np
from Parameters import *


Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")

import Globals
Globals.POST_PROCESSING=True

tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel('CRITICAL')

pd.set_option('display.max_columns', None)


# filepaths
folder_name = "comparison_ABC/ABC/"

filename_states = "simulated_states.csv"
filename_policies = "simulated_policies.csv"
filename_definitions = "simulated_definitions.csv"
filename_euler_discrepancies = "simulated_euler_discrepancies.csv"

# impulse response filepath
filename_irf_def = "irf_definitions.csv"
filename_irf_state = "irf_states.csv"
filename_irf_policies = "irf_policies.csv"
filename_irf_euler_discrepancies = "irf_euler_discrepancies.csv"


## file that contains all the policies along simulated path
tmp_state = pd.DataFrame()     
## file that contains all the policies along simulated path
tmp_policies = pd.DataFrame() 
## file that contains all the definitions along simulated path
tmp_def = pd.DataFrame() 
#file that contains all the EE along the simulated path       
tmp_EE = pd.DataFrame()

# do one random step to start the simulation in period 1
start_of_simulations = do_random_step(Parameters.starting_state)

starting_policy = Parameters.policy(start_of_simulations)

# import equations to calculate euler discrepancies
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

# read in parameters from csv file
param_df = pd.read_csv("comparison_ABC/alpha_beta_random.csv",usecols=range(1,3)).to_numpy()
N_param_combinations = param_df.shape[0]

N_simulated_episode_length = 1000
N_simulated_batch_size = 1


simulation_starting_state = tf.math.reduce_mean(start_of_simulations, axis = 0, keepdims=True)
#simulation_starting_state = tf.tile(simulation_starting_state, [N_param_combinations,1])

print("Turn off alpha and beta variations...")
Globals.DISABLE_SCHOCKS = True
Globals.PROD_SHOCK = True
Globals.PREF_SHOCK = True
Globals.PSEUDO_SHOCK = False
# +++++++++++++++++++++++++++++++++
# setup loop over combinations of alpha and beta
for i in range(N_param_combinations):
    updates_alpha = tf.constant(param_df[i,0], shape = 1, dtype=tf.float32)
    updates_beta = tf.constant(param_df[i,1], shape = 1,dtype=tf.float32)

    state_episode = State.update(simulation_starting_state, 'alpha_x',updates_alpha)
    state_episode = State.update(state_episode, 'beta_x',updates_beta)

    state_episode = tf.tile(tf.expand_dims(state_episode, axis = 0), [N_simulated_episode_length, 1, 1])

    print("Running episode in iteration ", i, "to get range of variables...")

    state_episode = tf.reshape(run_episode(state_episode), [N_simulated_episode_length * N_simulated_batch_size,len(Parameters.states)])


    ## calculate euler deviations
    policy_episode = Parameters.policy(state_episode)
    euler_discrepancies = pd.DataFrame(Equations.equations(state_episode, policy_episode))
    

    state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode) for s in Parameters.states})

    policy_episode_df = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})

    definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(state_episode, policy_episode) for d in Parameters.definitions})

    # save to latex

    # absolute values of euler discrepancies for export
    euler_discrepancies_abs = euler_discrepancies.abs() * 1000
    pseudo_name = 'pseudo_sim_summary_' + str(i+1) + '.csv'
    df_export = pd.concat([euler_discrepancies_abs, definition_episode_df, state_episode_df, policy_episode_df],axis=1)
    df_export = df_export.describe().round(3).drop(['alpha_x', 'beta_x', 'dummy_x'],axis=1)
    df_export.columns = df_export.columns.str.replace('_', '')
    df_export.to_csv("..//notebooks/ABC_model/latex/tables/post_process/" + pseudo_name) 

    # save all relevant quantities along the trajectory (starting state after burnin)
    tmp_state = pd.concat([tmp_state,state_episode_df.add_suffix("_" + str(i+1))],axis = 1)
    tmp_policies = pd.concat([tmp_policies,policy_episode_df.add_suffix("_" + str(i+1))], axis=1)
    tmp_def = pd.concat([tmp_def,definition_episode_df.add_suffix("_" + str(i+1))], axis=1)  
    tmp_EE = pd.concat([tmp_EE,euler_discrepancies.add_suffix("_" + str(i+1))], axis=1)

     

    # print out descriptives
    print("Iteration", i)
    print("State metrics")
    print(state_episode_df.describe(include='all'))

    print("Policy metrics")
    print(policy_episode_df.describe(include='all'))

    print("Definition metrics")
    print(definition_episode_df.describe(include='all'))

    # +++++++++++++++++++++++++++++++++
    # Now compare policies along a simulated path where we take the states from the basic model simulations
    # +++++++++++++++++++++++++++++++++
    # read in simulated states from basic model
    states_basic = pd.read_csv("comparison_ABC/ABC_basic/" + str(i+1) + "/simulated_states_simpath.csv")

    # prepare states for simulation
    state_episode = State.update(simulation_starting_state, 'alpha_x',updates_alpha)
    state_episode = State.update(state_episode, 'beta_x',updates_beta)

    state_episode = tf.tile(tf.expand_dims(state_episode, axis = 0), [N_simulated_episode_length, 1, 1])
    state_episode = tf.reshape(state_episode, [N_simulated_episode_length * N_simulated_batch_size,len(Parameters.states)])
    # take states from states_basic
    state_episode = State.update(state_episode, 'K_x',tf.constant(states_basic['K_x'].to_numpy(), shape = [N_simulated_episode_length * N_simulated_batch_size], dtype=tf.float32))
    state_episode = State.update(state_episode, 'a_x',states_basic['a_x'].to_numpy())
    state_episode = State.update(state_episode, 'd_x',states_basic['d_x'].to_numpy())

    ## calculate euler deviations
    policy_episode = Parameters.policy(state_episode)
    euler_discrepancies = pd.DataFrame(Equations.equations(state_episode, policy_episode))

    # create data frames
    state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode) for s in Parameters.states})
    policy_episode_df = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
    definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(state_episode, policy_episode) for d in Parameters.definitions})

    # save to csv to folder comparison_ABC/ABC
    export = pd.concat([euler_discrepancies, definition_episode_df, state_episode_df, policy_episode_df],axis=1)
    export.to_csv("comparison_ABC/ABC/"  + str(i+1) + "_results_simpath.csv", index=False)

tmp_state.to_csv(folder_name + filename_states, index=False)
tmp_policies.to_csv(folder_name + filename_policies, index=False)
tmp_def.to_csv(folder_name + filename_definitions, index=False)
tmp_EE.to_csv(folder_name + filename_euler_discrepancies, index=False)

# reset tmp files
## file that contains all the policies along simulated path
tmp_state = pd.DataFrame()     
## file that contains all the policies along simulated path
tmp_policies = pd.DataFrame() 
## file that contains all the definitions along simulated path
tmp_def = pd.DataFrame() 
#file that contains all the EE along the simulated path       
tmp_EE = pd.DataFrame()

irf_df_state = pd.DataFrame()
irf_df_policies = pd.DataFrame()
irf_df_def = pd.DataFrame()
irf_df_EE = pd.DataFrame()

# +++++++++++++++++++++++++++++++++
no_steps = 200 # need to run for a while to get to steady state
no_steps_decay = 40 # number of steps for IRFs

# +++++++++++++++++++++++++++++++++
# now do irfs
# +++++++++++++++++++++++++++++++++

# get to steady state

#no dynamics
Globals.PROD_SHOCK = False
Globals.PREF_SHOCK = False
Globals.DISABLE_SCHOCKS = True

for i in range(N_param_combinations):

    #no dynamics
    Globals.PROD_SHOCK = False
    Globals.PREF_SHOCK = False
    Globals.DISABLE_SCHOCKS = True

    ## file that contains all the policies along simulated path
    tmp_state = pd.DataFrame()     
    ## file that contains all the policies along simulated path
    tmp_policies = pd.DataFrame() 
    ## file that contains all the definitions along simulated path
    tmp_def = pd.DataFrame() 
    #file that contains all the EE along the simulated path       
    tmp_EE = pd.DataFrame()
    # reset starting state
    updates_alpha = tf.constant(param_df[i,0], shape = 1, dtype=tf.float32)
    updates_beta = tf.constant(param_df[i,1], shape = 1,dtype=tf.float32)

    state_episode = State.update(simulation_starting_state, 'alpha_x',updates_alpha)
    state_episode = State.update(state_episode, 'beta_x',updates_beta)

    for j in range(no_steps):
        # evaluate the state forward
        state_episode = do_random_step(state_episode)
        print("warm-up; shock, step = ", j, state_episode)    
        
        policy_episode = Parameters.policy(state_episode)

        # generate data frames for States, Polices, Definitions, EE
        state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode) for s in Parameters.states})
        #print(state_episode_df)
        policies_step = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
        #print(policies_step)
        definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(state_episode, policy_episode) for d in Parameters.definitions})
        #print(definition_episode_df)
        euler_discrepancies = pd.DataFrame(Equations.equations(state_episode, policy_episode))
        #print("Euler discrepancy (absolute value) metrics")
        #print(euler_discrepancies.abs().describe(include='all'))
        
        # save all relevant quantities along the trajectory 
        tmp_state = pd.concat([tmp_state,state_episode_df])
        tmp_policies = pd.concat([tmp_policies,policies_step])
        tmp_def = pd.concat([tmp_def,definition_episode_df])    
        tmp_EE = pd.concat([tmp_EE,euler_discrepancies])
        
    # generate data frames for States, Polices, Definitions, EE
    # state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode) for s in Parameters.states})
    # policies_step = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
    # definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(state_episode, policy_episode) for d in Parameters.definitions})
    # euler_discrepancies = pd.DataFrame(Equations.equations(state_episode, policy_episode)) 

    # # save all relevant quantities along the trajectory (starting state after burnin)
    # tmp_state = pd.concat([tmp_state,state_episode_df])
    # tmp_policies = pd.concat([tmp_policies,policies_step])
    # tmp_def = pd.concat([tmp_def,definition_episode_df])    
    # tmp_EE = pd.concat([tmp_EE,euler_discrepancies])

    # save steady state to csv
    #pd.concat([state_episode_df,policies_step,definition_episode_df,euler_discrepancies],axis=1).to_csv("../notebooks/ABC_model/latex/tables/steady_state/"  + str(i+1) + "_ABC_steadystate.csv", index=False)
    pd.concat([state_episode_df,policies_step,definition_episode_df,euler_discrepancies],axis=1).to_csv("comparison_ABC/ABC/"  + str(i+1) + "_steadystate.csv", index=False)

    # save initial state 
    initial_state_df = tmp_state
    initial_policies_df = tmp_policies
    initial_def_df = tmp_def
    initial_EE_df = tmp_EE

    # save steady state
    initial_state = state_episode


    # +++++++++++++++++++++++++++++++++
    # experiment 1
    # +++++++++++++++++++++++++++++++++


    print("one timestep with IRS -- ax shock - IRS config")       
    Globals.PROD_SHOCK = True
    Globals.PREF_SHOCK = False
    Globals.DISABLE_SCHOCKS = True

    print("starting state", state_episode)
    shocked_state = do_random_step(state_episode) 
    print("shocked state",shocked_state)    

    # policy at state "shocked_state"
    policy_episode = Parameters.policy(shocked_state)    
        # generate data frames for States, Polices, Definitions, EE
    state_episode_df = pd.DataFrame({s:getattr(State,s)(shocked_state) for s in Parameters.states})
        #print(state_episode_df)
    policies_step = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
        #print(policies_step)
    definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(shocked_state, policy_episode) for d in Parameters.definitions})
        #print(definition_episode_df)
    euler_discrepancies = pd.DataFrame(Equations.equations(shocked_state, policy_episode))
        #print("Euler discrepancy (absolute value) metrics")
        #print(euler_discrepancies.abs().describe(include='all'))
        # save all relevant quantities along the trajectory 
    tmp_state = pd.concat([tmp_state,state_episode_df])
    tmp_policies = pd.concat([tmp_policies,policies_step])
    tmp_def = pd.concat([tmp_def,definition_episode_df])    
    tmp_EE = pd.concat([tmp_EE,euler_discrepancies])
        
    ##===========================================
    ## 3. Recover from shock for e.g. 40 steps
    # turn off shocks
    Globals.PROD_SHOCK = False
    Globals.PREF_SHOCK = False
    Globals.DISABLE_SCHOCKS = True
    for j in range(no_steps_decay):
        # evaluate the state forward
        shocked_state = do_random_step(shocked_state)
        print("switched-off shock, step = ", j, shocked_state)    
        
        ## policy at state "shocked_state"
        policy_episode = Parameters.policy(shocked_state)
        
        # generate data frames for States, Polices, Definitions, EE
        state_episode_df = pd.DataFrame({s:getattr(State,s)(shocked_state) for s in Parameters.states})
        #print(state_episode_df)
        policies_step = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
        #print(policies_step)
        definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(shocked_state, policy_episode) for d in Parameters.definitions})
        #print(definition_episode_df)
        euler_discrepancies = pd.DataFrame(Equations.equations(shocked_state, policy_episode))
        #print("Euler discrepancy (absolute value) metrics")
        #print(euler_discrepancies.abs().describe(include='all'))
        
        # save all relevant quantities along the trajectory 
        tmp_state = pd.concat([tmp_state,state_episode_df])
        tmp_policies = pd.concat([tmp_policies,policies_step])
        tmp_def = pd.concat([tmp_def,definition_episode_df])    
        tmp_EE = pd.concat([tmp_EE,euler_discrepancies])

    if i == 0:
        irf_df_state = tmp_state.add_suffix("_"+"ax"+"_" + str(i+1))
        irf_df_policies = tmp_policies.add_suffix("_"+"ax"+"_" + str(i+1))
        irf_df_def = tmp_def.add_suffix("_"+"ax"+"_" + str(i+1))
        irf_df_EE = tmp_EE.add_suffix("_"+"ax"+"_" + str(i+1))
    else:
        # concat tmp_state to irf_df along axis 1
        irf_df_state = pd.concat([irf_df_state.reset_index(drop=True),tmp_state.add_suffix("_"+"ax"+"_" + str(i+1)).reset_index(drop=True)], axis = 1)
        irf_df_policies = pd.concat([irf_df_policies.reset_index(drop=True),tmp_policies.add_suffix("_"+"ax"+"_" + str(i+1)).reset_index(drop=True)], axis = 1)
        irf_df_def = pd.concat([irf_df_def.reset_index(drop=True),tmp_def.add_suffix("_"+"ax"+"_" + str(i+1)).reset_index(drop=True)], axis = 1)
        irf_df_EE = pd.concat([irf_df_EE.reset_index(drop=True),tmp_EE.add_suffix("_"+"ax"+"_" + str(i+1)).reset_index(drop=True)], axis = 1)

    # +++++++++++++++++++++++++++++++++
    # experiment 2
    # +++++++++++++++++++++++++++++++++
    # reset tmp files
    ## file that contains all the policies along simulated path
    tmp_state = initial_state_df 
    ## file that contains all the policies along simulated path
    tmp_policies = initial_policies_df
    ## file that contains all the definitions along simulated path
    tmp_def = initial_def_df
    #file that contains all the EE along the simulated path       
    tmp_EE = initial_EE_df

    # generate data frames for States, Polices, Definitions, EE
    # state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode) for s in Parameters.states})
    # policies_step = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
    # definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(state_episode, policy_episode) for d in Parameters.definitions})
    # euler_discrepancies = pd.DataFrame(Equations.equations(state_episode, policy_episode)) 

    # save all relevant quantities along the trajectory (starting state after burnin)
    # tmp_state = pd.concat([initial_state,state_episode_df])
    # tmp_policies = pd.concat([initial_policies,policies_step])
    # tmp_def = pd.concat([initial_def,definition_episode_df])    
    # tmp_EE = pd.concat([initial_EE,euler_discrepancies])

    

    print("one timestep with IRS -- dx shock - IRS config")       
    Globals.PROD_SHOCK = False
    Globals.PREF_SHOCK = True
    Globals.DISABLE_SCHOCKS = True

    print("starting state", state_episode)
    shocked_state = do_random_step(state_episode) 
    print("shocked state",shocked_state)    

    # policy at state "shocked_state"
    policy_episode = Parameters.policy(shocked_state)    
        # generate data frames for States, Polices, Definitions, EE
    state_episode_df = pd.DataFrame({s:getattr(State,s)(shocked_state) for s in Parameters.states})
        #print(state_episode_df)
    policies_step = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
        #print(policies_step)
    definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(shocked_state, policy_episode) for d in Parameters.definitions})
        #print(definition_episode_df)
    euler_discrepancies = pd.DataFrame(Equations.equations(shocked_state, policy_episode))
        #print("Euler discrepancy (absolute value) metrics")
        #print(euler_discrepancies.abs().describe(include='all'))
        # save all relevant quantities along the trajectory 
    tmp_state = pd.concat([tmp_state,state_episode_df])
    tmp_policies = pd.concat([tmp_policies,policies_step])
    tmp_def = pd.concat([tmp_def,definition_episode_df])    
    tmp_EE = pd.concat([tmp_EE,euler_discrepancies])

    ##===========================================
    ## 3. Recover from shock for e.g. 40 steps
    # turn off shocks
    Globals.PROD_SHOCK = False
    Globals.PREF_SHOCK = False
    Globals.DISABLE_SCHOCKS = True
    for j in range(no_steps_decay):
        # evaluate the state forward
        shocked_state = do_random_step(shocked_state)
        print("switched-off shock, step = ", j, shocked_state)    
        
        ## policy at state "shocked_state"
        policy_episode = Parameters.policy(shocked_state)
        
        # generate data frames for States, Polices, Definitions, EE
        state_episode_df = pd.DataFrame({s:getattr(State,s)(shocked_state) for s in Parameters.states})
        #print(state_episode_df)
        policies_step = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
        #print(policies_step)
        definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(shocked_state, policy_episode) for d in Parameters.definitions})
        #print(definition_episode_df)
        euler_discrepancies = pd.DataFrame(Equations.equations(shocked_state, policy_episode))
        #print("Euler discrepancy (absolute value) metrics")
        #print(euler_discrepancies.abs().describe(include='all'))
        
        # save all relevant quantities along the trajectory 
        tmp_state = pd.concat([tmp_state,state_episode_df])
        tmp_policies = pd.concat([tmp_policies,policies_step])
        tmp_def = pd.concat([tmp_def,definition_episode_df])    
        tmp_EE = pd.concat([tmp_EE,euler_discrepancies])

    # concat tmp_state to irf_df along axis 1
    irf_df_state = pd.concat([irf_df_state.reset_index(drop=True),tmp_state.add_suffix("_"+"dx"+"_" + str(i+1)).reset_index(drop=True)], axis = 1)
    irf_df_policies = pd.concat([irf_df_policies.reset_index(drop=True),tmp_policies.add_suffix("_"+"dx"+"_" + str(i+1)).reset_index(drop=True)], axis = 1)
    irf_df_def = pd.concat([irf_df_def.reset_index(drop=True),tmp_def.add_suffix("_"+"dx"+"_" + str(i+1)).reset_index(drop=True)], axis = 1)
    irf_df_EE = pd.concat([irf_df_EE.reset_index(drop=True),tmp_EE.add_suffix("_"+"dx"+"_" + str(i+1)).reset_index(drop=True)], axis = 1)


        

# save irf_df to csv
irf_df_state.to_csv(folder_name + filename_irf_state, index=False)
irf_df_policies.to_csv(folder_name + filename_irf_policies, index=False)
irf_df_def.to_csv(folder_name + filename_irf_def, index=False)
irf_df_EE.to_csv(folder_name + filename_irf_euler_discrepancies, index=False)

# updates_alpha = tf.constant(param_df[:,0], dtype=tf.float32)

# updates_beta = tf.constant(param_df[:,1], dtype=tf.float32)


# simulation_starting_state = State.update(simulation_starting_state, 'alpha_x',updates_alpha)
# simulation_starting_state = State.update(simulation_starting_state, 'beta_x',updates_beta)

# # take initial state and fix alpha_x and beta_x
# #state_episode = tf.expand_dims(simulation_starting_state, axis = 1)
# state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis = 1), [N_simulated_episode_length, 1, 1])
# print("state episode before simulation",state_episode)

# print("Turn off alpha and beta variations...")
# Globals.DISABLE_SCHOCKS = True
# Globals.PROD_SHOCK = True
# Globals.PREF_SHOCK = True

# print("Running episode to get range of variables...")


# state_episode = tf.reshape(run_episode(state_episode), [N_param_combinations * N_simulated_episode_length * N_simulated_batch_size,len(Parameters.states)])

# print("state episode:",state_episode)

# # Extract columns 4 and 5
# columns_4_and_5 = state_episode[:, 3:5]

# # Convert the columns to a NumPy array
# numpy_array = columns_4_and_5.numpy()

# # Find unique combinations of columns 4 and 5
# unique_combinations, indices = np.unique(numpy_array, axis=0, return_inverse=True)
# print(unique_combinations)
# # Split the original tensor based on unique combinations
# split_tensors = [state_episode[indices == i] for i in range(len(unique_combinations))]
# print(split_tensors)
# # print("Finished plots. Calculating Euler discrepancies...")




# ## calculate euler deviations
# policy_episode = Parameters.policy(state_episode)
# euler_discrepancies = pd.DataFrame(Equations.equations(state_episode, policy_episode))
# print("Euler discrepancy (absolute value) metrics")
# print(euler_discrepancies.abs().describe(include='all'))

# # save all relevant quantities along the trajectory 
# euler_discrepancies.to_csv(folder_name + filename_euler_discrepancies, index=False)

# state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode) for s in Parameters.states})
# state_episode_df.to_csv(folder_name + filename_states, index=False)

# policy_episode_df = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
# policy_episode_df.to_csv(folder_name + filename_policies, index=False)

# definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(state_episode, policy_episode) for d in Parameters.definitions})
# definition_episode_df.to_csv(folder_name + filename_definitions, index=False)

# print("State metrics")
# print(state_episode_df.describe(include='all'))

# print("Policy metrics")
# print(policy_episode_df.describe(include='all'))

# print("Definition metrics")
# print(definition_episode_df.describe(include='all'))