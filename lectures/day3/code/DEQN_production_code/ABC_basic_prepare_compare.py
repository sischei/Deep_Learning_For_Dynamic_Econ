# irfs and post_processing of ABC_basic

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
import os
from datetime import datetime 


import Globals
Globals.POST_PROCESSING=True 

## how many simulation steps to do impulse response
#no_steps = 50

# path wrangling, FOLDER variable needs to be set in the environment
folder_nr = os.environ["FOLDER"]


# Create the folder structure if it doesn't exist

# folder path
folder_path = "comparison_ABC/ABC_basic/" + folder_nr + "/"
os.makedirs(folder_path, exist_ok=True)



## filenames for the different quantities
filename_starting_state = folder_path + "starting_state_ABC_basic.csv"
filename_starting_policy = folder_path + "starting_policy_ABC_basic.csv"

filename_states_simpath   = folder_path + "simulated_states_simpath.csv"
filename_policies_simpath = folder_path + "simulated_policies_simpath.csv"
filename_def_simpath     = folder_path + "simulated_def_simpath.csv"
filename_EE_simpath      = folder_path + "simulated_euler_discrepancies_simpath.csv"

filename_states   = folder_path + "simulated_states_IRS.csv"
filename_policies = folder_path + "simulated_policies_IRS.csv"
filename_def      = folder_path + "simulated_def_IRS.csv"
filename_EE       = folder_path + "simulated_euler_discrepancies_IRS.csv"

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

## make sure we can change function run profiles (e.g. shocks or no shocks)
tf.config.experimental_run_functions_eagerly(True)
tf.get_logger().setLevel('CRITICAL')

#print("alpha value",Definitions.alpha)
## Hooks allows to load in a starting state for the simulation (if a particular choice is needed)
Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
starting_policy = Parameters.policy(Parameters.starting_state)

## Import EE to potentially compute 
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

## simulate from a single starting state which is the mean across the N_sim_batch individual trajectory starting states
simulation_starting_state = tf.math.reduce_mean(Parameters.starting_state, axis = 0, keepdims=True)



# ++++++++++++++++++++++++++++++++++++++++
# take initial states and policies from ABC_basic
# ++++++++++++++++++++++++++++++++++++++++
# write starting state to file
# generate data frames for States, Polices, Definitions, EE
state_episode_df = pd.DataFrame({s:getattr(State,s)(Parameters.starting_state) for s in Parameters.states})
state_episode_df.to_csv(filename_starting_state, index=False)

starting_policy_df = pd.DataFrame({ps:getattr(PolicyState,ps)(starting_policy) for ps in Parameters.policy_states})
starting_policy_df.to_csv(filename_starting_policy, index=False)

# ++++++++++++++++++++++++++++++++++++++++
# simulate to get moments
# ++++++++++++++++++++++++++++++++++++++++
N_simulated_episode_length = 1000
N_simulated_batch_size = 1

state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis = 0), [N_simulated_episode_length, 1, 1])

state_episode = tf.reshape(run_episode(state_episode), [N_simulated_episode_length * N_simulated_batch_size,len(Parameters.states)])
## calculate euler deviations
policy_episode = Parameters.policy(state_episode)
euler_discrepancies = pd.DataFrame(Equations.equations(state_episode, policy_episode))


state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode) for s in Parameters.states})

policy_episode_df = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})

definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(state_episode, policy_episode) for d in Parameters.definitions})

state_episode_df.to_csv(filename_states_simpath, index=False)
policy_episode_df.to_csv(filename_policies_simpath, index=False)
definition_episode_df.to_csv(filename_def_simpath, index=False)
euler_discrepancies.to_csv(filename_EE_simpath, index=False)

##===========================================

## 1. START OF IMPULSE RESPONSE EXPERIMENTS

## get init. distributions of states, polices, etc.
policy_episode = Parameters.policy(simulation_starting_state)
state_episode_df = pd.DataFrame({s:getattr(State,s)(simulation_starting_state) for s in Parameters.states})
policies_step = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
#print(policies_step)    
definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(simulation_starting_state, policy_episode) for d in Parameters.definitions})
euler_discrepancies = pd.DataFrame(Equations.equations(simulation_starting_state, policy_episode))

#append init. distributions to a DataFrame (not necessary for plots)
tmp_state = pd.concat([tmp_state,state_episode_df])
tmp_policies = pd.concat([tmp_policies,policies_step])
tmp_def = pd.concat([tmp_def,definition_episode_df])    
tmp_EE = pd.concat([tmp_EE,euler_discrepancies])

##===========================================
## to just perturb one single state, i.e, apply a single shock: do State.update in the impulse_reponse script, so you can manipulate the "fully" shocked state as you wish (e.g. take only a coordinate from it before doing the non-shocked rounds - and taking the others from the original starting point, or you can take multiple ones as well, etc...)
##===========================================

## 2. START OF IMPULSE RESPONSE EXPERIMENTS; 1 particular shock

## now do impulse response

no_steps = 200 #burn-in-steps
no_steps_decay = 40 #iteration steps after the pulse


#ordinary model dynamics, warm-up before shock
#DISABLE_SCHOCKS=False
#PROD_SHOCK=False
#PREF_SHOCK=False

#no dynamics
Globals.PROD_SHOCK = False
Globals.PREF_SHOCK = False
Globals.DISABLE_SCHOCKS = True

for i in range(no_steps):
    # evaluate the state forward
    simulation_starting_state = do_random_step(simulation_starting_state)
    print("warm-up; shock, step = ", i, simulation_starting_state)    
    
    ## policy at state "simulation_starting_state"
    policy_episode = Parameters.policy(simulation_starting_state)
    
    # generate data frames for States, Polices, Definitions, EE
    state_episode_df = pd.DataFrame({s:getattr(State,s)(simulation_starting_state) for s in Parameters.states})
    #print(state_episode_df)
    policies_step = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
    #print(policies_step)
    definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(simulation_starting_state, policy_episode) for d in Parameters.definitions})
    #print(definition_episode_df)
    euler_discrepancies = pd.DataFrame(Equations.equations(simulation_starting_state, policy_episode))
    #print("Euler discrepancy (absolute value) metrics")
    #print(euler_discrepancies.abs().describe(include='all'))
    
    # save all relevant quantities along the trajectory 
    tmp_state = pd.concat([tmp_state,state_episode_df])
    tmp_policies = pd.concat([tmp_policies,policies_step])
    tmp_def = pd.concat([tmp_def,definition_episode_df])    
    tmp_EE = pd.concat([tmp_EE,euler_discrepancies])

# export steady state values to csv
pd.concat([state_episode_df, policies_step, definition_episode_df, euler_discrepancies], axis=1).to_csv(folder_path + "steady_state.csv", index=False)

# # export tmpfiles
# tmp_state.to_csv(filename_states_simpath, index=False)
# tmp_policies.to_csv(filename_policies_simpath, index=False)
# tmp_def.to_csv(filename_def_simpath, index=False)
# tmp_EE.to_csv(filename_EE_simpath, index=False)

# save initial state after burn-in
initial_state = tmp_state
initial_policies = tmp_policies
initial_def = tmp_def
initial_EE = tmp_EE

# reset tmp to current episode
# tmp_state = state_episode_df
# tmp_policies = policies_step
# tmp_def = definition_episode_df
# tmp_EE = euler_discrepancies

################################################################################

## experiment 1: we shock productivity with a 2 sigma std, then switch off ALL shocks

################################################################################

print("starting state", simulation_starting_state)
# define the stochastic steady-state, before the shock happens
steady_state = tf.squeeze(simulation_starting_state).numpy()
steady_policy = tf.squeeze(policy_episode).numpy()
steady_def = [tf.squeeze(getattr(Definitions,d)(simulation_starting_state, policy_episode)).numpy() for d in Parameters.definitions]

##==========================================
## remind us about the stochastic steady-state:
print("**************************************")
print(f"state sequence {Parameters.states}")
print("stochastic steady state", steady_state)
print(f"policy names {Parameters.policy_states}")
print("stochastic steady state", steady_policy)
print(f"policy names {Parameters.definitions}")
print("stochastic steady state", steady_def)

print("**************************************")

## turn on shocks
print("one timestep with IRS -- ax shock - IRS config")       
Globals.PROD_SHOCK = True
Globals.PREF_SHOCK = False
Globals.DISABLE_SCHOCKS = True

shocked_state = do_random_step(simulation_starting_state) 
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
for i in range(no_steps_decay):
    # evaluate the state forward
    shocked_state = do_random_step(shocked_state)
    print("switched-off shock, step = ", i, shocked_state)    
    
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

################################################################################

## experiment 2: we shock d_t with a 2 sigma std, then switch off ALL shocks

################################################################################


## turn on shocks
print("one timestep with IRS -- dx shock - IRS config")       
Globals.PROD_SHOCK = False
Globals.PREF_SHOCK = True
Globals.DISABLE_SCHOCKS = True

print("starting state", simulation_starting_state)
shocked_state = do_random_step(simulation_starting_state) 
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
tmp_state_2 = pd.concat([initial_state,state_episode_df])
tmp_policies_2 = pd.concat([initial_policies,policies_step])
tmp_def_2 = pd.concat([initial_def,definition_episode_df])    
tmp_EE_2 = pd.concat([initial_EE,euler_discrepancies])
    
##===========================================
## 3. Recover from shock for e.g. 40 steps
# turn off shocks
Globals.PROD_SHOCK = False
Globals.PREF_SHOCK = False
Globals.DISABLE_SCHOCKS = True
for i in range(no_steps_decay):
    # evaluate the state forward
    shocked_state = do_random_step(shocked_state)
    print("switched-off shock, step = ", i, shocked_state)    
    
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
    tmp_state_2 = pd.concat([tmp_state_2,state_episode_df])
    tmp_policies_2 = pd.concat([tmp_policies_2,policies_step])
    tmp_def_2 = pd.concat([tmp_def_2,definition_episode_df])    
    tmp_EE_2 = pd.concat([tmp_EE_2,euler_discrepancies])
    

##===========================================
## 4. Files that contain simulations

# store state along simulated path
pd.concat([tmp_state.add_suffix('_ax',axis=1),tmp_state_2.add_suffix('_dx',axis=1)],axis = 1).to_csv(folder_path + filename_irf_state,index=False)

# store policies along simulated path
pd.concat([tmp_policies.add_suffix('_ax',axis=1),tmp_policies_2.add_suffix('_dx',axis=1)],axis = 1).to_csv(folder_path + filename_irf_policies,index=False)

# store def along simulated path
pd.concat([tmp_def.add_suffix('_ax',axis=1),tmp_def_2.add_suffix('_dx',axis=1)],axis = 1).to_csv(folder_path + filename_irf_def,index=False)

# store EE along simulated path
pd.concat([tmp_EE.add_suffix('_ax',axis=1),tmp_EE_2.add_suffix('_dx',axis=1)],axis = 1).abs().to_csv(folder_path + filename_irf_euler_discrepancies,index=False)

